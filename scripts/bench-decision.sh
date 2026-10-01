#!/usr/bin/env bash
# Decision-model CPU bench (DECISION.md, "Benchmark"): llama-decision-bench per model and
# thread count (K1 latency by group, K4 memory, K7 determinism, K9 CPU seconds, in-process
# K6), then llama-server --decision per model and thread count (K5 time to ready, K6 first
# request after idle over HTTP). Machine facts (CPU, power source, load) go next to the
# numbers; scripts/bench-decision-report.py turns a result folder into tables.
#
# Usage:
#   BIN_DIR=./build/bin MODELS="models-local/laya-f16.gguf models-local/laya-q8_0.gguf" bash scripts/bench-decision.sh
#   python3 scripts/bench-decision-report.py build/bench-decision
#
# Env:
#   BIN_DIR        directory with llama-decision-bench and llama-server   [./build/bin]
#   MODELS         space-separated decision GGUF paths                     (required)
#   THREADS        engine thread counts                                    [4 8]
#   SUITE          suite JSONL                                             [tests/decision/bench/suite.jsonl]
#   REPEAT         measured runs of every request                          [10]
#   WARMUP         unmeasured runs of every request first                  [1]
#   IDLE_S         idle seconds before the K6 request; 0 skips K6          [60]
#   SERVER         1: also measure llama-server --decision (K5, K6)        [1]
#   SERVER_THREADS thread counts of the server runs                        [$THREADS]
#   PORT           server port                                             [8093]
#   PLAN, SPEC     --decision-plan / --decision-spec for both tools        (empty: from the model)
#   KERNELS        --kernels / --decision-kernels: auto, default, repack,   (empty: from the model,
#                  blas, repack+blas (kernels change the logits slightly)    then auto)
#   LABEL          build label used in file names                          [local]
#   OUT_DIR        where the results go                                    [build/bench-decision]
#
# Close other apps first: the load average and the top CPU users are recorded before every
# run, and a busy machine makes p95 meaningless.
#
# Every result JSON gets an "identity" key (tests/laya/parity/identity.py): sha256 of the build
# tree (the tools, libggml* / libllama*, CMakeCache.txt), of the GGUF and of the suite, and the
# runtime (the bench result or the server's /props). bench-decision-report.py prints a speed row
# only when a passing parity gate of the same identity is given (--parity; DECISION.md "GPU
# backends"). CPU vs GPU comparisons: scripts/bench-decision-device.py (paired blocks).

set -uo pipefail

BIN_DIR="${BIN_DIR:-./build/bin}"
THREADS="${THREADS:-4 8}"
SUITE="${SUITE:-tests/decision/bench/suite.jsonl}"
REPEAT="${REPEAT:-10}"
WARMUP="${WARMUP:-1}"
IDLE_S="${IDLE_S:-60}"
SERVER="${SERVER:-1}"
SERVER_THREADS="${SERVER_THREADS:-$THREADS}"
PORT="${PORT:-8093}"
PLAN="${PLAN:-}"
SPEC="${SPEC:-}"
KERNELS="${KERNELS:-}"
LABEL="${LABEL:-local}"
OUT_DIR="${OUT_DIR:-build/bench-decision}"

export LC_ALL=C

if [[ -z "${MODELS:-}" ]]; then
  echo "error: MODELS is empty (space-separated GGUF paths)" >&2
  exit 1
fi
if [[ ! -x "$BIN_DIR/llama-decision-bench" ]]; then
  echo "error: $BIN_DIR/llama-decision-bench not found. Run:  cmake --build build --target llama-decision-bench llama-server" >&2
  exit 1
fi
if [[ "$SERVER" == 1 && ! -x "$BIN_DIR/llama-server" ]]; then
  echo "error: $BIN_DIR/llama-server not found (or set SERVER=0)" >&2
  exit 1
fi
if [[ ! -f "$SUITE" ]]; then
  echo "error: no $SUITE. Run:  python3 tests/decision/bench/gen_suite.py" >&2
  exit 1
fi
for m in $MODELS; do
  [[ -f "$m" ]] || { echo "error: model $m not found" >&2; exit 1; }
done
IDENTITY_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tests/laya/parity/identity.py"
[[ -f "$IDENTITY_PY" ]] || { echo "error: $IDENTITY_PY not found" >&2; exit 1; }
for tool in curl perl python3; do
  command -v "$tool" >/dev/null 2>&1 || { echo "error: $tool is needed" >&2; exit 1; }
done

HOST_TAG="$(hostname -s 2>/dev/null || hostname)"
STAMP="$(date -u +%Y%m%d-%H%M)"
# double underscores: bench-decision-report.py splits file names on "__"
PREFIX="$OUT_DIR/${HOST_TAG}__${LABEL}__${STAMP}"
FACTS="${PREFIX}__machine.txt"
mkdir -p "$OUT_DIR"

now() { perl -MTime::HiRes=time -e 'printf("%.6f\n", time)'; }

power_source() {
  case "$(uname -s)" in
    Darwin)
      pmset -g batt 2>/dev/null | head -1 | sed -e "s/.*'\(.*\)'.*/\1/" ;;
    Linux)
      if command -v upower >/dev/null 2>&1; then
        upower -i "$(upower -e | grep -m1 -E 'line_power|AC')" 2>/dev/null | awk '/online/ {print ($2 == "yes" ? "AC Power" : "Battery Power")}'
      else
        for s in /sys/class/power_supply/A*/online; do [[ -f "$s" ]] && { [[ "$(cat "$s")" == 1 ]] && echo "AC Power" || echo "Battery Power"; break; }; done
      fi ;;
    MINGW*|MSYS*|CYGWIN*)
      powercfg /getactivescheme 2>/dev/null | tr -d '\r'
      powershell -NoProfile -Command "(Get-CimInstance -ClassName BatteryStatus -Namespace root/wmi -ErrorAction SilentlyContinue).PowerOnline" 2>/dev/null | tr -d '\r' ;;
  esac
}

load_avg() {
  if [[ "$(uname -s)" == Darwin ]]; then sysctl -n vm.loadavg | tr -d '{}' | xargs; else cut -d' ' -f1-3 /proc/loadavg 2>/dev/null; fi
}

top_cpu() {
  if [[ "$(uname -s)" == Darwin ]]; then ps -Ao pcpu=,comm= -r; else ps -eo pcpu=,comm= --sort=-pcpu 2>/dev/null; fi |
    head -5 | awk '{p=$1; $1=""; n=split($0, a, "/"); print "    " p " " a[n]}'
}

# rss KB of a pid; macOS also phys_footprint KB (the number Activity Monitor shows)
proc_mem() {
  local pid=$1 rss fp=""
  rss=$(ps -o rss= -p "$pid" 2>/dev/null | xargs)
  if [[ "$(uname -s)" == Darwin ]] && command -v footprint >/dev/null 2>&1; then
    fp=$(footprint -p "$pid" 2>/dev/null | awk '/Footprint:/ {v=$(NF-5); u=$(NF-4); if (u=="MB") v*=1024; if (u=="GB") v*=1048576; print int(v); exit}')
  fi
  echo "${rss:-0} ${fp:-0}"
}

# machine facts, "key: value" lines (read by bench-decision-report.py)
{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "label: $LABEL"
  echo "bin_dir: $BIN_DIR"
  echo "git: $(git describe --always --dirty 2>/dev/null) $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "kernel: $(uname -srm)"
  case "$(uname -s)" in
    Darwin)
      echo "os: $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
      echo "cpu: $(sysctl -n machdep.cpu.brand_string)"
      echo "cores: $(sysctl -n hw.physicalcpu) ($(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null) P + $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null) E)"
      echo "memory_gb: $(( $(sysctl -n hw.memsize) / 1073741824 ))"
      echo "low_power_mode: $(pmset -g 2>/dev/null | awk '/lowpowermode|powermode/ {print $2; exit}')"
      echo "thermal: $(pmset -g therm 2>/dev/null | grep -v '^$' | tr '\n' ' ')"
      echo "battery: $(pmset -g batt 2>/dev/null | sed -n 2p | xargs)" ;;
    Linux)
      echo "os: $(. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME")"
      echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
      echo "cores: $(nproc) logical"
      echo "memory_gb: $(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1048576 ))"
      echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" ;;
    *)
      echo "cpu: $(powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name" 2>/dev/null | tr -d '\r')" ;;
  esac
  echo "power_source: $(power_source | xargs)"
  echo "loadavg_start: $(load_avg)"
  echo "threads: $THREADS"
  echo "server_threads: $SERVER_THREADS"
  echo "suite: $SUITE ($(wc -l < "$SUITE" | xargs) requests)"
  echo "repeat: $REPEAT  warmup: $WARMUP  idle_s: $IDLE_S  plan: ${PLAN:-spec}  spec: ${SPEC:-gguf}  kernels: ${KERNELS:-spec}"
  echo "models:"
  for m in $MODELS; do echo "  - $m ($(du -h "$m" 2>/dev/null | cut -f1))"; done
  echo "top_cpu_start:"
  top_cpu
} > "$FACTS"
echo "wrote $FACTS"

TOOL_ARGS=()
[[ -n "$PLAN" ]] && TOOL_ARGS+=(--plan "$PLAN")
[[ -n "$SPEC" ]] && TOOL_ARGS+=(--spec "$SPEC")
[[ -n "$KERNELS" ]] && TOOL_ARGS+=(--kernels "$KERNELS")

rc_all=0

# in-process: K1, K4, K7, K9 and K6 without HTTP
for m in $MODELS; do
  mname="$(basename "$m" .gguf)"
  for t in $THREADS; do
    out="${PREFIX}__${mname}__t${t}.json"
    echo "run ${mname} t${t}: loadavg $(load_avg), power $(power_source | xargs)" >> "$FACTS"
    echo "== llama-decision-bench $mname -t $t -> $out"
    "$BIN_DIR/llama-decision-bench" -m "$m" -f "$SUITE" -t "$t" --repeat "$REPEAT" --warmup "$WARMUP" \
      --idle-ms $(( IDLE_S * 1000 )) -o "$out" ${TOOL_ARGS[@]+"${TOOL_ARGS[@]}"} 2>&1 | tee "${out%.json}.log"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
      echo "  llama-decision-bench failed (rc=$rc), see ${out%.json}.log" >&2
      rc_all=$rc
    fi
    if [[ -f "$out" ]]; then
      # llama-server and llama-laya-cli too: the parity runs of the same tree used them
      exes=(--exe "$BIN_DIR/llama-decision-bench")
      for x in llama-server llama-laya-cli; do [[ -x "$BIN_DIR/$x" ]] && exes+=(--exe "$BIN_DIR/$x"); done
      python3 "$IDENTITY_PY" merge "$out" --kind bench "${exes[@]}" --model "$m" --corpus "$SUITE" || rc_all=1
    fi
  done
done

# one request body from the suite
suite_body() {
  python3 -c 'import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == sys.argv[2]:
        sys.stdout.write(json.dumps(r["body"], ensure_ascii=False))
        break' "$SUITE" "$1"
}

# curl one request: prints "<http code> <seconds>" and keeps the body in $2
post() {
  curl -s -o "$2" -w '%{http_code} %{time_total}\n' --max-time 60 -H 'content-type: application/json' \
    --data-binary @"$3" "http://127.0.0.1:$PORT$1"
}

# over HTTP: K5 time to ready, K6 first request after idle
if [[ "$SERVER" == 1 ]]; then
  tmp="$(mktemp -d)"
  suite_body so-1q-noul > "$tmp/so1.json"
  suite_body rt-n4-meeting > "$tmp/rt4.json"
  for m in $MODELS; do
    mname="$(basename "$m" .gguf)"
    for t in $SERVER_THREADS; do
      out="${PREFIX}__${mname}__t${t}__server.json"
      log="${out%.json}.log"
      echo "server ${mname} t${t}: loadavg $(load_avg), power $(power_source | xargs)" >> "$FACTS"
      echo "== llama-server --decision $mname -t $t -> $out"
      t0=$(now)
      "$BIN_DIR/llama-server" --decision -m "$m" --device none -t "$t" --host 127.0.0.1 --port "$PORT" \
        --decision-allow-uncalibrated ${PLAN:+--decision-plan "$PLAN"} ${SPEC:+--decision-spec "$SPEC"} ${KERNELS:+--decision-kernels "$KERNELS"} > "$log" 2>&1 &
      pid=$!
      t_listen="" t_ready=""
      for _ in $(seq 1 3000); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 1 "http://127.0.0.1:$PORT/health" 2>/dev/null)
        [[ -z "$t_listen" && "$code" != 000 ]] && t_listen=$(now)
        if [[ "$code" == 200 ]]; then t_ready=$(now); break; fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.01
      done
      if [[ -z "$t_ready" ]]; then
        echo "  server did not get ready, see $log" >&2
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        rc_all=1
        continue
      fi
      read -r rss_ready fp_ready <<< "$(proc_mem "$pid")"
      first_so1=$(post /v1/systemone "$tmp/first.json" "$tmp/so1.json")
      # warm: steady-state latency of both requests
      warm_so1="" warm_rt4=""
      for _ in 1 2 3 4 5; do
        warm_so1+="$(post /v1/systemone "$tmp/r.json" "$tmp/so1.json" | cut -d' ' -f2) "
        warm_rt4+="$(post /v1/router/score "$tmp/r.json" "$tmp/rt4.json" | cut -d' ' -f2) "
      done
      idle_so1="" idle_rt4=""
      if [[ "$IDLE_S" -gt 0 ]]; then
        sleep "$IDLE_S"
        idle_so1=$(post /v1/systemone "$tmp/idle1.json" "$tmp/so1.json")
        sleep "$IDLE_S"
        idle_rt4=$(post /v1/router/score "$tmp/idle4.json" "$tmp/rt4.json")
      fi
      read -r rss_end fp_end <<< "$(proc_mem "$pid")"
      curl -s --max-time 2 -o "$tmp/props.json" "http://127.0.0.1:$PORT/props"
      kernels=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["decision"]["plan"].get("kernels", ""))' "$tmp/props.json" 2>/dev/null)
      kill -INT "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      python3 - "$out" "$m" "$t" "$t0" "$t_listen" "$t_ready" "$rss_ready" "$fp_ready" "$rss_end" "$fp_end" \
        "$first_so1" "$warm_so1" "$warm_rt4" "$idle_so1" "$idle_rt4" "$IDLE_S" "$tmp" "$kernels" <<'EOF'
import json, os, sys
(out, model, t, t0, t_listen, t_ready, rss_ready, fp_ready, rss_end, fp_end,
 first_so1, warm_so1, warm_rt4, idle_so1, idle_rt4, idle_s, tmp, kernels) = sys.argv[1:]

def req(line, body):
    if not line:
        return None
    code, sec = line.split()
    r = {"http": int(code), "total_ms": float(sec) * 1000}
    try:
        b = json.load(open(os.path.join(tmp, body)))
        r["latency_ms"] = b.get("latency_ms")
        r["compute_ms"] = (b.get("timings") or {}).get("compute_ms")
    except Exception:
        pass
    return r

json.dump({
    "kind": "server", "model": model, "n_threads": int(t), "kernels": kernels or None,
    "listen_ms": (float(t_listen) - float(t0)) * 1000 if t_listen else None,
    "ready_ms": (float(t_ready) - float(t0)) * 1000,
    "rss_kb_ready": int(rss_ready), "footprint_kb_ready": int(fp_ready),
    "rss_kb_end": int(rss_end), "footprint_kb_end": int(fp_end),
    "first_so1": req(first_so1, "first.json"),
    "warm_so1_ms": [float(x) * 1000 for x in warm_so1.split()],
    "warm_rt4_ms": [float(x) * 1000 for x in warm_rt4.split()],
    "idle_s": int(idle_s),
    "idle_so1": req(idle_so1, "idle1.json"),
    "idle_rt4": req(idle_rt4, "idle4.json"),
}, open(out, "w"), indent=1)
print("  ready %.0f ms, rss %.0f MiB, footprint %.0f MiB" % ((float(t_ready) - float(t0)) * 1000, int(rss_ready) / 1024, int(fp_ready) / 1024), flush=True)
EOF
      if [[ -s "$tmp/props.json" ]]; then
        python3 "$IDENTITY_PY" merge "$out" --kind server --exe "$BIN_DIR/llama-server" --model "$m" --corpus "$SUITE" --props "$tmp/props.json" || rc_all=1
      else
        python3 "$IDENTITY_PY" merge "$out" --kind server --exe "$BIN_DIR/llama-server" --model "$m" --corpus "$SUITE" --threads "$t" || rc_all=1
      fi
    done
  done
  rm -rf "$tmp"
fi

echo "loadavg_end: $(load_avg)" >> "$FACTS"
echo "power_source_end: $(power_source | xargs)" >> "$FACTS"
echo "top_cpu_end:" >> "$FACTS"
top_cpu >> "$FACTS"
echo "done: ${PREFIX}*  (tables: python3 scripts/bench-decision-report.py $OUT_DIR)"
exit $rc_all
