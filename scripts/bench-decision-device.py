#!/usr/bin/env python3
"""Paired CPU vs GPU bench of the decision engine (DECISION.md, "GPU backends").

Usage:
    python3 scripts/bench-decision-device.py --model models/laya-f16.gguf [--model ...] \
        --config "cpu16 bin=build-cpu/bin device=cpu threads=16 kernels=blas" \
        --config "cuda  bin=build-cuda/bin device=gpu gpu=0 threads=4" \
        [--blocks 3] [--repeat 10] [--warmup 1] [--ready 3] [--out build/bench-device] [--label NAME]
    python3 scripts/bench-decision-report.py build/bench-device --parity <gate JSONs or folders>

A config is a name and key=value fields: bin (folder with llama-decision-bench and llama-server),
device (cpu / gpu / auto), gpu (index for gpu / auto), threads, kernels, precision, plan, and
env=NAME=VALUE (repeatable; for example VK_DRIVER_FILES or GGML_VK_VISIBLE_DEVICES).

Why paired blocks: the clocks, the load of other processes and the GPU boost state drift over
minutes, so a CPU block measured at 03:00 and a GPU block at 03:20 do not compare. Each block is
one llama-decision-bench process (warmup + repeat runs of every suite request); the configs run
one after the other inside a cycle, and the order flips every cycle (A B C, C B A, A B C, ...), so
a slow drift hits every config alike. The report pairs the blocks of one cycle.

POSIX only (Linux, macOS): it reads the load average and stops the chat server and its process group
with SIGINT; on Windows it exits with a message.

Disturbed blocks are measured again (up to --retries times): a block is disturbed when the 1-minute
load average before it is above --load-max, when other processes used more than --foreign-max CPU
cores on average during it (Linux: the CPU time of this cgroup, or of the machine from /proc/stat,
minus the bench process's own rusage; macOS: the ps %cpu of other processes), or when another process holds a CUDA context
(nvidia-smi). Every attempt is kept in the JSON ("ab.attempts").

Recorded per block: the bench JSON (latency per request and group, RSS / footprint, load time,
logits hash), load average before / after, foreign CPU, GPU utilization, SM / memory clocks, power
and the GPU memory of the bench process (nvidia-smi, sampled every 200 ms; on macOS ioreg gives
the GPU utilization and the memory the GPU driver has in use), and the identity of the run
(tests/laya/parity/identity.py: build tree incl. llama-server / llama-laya-cli, GGUF sha256,
device, kernels, precision). scripts/bench-decision-report.py prints a speed row only when a
passing parity gate JSON covers that identity.

Time to ready (--ready N per config and model, interleaved): spawn llama-server --decision ->
GET /health 200 (10 ms polling), then one so-1q-noul request; RSS and GPU memory at ready and
/props (placement, weights in device memory).

--background CMD runs a shell command for the whole session (restarted when it exits); its processes
do not count as foreign load.

--chat-model GGUF is the app case: a chat model generating on the same GPU for the whole session.
The script starts llama-server with that model (--chat-bin, default the first config's bin; all
layers on the GPU), measures --chat-alone completions with nothing else running, then keeps
issuing completions (--chat-tokens tokens each, ignore_eos, no prompt cache) until the end. Every
completion's generation speed (timings.predicted_per_second) is kept with its start and end time
(*__chat.json), so the report can give the chat speed during the blocks of each config next to
its speed alone. The chat server does not count as foreign load.
"""

import argparse
import json
import os
import platform
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests", "laya", "parity"))
import identity as ident  # noqa: E402

TOOLS = ("llama-decision-bench", "llama-server", "llama-laya-cli")


def log(*a):
    print(*a, flush=True)


def parse_config(text):
    parts = shlex.split(text)
    if not parts or "=" in parts[0]:
        raise ValueError("config needs a name first: %r" % text)
    c = {"name": parts[0], "device": "cpu", "gpu": None, "threads": None, "kernels": None, "precision": None,
         "plan": None, "bin": None, "env": {}}
    for p in parts[1:]:
        k, sep, v = p.partition("=")
        if not sep:
            raise ValueError("config %s: expected key=value, got %r" % (c["name"], p))
        if k == "env":
            ek, sep2, ev = v.partition("=")
            if not sep2:
                raise ValueError("config %s: env needs NAME=VALUE" % c["name"])
            c["env"][ek] = ev
        elif k in c:
            c[k] = v
        else:
            raise ValueError("config %s: unknown key %s" % (c["name"], k))
    if not c["bin"]:
        raise ValueError("config %s: bin= is required" % c["name"])
    if c["device"] not in ("cpu", "gpu", "auto"):
        raise ValueError("config %s: device must be cpu, gpu or auto" % c["name"])
    if not re.match(r"^[A-Za-z0-9.+-]+$", c["name"]):
        raise ValueError("config name %r: letters, digits, '.', '+', '-' only (file names split on '__')" % c["name"])
    return c


#
# machine state
#

def load_avg():
    if not hasattr(os, "getloadavg"):
        return None
    try:
        return list(os.getloadavg())
    except OSError:
        return None


def proc_stat_busy():
    """Busy CPU seconds of the whole machine (Linux), None elsewhere."""
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
    except OSError:
        return None
    hz = os.sysconf("SC_CLK_TCK")
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return (sum(v[:8]) - idle) / hz


def cgroup_cpu_seconds():
    """CPU seconds used by this container / cgroup (cgroup v2 cpu.stat, v1 cpuacct), None elsewhere."""
    try:
        with open("/sys/fs/cgroup/cpu.stat") as f:
            for line in f:
                if line.startswith("usage_usec"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/cpuacct/cpuacct.usage") as f:
            return int(f.read()) / 1e9
    except (OSError, ValueError):
        return None


def ps_cpu(exclude):
    """Sum of the ps %cpu of processes outside `exclude` (cores), macOS fallback."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,%cpu="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    tot = 0.0
    for line in out.splitlines():
        f = line.split()
        if len(f) == 2 and int(f[0]) not in exclude:
            tot += float(f[1])
    return tot / 100


def descendants(pids):
    """pids plus every descendant (ps), so a background command's children are excluded too."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return set(pids)
    kids = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) == 2:
            kids.setdefault(int(f[1]), []).append(int(f[0]))
    seen, todo = set(), list(pids)
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        todo += kids.get(p, [])
    return seen


def have(cmd):
    return any(os.access(os.path.join(d, cmd), os.X_OK) for d in os.environ.get("PATH", "").split(os.pathsep))


def nvidia_facts():
    if not have("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit,pcie.link.gen.max,pcie.link.width.max",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    keys = ["index", "name", "driver", "memory_total_mib", "max_sm_mhz", "max_mem_mhz", "power_limit_w", "pcie_gen", "pcie_width"]
    return [dict(zip(keys, [x.strip() for x in line.split(",")])) for line in out.strip().splitlines()]


def nvidia_apps():
    """[(pid, used MiB)] of processes with a CUDA context."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    apps = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 2 and f[0].isdigit():
            apps.append((int(f[0]), float(f[1]) if f[1].replace(".", "").isdigit() else 0.0))
    return apps


IOREG_KEYS = ("Device Utilization %", "Renderer Utilization %", "In use system memory", "Alloc system memory")


def ioreg_sample():
    """macOS GPU statistics (no sudo): utilization and driver memory from IOAccelerator."""
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    s = {}
    for k in IOREG_KEYS:
        m = re.search(r'"%s"=(\d+)' % re.escape(k), out)
        if m:
            s[k] = int(m.group(1))
    return s or None


class GpuSampler:
    """Samples the GPUs every `period` s while a block runs (nvidia-smi or ioreg)."""

    def __init__(self, period=0.2):
        self.period = period
        self.kind = "nvidia" if have("nvidia-smi") else ("ioreg" if sys.platform == "darwin" else None)
        self.samples, self.apps = [], []
        self._stop = threading.Event()
        self._proc = None
        self._thr = None

    def start(self):
        if self.kind == "nvidia":
            self._proc = subprocess.Popen(["nvidia-smi", "--query-gpu=index,utilization.gpu,clocks.sm,clocks.mem,memory.used,power.draw,temperature.gpu,clocks_throttle_reasons.active",
                                           "--format=csv,noheader,nounits", "-lms", str(int(self.period * 1000))],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._read_nvidia, daemon=True).start()
        if self.kind:
            self._thr = threading.Thread(target=self._poll, daemon=True)
            self._thr.start()

    def _read_nvidia(self):
        for line in self._proc.stdout:
            f = [x.strip() for x in line.split(",")]
            if len(f) < 7:
                continue
            try:
                self.samples.append({"t": time.time(), "gpu": int(f[0]), "util": float(f[1]), "sm_mhz": float(f[2]), "mem_mhz": float(f[3]),
                                     "mem_used_mib": float(f[4]), "power_w": float(f[5]) if f[5] not in ("[N/A]", "N/A") else None,
                                     "temp_c": float(f[6]), "throttle": f[7] if len(f) > 7 else None})
            except ValueError:
                continue

    def _poll(self):
        while not self._stop.is_set():
            if self.kind == "nvidia":
                self.apps.append((time.time(), nvidia_apps()))
                self._stop.wait(0.5)
            else:
                s = ioreg_sample()
                if s:
                    s["t"] = time.time()
                    self.samples.append(s)
                self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thr:
            self._thr.join(timeout=5)

    def summary(self, own_pids, gpu_index=None):
        def stats(v):
            v = sorted(x for x in v if x is not None)
            if not v:
                return None
            return {"min": v[0], "median": v[len(v) // 2], "max": v[-1], "mean": sum(v) / len(v), "n": len(v)}
        out = {"kind": self.kind}
        if self.kind == "nvidia":
            per = {}
            for s in self.samples:
                per.setdefault(s["gpu"], []).append(s)
            out["gpus"] = {str(g): {"util": stats([s["util"] for s in ss]), "sm_mhz": stats([s["sm_mhz"] for s in ss]),
                                    "mem_mhz": stats([s["mem_mhz"] for s in ss]), "mem_used_mib": stats([s["mem_used_mib"] for s in ss]),
                                    "power_w": stats([s["power_w"] for s in ss]), "temp_c": stats([s["temp_c"] for s in ss]),
                                    "throttle": sorted({s["throttle"] for s in ss if s["throttle"]})} for g, ss in per.items()}
            own, foreign = [], set()
            for _, apps in self.apps:
                own.append(sum(m for p, m in apps if p in own_pids))
                foreign |= {p for p, _ in apps if p not in own_pids}
            out["process_mem_mib_max"] = max(own) if own else None
            out["foreign_gpu_pids"] = sorted(foreign)
        elif self.kind == "ioreg":
            for k in IOREG_KEYS:
                v = stats([s.get(k) for s in self.samples])
                if v:
                    out[k] = v
        return out


#
# runs
#

def tool(cfg, name):
    return os.path.join(cfg["bin"], name)


def cfg_env(cfg):
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    env.update(cfg["env"])
    return env


def bench_cmd(cfg, model, args, out):
    cmd = [tool(cfg, "llama-decision-bench"), "-m", model, "-f", args.suite, "--repeat", str(args.repeat), "--warmup", str(args.warmup),
           "--idle-ms", "0", "--device", cfg["device"], "-o", out]
    for k, flag in (("threads", "-t"), ("gpu", "--gpu"), ("kernels", "--kernels"), ("precision", "--precision"), ("plan", "--plan")):
        if cfg[k] is not None:
            cmd += [flag, str(cfg[k])]
    return cmd


def run_identity(cfg, model, suite, runtime):
    exes = [tool(cfg, t) for t in TOOLS if os.path.isfile(tool(cfg, t))]
    return ident.make_identity("bench", build=ident.build_root_of(exes[0]), exes=exes, model=model, corpus=suite, runtime=runtime)


def run_block(cfg, model, args, out, background, keep=()):
    """One bench process; returns the attempt record (and leaves the bench JSON at out)."""
    la0 = load_avg()
    busy0 = proc_stat_busy()
    cg0 = cgroup_cpu_seconds()
    smp = GpuSampler()
    smp.start()
    t0 = time.time()
    with open(out[:-5] + ".log", "w") as logf:
        p = subprocess.Popen(bench_cmd(cfg, model, args, out), stdout=logf, stderr=subprocess.STDOUT, env=cfg_env(cfg))
        foreign_ps = []
        exclude = {os.getpid(), p.pid} | (descendants(background.pids()) if background else set()) | descendants(list(keep))
        last_ps = 0.0
        # reap with wait4 (not Popen.poll) to get the rusage of exactly this process
        while True:
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            if pid == p.pid:
                break
            if busy0 is None and cg0 is None and time.time() - last_ps >= 1.0:  # macOS: ps %cpu of the others
                last_ps = time.time()
                v = ps_cpu(exclude | descendants([os.getpid()]))
                if v is not None:
                    foreign_ps.append(v)
            time.sleep(0.05)
        p.returncode = rc = os.waitstatus_to_exitcode(status)
    wall = time.time() - t0
    smp.stop()
    busy1 = proc_stat_busy()
    cg1 = cgroup_cpu_seconds()
    la1 = load_avg()
    own_cpu = ru.ru_utime + ru.ru_stime
    rec = {"rc": rc, "wall_s": wall, "loadavg_before": la0, "loadavg_after": la1, "own_cpu_s": own_cpu,
           "gpu": smp.summary({p.pid}, cfg["gpu"]), "t_start": t0}
    # sampler / nvidia-smi / this script count as foreign too; they are small and the same for every config.
    # In a container /proc/stat is the whole host (other tenants included): the cgroup counter is what
    # this machine's processes used; the host figure is kept as a fact.
    if busy0 is not None and busy1 is not None:
        rec["host_busy_cores"] = (busy1 - busy0) / wall
    if cg0 is not None and cg1 is not None:
        rec["foreign_cores"] = max(0.0, (cg1 - cg0 - own_cpu) / wall)
    elif busy0 is not None and busy1 is not None:
        rec["foreign_cores"] = max(0.0, (busy1 - busy0 - own_cpu) / wall)
    elif foreign_ps:
        rec["foreign_cores"] = sum(foreign_ps) / len(foreign_ps)
    return rec


def disturbed(rec, args):
    why = []
    if rec["rc"] != 0:
        why.append("bench rc=%s" % rec["rc"])
    la = rec["loadavg_before"]
    if la and la[0] > args.load_max:
        why.append("load average %.2f before the block > %.2f" % (la[0], args.load_max))
    fc = rec.get("foreign_cores")
    if fc is not None and fc > args.foreign_max:
        why.append("other processes used %.2f cores > %.2f" % (fc, args.foreign_max))
    fg = (rec.get("gpu") or {}).get("foreign_gpu_pids")
    if fg:
        why.append("other processes on the GPU: %s" % fg)
    return why


def wait_quiet(args):
    """Waits (up to --quiet-wait s) for the 1-minute load average to fall to --load-max."""
    t0 = time.time()
    while True:
        la = load_avg()
        if not la or la[0] <= args.load_max or time.time() - t0 > args.quiet_wait:
            return la
        time.sleep(5)


def http(url, body=None, timeout=60):
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def rss_kb(pid):
    try:
        with open("/proc/%d/status" % pid) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    try:
        return int(subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def free_port(preferred):
    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port near %d" % preferred)


def suite_body(suite, rid):
    with open(suite, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] == rid:
                return json.dumps(r["body"], ensure_ascii=False).encode("utf-8")
    raise KeyError(rid)


def ready_once(cfg, model, args, logpath):
    port = free_port(args.port)
    cmd = [tool(cfg, "llama-server"), "--decision", "-m", model, "--host", "127.0.0.1", "--port", str(port),
           "--decision-allow-uncalibrated", "--decision-device", cfg["device"]]
    for k, flag in (("threads", "-t"), ("gpu", "--decision-gpu"), ("kernels", "--decision-kernels"),
                    ("precision", "--decision-precision"), ("plan", "--decision-plan")):
        if cfg[k] is not None:
            cmd += [flag, str(cfg[k])]
    la = load_avg()
    with open(logpath, "w") as logf:
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=cfg_env(cfg))
        t_listen = t_ready = None
        try:
            while time.perf_counter() - t0 < 300:
                if p.poll() is not None:
                    break
                try:
                    code, _ = http("http://127.0.0.1:%d/health" % port, timeout=1)
                except urllib.error.HTTPError as e:
                    code = e.code
                except (urllib.error.URLError, ConnectionError, socket.timeout, OSError):
                    code = None
                if code is not None and t_listen is None:
                    t_listen = time.perf_counter()
                if code == 200:
                    t_ready = time.perf_counter()
                    break
                time.sleep(0.01)
            rec = {"cmd": cmd, "loadavg_before": la, "rc": p.poll()}
            if t_ready is None:
                rec["error"] = "not ready (see %s)" % logpath
                return rec
            rec["listen_ms"] = (t_listen - t0) * 1000 if t_listen else None
            rec["ready_ms"] = (t_ready - t0) * 1000
            rec["rss_kb_ready"] = rss_kb(p.pid)
            if have("nvidia-smi"):
                rec["gpu_mem_mib_ready"] = sum(m for pid, m in nvidia_apps() if pid == p.pid)
            t1 = time.perf_counter()
            code, body = http("http://127.0.0.1:%d/v1/systemone" % port, suite_body(args.suite, "so-1q-noul"))
            rec["first_so1_ms"] = (time.perf_counter() - t1) * 1000
            rec["first_so1_http"] = code
            t2 = time.perf_counter()
            http("http://127.0.0.1:%d/v1/systemone" % port, suite_body(args.suite, "so-1q-noul"))
            rec["second_so1_ms"] = (time.perf_counter() - t2) * 1000
            rec["rss_kb_after"] = rss_kb(p.pid)
            if have("nvidia-smi"):
                rec["gpu_mem_mib_after"] = sum(m for pid, m in nvidia_apps() if pid == p.pid)
            _, props = http("http://127.0.0.1:%d/props" % port)
            d = json.loads(props).get("decision", {})
            rec["props"] = {k: d.get(k) for k in ("device", "device_description", "placement", "memory", "plan", "precision")}
            return rec
        finally:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()


CHAT_PROMPT = ("Write a long, detailed story about a lighthouse keeper on a small island who keeps a logbook of "
               "every ship that passes, the weather of each day and the letters she never sends.")


class ChatLoad:
    """A chat model in llama-server, generating completions in a loop (the app case)."""

    def __init__(self, bin_dir, model, port, ngl, n_predict, logpath):
        self.bin_dir, self.model, self.port, self.ngl, self.n_predict, self.logpath = bin_dir, model, port, ngl, n_predict, logpath
        self.proc = None
        self.samples, self.errors = [], []
        self._stop = threading.Event()
        self._thr = None

    def start(self):
        cmd = [os.path.join(self.bin_dir, "llama-server"), "-m", self.model, "--host", "127.0.0.1", "--port", str(self.port),
               "-ngl", str(self.ngl), "-c", "4096", "--no-webui"]
        env = dict(os.environ)
        env["LC_ALL"] = "C"
        self.proc = subprocess.Popen(cmd, stdout=open(self.logpath, "w"), stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while time.time() - t0 < 600:
            if self.proc.poll() is not None:
                raise RuntimeError("chat server exited (see %s)" % self.logpath)
            try:
                if http("http://127.0.0.1:%d/health" % self.port, timeout=2)[0] == 200:
                    return
            except (urllib.error.URLError, ConnectionError, socket.timeout, OSError):
                pass
            time.sleep(0.5)
        raise RuntimeError("chat server not ready after 600 s (see %s)" % self.logpath)

    def one(self, phase):
        body = json.dumps({"prompt": CHAT_PROMPT, "n_predict": self.n_predict, "ignore_eos": True, "cache_prompt": False,
                           "temperature": 0.0}).encode("utf-8")
        t0 = time.time()
        try:
            _, out = http("http://127.0.0.1:%d/completion" % self.port, body, timeout=600)
            tm = json.loads(out).get("timings") or {}
        except (urllib.error.URLError, ConnectionError, socket.timeout, OSError, ValueError) as e:
            self.errors.append({"t": t0, "error": str(e)})
            return None
        rec = {"phase": phase, "t0": t0, "t1": time.time(), "tg_ts": tm.get("predicted_per_second"), "tg_n": tm.get("predicted_n"),
               "pp_ts": tm.get("prompt_per_second")}
        self.samples.append(rec)
        return rec

    def alone(self, n):
        return [self.one("alone") for _ in range(n)]

    def run(self):
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        while not self._stop.is_set():
            if self.one("load") is None:
                self._stop.wait(1.0)

    def pids(self):
        return [self.proc.pid] if self.proc else []

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=self.n_predict)  # lets the running completion finish
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class Background:
    def __init__(self, cmd, logpath):
        self.cmd, self.logpath = cmd, logpath
        self.proc = None
        self.restarts = 0
        self._stop = threading.Event()
        self._thr = None

    def start(self):
        self._spawn()
        self._thr = threading.Thread(target=self._watch, daemon=True)
        self._thr.start()

    def _spawn(self):
        self.proc = subprocess.Popen(self.cmd, shell=True, stdout=open(self.logpath, "a"), stderr=subprocess.STDOUT, start_new_session=True)

    def _watch(self):
        while not self._stop.wait(1.0):
            if self.proc.poll() is not None:
                self.restarts += 1
                self._spawn()

    def pids(self):
        return [self.proc.pid] if self.proc else []

    def stop(self):
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=20)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)


def machine_facts(args, configs):
    f = {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
         "os": platform.platform(), "cpu": ident.cpu_brand(), "logical_cpus": os.cpu_count(), "python": platform.python_version(),
         "loadavg_start": load_avg(), "blocks": args.blocks, "repeat": args.repeat, "warmup": args.warmup, "ready": args.ready,
         "suite": os.path.abspath(args.suite), "load_max": args.load_max, "foreign_max": args.foreign_max,
         "background": args.background, "chat_model": args.chat_model and os.path.abspath(args.chat_model),
         "configs": configs, "models": [os.path.abspath(m) for m in args.model],
         "gpus": nvidia_facts()}
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            f["cgroup_cpu_max"] = fh.read().strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        for k, cmd in (("power_source", ["pmset", "-g", "batt"]), ("low_power_mode", ["pmset", "-g"])):
            try:
                f[k] = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[:2]
            except (OSError, subprocess.SubprocessError):
                pass
    return f


def main():
    if os.name != "posix":
        sys.exit("bench-decision-device.py is POSIX-only (process groups and SIGINT for the servers it starts)")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="decision GGUF (repeatable)")
    ap.add_argument("--config", action="append", required=True, help="NAME key=value ... (repeatable; see above)")
    ap.add_argument("--suite", default=os.path.join(HERE, "..", "tests", "decision", "bench", "suite.jsonl"))
    ap.add_argument("--blocks", type=int, default=3, help="cycles; each runs every config once per model (default 3)")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--ready", type=int, default=3, help="llama-server time-to-ready samples per config and model (0: none)")
    ap.add_argument("--load-max", type=float, default=4.0, help="1-minute load average a block may start at")
    ap.add_argument("--foreign-max", type=float, default=1.0, help="CPU cores other processes may use during a block")
    ap.add_argument("--retries", type=int, default=2, help="re-runs of a disturbed block")
    ap.add_argument("--quiet-wait", type=float, default=600, help="seconds to wait for a quiet machine before a block")
    ap.add_argument("--background", help="shell command kept running for the whole session (e.g. a chat model generating)")
    ap.add_argument("--chat-model", help="chat GGUF generating on the GPU for the whole session (the app case; see above)")
    ap.add_argument("--chat-bin", help="folder with the llama-server for --chat-model (default: the first config's bin)")
    ap.add_argument("--chat-ngl", type=int, default=99, help="GPU layers of the chat model (default 99: all)")
    ap.add_argument("--chat-tokens", type=int, default=256, help="tokens per chat completion (default 256)")
    ap.add_argument("--chat-alone", type=int, default=5, help="chat completions measured alone first (default 5)")
    ap.add_argument("--chat-port", type=int, default=18590)
    ap.add_argument("--port", type=int, default=18490)
    ap.add_argument("--out", default="build/bench-device")
    ap.add_argument("--label", default="device")
    args = ap.parse_args()

    try:
        configs = [parse_config(c) for c in args.config]
    except ValueError as e:
        sys.exit("error: %s" % e)
    if len({c["name"] for c in configs}) != len(configs):
        sys.exit("error: config names must be unique")
    for c in configs:
        for t in ("llama-decision-bench",) + (("llama-server",) if args.ready else ()):
            if not os.access(tool(c, t), os.X_OK):
                sys.exit("error: %s not found. Run:  cmake --build <build> --target llama-decision-bench llama-server" % tool(c, t))
    for m in args.model:
        if not os.path.isfile(m):
            sys.exit("error: model %s not found" % m)
    if args.chat_model:
        chat_bin = args.chat_bin or configs[0]["bin"]
        if not os.path.isfile(args.chat_model):
            sys.exit("error: chat model %s not found" % args.chat_model)
        if not os.access(os.path.join(chat_bin, "llama-server"), os.X_OK):
            sys.exit("error: %s/llama-server not found. Run:  cmake --build <build> --target llama-server" % chat_bin)
    if not os.path.isfile(args.suite):
        sys.exit("error: no %s. Run:  python3 tests/decision/bench/gen_suite.py" % args.suite)

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M", time.gmtime())
    host = socket.gethostname().split(".")[0].replace("__", "_")
    prefix = os.path.join(args.out, "%s__%s__%s" % (host, args.label, stamp))
    facts = machine_facts(args, configs)
    background = None
    chat = None
    if args.background:
        background = Background(args.background, prefix + "__background.log")
        background.start()
        log("background: %s (pid %d)" % (args.background, background.proc.pid))
        time.sleep(10)  # let it reach its steady state

    try:
        if args.chat_model:
            chat = ChatLoad(args.chat_bin or configs[0]["bin"], args.chat_model, free_port(args.chat_port), args.chat_ngl,
                            args.chat_tokens, prefix + "__chat-server.log")
            chat.start()
            alone = [r for r in chat.alone(args.chat_alone) if r]
            log("chat alone: %s tokens/s" % ", ".join("%.1f" % r["tg_ts"] for r in alone if r.get("tg_ts")))
            chat.run()
        # time to ready first (cold-ish processes; interleaved so every config sees the same cache state)
        ready = {}
        for i in range(args.ready):
            for m in args.model:
                for c in (configs if i % 2 == 0 else configs[::-1]):
                    mname = os.path.splitext(os.path.basename(m))[0]
                    wait_quiet(args)
                    r = ready_once(c, m, args, "%s__%s__%s__ready%d.log" % (prefix, mname, c["name"], i))
                    ready.setdefault((mname, c["name"]), []).append(r)
                    log("ready %s %s #%d: %s" % (mname, c["name"], i, "%.0f ms" % r["ready_ms"] if "ready_ms" in r else r.get("error")))
        for (mname, cname), rs in ready.items():
            c = next(x for x in configs if x["name"] == cname)
            m = next(x for x in args.model if os.path.splitext(os.path.basename(x))[0] == mname)
            with open("%s__%s__%s__ready.json" % (prefix, mname, cname), "w") as f:
                json.dump({"kind": "ready", "model": os.path.abspath(m), "config": c, "runs": rs}, f, indent=1)

        for b in range(args.blocks):
            order = configs if b % 2 == 0 else configs[::-1]
            for m in args.model:
                mname = os.path.splitext(os.path.basename(m))[0]
                for c in order:
                    out = "%s__%s__%s__b%d.json" % (prefix, mname, c["name"], b)
                    attempts = []
                    for a in range(args.retries + 1):
                        wait_quiet(args)
                        rec = run_block(c, m, args, out, background, chat.pids() if chat else ())
                        why = disturbed(rec, args)
                        rec["disturbed"] = why
                        attempts.append(rec)
                        log("block %d %s %s attempt %d: %.1f s, load %s -> %s, foreign %s cores%s" % (
                            b, mname, c["name"], a, rec["wall_s"], rec["loadavg_before"] and "%.2f" % rec["loadavg_before"][0],
                            rec["loadavg_after"] and "%.2f" % rec["loadavg_after"][0],
                            "%.2f" % rec["foreign_cores"] if rec.get("foreign_cores") is not None else "-",
                            "" if not why else "  DISTURBED: " + "; ".join(why)))
                        if not why or rec["rc"] != 0:
                            break
                    if rec["rc"] != 0 or not os.path.isfile(out):
                        log("error: bench failed for %s %s (see %s)" % (mname, c["name"], out[:-5] + ".log"))
                        continue
                    with open(out) as f:
                        d = json.load(f)
                    d["ab"] = {"config": c, "block": b, "position": order.index(c), "order": [x["name"] for x in order],
                               "attempts": attempts, "disturbed": bool(attempts[-1]["disturbed"]),
                               "background": args.background, "background_restarts": background.restarts if background else None,
                               "chat_model": args.chat_model}
                    d["identity"] = run_identity(c, m, args.suite, ident.runtime_from_bench(d))
                    with open(out, "w") as f:
                        json.dump(d, f, indent=1)
                        f.write("\n")
    except RuntimeError as e:
        log("error: %s" % e)
        return 1
    finally:
        if chat:
            chat.stop()
            with open(prefix + "__chat.json", "w") as f:
                json.dump({"kind": "chat", "model": os.path.abspath(args.chat_model), "n_predict": args.chat_tokens, "ngl": args.chat_ngl,
                           "samples": chat.samples, "errors": chat.errors}, f, indent=1)
        if background:
            background.stop()
        facts["loadavg_end"] = load_avg()
        with open(prefix + "__machine.json", "w") as f:
            json.dump(facts, f, indent=1)
    log("done: %s*  (tables: python3 scripts/bench-decision-report.py %s --parity <gate JSONs>)" % (prefix, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
