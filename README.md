# Atomic Engine

Run Qwen3.8 Flash Next on Apple Silicon with a single `llama-server` binary that serves a web chat and an OpenAI-compatible API. Set `LLAMA_EXPERIMENTAL_BUDGET_DFLASH2=1` at launch to let a small draft model propose tokens within the same memory budget. On the 912-token Handover prompt (M4 Max, 20 GiB budget, 4,096-token context), live decode peaked at **17.45 tokens/s**. The signed binary reached **15.58 tokens/s** and matched all **64/64** reference tokens. [Launch the server and open the chat](docs/atomic-engine-flash-next-dflash3-8-24gib.md#run-the-macos-binary).

Atomic Engine is an inference engine under development on top of llama.cpp,
targeting MoE expert streaming from SSD and adjustable managed memory budgets.
The experimental engine admits model, state, compute and expert buffers under a
shared budget with an explicit runtime reserve. On Apple M4 Max, original Qwen3.5
MoE prefill/decode matches resident logits and state under a 5 GiB engine budget.
Manual live shrink/grow releases backend memory while preserving the active
session. Opt-in automatic pressure control adds bounded recovery and manual
priority; see [the policy and qualification limits](docs/atomic-engine-pressure.md).
LFM2 MoE now has a [second-architecture qualification](docs/atomic-engine-second-architecture.md)
on the same Metal backend, including useful prefill and live resize.
The [fixed-budget CUDA server profile](docs/atomic-engine-cuda-server.md)
qualifies single-slot LFM2.5-24B with separate 4 GiB RAM / 2 GiB VRAM caps.
Qualification is limited to the measured model/configurations; this is
not an OS-wide RSS limiter or a throughput-optimized implementation.
See the [server build, launch commands and qualified profiles](docs/atomic-engine-release.md)
and the [implementation status](docs/atomic-engine.md).

## Run Inkling 2 Mini on Apple Silicon

The budgeted Metal profiles for AD-IQ2_M, AD-IQ3_S and AD-Q4_K use the same
AtomicChat Q8_0 DSpark GGUF. It contains its own Markov output projection
and can be passed directly to `-md`. Download the tested revision once
from the repository root. [AtomicChat DSpark weights](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF/blob/309509848b3151419aa26d069eb03db57fd29a00/dspark-Inkling-2-Mini-Q8_0.gguf).

```sh
hf download AtomicChat/Inkling-2-Mini-GGUF dspark-Inkling-2-Mini-Q8_0.gguf \
  --revision 309509848b3151419aa26d069eb03db57fd29a00 --local-dir .work/models/atomicchat-dspark-3095098
```

Start AD-IQ2_M with fused prefill attention, DSpark 7/32 (seven proposals per
verification round, draft microbatch 32) and a 14 GiB budget plus 100 MiB for KV.

```sh
./build-metal/bin/llama-server \
  -m .work/models/Inkling-2-Mini-AD-IQ2_M.gguf \
  -md .work/models/atomicchat-dspark-3095098/dspark-Inkling-2-Mini-Q8_0.gguf \
  --spec-type draft-dspark --engine-budget-mib 14436 \
  -fit off -c 4096 -b 1024 -ub 1024 -np 1 \
  -fa on -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 -ngl all -ngld 99 -t 8 -tb 8 \
  --temp 0 --cache-ram 0 --ctx-checkpoints 0 --no-warmup --metrics \
  --ui --alias inkling-2-mini --host 127.0.0.1 --port 8080
```

Open the chat at `http://127.0.0.1:8080/` or use the OpenAI-compatible API at
`http://127.0.0.1:8080/v1` with model `inkling-2-mini`. For this Metal
profile, set `--engine-budget-mib` to `1024 × GiB + 100`: 8292 for 8 GiB
through 20580 for 20 GiB. The server recognizes AD-IQ2_M by its GGUF metadata
and expert tensors, sets DSpark to seven proposals with a draft microbatch of
32, and selects fused attention for supported prompt blocks. Expert streaming
uses `pread` and KV uses Q4_0.

For greedy single-slot decoding with repeated passages, prefix the server
command with `LLAMA_EXPERIMENTAL_INKLING_PROMPT_LOOKUP=1`. When the last 32
tokens match an earlier passage, the server can propose the already-seen
continuation. The target checks every proposal. DSpark supplies the draft
when the search finds no match.

To compare prefill paths, prefix the same command with
`LLAMA_EXPERIMENTAL_INKLING_PREFILL_ATTN=0`. To use three draft proposals,
add `--spec-draft-n-max 3`. For the AD-Q4_K command and matching draft,
see the [Q4_K Metal profile](docs/atomic-engine-release.md#inkling-2-mini-ad-q4_k-with-dspark-3).

When an embedding host requests a live budget change from 20 to 8 GiB and
back, the engine releases idle expert-cache entries before restoring the
full banks. Generation continues in the same process.
See the [Metal build instructions](docs/atomic-engine-release.md#build-the-server)
and [Inkling profile options](docs/atomic-engine-release.md#inkling-2-mini-ad-q4_k-with-dspark-3).

### Benchmark Inkling against stock llama.cpp

Run from the repository root with both Metal servers built, the AtomicChat draft downloaded as above, and the three target files in `.work/models/`: `Inkling-2-Mini-AD-IQ2_M.gguf`, `Inkling-2-Mini-AD-IQ3_S.gguf` and `Inkling-2-Mini-AD-Q4_K.gguf`. The PNG renderer uses Pillow (`python3 -m pip install 'Pillow>=10.1'`).

```sh
python3 tools/expert-source/bench-inkling-metal.py
```

The default matrix covers **IQ2_M (DSpark 7/32 with fused prefill), IQ3_S (3/32) and Q4_K (3/32)**. For each GGUF the script measures uncapped stock llama.cpp once, then Atomic Engine at **8, 10, 12, 14, 16, 18 and 20 GiB**, each with another 100 MiB for KV. The 24 requests run sequentially. Each starts a fresh server and sends the same 4,000 uncached archive tokens for a 64-token greedy response. TTFT measures time from request submission to the first token. Decode uses the server's reported tokens/s. Each Atomic result must match all 64 stock token IDs for its own GGUF and stay within its managed memory cap.

The script prints progress during the run, then prints one table with a TTFT/decode column pair per quant and a separate stock row. The output directory printed at startup contains **`matrix.png`**, `results.json` and per-quant server logs. Append `--budgets 14` for a one-budget matrix across all three quants. Pass `--model PATH` to benchmark a single GGUF. Use `--server`, `--stock-server` and `--draft` to override the binary and draft paths.

#### Inkling 2 Mini results on Apple M4 Max

Apple M4 Max with 48 GB unified memory, November 23, 2026. Each cell shows **time to first token (seconds) / decode speed (tokens/s)** for one request with the model already loaded. All three engines received the same 4,000 archive tokens and produced the same 64 output token IDs.

| Engine and memory budget | IQ2_M | IQ3_S | Q4_K | SQ-10 |
|---|---:|---:|---:|---:|
| Atomic · 8 GiB + KV | 8.17 / 68.07 | 11.19 / 46.88 | 12.12 / 41.64 | — |
| Atomic · 10 GiB + KV | 7.83 / 76.28 | 10.92 / 49.74 | 11.92 / 45.23 | — |
| Atomic · 12 GiB + KV | 7.20 / 105.48 | 10.56 / 54.03 | 11.61 / 51.29 | — |
| Atomic · 14 GiB + KV | 7.15 / 105.71 | 10.48 / 64.04 | 11.09 / 49.85 | — |
| Atomic · 16 GiB + KV | 7.15 / 107.49 | 10.19 / 67.73 | 10.74 / 55.79 | — |
| Atomic · 18 GiB + KV | 7.09 / 109.53 | 10.36 / 69.40 | 10.12 / 70.06 | — |
| Atomic · 20 GiB + KV | 7.12 / 108.31 | 10.54 / 67.13 | 9.95 / 77.31 | — |
| Stock llama.cpp · uncapped | 9.57 / 33.11 | 9.54 / 31.63 | 9.46 / 32.38 | — |
| Strata · uncapped | — | — | — | 4.21 / 112.92 |

Atomic (`95b0cb23e`) and stock llama.cpp (`e412c6f46`) used Q4_0 KV. Each Atomic budget includes an additional 100 MiB for KV. Strata (`strata-inkling` 0.1.65) ran separately with its native SQ-10 pack, DSpark and FP16 KV.

#### Compare drafters or select one quant

To compare AtomicChat against a previous local draft on both 4,000-token fixtures,
set `PREVIOUS_DRAFT` to the path of the folded IQ2_M draft. Each budget gets two
runs per draft in opposite orders. The comparison requires all Atomic runs to
reproduce the first folded-draft run's 64 output IDs. It also reports the first
divergence from fresh stock, if any, and saves both checks alongside TTFT, decode
throughput, the managed memory peak and the token IDs.

```sh
PREVIOUS_DRAFT=.work/models/dspark-Inkling-2-Mini-IQ2_M-folded.gguf
(set -e
for prompt in archive handover; do
  python3 tools/expert-source/bench-inkling-metal.py --prompt "$prompt" --draft-depth 4 \
    --compare-draft "$PREVIOUS_DRAFT" --budgets 8 10 12 14 16
done)
```

To repeat the archive comparison at the default depth, use `--prompt archive --draft-depth 7`.
The server and benchmark use the AtomicChat draft in the commands above. With IQ2_M and
14 GiB + KV on an M4 Max, handover at 4/32 averaged 72.34 tok/s with AtomicChat and
70.74 with the folded draft. At 8 GiB + KV, archive at 7/32 averaged 66.25 and 67.34 tok/s,
respectively. All 60 qualification requests at 8–16 GiB + KV matched their archived
64-token references. A separate handover run of the command above at 14 GiB
averaged 70.22 tok/s with AtomicChat and 70.44 with the folded draft. Both drafts
produced identical IDs in both orders. Fresh stock first diverged at token 24.

For an 8 GiB run with AD-IQ3_S or AD-Q4_K, change the target and keep the
same AtomicChat draft. The benchmark uses DSpark 3/32 for these targets.

```sh
python3 tools/expert-source/bench-inkling-metal.py \
  --model .work/models/Inkling-2-Mini-AD-IQ3_S.gguf --budgets 8
```

For interactive AD-IQ3_S, change only `-m` in the server command above.
For AD-Q4_K, also supply `--spec-draft-n-max 3 --spec-draft-n-ubatch 32`.
With the AtomicChat draft and a managed budget, the single-slot Metal path
parallelizes the two additions over the vocabulary for all three target quants.
Set `LLAMA_EXPERIMENTAL_INKLING_W2_PARALLEL_ADD=0` to benchmark the original additions.
For IQ2_M on the handover prompt, add `--spec-draft-n-max 4` to run the 4/32 profile.

In the benchmark JSON, `expert_logical_read_bytes` counts streamed expert
weight bytes during the request, including reads served by the OS page
cache. `expert_cache_misses` counts misses in the engine's expert cache.
The historical matrix below used the earlier target-specific folded drafts.
It compares fused prefill on and off for AD-IQ2_M at the same draft depth. Each cell measures one uncached 4,000-token excerpt
from the repeated archive fixture and a 64-token greedy response. At
14 GiB + KV the folded IQ2_M draft accepted all 55 draft proposals in eight rounds (105.49 tok/s).
On the separate 4,000-token handover fixture, the same 7/32 profile accepted
47 of 105 proposals in sixteen rounds (53.91 tok/s across two runs).
The AD-IQ3_S columns use DSpark 3/32.
[Per-budget timings, token IDs and memory peaks](docs/atomic-engine-inkling-low-bit-metal.json).

![Inkling 2 Mini on Apple M4 Max, TTFT and decode at 8 to 20 GiB plus KV](docs/atomic-engine-inkling-low-bit-metal.png)

<details>
<summary>Build the stock Metal server used for the comparison</summary>

The reference is upstream llama.cpp revision `e412c6f46`. To build it once at the path the script uses:

```sh
mkdir -p .work/inkling-stock
git worktree add --detach .work/inkling-stock/upstream e412c6f46
cmake -S .work/inkling-stock/upstream -B .work/inkling-stock/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_CCACHE=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build .work/inkling-stock/build --target llama-server -j 6
```

</details>

The upstream llama.cpp documentation follows.

## llama.cpp

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
