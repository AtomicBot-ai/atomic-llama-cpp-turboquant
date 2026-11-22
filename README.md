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

Start AD-IQ2_M with its target-specific folded DSpark draft. Run from the repository
root and adjust the two GGUF paths if needed. This example gives the engine
14 GiB plus 100 MiB for KV:

```sh
./build-metal/bin/llama-server \
  -m .work/models/Inkling-2-Mini-AD-IQ2_M.gguf \
  -md .work/models/dspark-Inkling-2-Mini-IQ2_M-folded.gguf \
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

With both Metal servers built and the two GGUF files above in place, run from the repository root:

```sh
python3 tools/expert-source/bench-inkling-metal.py \
  --model .work/models/Inkling-2-Mini-AD-IQ2_M.gguf \
  --draft .work/models/dspark-Inkling-2-Mini-IQ2_M-folded.gguf
```

The script runs stock llama.cpp first, then Atomic Engine at **8, 10, 12, 14, 16, 18 and 20 GiB**, each with another 100 MiB for KV. Each row starts a fresh server on a private loopback port and sends the same 4,000 uncached input tokens. TTFT is the time from sending the request to receiving the first token. Decode uses the server's tokens/s for the 64-token greedy response. Each Atomic Engine row checks all 64 token IDs against stock and checks the managed memory peak against the cap. Run one benchmark at a time on the GPU.

The table prints as each row finishes. The script saves `results.json` and the server logs in a new directory under `.work/inkling-bench/`. To run just stock and 20 GiB, append `--budgets 20`. For other installation paths, pass `--server`, `--stock-server`, `--model` or `--draft`.

To benchmark the AD-IQ3_S or AD-IQ2_M target at 8 GiB plus KV, set the
quant below and place that target and the original Q8_0 DSpark draft in
`.work/models/`. Build the folded draft once per target (requires NumPy
and the bundled `gguf-py` package), then run the same stock comparison:

```sh
quant=IQ3_S  # or IQ2_M
model=".work/models/Inkling-2-Mini-AD-${quant}.gguf"
draft=".work/models/dspark-Inkling-2-Mini-${quant}-folded.gguf"
PYTHONPATH=gguf-py python3 tools/expert-source/fold-inkling-dspark.py \
  "$model" .work/models/dspark-Inkling-2-Mini-Q8_0.gguf "$draft"
python3 tools/expert-source/bench-inkling-metal.py --model "$model" --draft "$draft" --budgets 8
```

For interactive AD-IQ3_S, use its target and folded draft in the server
command above and add `--spec-draft-n-max 3 --spec-draft-n-ubatch 32`.

In the benchmark JSON, `expert_logical_read_bytes` counts streamed expert
weight bytes during the request, including reads served by the OS page
cache. `expert_cache_misses` counts misses in the engine's expert cache.
The matrix compares the AD-IQ2_M default with fused prompt attention disabled
at the same draft depth. Each measurement includes one uncached 4,000-token
prompt and a 64-token greedy response. The AD-IQ3_S columns use DSpark 3/32.
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
