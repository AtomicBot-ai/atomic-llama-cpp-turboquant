# TurboQuant fork — process & infrastructure

This is the `atomic-llama-cpp-turboquant` fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
used as the primary inference backend of Atomic Chat. This document describes
the branch model, the dev/staging channel, stable releases and the upstream
sync procedure. For what the fork changes technically (TurboQuant KV cache,
custom quant types, Inkling arch, merge history) see `MERGE_NOTES.md`.

## Branch model

| Branch | Role | Rules |
|---|---|---|
| `master` | **Stable.** What Atomic Chat ships. | Changes arrive only via PR from `dev`. Releases are tagged here. |
| `dev` | **Staging.** Feature/fix integration, no strict stability promise. | PRs land here first. Every push builds all platforms and republishes the rolling `dev-latest` prerelease. |
| `upstream` | **Pure mirror** of `ggml-org/llama.cpp` `master`. | Fast-forward only, never contains fork commits. Used as the merge source for upstream syncs. |
| `legacy/master-2025` | Archive of the pre-2026 `master` tip (`24cabf4d0`). | Frozen. |
| `feature/turboquant-kv-cache` | Former trunk, kept as an alias during the transition. | Do not push; will be deleted eventually. |

## Dev channel (staging binaries)

Workflow: `.github/workflows/dev-build.yml`.

Every push to `dev` builds all eleven archives and republishes the
[`dev-latest`](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant/releases/tag/dev-latest)
rolling prerelease with them. PRs into `dev` — and into `master`, so the
promotion PR reports the same checks — build everything but publish nothing.
If some backend fails, `dev-latest` is still published with the survivors and
the notes list what is missing.

| Platform | Archives |
|---|---|
| Linux x64 | `cpu`, `vulkan`, `cuda-12.4`, `cuda-13.3`, `rocm` |
| Linux arm64 | `cuda-13.3` — NVIDIA DGX Spark / GB10, sm_121 |
| Windows x64 | `cpu`, `vulkan`, `cuda-12.4`, `cuda-13.3`, `rocm` |
| macOS arm64 | `macos-arm64` (Metal) |

The CUDA archives bundle their own `libcudart`/`libcublas` and are linked with
an `$ORIGIN` RPATH, so they do not need a CUDA toolkit on the target machine —
only a recent enough driver. The Linux arm64 archive uses the arm64-SBSA CUDA
build; DGX Spark needs driver r580+ for the bundled CUDA 13.3 runtime.

Grab-and-test on any machine:

```bash
gh release download dev-latest -R AtomicBot-ai/atomic-llama-cpp-turboquant \
  -p 'llama-turboquant-linux-x64-vulkan.tar.gz'   # or your platform
tar -xzf llama-turboquant-linux-x64-vulkan.tar.gz
./build/bin/llama-server --version    # → version: turboquant-vX.Y.Z (<count>, <sha>)
```

macOS dev builds are signed but **not notarized** (release builds are):
`xattr -dr com.apple.quarantine build/` after unpacking.

### Build cache (ccache)

The CUDA, ROCm and Windows CUDA jobs take 5-15 min with a warm ccache and
1-2 h without it. The GitHub Actions cache is 10 GB per repository (least
recently used entries are evicted), and a run can only restore caches from
its own branch, the default branch and, for a PR, its base branch. The
repository's default branch is `dev`, so every run can read the `dev` cache:
PRs, manual `workflow_dispatch` runs on any branch, and release tag builds.

- `dev-build.yml` saves ccache only on a push to `dev`; every other run only
  restores. `release-turboquant.yml` never saves (a tag cache is never read
  again). This keeps the 10 GB for the `dev` caches.
- A release tag build restores the `dev` cache, so it is warm when `master`
  matches `dev` (the release branch only changes `TURBOQUANT_VERSION` and
  `CHANGELOG.md`; the commit hash only reaches `ggml-base`). Keep the ccache
  `key:` and the cmake flags of each job the same in `dev-build.yml` and
  `release-turboquant.yml`, or the release builds cold.
- A large upstream sync builds cold once whatever the cache, because most
  sources change; the first push to `dev` after it refills the cache.

## Versioning & stable releases

Version format: **`<upstream-base>-<fork-semver>`**, e.g. `b10018-1.2.0`:

- `b10018` — the upstream llama.cpp build the fork is based on
  (`git rev-list --count $(git merge-base master upstream)`, matching
  upstream's `b####` release tags). Changes only on upstream syncs; the sync
  PR updates it in `TURBOQUANT_VERSION` by hand.
- `1.2.0` — the fork's own semver: **major** for breaking changes, **minor**
  for features (e.g. implementing turbo-ops for a new backend), **patch**
  for fixes.

Single source of truth: the `TURBOQUANT_VERSION` file at the repo root.
CMake embeds it via `common/build-info.cpp.in`; `llama-server --version`
prints `version: b10018-1.2.0 (build <count>, commit <sha>)`. Note that
`llama_build_info()` (the OpenAI API `system_fingerprint`) intentionally
keeps the upstream `b<N>-<sha>` format — clients parse it.

Cut a release (from an up-to-date, clean `master` checkout):

```bash
# 1. Write the CHANGELOG section for the version you are about to cut,
#    commit it. `verify-version` refuses the release without it.
# 2. Then:
./scripts/turboquant-release.sh patch|minor|major|X.Y.Z
```

This bumps the fork-semver part, commits `release: b10018-X.Y.Z`, tags
`b10018-X.Y.Z` and pushes. The tag triggers
`.github/workflows/release-turboquant.yml`: all backends are built (macOS
fully notarized) and published as **one** GitHub release with all archives.
`verify-version` fails the release — in seconds, before three hours of
building — if the tag doesn't match `TURBOQUANT_VERSION` or if `CHANGELOG.md`
has no `## <tag>` section.

### Release notes

`CHANGELOG.md` has one section per tag, and the release notes are generated
from it verbatim: that section *is* the release announcement, so write it for
whoever downloads the build — what they get, what changed for them, what to
watch out for. Everything mechanical (the asset table, the full commit list
since the previous tag, versioning boilerplate) the workflow adds by itself,
collapsed below the fold. Do not paste a commit dump into the changelog; the
notes already carry one.

Consumers: `atomic-chat-conf/backends/turboquant-manifest.json` entries all
point at the same `b10018-X.Y.Z` tag; asset names
(`llama-turboquant-<backend>.zip|tar.gz`) are unchanged from the legacy
scheme, so the Atomic-Chat runtime URL builder needs no changes.

Legacy per-platform releases (`turboquant-<platform>-<sha>`) are kept for
old app versions; do not delete them.

## Upstream sync procedure

Small regular syncs instead of 130k-line big bangs:

```bash
# 1. Advance the mirror (fast-forward only — zero conflicts by definition)
git fetch upstream            # remote 'upstream' = https://github.com/ggml-org/llama.cpp.git
git push origin upstream/master:refs/heads/upstream

# 2. Merge into a sync branch off dev
git checkout -b sync/upstream-$(date +%Y-%m-%d) origin/dev
git merge origin/upstream     # resolve conflicts HERE, in the sync branch

# 3. PR the sync branch into dev → CI builds every platform
# 4. Test via dev-latest, then PR dev → master as usual
```

`git merge-base origin/master origin/upstream` always tells you exactly which
upstream commit the fork is based on.

Conflict hot-spots (see `MERGE_NOTES.md` for history): `ggml-cuda.cu`/`fattn.cu`,
`ggml-vulkan.cpp` (SET_ROWS/supports_op), `ggml-metal.metal` kernel naming,
`llama-kv-cache.cpp`, `gguf-py/gguf/constants.py` (quant type ids follow
upstream since the tqp-v0.4.0 rebase: `Q2_0` = 42, turbo KV types 43/44/47,
`TQ3_1S`/`TQ4_1S` = 45/46).

### TurboQuant+ (TheTom) base

Since 2026-10 `dev` is based on TheTom's `feature/turboquant-kv-cache` at tag
`tqp-v0.4.0` (remote `upstream` in this repo, `ggml` is ggml-org). TheTom
re-ported TurboQuant onto fresh ggml on 2026-07-31, so the older fork history
could not be merged; our own commits were re-applied on top of his tree
(cherry-pick with `-x`, the source sha is in each message). From now on sync
TheTom with a normal merge into a sync branch, the same way as above.

Kept on purpose when it differed from upstream:

- `ssm_alpha`/`ssm_beta` quantization exclusion only for `LLM_ARCH_QWEN4EXP`, so
  our quantize masks for Qwen3.5/3.6 and other GDN models reproduce.
- TQ WMMA prefill also checks the compiled CUDA arch (V100 on the CUDA 12.4
  archive runs `61-virtual` PTX).
- Vulkan moe cache checks device features before creating pipelines.
- Metal `LIGHTNING_INDEXER` restored (lost in an upstream merge).
- BoringSSL pin `0.20260803.0`.
- The `f32_pedantic` (strict decision precision) path keeps cuBLAS for F32
  transposed-vector matmuls.

Watch on the next sync:

- `vendors/hip.h` host-alloc APIs (kv-stream) must build on ROCm 7.2.1 and
  TheRock 10.0.0.
- `__ballot_sync` on HIP truncates to 32 bits on wave64; only NVIDIA calls it
  today.
- `tools/kv-stream-bench` links cudart; check CUDA archive contents and size.

## Known constraints

- Vulkan: banded-FA (Inkling) and lightning-indexer kernels are not
  implemented; those ops are rejected via `supports_op`. Turbo K/V flash
  attention, TURBO_WHT, turbo set_rows and GATED_DELTA_NET Vulkan kernels exist.
- Turbo K cannot be RoPE-shifted: the turbo dequant stays in its WHT basis, so
  `build_graph_shift` skips turbo K layers (context shift and `--cache-reuse`
  degrade output with turbo K). A correct shift needs an inverse WHT before RoPE
  on every backend.
- Inkling: no MTP/NextN support yet (heads in GGUF are ignored); the fork's
  MTP subsystem currently serves qwen35/step35/hy-v3. Planned work.
- Upstream removed `-sm row` (CUDA multi-GPU split-buffer) — gone since the
  inkling merge.
- CUDA-11: no TurboQuant build; the app maps such GPUs to the CPU backend.
