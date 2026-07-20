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

Every push to `dev` builds **linux-x64-vulkan, windows-x64
{cpu, vulkan, cuda-12.4, cuda-13.3}, macos-arm64** and republishes the
[`dev-latest`](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant/releases/tag/dev-latest)
rolling prerelease with all archives. PRs into `dev` build everything but
publish nothing. If some backend fails, `dev-latest` is still published with
the survivors and the notes list what is missing.

Grab-and-test on any machine:

```bash
gh release download dev-latest -R AtomicBot-ai/atomic-llama-cpp-turboquant \
  -p 'llama-turboquant-linux-x64-vulkan.tar.gz'   # or your platform
tar -xzf llama-turboquant-linux-x64-vulkan.tar.gz
./build/bin/llama-server --version    # → version: turboquant-vX.Y.Z (<count>, <sha>)
```

macOS dev builds are signed but **not notarized** (release builds are):
`xattr -dr com.apple.quarantine build/` after unpacking.

## Versioning & stable releases

Semver, single source of truth: the `TURBOQUANT_VERSION` file at the repo
root. CMake embeds it via `common/build-info.cpp.in`; `llama-server --version`
prints `version: turboquant-vX.Y.Z (<build-number>, <commit>)`.

Cut a release (from an up-to-date, clean `master` checkout):

```bash
./scripts/turboquant-release.sh patch|minor|major|X.Y.Z
```

This bumps `TURBOQUANT_VERSION`, commits `release: turboquant-vX.Y.Z`, tags
`turboquant-vX.Y.Z` and pushes. The tag triggers
`.github/workflows/release-turboquant.yml`: all backends are built (macOS
fully notarized) and published as **one** GitHub release with all archives.
A `verify-version` job fails the release if the tag doesn't match the file.

Consumers: `atomic-chat-conf/backends/turboquant-manifest.json` entries all
point at the same `turboquant-vX.Y.Z` tag; asset names
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
`llama-kv-cache.cpp`, `gguf-py/gguf/constants.py` (quant type ids — the fork
renumbered `Q2_0` to 47; upstream Q2_0 GGUFs are incompatible).

## Known constraints

- Vulkan: turbo3 flash-attn SPIR-V and banded-FA/lightning-indexer kernels are
  not implemented; those ops are rejected via `supports_op` (turbo KV cache
  falls back off on Vulkan). TURBO_WHT / turbo set_rows / GATED_DELTA_NET
  Vulkan kernels DO exist.
- Inkling: no MTP/NextN support yet (heads in GGUF are ignored); the fork's
  MTP subsystem currently serves qwen35/step35/hy-v3. Planned work.
- Upstream removed `-sm row` (CUDA multi-GPU split-buffer) — gone since the
  inkling merge.
- CUDA-11: no TurboQuant build; the app maps such GPUs to the CPU backend.
