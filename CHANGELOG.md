# Changelog

One section per release, keyed by the exact tag. `verify-version` refuses to cut
a release whose tag has no section here, so this gets written *before* the tag is
pushed — the release notes are generated from it verbatim.

Write it for the person who downloads the build: what they get, what changed for
them, what to watch out for. Not a commit dump — the notes already carry the full
commit list underneath.

Releases before `b10269-1.5.0` predate this file; see the git history.

## b10269-1.5.0

### Added

- **NVIDIA DGX Spark (GB10) support.** New archive
  `llama-turboquant-linux-arm64-cuda-13.3`, built natively for aarch64 with
  CUDA 13.3 and sm_121 SASS. Other arm64 NVIDIA machines (GH200, GB200, Jetson
  Thor) run it too, JITing the kernels from PTX on first launch. This is the
  first Linux arm64 build the fork ships — until now arm64 meant macOS only.
- **BailingMoeV3 (Ling 3.0) architecture support**, including the KDA gate
  handling.

### Changed

- **Linux CUDA archives are roughly half the size** — 1657 → 956 MB (12.4) and
  1879 → 1028 MB (13.3) measured across both the `.zip` and `.tar.gz`. The zips
  were storing `libcublas.so` → `.so.13` → `.so.13.5.1.27` as three full copies
  because `zip` followed the symlinks.
- **CUDA 13.3 builds ship Ampere PTX (`80-virtual`).** A100/H100/B200 were
  falling back to the Turing PTX floor, which silently disabled `cp.async` and
  the Ampere MMA path — both gated on `__CUDA_ARCH__ >= 800`. Those cards get
  Ampere-class kernels now. No architecture lost support in this release.
- Windows CUDA builds got their architecture lists pinned, all runner cores, a
  ccache that can actually hold a CUDA build, and 7-Zip instead of
  `Compress-Archive`. Release turnaround drops accordingly.

### Notes

- The DGX Spark archive has **not yet been validated on real GB10 hardware** —
  it is built and arch-checked in CI (`cuobjdump` asserts sm_121 SASS is
  present), but nobody has run it on a Spark yet. Treat this one as beta and
  report back.
- The CUDA 13.3 archives now use `-compress-mode=size`. Kernel SASS is
  unchanged and inference speed is unaffected; the fatbin is decompressed once
  at module load. It needs a driver from the CUDA 12.4 era or newer.
