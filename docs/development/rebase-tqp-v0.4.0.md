# Rebase onto TurboQuant+ tqp-v0.4.0 (2026-10)

`dev` moved from TheTom's old fork history (last pulled 2026-06-19, `4595fff0b`) to his re-ported tree, tag `tqp-v0.4.0` (`bcb85fc3a`, 2026-09-28). TheTom re-ported TurboQuant onto fresh ggml on 2026-07-31, so a merge saw every turbo file as added on both sides (75 files, 237 conflict hunks). Instead, our own commits were re-applied on top of his tree.

Our commits: `git log --no-merges <upstream tip>..<old dev> --not ggml/master 4595fff0b` (140 commits). Each re-applied commit keeps its author and names the source sha (`cherry picked from commit ...`).

## Result per group

| Group | Result |
|---|---|
| CI, releases, versioning, signing, ROCm/CUDA/DGX Spark archives | cherry-picked |
| NVFP4 quantize target, sched split-input limit, mtmd `--chat-template-file` | cherry-picked |
| Inkling, Kimi K3, BailingMoeV3, GDN rsqrt | cherry-picked, conflicts resolved (see below) |
| decision / laya | cherry-picked (clean) |
| Laguna | not re-applied: upstream port is a superset (DFlash capture, converter, autoparser tests for S-2.1 / XS-2.1 / XS.2) |
| Gemma4 UV/UA projectors, suppress tokens, Gemma4Unified | not re-applied: upstream has them |
| fattn-tile D=512, HIP f16-turbo instances, build_attn reshape | not re-applied: upstream has them |
| GQA ncols2 dispatch (`61ee3eb9d`) | not re-applied: upstream reverted it on purpose (AMD WMMA abort on gfx1201) |
| MTP / NextN / speculative | audited line by line, see below |

## Decisions

- `dd27e6d34`: dropped the dead `kv_only_nextn` branch in `llama_hparams::has_kv` (never set on dev).
- Inkling `a015409e6`: banded FA wired into upstream MMA signatures (`indices` + `rel_f`, `rel_extent`/`head_q0`/`ne11`); upstream transposed-vector MMVF path gated by `!f32_pedantic` so strict precision keeps the old cuBLAS path; mmq/mmvq dst offsets int64; `mtmd-image` keeps upstream rounding (Pillow-exact only for Lanczos); `init_inkling` taken from dev. Banded op is CPU + CUDA only, as before.
- `807cc6ec9` (host-staged cross-device copy helpers): skipped, no callers upstream; non-P2P copies go through `cudaMemcpyPeerAsync` or the scheduler fallback.
- `dee010c9e`: the SET_ROWS merge artifact does not exist upstream; only its CI concurrency and MERGE_NOTES parts remain.
- `4dfedb081`, `d70d2597a`: empty, upstream already has `LLAMA_MAX_EXPERTS 1024` and `add_vision_head_dim`.

MTP / NextN audit (lines added by each commit that are alive on old dev but absent now):

- Public C API (functions, `llama_context_params`, `llama_model_params`, quantize and sampler params, `LLAMA_*` enums): nothing lost.
- Ported: `--spec-type mtp|nextn` and `--mtp-head` aliases (`4c21bbc4f`), the clear error for pre-b10018 `gemma4_assistant` GGUFs (`29017fd55`), MTP.md / NEXTN.md, helper and UDT scripts.
- GDN partial `seq_rm`: upstream has the mainline `n_rs_seq` implementation.
- Draft mtmd processing (`TAG_MTMD_DRAFT_PROCESSING`): mainline replaced it with the mtmd post-decode callback (#24645).
- Turbo K-shift skip (`31df030fe`): not ported. Upstream shifts quantized K correctly (dequant, inverse Hadamard, RoPE, Hadamard, requant); the old skip left stale positions.
- iSWA `get_can_shift` without the size check (`d1333b0bc`): not ported. Mainline keeps the check for correctness; Gemma 4 with `--cache-reuse` needs `--swa-full` for fast TTFT.
- Removed upstream (mainline #26254): OuteTTS vocoder flags of `llama-tts`.

## Compatibility fixes on top of upstream

See `TURBOQUANT.md`, section "TurboQuant+ (TheTom) base": TQ WMMA compiled-arch check (Volta on CUDA 12.4), BoringSSL pin, Vulkan moe cache feature check, Metal `LIGHTNING_INDEXER` restored, `ssm_alpha`/`ssm_beta` exclusion limited to Qwen4-Exp, real-arch CUDA compile job in `dev-build.yml`.
