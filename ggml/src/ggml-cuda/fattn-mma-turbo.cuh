#pragma once

// fattn-mma-f16.cuh must be included before this header (provides flash_attn_ext_f16,
// launch_fattn, fattn_kernel_t, and the MMA helper functions).

// Fused MMA-native turbo flash attention: reads raw turbo-quantized K/V directly,
// dequants into half2 shmem tiles inside the attention loop. No intermediate fp16 buffers.
// Uses the same flash_attn_ext_f16 kernel with type_K/type_V template params.

template <int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    // Turbo forces nstages=0: cp.async can't do ALU dequant, so tiles load synchronously.
    // With nstages=0, tile_K and tile_V share the same shmem region (overlap).
    constexpr int nstages = 0;

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = false; // Turbo K/V are separate tensors.

    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(nbatch_K2 + 4,  nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nbytes_shared_KV_1stage; // nstages=0 → 1-stage layout

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, type_K, type_V>;

#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, type_K, type_V>;

#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
    }

    // need_f16_K=false, need_f16_V=false: raw turbo data passes through to kernel.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, warp_size_host);
}


#define DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, ncols1, ncols2, tK, tV)                                  \
    template void ggml_cuda_flash_attn_ext_mma_turbo_case                                            \
    <DKQ, DV, ncols1, ncols2, tK, tV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)            \

// Straight TurboQuant matched K/V at D=128 and D=256. ncols2 <= 8.
#define DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_NCOLS2(DKQ, DV, ncols, tK)                         \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/1, 1, tK, tK);                               \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/2, 2, tK, tK);                               \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/4, 4, tK, tK);                               \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/8, 8, tK, tK);                               \

#define DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(DKQ, DV, ncols)                               \
    DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_NCOLS2(DKQ, DV, ncols, GGML_TYPE_TURBO4_0)              \
    DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_NCOLS2(DKQ, DV, ncols, GGML_TYPE_TURBO3_0)              \
    DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_NCOLS2(DKQ, DV, ncols, GGML_TYPE_TURBO2_0)              \

DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(128, 128,  8)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(128, 128, 16)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(128, 128, 32)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(128, 128, 64)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(256, 256,  8)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(256, 256, 16)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(256, 256, 32)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(256, 256, 64)
