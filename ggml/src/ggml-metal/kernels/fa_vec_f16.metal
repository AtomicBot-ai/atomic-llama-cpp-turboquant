#include "common.h"
#include "dequantize.h"
#include "fa_vec_common.metal"

#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 32, 32, 4>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 64, 64, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 96, 96, 4>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 64, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 64, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 96, 64, 4>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 96, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk128_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 128, 96, 4>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 192, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 128, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 256, 256, 1>;

// Four query heads share one KV head. Each group covers 128 keys, and writes
// the partial sums in the ordinary vector-attention reduction layout.
kernel void kernel_flash_attn_ext_vec_bf16_dk64_dv64_gqa4(
        constant ggml_metal_kargs_flash_attn_ext_vec & a,
        device const char * q, device const char * k, device const char * v,
        device const char * mask, device const char * sinks, device const char * pad,
        device char * dst, device const char * idx,
        uint3 g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort D = 64, H = 4, T = 128, splits = 32;
    // Matrix tiles have eight rows. Only the first four are model heads.
    threadgroup float queries[8*D];
    threadgroup float scores[8*T];
    threadgroup float output[8*D];
    const uint head = g.y*H, split = g.z;
    device const bfloat * keys   = (device const bfloat *)(k + g.y*a.nb12 + split*T*a.nb11);
    device const bfloat * values = (device const bfloat *)(v + g.y*a.nb22 + split*T*a.nb21);
    device const half * masks = (device const half *)mask + split*T;
    device float * stats = (device float *)dst + a.ne02*D*splits;
    for (ushort i = tid; i < 8*D; i += H*32) {
        queries[i] = i < H*D ? ((device const float *)(q + (head+i/D)*a.nb02))[i%D] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort col = 8*sg; col < T; col += 8*H) {
        auto score = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (ushort d = 0; d < D; d += 8) {
            simdgroup_float8x8 query, key;
            simdgroup_bfloat8x8 raw;
            simdgroup_load(query, queries+d, D);
            simdgroup_load(raw, keys + col*(a.nb11/2) + d, a.nb11/2, 0, true);
            key.thread_elements()[0] = float(raw.thread_elements()[0]);
            key.thread_elements()[1] = float(raw.thread_elements()[1]);
            simdgroup_multiply_accumulate(score, query, key, score);
        }
        simdgroup_store(score, scores+col, T);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float peak = -FLT_MAX;
    for (ushort i = lane; i < T; i += 32) {
        const float s = fma(scores[sg*T+i], a.scale, float(masks[i]));
        scores[sg*T+i] = s;
        peak = max(peak, s);
    }
    peak = simd_max(peak);
    float sum = 0;
    for (ushort i = lane; i < T; i += 32) {
        const float p = exp(scores[sg*T+i] - peak);
        scores[sg*T+i] = p;
        sum += p;
    }
    sum = simd_sum(sum);
    if (!lane) {
        stats[((head+sg)*splits+split)*2] = sum;
        stats[((head+sg)*splits+split)*2+1] = peak;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort col = 8*sg; col < D; col += 8*H) {
        auto result = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (ushort i = 0; i < T; i += 8) {
            simdgroup_float8x8 probability, value;
            simdgroup_bfloat8x8 raw;
            simdgroup_load(probability, scores+i, T);
            simdgroup_load(raw, values + i*(a.nb21/2) + col, a.nb21/2);
            value.thread_elements()[0] = float(raw.thread_elements()[0]);
            value.thread_elements()[1] = float(raw.thread_elements()[1]);
            simdgroup_multiply_accumulate(result, probability, value, result);
        }
        simdgroup_store(result, output+col, D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < H*D/4) {
        ((device float4 *)dst)[(head*(D/4)+tid)*splits+split] = ((threadgroup float4 *)output)[tid];
    }
}

// One group handles eight query heads sharing a KV head, for one verify row
// and one 128-token split. The ordinary split reduction consumes its output.
kernel void kernel_flash_attn_ext_vec_bf16_dk256_dv256_gqa8(
        constant ggml_metal_kargs_flash_attn_ext_vec & a,
        device const char * q, device const char * k, device const char * v,
        device const char * mask, device const char * sinks, device const char * pad,
        device char * dst, device const char * idx,
        uint3 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort D = 256, H = 8, T = 128, splits = 32;
    threadgroup half queries[H * D];
    threadgroup float scores[H * T];
    threadgroup half probabilities[H * T];
    threadgroup float output[H * D];
    const uint qrow = g.x, kvhead = g.y, split = g.z;
    const uint first_head = kvhead * H, first_token = split * T;
    const ulong row = qrow * a.ne1 + first_head;
    device const bfloat * keys = (device const bfloat *) (k + kvhead*a.nb12 + first_token*a.nb11);
    device const bfloat * values = (device const bfloat *) (v + kvhead*a.nb22 + first_token*a.nb21);
    device const half * masks = (device const half *) (mask + qrow*a.nb31) + first_token;
    device float * stats = (device float *) dst + a.ne01*a.ne02*D*splits;
    for (uint h = 0; h < H; ++h) {
        device const float * source = (device const float *) (q + qrow*a.nb01 + (first_head+h)*a.nb02);
        queries[h*D+t] = half(source[t]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort col = 8*sg; col < T; col += 64) {
        auto score = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (ushort d = 0; d < D; d += 8) {
            simdgroup_half8x8 query, key;
            simdgroup_bfloat8x8 raw;
            simdgroup_load(query, queries + d, D);
            simdgroup_load(raw, keys + col*(a.nb11/2) + d, a.nb11/2, 0, true);
            key.thread_elements()[0] = half(raw.thread_elements()[0]);
            key.thread_elements()[1] = half(raw.thread_elements()[1]);
            simdgroup_multiply_accumulate(score, query, key, score);
        }
        simdgroup_store(score, scores + col, T);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float peak = -FLT_MAX;
    for (ushort i = lane; i < T; i += 32) {
        const float s = fma(scores[sg*T+i], a.scale, float(masks[i]));
        scores[sg*T+i] = s;
        peak = max(peak, s);
    }
    peak = simd_max(peak);
    float total = 0;
    for (ushort i = lane; i < T; i += 32) {
        const float p = exp(scores[sg*T+i] - peak);
        total += p;
        probabilities[sg*T+i] = half(p);
    }
    total = simd_sum(total);
    if (lane == 0) {
        stats[(row+sg)*2*splits + 2*split] = total;
        stats[(row+sg)*2*splits + 2*split+1] = peak;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort col = 8*sg; col < D; col += 64) {
        auto result = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (ushort i = 0; i < T; i += 8) {
            simdgroup_half8x8 p, value;
            simdgroup_bfloat8x8 raw;
            simdgroup_load(p, probabilities + i, T);
            simdgroup_load(raw, values + i*(a.nb21/2) + col, a.nb21/2);
            value.thread_elements()[0] = half(raw.thread_elements()[0]);
            value.thread_elements()[1] = half(raw.thread_elements()[1]);
            simdgroup_multiply_accumulate(result, p, value, result);
        }
        simdgroup_store(result, output + col, D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort i = t; i < H*D/4; i += 256) {
        ((device float4 *) dst)[((row+i/(D/4))*(D/4) + i%(D/4))*splits + split] =
                ((threadgroup float4 *) output)[i];
    }
}
#endif
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 320, 256, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 512, 512, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 576, 512, 2>;
#endif

#undef FA_TYPES
#undef FA_TYPES_F32

