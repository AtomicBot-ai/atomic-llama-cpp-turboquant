#pragma once

// Self-contained laya model: a ModernBERT encoder + a typed decision head for
// non-autoregressive decision making (choice / score / noul).
//
// This is a standalone ggml implementation following the tools/mtmd pattern:
// the model loads its own GGUF via gguf_init_from_file and builds the
// inference graph with ggml directly. It links only ggml.
//
// WARNING: This API is experimental and subject to change.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// number of marker (option) slots per sequence; must be >= max options per
// question (the Julia family allows 20). The reference collates marker_pos /
// marker_mask to this width dynamically, so keep this at least that high.
#define LAYA_MAX_MARKERS 20
// default BLAS backend threads (laya_context_params.n_threads_blas = 0)
#define LAYA_BLAS_THREADS_AUTO 8

struct laya_hparams {
    int32_t n_embd            = 0;   // hidden size (768)
    int32_t n_layer           = 0;   // encoder layers (22)
    int32_t n_head            = 0;   // attention heads (12)
    int32_t n_embd_head       = 0;   // head dim (64)
    int32_t n_ff              = 0;   // encoder intermediate size (1152)
    int32_t n_vocab           = 0;   // token embedding rows
    int32_t n_qtype           = 0;   // question types (choice/score/noul = 3)
    int32_t marker_token_id   = 0;   // <mask> token id
    int32_t max_len           = 0;   // max sequence length
    int32_t head_max_len      = 0;   // max option-region length
    int32_t n_act             = 0;   // action head classes
    int32_t n_head_layers     = 0;   // decision head transformer layers
    int32_t n_swa             = 0;   // encoder sliding window (128)
    int32_t swa_pattern       = 0;   // global_attn_every_n_layers (3)
    float   norm_eps          = 1e-5f;   // encoder RMSNorm epsilon
    float   norm_eps_layer    = 1e-5f;   // decision-head / scorer LayerNorm epsilon
    float   rope_freq_base    = 10000.0f;
    float   rope_freq_base_swa = 10000.0f;
    std::vector<float> temperature;  // per-qtype temperature (padded to n_qtype)
    // temperature_by_options of the checkpoint: "<qtype>:<bucket>" (buckets 2, 3-5, 6-10, 11+) -> T,
    // raw values (the reference clamps each one to [0.5, 5] when it applies it)
    std::vector<std::string> temperature_buckets;
    std::vector<float>       temperature_bucket_values;
};

// One batched forward pass: n_tokens packed tokens across n_seqs sequences.
// Sequences are packed without padding; separation is handled by an internal
// block-diagonal attention mask built from seq_id/positions.
struct laya_batch {
    int32_t           n_tokens = 0;   // total tokens across all sequences
    int32_t           n_seqs   = 0;   // number of sequences (questions)
    const int32_t *   tokens   = nullptr;   // [n_tokens] token ids
    const int32_t *   positions= nullptr;   // [n_tokens] per-token position
    const int32_t *   seq_id   = nullptr;   // [n_tokens] sequence id per token
    const int32_t *   qtype    = nullptr;   // [n_tokens] qtype id per token
    const int32_t *   marker_pos  = nullptr; // [n_markers_max, n_seqs] absolute token indices
    const int32_t *   marker_mask = nullptr; // [n_markers_max, n_seqs] 1 = valid marker
    const int32_t *   seq_start   = nullptr; // [n_seqs] first global token index of each seq
};

// Outputs of one forward pass.
struct laya_result {
    std::vector<float> logits;       // [n_markers_max, n_seqs] scorer logits at marker positions
    std::vector<float> act_logits;   // [n_act, n_seqs] raw action head logits
    int32_t n_markers_max = 0;
    int32_t n_seqs        = 0;
    int32_t n_act         = 0;
};

struct laya_model;
struct laya_context;

// how the weights are placed in memory
struct laya_model_params {
    // map the GGUF read-only and use token_embd in place: only get_rows reads it, so only the
    // rows of the tokens seen become resident (393 MB F16 / 209 MB Q8_0 for mmBERT-base).
    // Falls back to loading when the file cannot be mapped. Does not change any logits.
    bool use_mmap        = true;
    // lock the model memory in RAM (the mapping and the loaded weights); a failure only warns
    bool use_mlock       = false;
    // CPU repack buffers (ggml_backend_dev_get_extra_bufts) for the matmul weights that have a
    // repacked kernel (Q4_0 / Q4_K / Q5_K / Q6_K / Q8_0 ... on NEON dotprod / i8mm, AVX2).
    // Other kernels, other rounding: the logits change slightly (same accuracy class).
    bool use_extra_bufts = false;
    // compute device (laya_device_select): nullptr = the CPU. A GPU / iGPU device of the ggml backend
    // registry gets the weights in its buffer type; a weight whose op the device cannot run stays in host
    // memory (laya_placement.host_weights counts them). The load throws when the device buffer cannot be
    // allocated: computing from host weights would copy them to the device on every pass (the scheduler
    // offloads large-batch matmuls) or run them on the CPU. token_embd always stays on the host (file
    // mapping or CPU buffer): only get_rows reads it, on the CPU, and that keeps the 393 MB (F16 mmBERT)
    // table out of device memory. use_extra_bufts is CPU-only (throws with a device).
    ggml_backend_dev_t device = nullptr;
    // test hook (tests/test-laya-backend-parity.cpp): names of weights a device model keeps in host memory
    // as if the device could not run their op
    std::vector<std::string> host_weights;
};

// load a laya GGUF (F16 or quantized; dequantization happens inside ggml mul_mat)
laya_model * laya_model_load_from_file(const char * fname);
laya_model * laya_model_load_from_file_ext(const char * fname, const laya_model_params & params);

void laya_model_free(laya_model * model);

const laya_hparams & laya_model_hparams(const laya_model * model);

// where the weights went: loaded into the CPU buffer, used in place from the file mapping
// (token_embd with use_mmap), converted into repack buffers (use_extra_bufts)
struct laya_model_memory {
    size_t  device     = 0;  // bytes in the buffer of the compute device (0 on the CPU)
    size_t  loaded     = 0;  // bytes (host memory)
    size_t  mapped     = 0;  // bytes (virtual: only the rows of the tokens seen become resident)
    size_t  repacked   = 0;  // bytes before repacking
    int32_t n_repacked = 0;  // tensors
};
laya_model_memory laya_model_memory_info(const laya_model * model);

// ---- compute device ----
// The GPU / iGPU devices of the ggml backend registry, in llama.cpp order: the discrete GPUs first, then
// the integrated ones (ggml_backend_load_all first in GGML_BACKEND_DL builds).
std::vector<ggml_backend_dev_t> laya_gpu_devices();

// device for laya_model_params.device from a mode: "cpu" (nullptr), "gpu" (laya_gpu_devices()[gpu_index],
// throws when there is no such device), "auto" (laya_gpu_devices()[gpu_index] when it exists, else the CPU,
// with a log line). Throws on any other mode. A caller in auto mode also falls back to the CPU when the
// device then fails to load, initialize or warm up (engine-laya.cpp, laya-cli.cpp).
ggml_backend_dev_t laya_device_select(const std::string & mode, int32_t gpu_index = 0);

// "cpu" or the ggml device name of a device ("MTL0", "CUDA0", "Vulkan0", ...)
std::string laya_device_name(ggml_backend_dev_t device);

// the device a model was loaded for (nullptr = the CPU)
ggml_backend_dev_t laya_model_device(const laya_model * model);

// ---- tokenizer: self-contained ports of the two HF fast tokenizers of the laya checkpoints ----
// GGUF key decision.laya.tokenizer picks one (absent: metaspace-bpe, the only kind before it existed):
//   metaspace-bpe  laya-multilingual (mmBERT): split at added tokens, " " -> U+2581, prepend U+2581,
//                  split into words, BPE with byte fallback
//   bytelevel-bpe  laya, laya-typed-decisions (ModernBERT / OLMo BPE): split at special added tokens,
//                  NFC (decision.laya.normalizer), split at normalized added tokens, GPT-2 regex,
//                  byte-to-unicode, BPE
// Matches `tokenizer(text, add_special_tokens=False)["input_ids"]` from the reference; O(n log n)
// in the text length.

// tokenize with add_special_tokens=False (no <bos>/<eos>)
std::vector<int32_t> laya_tokenize(const laya_model * model, const std::string & text);

// the added-token strings (matched on the raw text) and whether each is a CONTROL token
void laya_vocab_added_tokens(const laya_model * model, std::vector<std::string> & text, std::vector<bool> & control);

// special token ids from the GGUF tokenizer metadata
int32_t laya_vocab_bos (const laya_model * model);
int32_t laya_vocab_sep (const laya_model * model);
int32_t laya_vocab_mask(const laya_model * model);

// the tokenizer string of the mask token (e.g. "<mask>"); the reference
// build_sequence replaces this literal string with a space in the inputs
std::string laya_vocab_mask_token(const laya_model * model);

// matmul precision request (ggml_mul_mat_set_prec) of the graph. The CPU and BLAS backends ignore it:
// on the CPU both modes compute the same bits. It matters on devices that pick lower-precision
// kernels by default (CUDA: TF32 / F16 accumulation, DECISION.md "Backend parity tiers").
//   default  KQ and PV (both operands F32 activations): GGML_PREC_F32_PEDANTIC; the matmuls with a
//            weight matrix: GGML_PREC_F32 (F32 accumulation; an F32 weight may still run in TF32)
//   strict   GGML_PREC_F32_PEDANTIC on every matmul (the strict-f32 parity mode)
// On CUDA the graph also dequantizes some weights to F32 before their matmul and, in strict, applies
// RoPE from host tables (laya_weight_dequantized; DECISION.md "Backend parity tiers").
enum laya_precision {
    LAYA_PRECISION_DEFAULT = 0,
    LAYA_PRECISION_STRICT  = 1,
};

// "default" / "strict"
const char * laya_precision_name(laya_precision precision);
// false for any other name
bool         laya_precision_from_name(const std::string & name, laya_precision & precision);

// true when the graph on this device dequantizes a weight of this type to F32 before its matmul at
// this precision (CUDA: F16 in strict, Q8_0 always), so the matmul runs F32 x F32
bool laya_weight_dequantized(ggml_backend_dev_t device, laya_precision precision, ggml_type type);

struct laya_context_params {
    int32_t n_threads = 1;
    // polling level of the persistent threadpool between graphs (0..100, see
    // ggml_threadpool_params.poll); within a graph the workers always spin on the barrier
    int32_t poll      = 0;
    // add the BLAS backend (Accelerate on macOS) to the scheduler: it takes the matmuls it
    // supports (F32/F16/quantized weights converted to F32, sgemm). No activation rounding and
    // F32 accumulation, so the logits change (with Accelerate: closer to the PyTorch reference).
    // Off here; the decision engine's kernels "auto" turns it on for Accelerate. laya_init_ext
    // throws when the build has no BLAS backend, and for a device model (BLAS is CPU-only).
    // ggml-blas takes a matmul only when all its dimensions are >= 32: sequences shorter than 32
    // tokens and the marker-row head matmuls still run on the ggml CPU kernels. Repacked weights
    // (use_extra_bufts) are not host memory, BLAS never takes them.
    bool    use_blas  = false;
    // threads of the BLAS backend: they convert the F16 / quantized weights to F32 before every
    // sgemm, and without OpenMP ggml-blas starts n_threads_blas - 1 new std::async threads for
    // each of those conversions (Accelerate runs the sgemm on its own threads). 0: auto,
    // min(n_threads, LAYA_BLAS_THREADS_AUTO) (measured, DECISION.md "CPU kernels and memory")
    int32_t n_threads_blas = 0;
    // macOS: raise the QoS of the thread that runs laya_encode to USER_INITIATED (never lowers it)
    bool    qos       = true;
    // matmul precision request, see laya_precision
    laya_precision precision = LAYA_PRECISION_DEFAULT;
    // with a device model: laya_init_ext throws when a graph node outside the allowlist runs on the CPU or
    // a weight stays in host memory (see laya_placement). Off: such nodes run on the CPU
    // (ggml_backend_sched copies their inputs).
    bool    strict_placement = false;
    // layer trace: an existing directory; every laya_encode of the context dumps the outputs of the layers,
    // the marker rows and the logits there (trace.jsonl + raw float32 files, compared with
    // tools/laya/trace_diff.py). Empty: off. The CPU computes the same bits with and without it. The library
    // reads no environment: llama-laya-cli and llama-decision-bench take LAYA_TRACE_DIR, llama-server only
    // with --decision-debug (a trace holds the activations of every request).
    std::string trace_dir;
};

// Where ggml_backend_sched put the nodes of a graph. laya_init_ext builds the graph of the warm-up input
// (64 tokens, one sequence), lets the scheduler place it (ggml_backend_sched_alloc_graph, no compute) and
// keeps the report. The placement can depend on the input size: ggml-blas takes a matmul only when all
// its dimensions are >= 32, the scheduler offloads an op on host weights to the device only from a
// minimum batch (op_offload; host_weights counts such weights instead), and a backend's supports_op may
// look at the shapes. The probe sees the 64-token graph only.
//   nodes        compute nodes (views, reshapes, permutes and transposes compute nothing: not counted)
//   backends     compute nodes per backend name, in the scheduler's order ("MTL0", "BLAS", "CPU")
//   n_splits     graph splits (every split boundary copies the inputs of the next backend)
//   cpu_fallback compute nodes of a device context that run on the CPU although the design does not
//                put them there. Allowed on the CPU: get_rows of token_embd (host memory by design),
//                casts of graph inputs (the I32 marker_mask to F32) and of that get_rows result, and
//                every node of a CPU context (BLAS included). fallback_ops: "OP xN" per op.
//   host_weights weights of a device model that stay in host memory because the device cannot run their
//                op (laya_model_params.device); their nodes may still show on the device (op_offload)
struct laya_placement {
    std::string device;          // laya_device_name of the context's device
    int32_t     n_nodes      = 0;
    int32_t     n_splits     = 0;
    std::vector<std::pair<std::string, int32_t>> backends;
    int32_t     cpu_fallback = 0;
    std::vector<std::string> fallback_ops;
    int32_t     host_weights = 0;
};
laya_placement laya_context_placement(const laya_context * ctx);
// one line for the log: "MTL0: 1203 nodes (MTL0 1170, CPU 33), 3 splits, cpu fallback 0"
std::string laya_placement_str(const laya_placement & p);

// one persistent ggml threadpool per context (created here, freed by laya_free).
laya_context * laya_init(const laya_model * model, int n_threads);
laya_context * laya_init_ext(const laya_model * model, const laya_context_params & params);

// one forward pass of n_tokens (capped at the model's max_len): starts the threads, faults in
// the weights and sizes the compute buffer for n_tokens. A longer request later grows the buffer
// again (ggml_backend_sched_alloc_graph), so this does not bound the compute memory.
int laya_warmup(laya_context * ctx, int32_t n_tokens = 64);

// "cpu", "cpu+repack", "cpu+blas", ...: the kernels a context computes with; a device context: the
// lower-case name of the device's backend ("metal", "cuda", "vulkan")
std::string laya_context_kernels(const laya_context * ctx);

// threads of the BLAS backend of a context; 0 without BLAS
int32_t laya_context_n_threads_blas(const laya_context * ctx);

// matmul precision request of a context
laya_precision laya_context_precision(const laya_context * ctx);

// description of the BLAS backend of this build ("Accelerate", "OpenBLAS", ...), "" if none
std::string laya_blas_description();

// performance cores of this machine: macOS hw.perflevel0.physicalcpu, Windows the cores of
// the highest EfficiencyClass; 0 when unknown (other systems: keep the caller's default)
int32_t laya_cpu_perf_cores();

// OpenMP builds: KMP_BLOCKTIME=0 and OMP_WAIT_POLICY=passive unless the user set them, so idle
// OpenMP workers sleep instead of spinning between requests. Call before the first compute.
// (LLVM libomp reads them at its first parallel region; GNU libgomp only at program start.)
void laya_cpu_env_defaults();

#if defined(_WIN32)
// File names are UTF-8 here, as gguf_init_from_file / ggml_fopen take them. The narrow Win32 and
// CRT calls (CreateFileA, std::ifstream(const char *)) read a name in the ANSI code page, so the
// loader opens files through this UTF-16 copy. Empty for invalid UTF-8.
std::wstring laya_utf8_to_wide(const std::string & utf8);
#endif

void laya_free(laya_context * ctx);

// run one forward pass; returns 0 on success, non-zero on failure. The batch is checked first
// (O(n_tokens)): token ids within the token_embd rows, qtype within type_emb, seq_start and every
// marker_pos slot (masked ones too: get_rows gathers them) within [0, n_tokens). A device get_rows
// has no bounds check, so an index outside would read outside the tensor.
int laya_encode(laya_context * ctx, const laya_batch & batch, laya_result & result);
