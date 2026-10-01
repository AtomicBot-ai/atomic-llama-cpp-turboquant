#pragma once

// laya HF checkpoint directory -> GGUF, without Python.
//
// Port of convert_hf_to_gguf.py for the laya checkpoints (conversion/laya.py and the ModernBert /
// Bert / TextModel / ModelBase classes it inherits, gguf-py SpecialVocab and Metadata). The Python
// converter is the reference: for the inputs listed below the output is byte-identical to
//   convert_hf_to_gguf.py <dir> --outfile <out> --outtype <f32|f16|q8_0> [--model-name NAME]
//
// Reads (like Python):
//   rl_agent_config.json, encoder/config.json      hparams (the laya hparams loader)
//   model*.safetensors (+ model.safetensors.index.json)   tensors, sorted by name within each file
//   tokenizer/tokenizer.json, tokenizer_config.json      vocab, merges, special ids (tokenizer/ or the root)
//   tokenizer/config.json, chat_template.jinja|json      special ids, chat template (string form)
//   README.md front matter (YAML subset), generation_config.json
//   the directory name                               general.name / finetune / basename / version / size_label
// Refuses (clear error) what the port does not cover instead of writing a different file:
//   a root config.json (Python would not use the laya loader), pytorch_model*.bin (also a
//   model.safetensors.index.json without any model*.safetensors file), non-float dtypes,
//   tensor names outside the laya table (Python also accepts aliases of other architectures),
//   modules.json (sentence-transformers pooling), a tokenizer_class other than
//   PreTrainedTokenizerFast / TokenizersBackend (or, without one, a tokenizer/config.json with
//   model_type or tokenizer_class: AutoTokenizer would pick a model-specific class), auto_map,
//   rope scaling / experts / quantization_config / id2label / text_config in the encoder config,
//   added tokens whose AutoTokenizer round trip is not a known identity or U+2581 -> space,
//   special tokens that AutoTokenizer would add, list-form chat templates, YAML beyond the subset
//   (for the keys that are used), non-finite weights with q8_0, absurd sizes that Python could not
//   finish either (block_count > 65536, vocab_size > 2^24) and a vocab_size above the token_embd
//   rows (the padded vocab would not load), JSON nested deeper than 127 levels (what the tokenizers
//   library accepts; copying such a value would overflow the stack), and a generation_config.json
//   with NaN / Infinity / lone surrogate escapes (Python reads them, nlohmann does not; a plain
//   syntax error is ignored as in Python).
// q8_0 is convert_hf_to_gguf.py --outtype q8_0: every 2-D weight whose rows are whole blocks,
// including token_embd, type_emb, scorer.* and act_head.*. That is NOT the precision-protected recipe
// the laya accuracy figures were measured on (tests/laya/quantize.sh keeps those at F16), so
// llama-server --decision -m DIR does not offer it.
// Not offered: --metadata overrides, splitting, bf16 / tq outtypes.
//
// The GGUF bytes are written here, not with gguf_write_to_file: ggml's writer stores
// n_dims = ggml_n_dims() and drops trailing 1-dims, gguf-py stores len(shape) (scorer.3.weight
// {1024, 1} has n_dims 2). The result is read back with gguf_init_from_file_ptr and checked.
//
// Paths are UTF-8 (UTF-16 file APIs on Windows). All checkpoint input is untrusted: sizes, offsets,
// dtypes and shapes are checked before use.

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>

// Version of the bytes this converter writes. Part of the key of the decision GGUF cache
// (tools/decision/decision-checkpoint.h): bump it with every change that can change the output
// for some input, or that refuses an input an older build converted, so that cached conversions
// made by an older build are not reused.
//   2: refuses tokenizer_class / modules.json / deep JSON / index without model*.safetensors, and a
//      generation_config.json with NaN / Infinity (version 1 wrote those files)
#define LAYA_CONVERT_VERSION 2

// values are general.file_type
enum laya_convert_outtype {
    LAYA_CONVERT_F32  = 0,
    LAYA_CONVERT_F16  = 1,
    LAYA_CONVERT_Q8_0 = 7,
};

struct laya_convert_params {
    laya_convert_outtype outtype = LAYA_CONVERT_F16;
    // --model-name; an empty name writes no general.name (gguf-py skips empty strings)
    bool        has_model_name = false;
    std::string model_name;
    // one line per step and per tensor; empty: silent
    std::function<void(const std::string &)> log;
};

// convert the checkpoint directory `dir` into the GGUF file `out`. Returns "" on success, else the
// error. The output is written to a new file `out`.<random hex>.tmp (created exclusively: never an
// existing file or a symlink), synced to disk and renamed at the end; nothing is left on failure.
std::string laya_convert(const std::string & dir, const std::string & out, const laya_convert_params & params);

// compare two GGUF files. Returns "" when they are byte-identical, else the first difference:
// header, KV (in order), tensor info, tensor data, then the first differing byte. When the files
// cannot be compared (missing, unreadable, not a GGUF) the message says why and *failed is set.
std::string laya_convert_compare(const std::string & a, const std::string & b, bool * failed = nullptr);

// "f32" | "f16" | "q8_0"
bool laya_convert_parse_outtype(const std::string & s, laya_convert_outtype & out);
const char * laya_convert_outtype_name(laya_convert_outtype t);

// Path(dir).name as the converter sees it: the last component that is not empty or ".", "" for
// "." or "/". general.name / basename / finetune / version / size_label are derived from it, so
// the same checkpoint under another directory name converts to different bytes.
std::string laya_convert_path_name(const std::string & path);

// rename with replace (MoveFileExW on Windows); false on failure
bool laya_convert_rename(const std::string & from, const std::string & to);

// helpers, public for the tests
FILE *   laya_convert_fopen(const std::string & utf8_path, const char * mode);
bool     laya_convert_remove(const std::string & utf8_path);
// numpy astype(float16) on AArch64 / x86-64 F16C: round to nearest even, NaN quieted with the top payload bits
uint16_t laya_convert_fp32_to_fp16(float f);
// numpy astype(float32) of float16 on the same machines: exact, NaN quieted
float    laya_convert_fp16_to_fp32(uint16_t h);
// gguf-py Metadata.get_model_id_components + id_to_title for a directory name, for the tests:
// "name|basename|finetune|version|size_label" with "-" for None
std::string laya_convert_dir_metadata(const std::string & dir_name, int64_t total_params);
