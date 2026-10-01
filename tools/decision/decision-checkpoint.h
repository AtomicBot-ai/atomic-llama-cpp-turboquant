#pragma once

// -m DIR for the decision engine: a laya Hugging Face checkpoint directory is converted once into a
// cached GGUF (tools/laya/laya-convert.h, byte-identical to convert_hf_to_gguf.py) and the GGUF is
// then loaded exactly like one given with -m FILE. One execution path: the engine never reads the
// checkpoint itself.
//
// Cache: <cache_dir>/<key>/<name>.gguf, never inside the checkpoint directory (checked on the
// deepest existing ancestor before anything is created, and again after).
//   cache_dir  --decision-convert-cache, else decision_checkpoint_default_cache()
//   name       the directory name (laya_convert_path_name, "checkpoint" when it is empty or not a
//              portable file name), so that the default model name and spec model_id are the
//              directory name, as for a GGUF named <name>.gguf
//   key        first 32 hex digits of the sha256 of a manifest of:
//                cache format, LAYA_CONVERT_VERSION, outtype, the directory name (general.* come from it),
//                the files the converter can read: those directly in the checkpoint root, in encoder/
//                and in tokenizer/ (sorted relative paths, dot-entries skipped):
//                  files <= 8 MiB (configs, README, the English tokenizer.json): size + sha256 of the content
//                  larger files (weights, the 256k-vocab tokenizer.json): size + mtime in ns + symlink target
//                any other subdirectory by its name only (the converter never reads below it; the laya
//                repo keeps two more checkpoints and eval data there). At most 20000 entries.
// Why not hash every byte: the weights are 0.6-0.8 GB, and hashing them at every start costs more
// than converting them (sha256 of 644 MB: 1.43 s; the f16 conversion: 0.2-1.8 s); size + ns mtime
// changes on every rewrite, and in the HF cache the symlink target is the blob name, which is the
// content hash. The small files, where an edit can keep the size and land within the mtime
// granularity (a digit in a config), are hashed by content.
// Writes are atomic and durable: convert to a unique temporary name in the key directory (synced to
// disk), then rename and sync the directory. Temporary files older than 10 minutes in the key
// directory (a killed conversion) are removed before converting there. A cache entry that is not a
// readable laya GGUF (size, tensor data) is converted again. On POSIX a key directory or entry owned
// by another user, writable by others or a symlink is refused (a shared LLAMA_CACHE).
// -m DIR offers f16 and f32: the converter's q8_0 is not the precision-protected recipe (laya-convert.h).

#include "laya-convert.h"

#include <cstdint>
#include <functional>
#include <string>

struct decision_checkpoint_params {
    std::string          cache_dir;                  // empty: decision_checkpoint_default_cache()
    laya_convert_outtype outtype = LAYA_CONVERT_F16;
    std::function<void(const std::string &)> log;    // progress lines, may be empty
};

// where a model came from; for /props and the logs
struct decision_model_source {
    std::string kind = "gguf";   // "gguf" | "checkpoint-dir"
    std::string input;           // -m as given
    std::string gguf_path;       // the file the engine loads (input for "gguf")
    // "checkpoint-dir" only
    std::string cache_dir;
    std::string key;
    std::string outtype;
    bool        cache_hit  = false;
    double      convert_ms = 0.0;
};

// true when path names an existing directory (UTF-8, symlinks followed)
bool decision_path_is_dir(const std::string & path);

// true when dir looks like a laya checkpoint (rl_agent_config.json at its top); for hints only,
// the converter does the real checks
bool decision_is_laya_checkpoint_dir(const std::string & dir);

// file name stem of the cached GGUF of dir, which is also the default model name
std::string decision_checkpoint_name(const std::string & dir);

// $LLAMA_CACHE/laya/gguf-cache, else the platform user cache + llama.cpp/laya/gguf-cache:
// macOS ~/Library/Caches, Linux $XDG_CACHE_HOME or ~/.cache, Windows %LOCALAPPDATA%. "" when unknown.
std::string decision_checkpoint_default_cache();

// cache key of dir for params (see the top of this file); "" and err on failure
std::string decision_checkpoint_key(const std::string & dir, const decision_checkpoint_params & params, std::string & err);

// -m path -> the GGUF to load. A file is used as is (kind "gguf"); a directory is converted into
// the cache, or the cached conversion is reused. false with err on failure (nothing is left behind).
bool decision_model_source_resolve(const std::string & path, const decision_checkpoint_params & params,
                                   decision_model_source & out, std::string & err);
