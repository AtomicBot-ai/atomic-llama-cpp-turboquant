#pragma once

// llama-server --decision: Decision API v1 (POST /v1/systemone, POST /v1/router/score)
// over a decision model, see DECISION.md. No chat routes and no llama_context:
// the engines come from tools/decision.

#include "server-http.h"

#include <functional>
#include <string>

struct common_params;

// check and adjust params before the HTTP server is set up; false on a fatal error
bool server_decision_prepare(common_params & params);

// general.architecture when it is the first GGUF key (as all writers do), else empty; for a quick hint only
std::string server_decision_gguf_arch(const std::string & path);

// true when path is a laya Hugging Face checkpoint directory (what --decision -m DIR converts); for a hint only
bool server_decision_is_checkpoint_dir(const std::string & path);

// without --decision: what to do with a Laya repackaging llama.cpp cannot load (a ggmlc or laya-head
// GGUF); "" for anything else, including modern-bert encoders (valid embedding models)
std::string server_decision_foreign_laya_hint(const std::string & path);

// register routes -> start HTTP -> load the model -> serve until the HTTP server stops.
// register_signals (may be empty) is called once shutdown_handler is set.
int server_decision_main(
        common_params & params,
        server_http_context & ctx_http,
        std::function<void(int)> & shutdown_handler,
        const std::function<void()> & register_signals);
