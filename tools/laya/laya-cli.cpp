// llama-laya-cli: run the self-contained laya decision model.
//
// Loads a laya GGUF through the tools/laya library (self-contained ggml graph
// and tokenizer).
// Reads a JSON file containing `state` + `questions`, builds the inputs with
// laya-decide (port of the PyTorch `build_sequence`), runs one forward pass through the
// encoder + decision head, and prints per-question answers
// (choice/score/noul + probabilities + confidence + action) plus the raw
// per-question outputs used to compare against tests/laya/golden.
//
// Usage:
//   llama-laya-cli -m laya-f16.gguf -f input.json [-t N] [-b RUNS] [--kernels NAME] [--no-mmap] [--mlock]
//       --kernels: default (ggml CPU kernels; the golden files and tests/laya/verify_precision.py
//       use it), auto (what llama-server --decision runs: blas when the BLAS backend is
//       Accelerate, else default), repack (CPU repack buffers), blas (BLAS backend), repack+blas.
//       Kernels change the logits slightly.
//   llama-laya-cli -m laya-f16.gguf --tokenize strings.jsonl   (one JSON string per line;
//       prints the token ids of each, add_special_tokens=False, for tests/laya/verify_tokenizer.py)
//   -m may also be a laya Hugging Face checkpoint directory: it is converted once into the GGUF
//       cache and the cached GGUF is loaded (as llama-server --decision -m DIR, see
//       tools/decision/decision-checkpoint.h); --decision-convert-cache DIR, --decision-convert-type
//       f16|f32 (default f16). "model" in the output is -m as given.
//
// input.json format (same as the golden fixtures under tests/laya/golden):
//   {
//     "state": <string | dict | list>,
//     "questions": {
//       "q1": {"type": "choice", "instructions": "...", "criteria": {...}},
//       "q2": {"type": "score",  "instructions": "...", "criteria": [...]},
//       "q3": {"type": "noul",   "instructions": "..."}
//     }
//   }
// Questions are read as the PyTorch reference reads them (laya_question_parse in
// tools/decision/laya-decide.h: noul labels, capitalised noul keys, non-string
// instructions, list choice labels, ...); a missing or null state is an error and an
// empty "questions" object prints empty answers. Answers have the reference Agent shape.

#include "laya.h"
#include "laya-decide.h"
#include "decision.h"
#include "decision-checkpoint.h"
#include "decision-json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <fcntl.h>
#    include <io.h>
#endif

// --tokenize: token ids of each JSON string line, timings on stderr
static int laya_cli_tokenize(const laya_model * model, const std::string & path) {
    FILE * f = ggml_fopen(path.c_str(), "rb"); // UTF-8 name, also on Windows
    if (!f) {
        fprintf(stderr, "laya: failed to open '%s'\n", path.c_str());
        return 1;
    }
    std::string line;
    size_t n_lines = 0, n_bytes = 0;
    double total_ms = 0.0, max_ms = 0.0;
    int c;
    bool eof = false;
    while (!eof) {
        line.clear();
        while ((c = fgetc(f)) != EOF && c != '\n') {
            line += (char) c;
        }
        eof = c == EOF;
        if (line.empty()) {
            continue;
        }
        json s;
        std::string err;
        if (decision_json_parse(line, s, err) != DECISION_JSON_OK || !s.is_string()) {
            fprintf(stderr, "laya: line %zu is not a JSON string: %s\n", n_lines + 1, err.c_str());
            fclose(f);
            return 1;
        }
        const std::string text = s.get<std::string>();
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<int32_t> ids = laya_tokenize(model, text);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        total_ms += ms;
        max_ms    = std::max(max_ms, ms);
        n_bytes  += text.size();
        ++n_lines;
        std::string out = "[";
        for (size_t i = 0; i < ids.size(); ++i) {
            out += (i ? "," : "") + std::to_string(ids[i]);
        }
        printf("%s]\n", out.c_str());
    }
    fclose(f);
    fprintf(stderr, "laya tokenize: %zu strings, %zu bytes, total %.2f ms, max %.2f ms\n", n_lines, n_bytes, total_ms, max_ms);
    return 0;
}

int main(int argc, char ** argv) {
#if defined(_WIN32)
    // "\n", not "\r\n": the output matches the golden files byte for byte on every platform
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    std::string model_path;
    std::string input_path;
    std::string tokenize_path;
    int n_threads = 1;
    int n_bench   = 0; // >0: repeat the forward pass in-process for timing/stability
    std::string kernels = "default";
    laya_model_params mparams;
    decision_checkpoint_params cparams;
    bool convert_opts = false; // --decision-convert-* given

    // Windows: argv is in the ANSI code page; paths go on as UTF-8 (ggml_fopen, gguf, laya)
    const std::vector<std::string> args = decision_utf8_args(argc, argv);
    for (int i = 1; i < argc; ++i) {
        const std::string & arg = args[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) model_path = args[++i];
        } else if (arg == "-f" || arg == "--file") {
            if (i + 1 < argc) input_path = args[++i];
        } else if (arg == "--tokenize") {
            if (i + 1 < argc) tokenize_path = args[++i];
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) n_threads = std::max(1, std::atoi(args[++i].c_str()));
        } else if (arg == "-b" || arg == "--bench") {
            if (i + 1 < argc) n_bench = std::max(0, std::atoi(args[++i].c_str()));
        } else if (arg == "--kernels") {
            if (i + 1 < argc) kernels = args[++i];
            if (kernels != "auto" && kernels != "default" && kernels != "repack" && kernels != "blas" && kernels != "repack+blas") {
                fprintf(stderr, "laya: --kernels must be auto, default, repack, blas or repack+blas\n");
                return 1;
            }
        } else if (arg == "--decision-convert-cache") {
            if (i + 1 >= argc) {
                fprintf(stderr, "laya: --decision-convert-cache needs a value\n");
                return 1;
            }
            cparams.cache_dir = args[++i];
            convert_opts      = true;
        } else if (arg == "--decision-convert-type") {
            const std::string t = i + 1 < argc ? args[++i] : "";
            if (t != "f16" && t != "f32") {
                fprintf(stderr, "laya: --decision-convert-type must be f16 or f32\n");
                return 1;
            }
            laya_convert_parse_outtype(t, cparams.outtype);
            convert_opts = true;
        } else if (arg == "--no-mmap") {
            mparams.use_mmap = false;
        } else if (arg == "--mlock") {
            mparams.use_mlock = true;
        } else if (arg == "-h" || arg == "--help") {
            printf("Usage: %s -m <laya-f16.gguf> -f <input.json> [-t threads] [-b bench_runs] [--kernels default|auto|repack|blas|repack+blas] [--no-mmap] [--mlock]\n"
                   "       %s -m <laya-f16.gguf> --tokenize <strings.jsonl>\n"
                   "       -m <checkpoint-dir> [--decision-convert-cache DIR] [--decision-convert-type f16|f32]: convert once into the GGUF cache, then load\n",
                   argv[0], argv[0]);
            return 0;
        } else {
            fprintf(stderr, "laya: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (model_path.empty() || (input_path.empty() && tokenize_path.empty())) {
        fprintf(stderr, "laya: missing -m <model> or -f <input>\n");
        return 1;
    }

    // DL builds have no static CPU backend
    ggml_backend_load_all();

    // -m DIR: convert once into the GGUF cache (or reuse it), then load that GGUF
    decision_model_source source;
    if (convert_opts && !decision_path_is_dir(model_path)) {
        fprintf(stderr, "laya: warning: --decision-convert-cache/--decision-convert-type apply to -m DIR only and are ignored for a GGUF file\n");
    }
    {
        cparams.log = [](const std::string & msg) { fprintf(stderr, "laya: %s\n", msg.c_str()); };
        std::string err;
        if (!decision_model_source_resolve(model_path, cparams, source, err)) {
            fprintf(stderr, "laya: failed to load model: %s\n", err.c_str());
            return 1;
        }
    }

    // load the self-contained model
    laya_model * model = nullptr;
    try {
        if (kernels == "auto") {
            kernels = laya_blas_description() == "Accelerate" ? "blas" : "default";
        }
        mparams.use_extra_bufts = kernels == "repack" || kernels == "repack+blas";
        model = laya_model_load_from_file_ext(source.gguf_path.c_str(), mparams);
    } catch (const std::exception & e) {
        fprintf(stderr, "laya: failed to load model: %s\n", e.what());
        return 1;
    }
    const laya_hparams & hp = laya_model_hparams(model);

    if (!tokenize_path.empty()) {
        const int rc = laya_cli_tokenize(model, tokenize_path);
        laya_model_free(model);
        return rc;
    }

    // read input JSON
    FILE * fin = ggml_fopen(input_path.c_str(), "rb");
    if (!fin) {
        fprintf(stderr, "laya: failed to open input file '%s'\n", input_path.c_str());
        laya_model_free(model);
        return 1;
    }
    fseek(fin, 0, SEEK_END);
    const long fsize = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    std::string text((size_t) fsize, '\0');
    const size_t nread = fread(&text[0], 1, (size_t) fsize, fin);
    fclose(fin);
    text.resize(nread);

    json input;
    try {
        std::string err;
        if (decision_json_parse(text, input, err) != DECISION_JSON_OK) {
            throw std::runtime_error(err);
        }
        if (!input.is_object() || !input.contains("questions") || !input.at("questions").is_object()) {
            throw std::runtime_error("expected an object with a \"questions\" object");
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "laya: failed to parse JSON: %s\n", e.what());
        laya_model_free(model);
        return 1;
    }

    // the reference refuses a missing state: serialize_state(None) would be the text "null"
    if (!input.contains("state") || input.at("state").is_null()) {
        fprintf(stderr, "laya: state must not be None; pass a string, object or list\n");
        laya_model_free(model);
        return 1;
    }
    const json state = input.at("state");
    const json & questions = input.at("questions");

    // every question is checked (laya_question_parse) before anything is built, as in the reference
    std::vector<laya_question> parsed;
    try {
        for (auto it = questions.begin(); it != questions.end(); ++it) {
            parsed.push_back(laya_question_parse(it.key(), it.value(), /*raw_options =*/ true));
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "laya: invalid question: %s\n", e.what());
        laya_model_free(model);
        return 1;
    }

    // no questions: empty answers and zero usage, without a forward pass (reference predict_batch)
    if (parsed.empty()) {
        json out = json::object();
        out["model"]        = model_path;
        out["state"]        = state;
        out["questions"]    = questions;
        out["per_question"] = json::object();
        out["answers"]      = json::object();
        out["usage"]        = { {"input_tokens", 0}, {"output_tokens", 0} };
        printf("%s\n", out.dump(2, ' ', false, json::error_handler_t::replace).c_str());
        laya_model_free(model);
        return 0;
    }

    // one sequence per question; the state is tokenized once
    const laya_escape escape = laya_escape_init(model);
    const std::vector<int32_t> state_ids = laya_tokenize(model, laya_escape_text(escape, laya_serialize_state(state), false));

    std::vector<laya_seq> seqs;
    try {
        size_t i = 0;
        for (auto it = questions.begin(); it != questions.end(); ++it, ++i) {
            const laya_question & q = parsed[i];
            laya_seq seq = laya_build_sequence(model, q.qtype, q.instructions, q.options, state_ids, state.is_array());
            if ((int32_t) seq.markers.size() < seq.n_options) {
                throw std::runtime_error("question '" + it.key() + "' options exceed head_max_len / max_len");
            }
            seqs.push_back(std::move(seq));
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "laya: invalid question: %s\n", e.what());
        laya_model_free(model);
        return 1;
    }

    const int32_t n_seqs = (int32_t) seqs.size();

    for (int32_t s = 0; s < n_seqs; ++s) {
        if ((int32_t) seqs[s].markers.size() > LAYA_MAX_MARKERS) {
            fprintf(stderr, "laya: question %d has %d options > LAYA_MAX_MARKERS=%d\n",
                    s, (int) seqs[s].markers.size(), LAYA_MAX_MARKERS);
            laya_model_free(model);
            return 1;
        }
    }

    // pack all sequences into one batch
    std::vector<const laya_seq *> seq_ptrs;
    for (const auto & seq : seqs) {
        seq_ptrs.push_back(&seq);
    }
    laya_batch_data batch_data;
    laya_batch_pack(seq_ptrs, batch_data);
    const laya_batch & batch = batch_data.batch;
    const int32_t n_tokens = batch.n_tokens;

    laya_context * ctx = nullptr;
    try {
        laya_context_params cparams;
        cparams.n_threads = n_threads;
        cparams.use_blas  = kernels == "blas" || kernels == "repack+blas";
        ctx = laya_init_ext(model, cparams);
    } catch (const std::exception & e) {
        fprintf(stderr, "laya: failed to init context: %s\n", e.what());
        laya_model_free(model);
        return 1;
    }

    laya_result result;
    const int n_runs = n_bench > 0 ? n_bench : 1;
    std::vector<double> run_ms;
    bool deterministic = true;
    std::vector<float> first_logits, first_act;
    for (int run = 0; run < n_runs; ++run) {
        const auto t0 = std::chrono::steady_clock::now();
        if (laya_encode(ctx, batch, result) != 0) {
            fprintf(stderr, "laya: forward pass failed\n");
            laya_free(ctx);
            laya_model_free(model);
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        run_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (run == 0) {
            first_logits = result.logits;
            first_act    = result.act_logits;
        } else if (result.logits != first_logits || result.act_logits != first_act) {
            deterministic = false;
        }
    }

    // bench statistics
    double sum_ms = 0.0, min_ms = run_ms[0], max_ms = run_ms[0];
    for (double v : run_ms) { sum_ms += v; min_ms = std::min(min_ms, v); max_ms = std::max(max_ms, v); }
    const double mean_ms = sum_ms / run_ms.size();
    double var_ms = 0.0;
    for (double v : run_ms) { var_ms += (v - mean_ms) * (v - mean_ms); }
    const double std_ms = std::sqrt(var_ms / run_ms.size());
    std::vector<double> sorted_ms = run_ms;
    std::sort(sorted_ms.begin(), sorted_ms.end());
    const double median_ms = sorted_ms[sorted_ms.size() / 2];

    if (n_bench > 0) {
        fprintf(stderr,
                "laya bench: tokens=%d seqs=%d runs=%d threads=%d kernels=%s  "
                "mean=%.2fms median=%.2fms min=%.2fms max=%.2fms std=%.2fms  deterministic=%s\n",
                n_tokens, n_seqs, n_runs, n_threads, laya_context_kernels(ctx).c_str(),
                mean_ms, median_ms, min_ms, max_ms, std_ms, deterministic ? "yes" : "no");
    }

    // per-question answers
    json answers = json::object();
    json per_question = json::object();
    json collapsed = json::object();

    int32_t qid = 0;
    for (auto it = questions.begin(); it != questions.end(); ++it, ++qid) {
        json ans;
        json pq;
        laya_postprocess(parsed[qid], seqs[qid], result, qid, hp, ans, pq);
        answers[it.key()]      = ans;
        per_question[it.key()] = pq;
        // reference usage["options"]: questions whose options lost their own token span to the head budget
        if (seqs[qid].n_options_distinct < seqs[qid].n_options) {
            collapsed[it.key()] = {
                {"total",            seqs[qid].n_options},
                {"distinct",         seqs[qid].n_options_distinct},
                {"tokens_per_option", seqs[qid].tokens_per_option < 0 ? json() : json(seqs[qid].tokens_per_option)},
            };
        }
    }

    json usage = { {"input_tokens", n_tokens}, {"output_tokens", 0} };
    if (!collapsed.empty()) {
        usage["options"] = collapsed;
    }

    json out = json::object();
    out["model"] = model_path;
    out["state"] = state;
    out["questions"] = questions;
    out["per_question"] = per_question;
    out["answers"] = answers;
    out["usage"] = usage;

    if (n_bench > 0) {
        json b = json::object();
        b["runs"]   = n_runs;
        b["tokens"] = n_tokens;
        b["seqs"]   = n_seqs;
        b["threads"] = n_threads;
        b["mean_ms"] = mean_ms;
        b["median_ms"] = median_ms;
        b["min_ms"] = min_ms;
        b["max_ms"] = max_ms;
        b["std_ms"] = std_ms;
        b["deterministic"] = deterministic;
        out["bench"] = b;
    }

    printf("%s\n", out.dump(2, ' ', false, json::error_handler_t::replace).c_str());

    laya_free(ctx);
    laya_model_free(model);
    return 0;
}
