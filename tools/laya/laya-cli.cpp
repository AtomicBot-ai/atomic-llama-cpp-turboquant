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
//   llama-laya-cli -m laya-f16.gguf -f input.json [-t N] [-b RUNS] [--kernels NAME] [--precision NAME] [--no-mmap] [--mlock]
//                  [--device cpu|gpu|auto] [--gpu N] [--strict-placement] [--plan packed|sequential]
//   llama-laya-cli -m laya-f16.gguf --jsonl inputs.jsonl [...same options, no -b]
//       --kernels: default (ggml CPU kernels; the golden files and tests/laya/verify_precision.py
//       use it), auto (what llama-server --decision runs: blas when the BLAS backend is
//       Accelerate, else default), repack (CPU repack buffers), blas (BLAS backend), repack+blas.
//       Kernels change the logits slightly.
//       --precision: default or strict, the matmul precision request (laya_precision); the CPU and
//       BLAS kernels compute the same bits in both modes.
//       --device: cpu (default), gpu (the GPU / iGPU device --gpu N of the ggml backend registry, 0 =
//       the first; an error when there is none) or auto (that device when it exists, else the CPU; also
//       the CPU, with a warning, when the device fails to load, initialize or warm up, and when --kernels
//       names CPU-only kernels).
//       A device computes the graph with the stock ggml backend, weights in device memory, token_embd
//       and its get_rows on the CPU (laya.h). --kernels then must be default or auto (repack and
//       BLAS are CPU-only). Stderr gets the placement of the graph nodes (laya_placement) and one
//       line "laya-cli: runtime {json}" with the device, kernels and placement.
//       --strict-placement: fail at load when a graph node outside the allowlist would run on the CPU or
//       a weight stays in host memory.
//       --plan: packed (default: all questions of the input in one graph, block-diagonal mask) or
//       sequential (one graph per question, the plan llama-server --decision runs by default). On a
//       device the two plans can compute different bits: the kernel of a matmul depends on its row
//       count (Metal: mat-vec below 9 rows, matrix-matrix above).
//       A device run warms up once at load (64 tokens: pipeline compilation, weight residency).
//   --jsonl FILE ("-": stdin): the model and context are loaded once, then every line of FILE is one
//       input object (the -f format) and prints exactly one line: the same JSON as -f would print, on
//       one line (compact), or {"error": "..."} for a line that fails (the message -f prints on
//       stderr, an empty line included). Each line is its own forward pass, exactly as a -f run of
//       that input: no batching across lines. Output is flushed after every line.
//   LAYA_TRACE_DIR=<dir> llama-laya-cli ...: dump the per-layer outputs of the forward pass into dir
//       (laya_context_params.trace_dir; tools/laya/trace_diff.py compares two dumps)
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
#include <cstdlib>
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

// the output of one input text (the -f format): false with err set when the input is refused
// or the forward pass fails
// one forward pass per question (--plan sequential, the plan llama-server --decision runs), the
// results gathered into one laya_result as a packed pass would return them
static int laya_cli_encode_sequential(laya_context * ctx, const std::vector<laya_seq> & seqs, laya_result & result) {
    laya_result one;
    for (size_t s = 0; s < seqs.size(); ++s) {
        laya_batch_data data;
        laya_batch_pack({ &seqs[s] }, data);
        if (laya_encode(ctx, data.batch, one) != 0) {
            return 1;
        }
        if (s == 0) {
            result.n_markers_max = one.n_markers_max;
            result.n_act         = one.n_act;
            result.n_seqs        = (int32_t) seqs.size();
            result.logits.assign((size_t) one.n_markers_max * seqs.size(), 0.0f);
            result.act_logits.assign((size_t) one.n_act * seqs.size(), 0.0f);
        }
        std::copy(one.logits.begin(), one.logits.begin() + one.n_markers_max, result.logits.begin() + s * one.n_markers_max);
        std::copy(one.act_logits.begin(), one.act_logits.begin() + one.n_act, result.act_logits.begin() + s * one.n_act);
    }
    return 0;
}

static bool laya_cli_run(laya_model * model, laya_context * ctx, const std::string & model_path, const std::string & text,
                         int n_bench, int n_threads, bool sequential, json & out, std::string & err) {
    const laya_hparams & hp = laya_model_hparams(model);

    json input;
    try {
        std::string perr;
        if (decision_json_parse(text, input, perr) != DECISION_JSON_OK) {
            throw std::runtime_error(perr);
        }
        if (!input.is_object() || !input.contains("questions") || !input.at("questions").is_object()) {
            throw std::runtime_error("expected an object with a \"questions\" object");
        }
    } catch (const std::exception & e) {
        err = std::string("failed to parse JSON: ") + e.what();
        return false;
    }

    // the reference refuses a missing state: serialize_state(None) would be the text "null"
    if (!input.contains("state") || input.at("state").is_null()) {
        err = "state must not be None; pass a string, object or list";
        return false;
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
        err = std::string("invalid question: ") + e.what();
        return false;
    }

    // no questions: empty answers and zero usage, without a forward pass (reference predict_batch)
    if (parsed.empty()) {
        out = json::object();
        out["model"]        = model_path;
        out["state"]        = state;
        out["questions"]    = questions;
        out["per_question"] = json::object();
        out["answers"]      = json::object();
        out["usage"]        = { {"input_tokens", 0}, {"output_tokens", 0} };
        return true;
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
        err = std::string("invalid question: ") + e.what();
        return false;
    }

    const int32_t n_seqs = (int32_t) seqs.size();

    for (int32_t s = 0; s < n_seqs; ++s) {
        if ((int32_t) seqs[s].markers.size() > LAYA_MAX_MARKERS) {
            err = "question " + std::to_string(s) + " has " + std::to_string(seqs[s].markers.size()) +
                  " options > LAYA_MAX_MARKERS=" + std::to_string(LAYA_MAX_MARKERS);
            return false;
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

    laya_result result;
    const int n_runs = n_bench > 0 ? n_bench : 1;
    std::vector<double> run_ms;
    bool deterministic = true;
    std::vector<float> first_logits, first_act;
    for (int run = 0; run < n_runs; ++run) {
        const auto t0 = std::chrono::steady_clock::now();
        if ((sequential ? laya_cli_encode_sequential(ctx, seqs, result) : laya_encode(ctx, batch, result)) != 0) {
            err = "forward pass failed";
            return false;
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

    out = json::object();
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
    return true;
}

// whole file, false if it cannot be opened
static bool laya_cli_read_file(const std::string & path, std::string & text) {
    FILE * fin = ggml_fopen(path.c_str(), "rb"); // UTF-8 name, also on Windows
    if (!fin) {
        return false;
    }
    fseek(fin, 0, SEEK_END);
    const long fsize = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    text.assign((size_t) std::max(0L, fsize), '\0');
    const size_t nread = fread(&text[0], 1, text.size(), fin);
    fclose(fin);
    text.resize(nread);
    return true;
}

// --jsonl: one output line per input line; 0 unless the file cannot be read
static int laya_cli_jsonl(laya_model * model, laya_context * ctx, const std::string & model_path, const std::string & path, int n_threads,
                          bool sequential) {
    FILE * f = path == "-" ? stdin : ggml_fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "laya: failed to open input file '%s'\n", path.c_str());
        return 1;
    }
    std::string line;
    size_t n_lines = 0, n_errors = 0;
    const auto t0 = std::chrono::steady_clock::now();
    int c;
    bool eof = false;
    while (!eof) {
        line.clear();
        while ((c = fgetc(f)) != EOF && c != '\n') {
            line += (char) c;
        }
        eof = c == EOF;
        if (eof && line.empty()) {
            break;  // the newline of the last line
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        ++n_lines;
        json out;
        std::string err;
        if (line.empty()) {
            err = "empty line";
        }
        if (!err.empty() || !laya_cli_run(model, ctx, model_path, line, 0, n_threads, sequential, out, err)) {
            out = json{ {"error", err} };
            ++n_errors;
        }
        printf("%s\n", out.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
        fflush(stdout);
    }
    if (f != stdin) {
        fclose(f);
    }
    fprintf(stderr, "laya jsonl: %zu lines, %zu errors, %.1f s\n", n_lines, n_errors,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}

int main(int argc, char ** argv) {
#if defined(_WIN32)
    // "\n", not "\r\n": the output matches the golden files byte for byte on every platform
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    std::string model_path;
    std::string input_path;
    std::string jsonl_path;
    std::string tokenize_path;
    int n_threads = 1;
    int n_bench   = 0; // >0: repeat the forward pass in-process for timing/stability
    std::string kernels = "default";
    std::string device  = "cpu";
    int gpu_index = 0;
    bool strict_placement = false;
    bool sequential = false;  // --plan
    laya_precision precision = LAYA_PRECISION_DEFAULT;
    laya_model_params mparams;
    decision_checkpoint_params ckparams;
    bool convert_opts = false; // --decision-convert-* given

    // Windows: argv is in the ANSI code page; paths go on as UTF-8 (ggml_fopen, gguf, laya)
    const std::vector<std::string> args = decision_utf8_args(argc, argv);
    for (int i = 1; i < argc; ++i) {
        const std::string & arg = args[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) model_path = args[++i];
        } else if (arg == "-f" || arg == "--file") {
            if (i + 1 < argc) input_path = args[++i];
        } else if (arg == "--jsonl") {
            if (i + 1 < argc) jsonl_path = args[++i];
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
        } else if (arg == "--precision") {
            if (i + 1 >= argc || !laya_precision_from_name(args[++i], precision)) {
                fprintf(stderr, "laya: --precision must be default or strict\n");
                return 1;
            }
        } else if (arg == "--plan") {
            const std::string plan = i + 1 < argc ? args[++i] : "";
            if (plan != "packed" && plan != "sequential") {
                fprintf(stderr, "laya: --plan must be packed or sequential\n");
                return 1;
            }
            sequential = plan == "sequential";
        } else if (arg == "--device") {
            if (i + 1 < argc) device = args[++i];
            if (device != "cpu" && device != "gpu" && device != "auto") {
                fprintf(stderr, "laya: --device must be cpu, gpu or auto\n");
                return 1;
            }
        } else if (arg == "--gpu") {
            if (i + 1 < argc) gpu_index = std::atoi(args[++i].c_str());
            if (gpu_index < 0) {
                fprintf(stderr, "laya: --gpu must be >= 0\n");
                return 1;
            }
        } else if (arg == "--strict-placement") {
            strict_placement = true;
        } else if (arg == "--decision-convert-cache") {
            if (i + 1 >= argc) {
                fprintf(stderr, "laya: --decision-convert-cache needs a value\n");
                return 1;
            }
            ckparams.cache_dir = args[++i];
            convert_opts       = true;
        } else if (arg == "--decision-convert-type") {
            const std::string t = i + 1 < argc ? args[++i] : "";
            if (t != "f16" && t != "f32") {
                fprintf(stderr, "laya: --decision-convert-type must be f16 or f32\n");
                return 1;
            }
            laya_convert_parse_outtype(t, ckparams.outtype);
            convert_opts = true;
        } else if (arg == "--no-mmap") {
            mparams.use_mmap = false;
        } else if (arg == "--mlock") {
            mparams.use_mlock = true;
        } else if (arg == "-h" || arg == "--help") {
            printf("Usage: %s -m <laya-f16.gguf> -f <input.json> [-t threads] [-b bench_runs] [--kernels default|auto|repack|blas|repack+blas] [--precision default|strict]\n"
                   "          [--plan packed|sequential]\n"
                   "          [--device cpu|gpu|auto] [--gpu N] [--strict-placement] [--no-mmap] [--mlock]\n"
                   "       %s -m <laya-f16.gguf> --jsonl <inputs.jsonl | -> [same options, no -b]\n"
                   "       %s -m <laya-f16.gguf> --tokenize <strings.jsonl>\n"
                   "       -m <checkpoint-dir> [--decision-convert-cache DIR] [--decision-convert-type f16|f32]: convert once into the GGUF cache, then load\n",
                   argv[0], argv[0], argv[0]);
            return 0;
        } else {
            fprintf(stderr, "laya: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (model_path.empty() || (input_path.empty() && jsonl_path.empty() && tokenize_path.empty())) {
        fprintf(stderr, "laya: missing -m <model> or -f <input>\n");
        return 1;
    }
    if (!jsonl_path.empty() && (!input_path.empty() || n_bench > 0)) {
        fprintf(stderr, "laya: --jsonl cannot be combined with -f or -b\n");
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
        ckparams.log = [](const std::string & msg) { fprintf(stderr, "laya: %s\n", msg.c_str()); };
        std::string err;
        if (!decision_model_source_resolve(model_path, ckparams, source, err)) {
            fprintf(stderr, "laya: failed to load model: %s\n", err.c_str());
            return 1;
        }
    }

    // load the self-contained model. auto: a device that fails to load, initialize or warm up gives way to
    // the CPU (with a warning), as do CPU-only kernels; gpu: such a failure is an error
    const std::string kernels_arg = kernels;
    ggml_backend_dev_t dev_sel = nullptr;
    try {
        dev_sel = laya_device_select(device, gpu_index);
    } catch (const std::exception & e) {
        fprintf(stderr, "laya: failed to load model: %s\n", e.what());
        return 1;
    }
    if (dev_sel && device == "auto" && kernels_arg != "auto" && kernels_arg != "default") {
        fprintf(stderr, "laya: --kernels %s is CPU-only: --device auto uses the CPU\n", kernels_arg.c_str());
        dev_sel = nullptr;
    }
    auto load_model = [&](ggml_backend_dev_t d) {
        kernels = kernels_arg;
        if (kernels == "auto") {
            // a device computes with its own kernels; on the CPU what llama-server runs
            kernels = !d && laya_blas_description() == "Accelerate" ? "blas" : "default";
        }
        if (d && kernels != "default") {
            throw std::runtime_error("--kernels " + kernels + " is CPU-only, the device " + laya_device_name(d) +
                                     " takes default or auto");
        }
        mparams.device          = d;
        mparams.use_extra_bufts = kernels == "repack" || kernels == "repack+blas";
        return laya_model_load_from_file_ext(source.gguf_path.c_str(), mparams);
    };
    auto can_fall_back = [&](const char * what, const std::exception & e) {
        if (device != "auto" || !mparams.device) {
            return false;
        }
        fprintf(stderr, "laya: warning: the device %s failed to %s (%s); --device auto uses the CPU\n",
                laya_device_name(mparams.device).c_str(), what, e.what());
        return true;
    };

    laya_model * model = nullptr;
    try {
        try {
            model = load_model(dev_sel);
        } catch (const std::exception & e) {
            if (!can_fall_back("load the model", e)) {
                throw;
            }
            model = load_model(nullptr);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "laya: failed to load model: %s\n", e.what());
        return 1;
    }

    if (!tokenize_path.empty()) {
        const int rc = laya_cli_tokenize(model, tokenize_path);
        laya_model_free(model);
        return rc;
    }

    std::string text;
    if (!input_path.empty() && !laya_cli_read_file(input_path, text)) {
        fprintf(stderr, "laya: failed to open input file '%s'\n", input_path.c_str());
        laya_model_free(model);
        return 1;
    }

    // a device compiles its pipelines and makes the weights resident on the first graph
    const char * trace_dir = getenv("LAYA_TRACE_DIR");
    auto init_ctx = [&]() {
        laya_context_params cparams;
        cparams.n_threads = n_threads;
        cparams.use_blas  = kernels == "blas" || kernels == "repack+blas";
        cparams.precision = precision;
        cparams.strict_placement = strict_placement;
        cparams.trace_dir = trace_dir ? trace_dir : "";
        laya_context * c = laya_init_ext(model, cparams);
        if (laya_model_device(model) && laya_warmup(c, 64) != 0) {
            laya_free(c);
            throw std::runtime_error("warm-up forward pass failed");
        }
        return c;
    };
    laya_context * ctx = nullptr;
    try {
        try {
            ctx = init_ctx();
        } catch (const std::exception & e) {
            if (!can_fall_back("initialize", e)) {
                throw;
            }
            laya_model_free(model);
            model = nullptr;
            model = load_model(nullptr);
            ctx = init_ctx();
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "laya: failed to init context: %s\n", e.what());
        laya_model_free(model);
        return 1;
    }

    // what computes, for the parity identity records (tests/laya/verify_reference.py)
    {
        const laya_placement pl = laya_context_placement(ctx);
        json backends = json::object();
        for (const auto & b : pl.backends) {
            backends[b.first] = b.second;
        }
        ggml_backend_dev_t dev = laya_model_device(model);
        const json runtime = {
            {"device",       laya_device_name(dev)},
            {"device_description", dev ? ggml_backend_dev_description(dev) : ""},
            {"kernels",      laya_context_kernels(ctx)},
            {"precision",    laya_precision_name(laya_context_precision(ctx))},
            {"plan",         sequential ? "sequential" : "packed"},
            {"n_threads",    n_threads},
            {"placement",    {
                {"nodes", pl.n_nodes}, {"splits", pl.n_splits}, {"backends", backends},
                {"cpu_fallback", pl.cpu_fallback}, {"fallback_ops", pl.fallback_ops}, {"host_weights", pl.host_weights},
            }},
        };
        fprintf(stderr, "laya-cli: runtime %s\n", runtime.dump().c_str());
    }

    int rc = 0;
    if (!jsonl_path.empty()) {
        rc = laya_cli_jsonl(model, ctx, model_path, jsonl_path, n_threads, sequential);
    } else {
        json out;
        std::string err;
        if (laya_cli_run(model, ctx, model_path, text, n_bench, n_threads, sequential, out, err)) {
            printf("%s\n", out.dump(2, ' ', false, json::error_handler_t::replace).c_str());
        } else {
            fprintf(stderr, "laya: %s\n", err.c_str());
            rc = 1;
        }
    }

    laya_free(ctx);
    laya_model_free(model);
    return rc;
}
