// llama-decision-bench: in-process latency, CPU, memory and determinism bench of the decision API.
//
// Runs a suite of /v1/systemone and /v1/router/score request bodies through the same code as
// llama-server --decision (decision library + engine), without HTTP and without the queue.
//
// Usage:
//   llama-decision-bench -m laya-q8_0.gguf -f tests/decision/bench/suite.jsonl [-t 4] [--kernels NAME]
//                        [--repeat 20] [--warmup 2] [--plan NAME] [--spec FILE]
//                        [--idle-ms 60000] [--filter TEXT] [--poll N] [--no-mmap] [--mlock] [--no-warmup]
//                        [--warmup-tokens N] [--blas-threads N] [--device cpu|gpu|auto] [--gpu N]
//                        [--strict-placement] [-o result.json]
//   LAYA_TRACE_DIR=<dir>: the laya layer trace (decision_engine_params.trace_dir)
//
// Suite: one JSON object per line
//   {"id": "router-n4-a", "endpoint": "systemone" | "router", "group": "router N=4", "body": {...}}
// Router requests run without a router calibration too (p = sigmoid(z), "calibrated": false).
//
// Phases of one request (wall clock, steady_clock):
//   parse    - body text -> JSON -> checked request
//   render   - card checks, items, tokenize and build every sequence (ends at the last render check)
//   compute  - forward passes
//   post     - calibration, answers, response text
//   total    - parse + render + compute + post
//   tokenize - separate pass after the request: tokenize its distinct state pieces once (a part of render)
// cpu_s is process user + system time over the total window (all threads).
// hash is SHA-256 of the raw logits (double bytes, item order); the same request must give the
// same hash in every warmup and repeat run, and at every thread count for a thread-invariant engine.
// Requests run interleaved: repeat r runs every request once, so slow drift hits all requests alike.
// Percentiles interpolate linearly between ranks (numpy "linear").

#include "decision.h"
#include "decision-calib.h"
#include "decision-request.h"
#include "decision-router.h"
#include "decision-spec.h"
#include "laya-decide.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#else
#    include <sys/resource.h>
#    include <sys/time.h>
#endif

#if defined(__APPLE__)
#    include <mach/mach.h>
#endif

using bench_clock = std::chrono::steady_clock;

static double bench_ms(bench_clock::time_point t0, bench_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

//
// process facts
//

// user + system CPU seconds of the whole process
static double bench_cpu_seconds() {
#if defined(_WIN32)
    FILETIME t_create, t_exit, t_kernel, t_user;
    if (!GetProcessTimes(GetCurrentProcess(), &t_create, &t_exit, &t_kernel, &t_user)) {
        return 0.0;
    }
    const auto sec = [](const FILETIME & ft) {
        return (double) (((uint64_t) ft.dwHighDateTime << 32) | ft.dwLowDateTime) * 1e-7;
    };
    return sec(t_kernel) + sec(t_user);
#else
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double) ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + (double) ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
#endif
}

// bytes, -1 when unknown. footprint: macOS phys_footprint (what Activity Monitor shows),
// Linux RssAnon, Windows PrivateUsage
struct bench_mem {
    int64_t rss            = -1;
    int64_t rss_peak       = -1;
    int64_t footprint      = -1;
    int64_t footprint_peak = -1;
};

static bench_mem bench_mem_now() {
    bench_mem m;
#if defined(__APPLE__)
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count) == KERN_SUCCESS) {
        m.rss       = (int64_t) info.resident_size;
        m.rss_peak  = (int64_t) info.resident_size_peak;
        m.footprint = (int64_t) info.phys_footprint;
        if (count * sizeof(natural_t) >= offsetof(task_vm_info_data_t, ledger_phys_footprint_peak) + sizeof(int64_t)) {
            m.footprint_peak = info.ledger_phys_footprint_peak;
        }
    }
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc))) {
        m.rss       = (int64_t) pmc.WorkingSetSize;
        m.rss_peak  = (int64_t) pmc.PeakWorkingSetSize;
        m.footprint = (int64_t) pmc.PrivateUsage;
    }
#else
    std::ifstream f("/proc/self/status");
    std::string key;
    while (f >> key) {
        int64_t kb = 0;
        if (key == "VmRSS:" && f >> kb) {
            m.rss = kb * 1024;
        } else if (key == "VmHWM:" && f >> kb) {
            m.rss_peak = kb * 1024;
        } else if (key == "RssAnon:" && f >> kb) {
            m.footprint = kb * 1024;
        }
        f.ignore(1 << 20, '\n');
    }
#endif
    return m;
}

static json bench_mem_json(const bench_mem & m) {
    return json{
        {"rss",            m.rss},
        {"rss_peak",       m.rss_peak},
        {"footprint",      m.footprint},
        {"footprint_peak", m.footprint_peak},
    };
}

static double bench_mib(int64_t bytes) {
    return bytes < 0 ? -1.0 : bytes / (1024.0 * 1024.0);
}

//
// statistics
//

static double bench_percentile(std::vector<double> v, double p) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const double pos = p / 100.0 * (double) (v.size() - 1);
    const size_t lo  = (size_t) std::floor(pos);
    const size_t hi  = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - (double) lo);
}

static double bench_mean(const std::vector<double> & v) {
    double s = 0.0;
    for (double x : v) {
        s += x;
    }
    return v.empty() ? 0.0 : s / (double) v.size();
}

static json bench_stats(const std::vector<double> & v) {
    return json{
        {"n",    v.size()},
        {"mean", bench_mean(v)},
        {"min",  v.empty() ? 0.0 : *std::min_element(v.begin(), v.end())},
        {"p50",  bench_percentile(v, 50)},
        {"p95",  bench_percentile(v, 95)},
        {"p99",  bench_percentile(v, 99)},
        {"max",  v.empty() ? 0.0 : *std::max_element(v.begin(), v.end())},
    };
}

//
// suite and runs
//

struct bench_request {
    std::string id;
    std::string endpoint;   // "systemone" or "router"
    std::string group;
    std::string body;       // request text, as a client would send it
};

struct bench_sample {
    double parse_ms    = 0.0;
    double render_ms   = 0.0;
    double compute_ms  = 0.0;
    double post_ms     = 0.0;
    double total_ms    = 0.0;
    double tokenize_ms = 0.0;
    double cpu_s       = 0.0;
    std::string hash;
    int64_t n_items     = 0;
    int64_t n_tokens    = 0;    // sum of full prompt lengths
    int64_t n_state_cut = 0;    // state tokens dropped, all items
    int64_t n_res_len   = 0;    // response text bytes
};

struct bench_series {
    std::vector<double> parse_ms, render_ms, compute_ms, post_ms, total_ms, tokenize_ms, cpu_s;
    std::set<std::string> hashes;    // warmup and measured runs
    bench_sample last;

    void add(const bench_sample & s) {
        parse_ms.push_back(s.parse_ms);
        render_ms.push_back(s.render_ms);
        compute_ms.push_back(s.compute_ms);
        post_ms.push_back(s.post_ms);
        total_ms.push_back(s.total_ms);
        tokenize_ms.push_back(s.tokenize_ms);
        cpu_s.push_back(s.cpu_s);
        last = s;
    }
};

static bool bench_read_suite(const std::string & path, const std::string & filter, std::vector<bench_request> & out, std::string & err) {
    std::ifstream f = decision_ifstream(path);
    if (!f) {
        err = "cannot open '" + path + "'";
        return false;
    }
    std::set<std::string> ids;
    std::string line;
    int n_line = 0;
    while (std::getline(f, line)) {
        ++n_line;
        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }
        json j;
        std::string perr;
        if (decision_json_parse(line, j, perr) != DECISION_JSON_OK || !j.is_object()) {
            err = path + ":" + std::to_string(n_line) + ": not a JSON object: " + perr;
            return false;
        }
        bench_request r;
        if (!j.contains("id") || !j.at("id").is_string() || !j.contains("body") || !j.at("body").is_object()) {
            err = path + ":" + std::to_string(n_line) + ": needs a string \"id\" and an object \"body\"";
            return false;
        }
        r.id       = j.at("id").get<std::string>();
        r.endpoint = j.contains("endpoint") && j.at("endpoint").is_string() ? j.at("endpoint").get<std::string>() : "systemone";
        r.group    = j.contains("group") && j.at("group").is_string() ? j.at("group").get<std::string>() : r.endpoint;
        r.body     = j.at("body").dump();
        if (r.endpoint != "systemone" && r.endpoint != "router") {
            err = path + ":" + std::to_string(n_line) + ": endpoint must be \"systemone\" or \"router\"";
            return false;
        }
        if (!ids.insert(r.id).second) {
            err = path + ":" + std::to_string(n_line) + ": duplicate id '" + r.id + "'";
            return false;
        }
        if (!filter.empty() && r.id.find(filter) == std::string::npos && r.group.find(filter) == std::string::npos) {
            continue;
        }
        out.push_back(std::move(r));
    }
    if (out.empty()) {
        err = "no requests in '" + path + "'" + (filter.empty() ? "" : " match '" + filter + "'");
        return false;
    }
    return true;
}

struct bench_ctx {
    decision_spec                    spec;
    std::unique_ptr<decision_engine> engine;
    decision_caps                    caps;
    decision_limits                  limits;
    decision_confidence_mode         confidence = DECISION_CONFIDENCE_LAYA;
    decision_router_spec             router;
    int32_t                          max_candidates = DECISION_MAX_CANDIDATES;

    // same limits and router defaults as server_decision::load
    bool init(const std::string & spec_path, const decision_engine_params & ep, int32_t max_items, std::string & err) {
        const std::string & model = ep.model_path;
        if (!decision_spec_load(model, spec_path, spec, err)) {
            return false;
        }
        if (!decision_confidence_from_name(spec.confidence, confidence)) {
            err = "unknown confidence mode '" + spec.confidence + "'";
            return false;
        }
        try {
            engine = decision_engine_init(spec, ep);
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
        caps = engine->caps();

        limits.max_items   = max_items;
        limits.max_options = caps.max_options;
        limits.layout      = spec.layout;
        if (spec.max_options > 0) {
            limits.max_options = std::min(limits.max_options, spec.max_options);
        }
        max_candidates = std::min(max_items, (int32_t) DECISION_MAX_CANDIDATES);
        if (spec.max_candidates > 0) {
            max_candidates = std::min(max_candidates, spec.max_candidates);
        }
        router = spec.router;
        if (router.max_card_tokens < 0) {
            router.max_card_tokens = caps.max_tokens * 3 / 8;
        }
        return true;
    }

    // tokenize the distinct state pieces of the items once, as the engine does with its cache
    double tokenize_pass(const std::vector<decision_item> & items) const {
        std::vector<std::pair<std::string, bool>> pieces;
        std::set<std::pair<std::string, bool>> seen;
        for (const auto & item : items) {
            const std::string text = laya_serialize_state(item.state);
            std::vector<size_t> cuts;
            if (caps.state_split && item.state.is_string()) {
                cuts = item.state_splits;
            }
            size_t start = 0;
            for (size_t k = 0; k <= cuts.size(); ++k) {
                const size_t end = k < cuts.size() ? std::min(cuts[k], text.size()) : text.size();
                if (end > start) {
                    auto piece = std::make_pair(text.substr(start, end - start), item.escape_control);
                    if (seen.insert(piece).second) {
                        pieces.push_back(std::move(piece));
                    }
                    start = end;
                }
            }
        }
        const auto t0 = bench_clock::now();
        for (const auto & p : pieces) {
            engine->n_tokens(p.first, p.second);
        }
        return bench_ms(t0, bench_clock::now());
    }

    // one request through the server code path; throws decision_error like the server would answer 4xx/5xx
    bench_sample run(const bench_request & r) {
        bench_sample s;
        std::vector<decision_item>   items;
        std::vector<decision_output> outs;
        decision_render_check        check;
        std::unique_ptr<decision_request>        sreq;
        std::unique_ptr<decision_router_request> rreq;
        const bool router_req = r.endpoint == "router";

        const double cpu0 = bench_cpu_seconds();
        const auto t0 = bench_clock::now();

        const json body = decision_parse_body(r.body);
        if (router_req) {
            rreq.reset(new decision_router_request(decision_parse_router(body, max_candidates)));
        } else {
            sreq.reset(new decision_request(decision_parse_systemone(body, limits)));
        }
        const auto t1 = bench_clock::now();

        if (router_req) {
            decision_router_check_cards(*rreq, router, *engine);
            items = decision_router_items(*rreq, router, limits);
            check = decision_router_render_check(*rreq, router, *engine);
        } else {
            items = decision_request_items(*sreq);
            check = decision_request_render_check(*sreq);
        }
        auto t_rendered = t1;
        const auto check_timed = [&](size_t idx, const decision_output & o) {
            check(idx, o);
            t_rendered = bench_clock::now();
        };
        if (!engine->evaluate(items, router_req, nullptr, check_timed, outs)) {
            throw decision_error(DECISION_REASON_INTERNAL, "evaluate was cancelled");
        }
        const auto t2 = bench_clock::now();

        json res;
        if (router_req) {
            json scores = json::array();
            for (size_t i = 0; i < outs.size(); ++i) {
                const double z = decision_router_logit(outs[i]);
                scores.push_back({
                    {"id",               rreq->candidates[i].id},
                    {"p_success",        decision_platt(router.calibrated ? router.platt_a : 1.0, router.calibrated ? router.platt_b : 0.0, z)},
                    {"logit",            z},
                    {"calibrated",       router.calibrated},
                    {"input_tokens",     outs[i].n_tokens},
                    {"truncated_tokens", outs[i].n_state_cut},
                });
            }
            res = json{{"object", "router.scores"}, {"scores", scores}};
        } else {
            json answers = json::object();
            for (size_t i = 0; i < outs.size(); ++i) {
                const decision_question & q = sreq->questions[i];
                const double t = decision_temperature(spec.calibration, q.type, (int32_t) q.keys.size());
                answers[q.id] = decision_answer(q, decision_softmax(outs[i].logits, t), confidence);
            }
            res = json{{"answers", answers}};
        }
        const std::string text = res.dump();
        const auto t3 = bench_clock::now();
        const double cpu1 = bench_cpu_seconds();

        s.parse_ms   = bench_ms(t0, t1);
        s.render_ms  = bench_ms(t1, t_rendered);
        s.compute_ms = bench_ms(t_rendered, t2);
        s.post_ms    = bench_ms(t2, t3);
        s.total_ms   = bench_ms(t0, t3);
        s.cpu_s      = cpu1 - cpu0;

        std::string raw;
        for (const auto & o : outs) {
            raw.append((const char *) o.logits.data(), o.logits.size() * sizeof(double));
            s.n_tokens    += o.n_tokens;
            s.n_state_cut += o.n_state_cut;
        }
        s.n_items   = (int64_t) outs.size();
        s.n_res_len = (int64_t) text.size();
        s.hash      = decision_sha256_hex(raw);

        s.tokenize_ms = tokenize_pass(items);
        return s;
    }
};

static void bench_usage(const char * argv0) {
    printf("usage: %s -m MODEL -f SUITE.jsonl [options]\n"
           "\n"
           "  -m, --model FILE      decision model (GGUF)\n"
           "  -f, --file FILE       suite JSONL: {\"id\", \"endpoint\": \"systemone\"|\"router\", \"group\", \"body\"}\n"
           "  -t, --threads N       engine threads (default: performance cores, else 4)\n"
           "  --repeat N            measured runs of every request (default: 10)\n"
           "  --warmup N            unmeasured runs of every request first (default: 1)\n"
           "  --plan NAME           decision plan (default: from the spec)\n"
           "  --spec FILE           decision spec sidecar (default: from the GGUF)\n"
           "  --max-items N         questions / candidates per request (default: 16)\n"
           "  --kernels NAME        laya matmul kernels: auto, default, repack, blas, repack+blas (default: from the spec, then auto)\n"
           "  --precision NAME      matmul precision request: default, strict (the CPU computes the same bits)\n"
           "  --device NAME         compute device: cpu, gpu, auto (default: cpu; see llama-server --decision-device)\n"
           "  --gpu N               GPU / iGPU device for --device gpu / auto (default: 0)\n"
           "  --strict-placement    fail at load when a graph node outside the allowlist runs on the CPU\n"
           "  --poll N              threadpool polling level between graphs, 0..100 (default: 0)\n"
           "  --no-mmap             load token_embd instead of mapping it\n"
           "  --mlock               lock the model memory in RAM\n"
           "  --no-warmup           no warm-up forward pass at load\n"
           "  --warmup-tokens N     tokens of the warm-up forward pass (default: 64)\n"
           "  --blas-threads N      BLAS backend threads (weight conversion; default: 0 = min(threads, 8))\n"
           "  --idle-ms N           after the runs, sleep N ms and run every request once more (K6)\n"
           "  --filter TEXT         only requests whose id or group contains TEXT\n"
           "  -o, --output FILE     write the full result as JSON\n",
           argv0);
}

int main(int argc, char ** argv) {
    // number text must not follow the user locale
    setlocale(LC_NUMERIC, "C");

    std::string model_path, suite_path, spec_path, plan, out_path, filter;
    int32_t n_threads = decision_cpu_perf_cores() > 0 ? decision_cpu_perf_cores() : 4, n_repeat = 10, n_warmup = 1, max_items = 16;
    decision_engine_params ep;
    int64_t idle_ms = 0;

    const std::vector<std::string> args = decision_utf8_args(argc, argv); // paths are UTF-8
    for (int i = 1; i < argc; ++i) {
        const std::string arg = args[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s needs a value\n", arg.c_str());
                exit(1);
            }
            return args[++i];
        };
        if (arg == "-m" || arg == "--model") {
            model_path = next();
        } else if (arg == "-f" || arg == "--file") {
            suite_path = next();
        } else if (arg == "-t" || arg == "--threads") {
            n_threads = std::max(1, atoi(next().c_str()));
        } else if (arg == "--repeat") {
            n_repeat = std::max(1, atoi(next().c_str()));
        } else if (arg == "--warmup") {
            n_warmup = std::max(0, atoi(next().c_str()));
        } else if (arg == "--plan") {
            plan = next();
        } else if (arg == "--spec") {
            spec_path = next();
        } else if (arg == "--max-items") {
            max_items = std::max(1, atoi(next().c_str()));
        } else if (arg == "--idle-ms") {
            idle_ms = std::max(0LL, atoll(next().c_str()));
        } else if (arg == "--kernels") {
            ep.kernels = next();
        } else if (arg == "--precision") {
            ep.precision = next();
            if (ep.precision != "default" && ep.precision != "strict") {
                fprintf(stderr, "error: --precision must be default or strict\n");
                return 1;
            }
        } else if (arg == "--device") {
            ep.device = next();
            if (ep.device != "cpu" && ep.device != "gpu" && ep.device != "auto") {
                fprintf(stderr, "error: --device must be cpu, gpu or auto\n");
                return 1;
            }
        } else if (arg == "--gpu") {
            ep.gpu_index = atoi(next().c_str());
            if (ep.gpu_index < 0) {
                fprintf(stderr, "error: --gpu must be >= 0\n");
                return 1;
            }
        } else if (arg == "--strict-placement") {
            ep.strict_placement = true;
        } else if (arg == "--poll") {
            ep.poll = std::min(100, std::max(0, atoi(next().c_str())));
        } else if (arg == "--no-mmap") {
            ep.use_mmap = false;
        } else if (arg == "--mlock") {
            ep.use_mlock = true;
        } else if (arg == "--no-warmup") {
            ep.warmup = false;
        } else if (arg == "--warmup-tokens") {
            ep.warmup_tokens = std::max(8, atoi(next().c_str()));
        } else if (arg == "--blas-threads") {
            ep.n_threads_blas = std::max(0, atoi(next().c_str()));
        } else if (arg == "--filter") {
            filter = next();
        } else if (arg == "-o" || arg == "--output") {
            out_path = next();
        } else if (arg == "-h" || arg == "--help") {
            bench_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            bench_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || suite_path.empty()) {
        bench_usage(argv[0]);
        return 1;
    }

    {
        const char * trace_dir = getenv("LAYA_TRACE_DIR");
        ep.trace_dir = trace_dir ? trace_dir : "";
    }

    // inputs first, before the model loads
    std::vector<bench_request> reqs;
    std::string err;
    if (!bench_read_suite(suite_path, filter, reqs, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    // before any thread exists (setenv); OpenMP builds read these at their first parallel region
    decision_cpu_env_defaults();

    // DL builds have no static CPU backend
    ggml_backend_load_all();

    const bench_mem mem_start = bench_mem_now();
    const double    cpu_start = bench_cpu_seconds();
    const auto      t_load0   = bench_clock::now();
    bench_ctx ctx;
    ep.model_path = model_path;
    ep.plan       = plan;
    ep.n_threads  = n_threads;
    if (!ctx.init(spec_path, ep, max_items, err)) {
        fprintf(stderr, "error: failed to load the decision model: %s\n", err.c_str());
        return 1;
    }
    const double    load_ms   = bench_ms(t_load0, bench_clock::now());
    const double    load_cpu  = bench_cpu_seconds() - cpu_start;
    const bench_mem mem_load  = bench_mem_now();

    fprintf(stderr, "decision-bench: %s layout %s, plan %s (router %s), device %s, kernels %s (blas threads %d), %d threads, spec %s (%s), %zu requests, warmup %d, repeat %d, load %.1f ms\n",
            model_path.c_str(), ctx.spec.layout.c_str(), ctx.caps.plan.c_str(), ctx.caps.plan_independent.c_str(), ctx.caps.device.c_str(), ctx.caps.kernels.c_str(), ctx.caps.n_threads_blas, ctx.caps.n_threads,
            ctx.spec.source.c_str(), ctx.spec.sha256.substr(0, 12).c_str(), reqs.size(), n_warmup, n_repeat, load_ms);

    std::vector<bench_series> series(reqs.size());
    double first_ms = -1.0;
    bench_mem mem_warm;
    try {
        for (int w = 0; w < n_warmup; ++w) {
            for (size_t i = 0; i < reqs.size(); ++i) {
                const bench_sample s = ctx.run(reqs[i]);
                if (first_ms < 0.0) {
                    first_ms = s.total_ms;
                }
                series[i].hashes.insert(s.hash);
            }
        }
        mem_warm = bench_mem_now();
        for (int r = 0; r < n_repeat; ++r) {
            for (size_t i = 0; i < reqs.size(); ++i) {
                const bench_sample s = ctx.run(reqs[i]);
                if (first_ms < 0.0) {
                    first_ms = s.total_ms;
                }
                series[i].add(s);
                series[i].hashes.insert(s.hash);
            }
            fprintf(stderr, "decision-bench: repeat %d/%d\n", r + 1, n_repeat);
        }
    } catch (const decision_error & e) {
        fprintf(stderr, "error: request failed: %s (%s)\n", e.what(), decision_reason_name(e.reason));
        return 1;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: request failed: %s\n", e.what());
        return 1;
    }
    const bench_mem mem_end = bench_mem_now();

    // K6: first request after an idle period
    json after_idle = nullptr;
    if (idle_ms > 0) {
        fprintf(stderr, "decision-bench: idle %lld ms\n", (long long) idle_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));
        json runs = json::array();
        try {
            for (size_t i = 0; i < reqs.size(); ++i) {
                const bench_sample s = ctx.run(reqs[i]);
                series[i].hashes.insert(s.hash);
                runs.push_back({
                    {"id", reqs[i].id}, {"total_ms", s.total_ms}, {"compute_ms", s.compute_ms}, {"cpu_s", s.cpu_s},
                    {"p50_ms", bench_percentile(series[i].total_ms, 50)},
                });
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "error: request after idle failed: %s\n", e.what());
            return 1;
        }
        after_idle = json{{"idle_ms", idle_ms}, {"runs", runs}};
    }

    // per request and per group
    json jreqs = json::array();
    std::map<std::string, std::vector<double>> g_total, g_compute, g_cpu;
    std::vector<std::string> g_order;
    std::string suite_raw;
    bool deterministic = true;
    for (size_t i = 0; i < reqs.size(); ++i) {
        const bench_series & se = series[i];
        const bool det = se.hashes.size() == 1;
        deterministic = deterministic && det;
        suite_raw += reqs[i].id + ":" + se.last.hash + "\n";
        if (!g_total.count(reqs[i].group)) {
            g_order.push_back(reqs[i].group);
        }
        auto & gt = g_total[reqs[i].group];
        auto & gc = g_compute[reqs[i].group];
        auto & gu = g_cpu[reqs[i].group];
        gt.insert(gt.end(), se.total_ms.begin(), se.total_ms.end());
        gc.insert(gc.end(), se.compute_ms.begin(), se.compute_ms.end());
        gu.insert(gu.end(), se.cpu_s.begin(), se.cpu_s.end());
        jreqs.push_back({
            {"id",            reqs[i].id},
            {"endpoint",      reqs[i].endpoint},
            {"group",         reqs[i].group},
            {"n_items",       se.last.n_items},
            {"n_tokens",      se.last.n_tokens},
            {"n_state_cut",   se.last.n_state_cut},
            {"response_bytes", se.last.n_res_len},
            {"hash",          se.last.hash},
            {"deterministic", det},
            {"n_hashes",      se.hashes.size()},
            {"parse_ms",      bench_stats(se.parse_ms)},
            {"tokenize_ms",   bench_stats(se.tokenize_ms)},
            {"render_ms",     bench_stats(se.render_ms)},
            {"compute_ms",    bench_stats(se.compute_ms)},
            {"post_ms",       bench_stats(se.post_ms)},
            {"total_ms",      bench_stats(se.total_ms)},
            {"cpu_s",         bench_stats(se.cpu_s)},
            {"runs_total_ms", se.total_ms},
            {"runs_cpu_s",    se.cpu_s},
        });
    }
    json jgroups = json::array();
    for (const auto & g : g_order) {
        jgroups.push_back({
            {"group",      g},
            {"total_ms",   bench_stats(g_total[g])},
            {"compute_ms", bench_stats(g_compute[g])},
            {"cpu_s",      bench_stats(g_cpu[g])},
        });
    }
    const std::string suite_hash = decision_sha256_hex(suite_raw);

    // tables
    printf("\nmodel %s, %d threads, plan %s / router %s, kernels %s, load %.1f ms, first request %.1f ms\n",
           model_path.c_str(), ctx.caps.n_threads, ctx.caps.plan.c_str(), ctx.caps.plan_independent.c_str(), ctx.caps.kernels.c_str(), load_ms, first_ms);
    printf("\n| %-26s | %5s | %6s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %6s | %-16s |\n",
           "request", "items", "tokens", "parse", "tokenize", "render", "compute", "total", "p95", "p99", "cpu_s", "det", "logits sha256");
    printf("|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|\n", std::string(28, '-').c_str(), std::string(7, '-').c_str(), std::string(8, '-').c_str(),
           std::string(10, '-').c_str(), std::string(10, '-').c_str(), std::string(10, '-').c_str(), std::string(10, '-').c_str(),
           std::string(10, '-').c_str(), std::string(10, '-').c_str(), std::string(10, '-').c_str(), std::string(10, '-').c_str(),
           std::string(8, '-').c_str(), std::string(18, '-').c_str());
    for (size_t i = 0; i < reqs.size(); ++i) {
        const bench_series & se = series[i];
        printf("| %-26s | %5lld | %6lld | %8.3f | %8.3f | %8.3f | %8.2f | %8.2f | %8.2f | %8.2f | %8.3f | %6s | %-16s |\n",
               reqs[i].id.c_str(), (long long) se.last.n_items, (long long) se.last.n_tokens,
               bench_percentile(se.parse_ms, 50), bench_percentile(se.tokenize_ms, 50), bench_percentile(se.render_ms, 50),
               bench_percentile(se.compute_ms, 50), bench_percentile(se.total_ms, 50), bench_percentile(se.total_ms, 95),
               bench_percentile(se.total_ms, 99), bench_mean(se.cpu_s), se.hashes.size() == 1 ? "yes" : "NO",
               se.last.hash.substr(0, 16).c_str());
    }
    printf("\n| %-20s | %5s | %8s | %8s | %8s | %8s | %8s |\n", "group", "runs", "p50 ms", "p95 ms", "p99 ms", "mean ms", "cpu_s");
    printf("|%s|%s|%s|%s|%s|%s|%s|\n", std::string(22, '-').c_str(), std::string(7, '-').c_str(), std::string(10, '-').c_str(),
           std::string(10, '-').c_str(), std::string(10, '-').c_str(), std::string(10, '-').c_str(), std::string(10, '-').c_str());
    for (const auto & g : g_order) {
        printf("| %-20s | %5zu | %8.2f | %8.2f | %8.2f | %8.2f | %8.3f |\n", g.c_str(), g_total[g].size(),
               bench_percentile(g_total[g], 50), bench_percentile(g_total[g], 95), bench_percentile(g_total[g], 99),
               bench_mean(g_total[g]), bench_mean(g_cpu[g]));
    }
    printf("\nmemory MiB: rss after load %.1f, rss peak %.1f, footprint after load %.1f, footprint peak %.1f\n",
           bench_mib(mem_load.rss), bench_mib(mem_end.rss_peak), bench_mib(mem_load.footprint), bench_mib(mem_end.footprint_peak));
    printf("determinism: %s across %d warmup + %d repeat runs%s, suite logits sha256 %s\n",
           deterministic ? "bitwise identical" : "DIFFERENT", n_warmup, n_repeat, idle_ms > 0 ? " + 1 after idle" : "", suite_hash.c_str());
    if (!after_idle.is_null()) {
        const json & first = after_idle.at("runs").at(0);
        printf("after %lld ms idle: first request %s %.2f ms (its p50 %.2f ms)\n", (long long) idle_ms,
               first.at("id").get<std::string>().c_str(), first.at("total_ms").get<double>(), first.at("p50_ms").get<double>());
    }
    fflush(stdout);

    if (!out_path.empty()) {
        const json out = {
            {"bench",   "llama-decision-bench"},
            {"version", 1},
            {"model",   model_path},
            {"suite",   suite_path},
            {"n_threads", ctx.caps.n_threads},
            {"plan",    ctx.caps.plan},
            {"plan_independent", ctx.caps.plan_independent},
            {"layout",  ctx.spec.layout},
            {"format",  ctx.caps.format},
            {"device",  ctx.caps.device},
            {"device_description", ctx.caps.device_description},
            {"placement", ctx.caps.placement},
            {"kernels", ctx.caps.kernels},
            {"n_threads_blas", ctx.caps.n_threads_blas},
            {"precision", ctx.caps.precision},
            {"engine_params", {
                {"mmap", ep.use_mmap}, {"mlock", ep.use_mlock}, {"warmup", ep.warmup}, {"poll", ep.poll},
                {"warmup_tokens", ep.warmup_tokens}, {"blas_threads", ep.n_threads_blas},
            }},
            {"state_split", ctx.caps.state_split},
            {"spec", {
                {"sha256", ctx.spec.sha256}, {"source", ctx.spec.source},
                {"calibrated", ctx.spec.calibration.calibrated}, {"router_calibrated", ctx.router.calibrated},
            }},
            {"repeat",  n_repeat},
            {"warmup",  n_warmup},
            {"load", {
                {"ms", load_ms}, {"cpu_s", load_cpu}, {"first_request_ms", first_ms},
            }},
            {"memory", {
                {"start", bench_mem_json(mem_start)},
                {"after_load", bench_mem_json(mem_load)},
                {"after_warmup", bench_mem_json(mem_warm)},
                {"end", bench_mem_json(mem_end)},
            }},
            {"deterministic", deterministic},
            {"suite_hash", suite_hash},
            {"groups",   jgroups},
            {"requests", jreqs},
            {"after_idle", after_idle},
        };
        std::ofstream f = decision_ofstream(out_path);
        f << out.dump(1) << "\n";
        if (!f) {
            fprintf(stderr, "error: cannot write '%s'\n", out_path.c_str());
            return 1;
        }
        fprintf(stderr, "decision-bench: wrote %s\n", out_path.c_str());
    }
    return 0;
}
