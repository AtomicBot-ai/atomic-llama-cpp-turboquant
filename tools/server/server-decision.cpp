#include "server-decision.h"
#include "server-common.h"

#include "decision.h"
#include "decision-calib.h"
#include "decision-checkpoint.h"
#include "decision-request.h"
#include "decision-router.h"
#include "decision-spec.h"
#include "laya.h"

#include "build-info.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

using dec_clock = std::chrono::steady_clock;

static double dec_ms(dec_clock::time_point t0, dec_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

//
// responses
//

static server_http_res_ptr dec_json_res(int status, const json & body) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->data   = safe_json_to_str(body);
    return res;
}

static server_http_res_ptr dec_error_res(const decision_error & err) {
    auto res = dec_json_res(decision_reason_status(err.reason), decision_error_body(err));
    if (err.reason == DECISION_REASON_OVERLOADED) {
        res->headers["Retry-After"] = "1";
    }
    return res;
}

// the message of a 500 for an unexpected exception: its text can hold file paths, library internals
// or fragments of the input, so it goes to the server log only
static const char * DEC_INTERNAL_MESSAGE = "internal error while processing the request (details in the server log)";

static server_http_res_ptr dec_internal_res(const char * what) {
    SRV_ERR("decision: internal error: %s\n", what);
    return dec_error_res(decision_error(DECISION_REASON_INTERNAL, DEC_INTERNAL_MESSAGE));
}

// --decision-debug with LLAMA_DECISION_DEBUG_THROW=worker|http: a systemone request throws this on the
// worker / on the HTTP thread (tests of the 500 path)
static const char * DEC_DEBUG_THROW_TEXT = "debug: injected failure at /private/decision-debug-secret";

// accepted requests that were not run (server stop, client gone); no decision reason code
static server_http_res_ptr dec_unavailable_res(const std::string & msg) {
    return dec_json_res(503, json{{"error", {
        {"code",    503},
        {"type",    "unavailable_error"},
        {"reason",  "UNAVAILABLE"},
        {"message", msg},
    }}});
}

// progress of -m DIR (converted / cache hit)
static void dec_log_checkpoint(const std::string & msg) {
    SRV_INF("decision: %s\n", msg.c_str());
}

//
// FIFO worker: one engine, one request at a time, in arrival order
//

struct dec_job {
    // runs on the worker; returns the response body, or null when cancelled
    std::function<json(dec_job & job)> run;

    dec_clock::time_point t_received;
    dec_clock::time_point t_start;
    std::atomic<bool>     cancel { false }; // set by the HTTP thread when the client is gone

    std::mutex              mtx;
    std::condition_variable cv;
    bool                    done = false;
    server_http_res_ptr     res;
};

using dec_job_ptr = std::shared_ptr<dec_job>;

static void dec_job_finish(dec_job & job, server_http_res_ptr res) {
    std::lock_guard<std::mutex> lock(job.mtx);
    job.res  = std::move(res);
    job.done = true;
    job.cv.notify_all();
}

enum dec_push_result {
    DEC_PUSH_OK,
    DEC_PUSH_FULL,      // 429
    DEC_PUSH_STOPPING,  // 503: cancel_all ran, nothing new is accepted
};

struct dec_worker {
    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<dec_job_ptr> queue;
    dec_job_ptr             running;
    bool                    busy     = false;
    bool                    stopping = false;
    int32_t                 capacity = 4;
    int32_t                 delay_ms = 0;   // debug: sleep before each job, makes 429 tests deterministic
    std::thread             thread;

    void start() {
        thread = std::thread([this]() { loop(); });
    }

    // capacity counts waiting requests; an idle worker always takes one
    dec_push_result push(const dec_job_ptr & job) {
        std::lock_guard<std::mutex> lock(mtx);
        if (stopping) {
            return DEC_PUSH_STOPPING;
        }
        // jobs whose client is gone do not take a place
        for (auto it = queue.begin(); it != queue.end();) {
            if ((*it)->cancel.load()) {
                dec_job_finish(**it, dec_unavailable_res("request was cancelled"));
                it = queue.erase(it);
            } else {
                ++it;
            }
        }
        // a running job whose client is gone stops at its next check, so it does not
        // make the worker busy (matters with --decision-queue 0)
        const bool idle = (!busy || (running && running->cancel.load())) && queue.empty();
        if (!idle && (int32_t) queue.size() >= capacity) {
            return DEC_PUSH_FULL;
        }
        queue.push_back(job);
        cv.notify_one();
        return DEC_PUSH_OK;
    }

    size_t n_waiting() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }

    // shutdown: no new jobs, waiting jobs get 503 when the worker leaves its loop,
    // the running job stops at its next check. Under the mutex, so a push either
    // lands before (and is cancelled here) or sees stopping.
    void cancel_all() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stopping = true;
            for (auto & job : queue) {
                job->cancel.store(true);
            }
            if (running) {
                running->cancel.store(true);
            }
        }
        cv.notify_all();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stopping = true;
        }
        cv.notify_all();
        if (thread.joinable()) {
            thread.join();
        }
    }

    server_http_res_ptr execute(dec_job & job) const {
        if (delay_ms > 0 && !job.cancel.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
        if (job.cancel.load()) {
            return dec_unavailable_res("request was cancelled");
        }
        job.t_start = dec_clock::now();
        try {
            json body = job.run(job);
            return body.is_null() ? dec_unavailable_res("request was cancelled") : dec_json_res(200, body);
        } catch (const decision_error & e) {
            return dec_error_res(e);
        } catch (const std::exception & e) {
            return dec_internal_res(e.what());
        } catch (...) {
            return dec_internal_res("unknown exception");
        }
    }

    void loop() {
        while (true) {
            dec_job_ptr job;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this]() { return stopping || !queue.empty(); });
                if (stopping) {
                    break;
                }
                job = queue.front();
                queue.pop_front();
                running = job;
                busy = true;
            }
            server_http_res_ptr res = execute(*job);
            // idle before the client sees the answer, so its next request is not a 429
            {
                std::lock_guard<std::mutex> lock(mtx);
                running.reset();
                busy = false;
            }
            dec_job_finish(*job, std::move(res));
        }
        std::deque<dec_job_ptr> rest;
        {
            std::lock_guard<std::mutex> lock(mtx);
            rest.swap(queue);
        }
        for (auto & job : rest) {
            dec_job_finish(*job, dec_unavailable_res("server is stopping"));
        }
    }
};

//
// server state
//

struct dec_stats {
    std::atomic<uint64_t> n_requests  { 0 };
    std::atomic<uint64_t> n_errors    { 0 };
    std::atomic<uint64_t> n_rejected  { 0 };
    std::atomic<uint64_t> n_items     { 0 };
    std::atomic<uint64_t> n_tokens    { 0 };
    std::atomic<uint64_t> compute_us  { 0 };
};

struct server_decision {
    common_params & params;

    // set by load(), read-only once the server is ready
    decision_spec                    spec;
    std::unique_ptr<decision_engine> engine;
    decision_caps                    caps;
    decision_limits                  limits;
    decision_confidence_mode         confidence = DECISION_CONFIDENCE_LAYA;
    decision_router_spec             router;
    int32_t                          max_candidates   = DECISION_MAX_CANDIDATES;
    bool                             router_available = false;
    std::string                      calibration_id;
    std::string                      model_name;
    decision_model_source            source;   // -m FILE, or -m DIR and its cached GGUF
    std::string                      debug_throw;   // LLAMA_DECISION_DEBUG_THROW (--decision-debug only)

    dec_worker        worker;
    dec_stats         stats;
    std::atomic<bool> stopping { false };

    explicit server_decision(common_params & params) : params(params) {
        worker.capacity = params.decision.queue;
        model_name = params.model_alias.empty() ? decision_model_name(params.model.path) : *params.model_alias.begin();
    }

    bool load(std::string & err) {
        // -m DIR: convert the checkpoint once into the GGUF cache (or reuse it), then load that GGUF
        // like -m FILE. Here, after the HTTP server started: /health answers 503 while converting.
        decision_checkpoint_params cparams;
        cparams.cache_dir = params.decision.convert_cache;
        if (!laya_convert_parse_outtype(params.decision.convert_type, cparams.outtype)) {
            err = "unknown --decision-convert-type '" + params.decision.convert_type + "'";
            return false;
        }
        cparams.log = dec_log_checkpoint;
        if (!decision_model_source_resolve(params.model.path, cparams, source, err)) {
            return false;
        }
        const std::string & gguf_path = source.gguf_path;

        if (!decision_spec_load(gguf_path, params.decision.spec_path, spec, err)) {
            return false;
        }

        // lets tests see the 503 of the loading phase and the 429 of a full queue without a race
        if (params.decision.debug) {
            const char * load_delay = std::getenv("LLAMA_DECISION_DEBUG_LOAD_DELAY_MS");
            const char * job_delay  = std::getenv("LLAMA_DECISION_DEBUG_JOB_DELAY_MS");
            worker.delay_ms = job_delay ? std::max(0, atoi(job_delay)) : 0;
            const char * throw_at = std::getenv("LLAMA_DECISION_DEBUG_THROW");
            debug_throw = throw_at ? throw_at : "";
            const auto t_end = dec_clock::now() + std::chrono::milliseconds(load_delay ? atoi(load_delay) : 0);
            while (dec_clock::now() < t_end && !stopping.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }

        if (!decision_confidence_from_name(spec.confidence, confidence)) {
            err = "unknown confidence mode '" + spec.confidence + "'";
            return false;
        }

        decision_engine_params eparams;
        eparams.model_path = gguf_path;
        eparams.plan       = params.decision.plan;
        eparams.n_threads  = params.cpuparams.n_threads;
        if (!params.decision.threads_set && decision_cpu_perf_cores() > 0) {
            // performance cores only: macOS perflevel0 (as the common default), Windows the
            // highest EfficiencyClass (the common default counts the efficiency cores too)
            eparams.n_threads = decision_cpu_perf_cores();
        }
        eparams.kernels    = params.decision.kernels;
        eparams.precision  = params.decision.precision;
        eparams.device     = params.decision.device;
        eparams.gpu_index  = params.decision.gpu;
        eparams.strict_placement = params.decision.strict_placement;
        if (params.decision.debug) {
            // the layer trace writes the activations of every request to disk: a debugging aid only
            const char * trace_dir = std::getenv("LAYA_TRACE_DIR");
            eparams.trace_dir = trace_dir ? trace_dir : "";
        }
        eparams.use_mmap   = params.load_mode == LLAMA_LOAD_MODE_MMAP || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
        eparams.use_mlock  = params.load_mode == LLAMA_LOAD_MODE_MLOCK || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
        eparams.warmup     = params.warmup;
        try {
            engine = decision_engine_init(spec, eparams);
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
        caps = engine->caps();

        limits.max_items   = params.decision.max_items;
        limits.max_options = caps.max_options;
        limits.layout      = spec.layout;
        if (spec.max_options > 0) {
            limits.max_options = std::min(limits.max_options, spec.max_options);
        }
        max_candidates = std::min(params.decision.max_items, (int32_t) DECISION_MAX_CANDIDATES);
        if (spec.max_candidates > 0) {
            max_candidates = std::min(max_candidates, spec.max_candidates);
        }

        router = spec.router;
        if (router.max_card_tokens < 0) {
            router.max_card_tokens = caps.max_tokens * 3 / 8;
        }
        try {
            decision_normalize_question("", router.question, limits, "router.question");
        } catch (const decision_error & e) {
            err = std::string("decision spec router.question: ") + e.what();
            return false;
        }
        router_available = router.calibrated || params.decision.allow_uncalibrated;

        if (!spec.calibration.version.empty()) {
            calibration_id = spec.calibration.version;
        } else if (spec.calibration.calibrated) {
            calibration_id = "cal-" + decision_sha256_hex(decision_py_dumps(spec.calibration.temperature)).substr(0, 12);
        } else {
            calibration_id = "none";
        }

        if (!params.devices.empty() && params.devices[0] != nullptr) {
            SRV_WRN("%s", "decision: --device is ignored, the decision engine takes --decision-device and --decision-gpu\n");
        }
        if (spec.plan.contains("n_threads") && spec.plan.at("n_threads").is_number_integer() &&
            spec.plan.at("n_threads").get<int64_t>() != caps.n_threads) {
            SRV_WRN("decision: the spec plan was calibrated with n_threads = %lld, running with -t %d; logits can differ slightly\n",
                    (long long) spec.plan.at("n_threads").get<int64_t>(), caps.n_threads);
        }

        SRV_INF("decision: model '%s' layout %s, format %s, plan %s, kernels %s, precision %s, %d threads, spec %s (%s), calibration %s\n",
                spec.model_id.c_str(), spec.layout.c_str(), caps.format.c_str(), caps.plan.c_str(), caps.kernels.c_str(), caps.precision.c_str(),
                caps.n_threads, spec.source.c_str(), spec.sha256.substr(0, 12).c_str(), calibration_id.c_str());
        SRV_INF("decision: weights %.1f MiB loaded, %.1f MiB mapped (read-only, resident as used), %.1f MiB repacked, blas threads %d\n",
                caps.weights_loaded / 1048576.0, caps.weights_mapped / 1048576.0, caps.weights_repacked / 1048576.0, caps.n_threads_blas);
        SRV_INF("decision: device %s%s%s, %.1f MiB of weights in device memory, placement %s\n",
                caps.device.c_str(), caps.device_description.empty() ? "" : " ", caps.device_description.c_str(),
                caps.weights_device / 1048576.0, caps.placement.dump().c_str());
        if (!spec.calibration.calibrated) {
            SRV_WRN("%s", "decision: the model has no calibration, probabilities are softmax(logits) with T = 1\n");
        }
        if (!caps.state_split) {
            SRV_INF("%s", "decision: router states are tokenized whole (splitting them is not exact for this vocabulary)\n");
        }
        if (router.calibrated) {
            SRV_INF("decision: router calibration platt a = %g, b = %g\n", router.platt_a, router.platt_b);
        } else if (params.decision.allow_uncalibrated) {
            SRV_WRN("%s", "decision: router scores are not calibrated (--decision-allow-uncalibrated)\n");
        }
        return true;
    }

    //
    // blocks
    //

    json runtime_block(const std::string & calibration, bool independent = false) const {
        return json{
            {"layout",      spec.layout},
            {"format",      caps.format},
            {"plan",        independent ? caps.plan_independent : caps.plan},
            {"spec_sha256", spec.sha256},
            {"calibration", calibration},
        };
    }

    // with_act: systemone adds the raw act-head logits (the router has no action)
    static json debug_block(const decision_output & o, double temperature, bool with_act = false) {
        json d = {
            {"logits",      o.logits},
            {"temperature", temperature},
            {"tokens",      o.tokens},
            {"n_state",     o.n_state},
            {"n_state_cut", o.n_state_cut},
            {"options_cut", o.options_cut},
            {"head_cut",    o.head_cut},
        };
        if (with_act && !o.act_logits.empty()) {
            d["act_logits"] = o.act_logits;
        }
        return d;
    }

    json limits_block() const {
        return json{
            {"max_questions",   limits.max_items},
            {"max_candidates",  max_candidates},
            {"max_options",     limits.max_options},
            {"max_checks",      DECISION_MAX_CHECKS},
            {"max_tokens",      caps.max_tokens},
            {"max_card_tokens", router.max_card_tokens},
            {"max_card_field_bytes", DECISION_MAX_CARD_FIELD_BYTES},
            {"max_body_bytes",  DECISION_MAX_BODY_BYTES},
        };
    }

    json capabilities() const {
        json caps_list = json::array({"decision", "systemone"});
        if (router.calibrated) {
            caps_list.push_back("router_score");
        }
        return caps_list;
    }

    json props_block() const {
        json endpoints = json::array({"/v1/systemone"});
        if (router_available) {
            endpoints.push_back("/v1/router/score");
        }
        if (params.decision.debug) {
            endpoints.push_back("/v1/decision/render");
        }
        return json{
            {"api_version",    DECISION_API_VERSION},
            {"endpoints",      endpoints},
            {"layout",         spec.layout},
            {"format",         caps.format},
            {"model_id",       spec.model_id},
            {"model_version",  spec.model_version},
            {"spec_version",   spec.spec_version},
            {"spec_sha256",    spec.sha256},
            {"spec_source",    spec.source},
            {"question_types", caps.question_types},
            {"limits",         limits_block()},
            {"confidence",     spec.confidence},
            {"calibration", {
                {"method",     spec.calibration.method},
                {"calibrated", spec.calibration.calibrated},
                {"version",    calibration_id},
            }},
            {"router", {
                {"available",       router_available},
                {"calibrated",      router.calibrated},
                {"method",          router.calibrated ? "platt" : "none"},
                {"card_schema",     router.card_schema},
                {"card_renderer",   router.card_renderer},
            }},
            {"plan", {
                {"name",      caps.plan},
                {"router",    caps.plan_independent},
                {"plans",     caps.plans},
                {"n_threads", caps.n_threads},
                {"kernels",   caps.kernels},
                {"n_threads_blas", caps.n_threads_blas},
                {"precision", caps.precision},
                {"state_split", caps.state_split},
                {"recipe",    spec.plan.contains("recipe") ? spec.plan.at("recipe") : json()},
                {"fa",        nullptr},
                {"n_batch",   nullptr},
                {"n_ubatch",  nullptr},
            }},
            {"memory", {
                {"weights_device_bytes",   caps.weights_device},
                {"weights_loaded_bytes",   caps.weights_loaded},
                {"weights_mapped_bytes",   caps.weights_mapped},
                {"weights_repacked_bytes", caps.weights_repacked},
            }},
            {"device",         caps.device},
            {"device_description", caps.device_description},
            {"placement",      caps.placement},
            {"kv_type",        nullptr},
            {"queue_capacity", params.decision.queue},
            {"input_contract", spec.input_contract},
            {"special_tokens", spec.special_tokens},
            {"debug",          params.decision.debug},
            // "gguf": -m FILE, loaded as is; "checkpoint-dir": -m DIR, converted into cache_path
            {"source",         source.kind},
            {"cache_path",     source.kind == "gguf" ? json() : json(source.gguf_path)},
            {"checkpoint",     source.kind == "gguf" ? json() : json{
                {"dir",        source.input},
                {"cache_dir",  source.cache_dir},
                {"key",        source.key},
                {"outtype",    source.outtype},
                {"cache_hit",  source.cache_hit},
                {"convert_ms", source.convert_ms},
                {"converter",  LAYA_CONVERT_VERSION},
            }},
        };
    }

    //
    // work (runs on the worker thread)
    //

    // evaluate with timings: render_ms ends at the last render check
    bool evaluate(const std::vector<decision_item> & items, bool independent, const decision_render_check & check, dec_job & job,
                  std::vector<decision_output> & outs, json & timings) {
        dec_clock::time_point t_rendered = job.t_start;
        const auto check_timed = [&](size_t idx, const decision_output & r) {
            if (check) {
                check(idx, r);
            }
            t_rendered = dec_clock::now();
        };
        if (!engine->evaluate(items, independent, &job.cancel, check_timed, outs)) {
            return false;
        }
        const auto t_end = dec_clock::now();
        stats.compute_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_rendered).count();
        stats.n_items    += items.size();
        timings = json{
            {"queue_ms",   dec_ms(job.t_received, job.t_start)},
            {"render_ms",  dec_ms(job.t_start, t_rendered)},
            {"compute_ms", dec_ms(t_rendered, t_end)},
        };
        return true;
    }

    json run_systemone(const decision_request & req, dec_job & job) {
        if (debug_throw == "worker") {
            throw std::runtime_error(DEC_DEBUG_THROW_TEXT);
        }
        const std::vector<decision_item> items = decision_request_items(req);
        std::vector<decision_output> outs;
        json timings;
        if (!evaluate(items, false, decision_request_render_check(req), job, outs, timings)) {
            return json();
        }

        json answers  = json::object();
        json warnings = json::array();
        int64_t n_input = 0;
        int64_t n_eval  = 0;
        bool state_cut = false, options_cut = false, head_cut = false;
        for (size_t i = 0; i < items.size(); ++i) {
            const decision_question & q = req.questions[i];
            const decision_output   & o = outs[i];
            const double t = decision_temperature(spec.calibration, q.type, (int32_t) q.keys.size());
            json ans = decision_answer(q, decision_softmax(o.logits, t), confidence);
            if (o.act_probability >= 0.0) {
                // the reference Agent's action head, rounded to 4 digits like llama-laya-cli
                ans["action"] = { {"act_probability", o.act_probability} };
            }
            if (params.decision.debug) {
                ans["debug"] = debug_block(o, t, true);
            }
            answers[q.id] = std::move(ans);
            n_input     += o.n_tokens;
            n_eval      += o.n_evaluated;
            state_cut   |= o.n_state_cut > 0;
            options_cut |= o.options_cut;
            head_cut    |= o.head_cut;
        }
        if (state_cut)   warnings.push_back("state_truncated");
        if (options_cut) warnings.push_back("options_truncated");
        if (head_cut)    warnings.push_back("instructions_truncated");
        stats.n_tokens += (uint64_t) n_eval;

        return json{
            {"model",      model_name},
            {"answers",    answers},
            {"usage",      {{"input_tokens", n_input}, {"output_tokens", 0}, {"evaluated_tokens", n_eval}}},
            {"latency_ms", dec_ms(job.t_received, dec_clock::now())},
            {"timings",    timings},
            {"warnings",   warnings},
            {"runtime",    runtime_block(calibration_id)},
        };
    }

    json run_router(const decision_router_request & req, dec_job & job) {
        decision_router_check_cards(req, router, *engine);
        const std::vector<decision_item> items = decision_router_items(req, router, limits);
        std::vector<decision_output> outs;
        json timings;
        // router scores must not depend on the other candidates: never packed
        if (!evaluate(items, true, decision_router_render_check(req, router, *engine), job, outs, timings)) {
            return json();
        }

        json scores   = json::array();
        json warnings = json::array();
        for (const auto & w : req.warnings) {
            warnings.push_back(w);
        }
        int64_t n_input = 0;
        bool cut = false;
        for (size_t i = 0; i < items.size(); ++i) {
            const decision_output & o = outs[i];
            const double z = decision_router_logit(o);
            const double p = router.calibrated ? decision_platt(router.platt_a, router.platt_b, z) : decision_platt(1.0, 0.0, z);
            json s = {
                {"id",               req.candidates[i].id},
                {"p_success",        p},
                {"logit",            z},
                {"calibrated",       router.calibrated},
                {"input_tokens",     o.n_tokens},
                {"truncated_tokens", o.n_state_cut},
            };
            if (params.decision.debug) {
                s["debug"] = debug_block(o, 1.0);
            }
            scores.push_back(std::move(s));
            n_input += o.n_tokens;
            cut     |= o.n_state_cut > 0;
        }
        if (cut) {
            warnings.push_back("task_truncated");
        }
        stats.n_tokens += (uint64_t) n_input;

        return json{
            {"object",     "router.scores"},
            {"model",      model_name},
            {"scores",     scores},
            {"usage",      {{"input_tokens", n_input}, {"output_tokens", 0}, {"passes", (int64_t) items.size()}}},
            {"latency_ms", dec_ms(job.t_received, dec_clock::now())},
            {"timings",    timings},
            {"runtime",    runtime_block(router.calibrated ? "platt" : "none", true)},
            {"warnings",   warnings},
        };
    }

    json run_render(const std::vector<decision_item> & items) {
        json out = json::array();
        for (const auto & item : items) {
            const decision_output r = engine->render(item);
            out.push_back({
                {"id",           item.q.id},
                {"type",         decision_qtype_name(item.q.type)},
                {"instructions", item.q.instructions},
                {"keys",         item.q.keys},
                {"state",        item.state},
                {"tokens",       r.tokens},
                {"n_tokens",     r.n_tokens},
                {"n_state",      r.n_state},
                {"n_state_cut",  r.n_state_cut},
                {"options_cut",  r.options_cut},
                {"head_cut",     r.head_cut},
            });
        }
        return json{{"items", out}, {"runtime", runtime_block(calibration_id)}};
    }

    //
    // HTTP side
    //

    server_http_res_ptr submit(const server_http_req & req, const dec_job_ptr & job) {
        if (stopping.load()) {
            return dec_unavailable_res("server is stopping");
        }
        switch (worker.push(job)) {
            case DEC_PUSH_OK:
                break;
            case DEC_PUSH_STOPPING:
                return dec_unavailable_res("server is stopping");
            case DEC_PUSH_FULL:
                stats.n_rejected++;
                throw decision_error(DECISION_REASON_OVERLOADED, "decision queue is full (" + std::to_string(worker.capacity) + " waiting), retry later");
        }
        std::unique_lock<std::mutex> lock(job->mtx);
        while (!job->cv.wait_for(lock, std::chrono::milliseconds(50), [&job]() { return job->done; })) {
            if (req.should_stop()) {
                // the worker only sees the job, never req, so returning here is safe
                job->cancel.store(true);
                return dec_unavailable_res("client disconnected");
            }
        }
        return std::move(job->res);
    }

    void log_request(const server_http_req & req) const {
        if (params.decision.debug) {
            SRV_INF("decision %s: %.*s\n", req.path.c_str(), (int) std::min<size_t>(req.body.size(), 4096), req.body.c_str());
        }
    }

    server_http_context::handler_t wrap(server_http_context::handler_t func) {
        return [this, func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
            server_http_res_ptr res;
            try {
                res = func(req);
            } catch (const decision_error & e) {
                res = dec_error_res(e);
            } catch (const std::exception & e) {
                res = dec_internal_res(e.what());
            } catch (...) {
                res = dec_internal_res("unknown exception");
            }
            if (res->status >= 400) {
                stats.n_errors++;
            }
            return res;
        };
    }

    std::string metrics_text() {
        const std::pair<const char *, std::string> gauges[] = {
            { "decision_queue_waiting", std::to_string(worker.n_waiting()) },
        };
        const struct { const char * name; const char * help; std::string value; } counters[] = {
            { "decision_requests_total",       "Decision requests received.",                     std::to_string(stats.n_requests.load()) },
            { "decision_errors_total",         "Decision requests answered with an error.",       std::to_string(stats.n_errors.load()) },
            { "decision_rejected_total",       "Decision requests rejected with 429.",            std::to_string(stats.n_rejected.load()) },
            { "decision_items_total",          "Questions and candidates evaluated.",             std::to_string(stats.n_items.load()) },
            { "decision_tokens_total",         "Tokens evaluated.",                               std::to_string(stats.n_tokens.load()) },
            { "decision_compute_seconds_total","Engine compute time.",                            std::to_string(stats.compute_us.load() / 1e6) },
        };
        std::string out;
        for (const auto & c : counters) {
            out += std::string("# HELP llamacpp:") + c.name + " " + c.help + "\n";
            out += std::string("# TYPE llamacpp:") + c.name + " counter\n";
            out += std::string("llamacpp:") + c.name + " " + c.value + "\n";
        }
        for (const auto & g : gauges) {
            out += std::string("# HELP llamacpp:") + g.first + " Decision requests waiting in the queue.\n";
            out += std::string("# TYPE llamacpp:") + g.first + " gauge\n";
            out += std::string("llamacpp:") + g.first + " " + g.second + "\n";
        }
        return out;
    }

    void register_routes(const server_http_context & ctx_http) {
        const auto get_health = [this](const server_http_req &) {
            return dec_json_res(200, json{
                {"status", "ok"},
                {"ok",     true},
                {"model",  model_name},
                {"layout", spec.layout},
            });
        };
        ctx_http.get("/health",    wrap(get_health)); // public endpoint (no API key check)
        ctx_http.get("/v1/health", wrap(get_health)); // public endpoint (no API key check)

        ctx_http.get("/props", wrap([this](const server_http_req &) {
            return dec_json_res(200, json{
                {"model_alias",      model_name},
                {"model_path",       params.model.path},
                {"build_info",       llama_build_info()},
                {"endpoint_metrics", params.endpoint_metrics},
                {"decision",         props_block()},
            });
        }));

        const auto get_models = [this](const server_http_req &) {
            return dec_json_res(200, json{
                {"object", "list"},
                {"data", json::array({json{
                    {"id",           model_name},
                    {"aliases",      params.model_alias},
                    {"object",       "model"},
                    {"created",      std::time(0)},
                    {"owned_by",     "llamacpp"},
                    {"capabilities", capabilities()},
                    {"decision", {
                        {"api_version",   DECISION_API_VERSION},
                        {"layout",        spec.layout},
                        {"model_id",      spec.model_id},
                        {"model_version", spec.model_version},
                    }},
                }})},
            });
        };
        ctx_http.get("/models",    wrap(get_models)); // public endpoint (no API key check)
        ctx_http.get("/v1/models", wrap(get_models)); // public endpoint (no API key check)

        ctx_http.post("/v1/systemone", wrap([this](const server_http_req & req) {
            const auto t0 = dec_clock::now();
            stats.n_requests++;
            log_request(req);
            if (debug_throw == "http") {
                throw std::runtime_error(DEC_DEBUG_THROW_TEXT);
            }
            const json body = decision_parse_body(req.body);
            auto parsed = std::make_shared<decision_request>(decision_parse_systemone(body, limits));
            auto job = std::make_shared<dec_job>();
            job->t_received = t0;
            job->run = [this, parsed](dec_job & j) { return run_systemone(*parsed, j); };
            return submit(req, job);
        }));

        ctx_http.post("/v1/router/score", wrap([this](const server_http_req & req) {
            const auto t0 = dec_clock::now();
            stats.n_requests++;
            log_request(req);
            if (!router_available) {
                throw decision_error(DECISION_REASON_ROUTER_NOT_CALIBRATED,
                                     "the model has no router calibration; start the server with --decision-allow-uncalibrated for uncalibrated scores");
            }
            const json body = decision_parse_body(req.body);
            auto parsed = std::make_shared<decision_router_request>(decision_parse_router(body, max_candidates));
            auto job = std::make_shared<dec_job>();
            job->t_received = t0;
            job->run = [this, parsed](dec_job & j) { return run_router(*parsed, j); };
            return submit(req, job);
        }));

        if (params.decision.debug) {
            ctx_http.post("/v1/decision/render", wrap([this](const server_http_req & req) {
                const auto t0 = dec_clock::now();
                log_request(req);
                const json body = decision_parse_body(req.body);
                auto items = std::make_shared<std::vector<decision_item>>();
                if (body.contains("candidates")) {
                    *items = decision_router_items(decision_parse_router(body, max_candidates), router, limits);
                } else {
                    *items = decision_request_items(decision_parse_systemone(body, limits));
                }
                auto job = std::make_shared<dec_job>();
                job->t_received = t0;
                job->run = [this, items](dec_job &) { return run_render(*items); };
                return submit(req, job);
            }));
        }

        if (params.endpoint_metrics) {
            ctx_http.get("/metrics", wrap([this](const server_http_req &) {
                auto res = std::make_unique<server_http_res>();
                res->content_type = "text/plain; version=0.0.4";
                res->data = metrics_text();
                return res;
            }));
        }
    }
};

//
// entry points
//

bool server_decision_prepare(common_params & params) {
    if (std::getenv("LLAMA_SERVER_ROUTER_PORT") != nullptr) {
        SRV_ERR("%s", "--decision is not supported for router mode instances; start a separate llama-server --decision process\n");
        return false;
    }
    if (!params.model.hf_repo.empty() || !params.model.docker_repo.empty() || !params.model.url.empty()) {
        SRV_ERR("%s", "--decision serves a local file only: use -m FILE (-hf, -dr and -mu are not supported)\n");
        return false;
    }
    if (params.model.path.empty()) {
        SRV_ERR("%s", "--decision requires -m FILE or -m DIR (without a model the server would start in router mode)\n");
        return false;
    }
    const bool is_dir = decision_path_is_dir(params.model.path);
    if (!is_dir && (!params.decision.convert_cache.empty() || params.decision.convert_type != common_params().decision.convert_type)) {
        SRV_WRN("%s", "decision mode: --decision-convert-cache/--decision-convert-type apply to -m DIR only and are ignored for a GGUF file\n");
    }

    // chat, embedding and UI options have no meaning here
    const common_params def = common_params();
    std::vector<std::string> ignored;
    if (params.n_parallel != -1)                                                  ignored.push_back("--parallel");
    if (params.cache_type_k != def.cache_type_k || params.cache_type_v != def.cache_type_v) ignored.push_back("-ctk/-ctv");
    if (params.n_ctx_checkpoints != def.n_ctx_checkpoints)                        ignored.push_back("--ctx-checkpoints");
    if (params.speculative.has_dft() || params.speculative.types != def.speculative.types) ignored.push_back("--spec-*/-md");
    if (params.embedding || params.pooling_type != def.pooling_type)              ignored.push_back("--embedding/--pooling");
    if (!params.mmproj.empty())                                                   ignored.push_back("--mmproj");
    if (!params.lora_adapters.empty())                                            ignored.push_back("--lora");
    if (params.sleep_idle_seconds != def.sleep_idle_seconds)                      ignored.push_back("--sleep-idle-seconds");
    if (!params.mcp_servers_config.empty() || !params.mcp_servers_json.empty())   ignored.push_back("--mcp-servers-*");
    if (!params.server_tools.empty())                                             ignored.push_back("--tools");
    if (params.ui_mcp_proxy)                                                      ignored.push_back("--ui-mcp-proxy");
    if (!params.public_path.empty())                                              ignored.push_back("--path");
    for (const auto & name : ignored) {
        SRV_WRN("decision mode: %s is ignored\n", name.c_str());
    }

    // default model name: the file name without .gguf (llama_server would use the path as given),
    // the same derivation as the default spec model_id; for -m DIR the directory name, which is
    // also the file name of its cached GGUF (<cache>/<key>/<name>.gguf)
    if (params.model_alias.empty()) {
        params.model_alias.insert(is_dir ? decision_checkpoint_name(params.model.path) : decision_model_name(params.model.path));
    }

    // ui and public_path are read by ctx_http.init
    params.ui           = false;
    params.ui_mcp_proxy = false;
    params.embedding    = false;
    params.public_path.clear();

    // setenv: here, before ctx_http.start() creates the HTTP threads (getenv in them would race)
    decision_cpu_env_defaults();
    return true;
}

bool server_decision_is_checkpoint_dir(const std::string & path) {
    return decision_is_laya_checkpoint_dir(path);
}

std::string server_decision_foreign_laya_hint(const std::string & path) {
    const std::string arch = server_decision_gguf_arch(path);
    return arch == "ggmlc" || arch == "laya-head" ? laya_foreign_gguf_hint(arch) : std::string();
}

std::string server_decision_gguf_arch(const std::string & path) {
    std::ifstream f = decision_ifstream(path);
    char     magic[4];
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0, n_key = 0;
    if (!f.read(magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        !f.read((char *) &version, sizeof(version)) || !f.read((char *) &n_tensors, sizeof(n_tensors)) ||
        !f.read((char *) &n_kv, sizeof(n_kv)) || n_kv == 0 || !f.read((char *) &n_key, sizeof(n_key))) {
        return "";
    }
    const std::string want = "general.architecture";
    std::string key(want.size(), '\0');
    uint32_t type = 0;
    uint64_t n_val = 0;
    if (n_key != want.size() || !f.read(&key[0], n_key) || key != want ||
        !f.read((char *) &type, sizeof(type)) || type != 8 /* GGUF_TYPE_STRING */ ||
        !f.read((char *) &n_val, sizeof(n_val)) || n_val > 256) {
        return "";
    }
    std::string arch(n_val, '\0');
    if (n_val > 0 && !f.read(&arch[0], n_val)) {
        return "";
    }
    return arch;
}

int server_decision_main(
        common_params & params,
        server_http_context & ctx_http,
        std::function<void(int)> & shutdown_handler,
        const std::function<void()> & register_signals) {
    server_decision dec(params);
    dec.register_routes(ctx_http);

    shutdown_handler = [&dec, &ctx_http](int) {
        dec.stopping.store(true);
        dec.worker.cancel_all();
        ctx_http.stop();
    };
    // dec dies with this function; a late signal must not reach it
    struct handler_reset {
        std::function<void(int)> & h;
        ~handler_reset() { h = [](int) {}; }
    } reset { shutdown_handler };
    if (register_signals) {
        register_signals();
    }

    // start the HTTP server before loading the model to be able to serve 503 on /health
    if (!ctx_http.start()) {
        SRV_ERR("%s", "exiting due to HTTP server error\n");
        return 1;
    }
    SRV_INF("decision mode, listening on %s\n", ctx_http.listening_address.c_str());

    std::string err;
    const bool ok = dec.load(err);
    if (!ok || dec.stopping.load()) {
        if (!ok) {
            SRV_ERR("failed to load the decision model: %s\n", err.c_str());
        }
        // a joinable std::thread in the destructor would call std::terminate
        ctx_http.stop();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        return ok ? 0 : 1;
    }

    dec.worker.start();
    ctx_http.is_ready.store(true);
    SRV_INF("%s", "decision model loaded, ready\n");

    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }
    dec.worker.stop();
    SRV_INF("%s", "decision server stopped\n");
    return 0;
}
