// clef engine: a Clef GGUF (llama arch "clef": Qwen3.5 backbone + joint schema head, upstream PR #29831)
// run by libllama behind decision_engine.
//
// Plan "joint": the questions of a systemone request share one prompt (encode_record of
// joint_schema_model.py) and one forward pass; the head scores all options of all questions together,
// so a question's logits depend on the other questions of the request, as in the reference.
// Independent items (router candidates) get one prompt each.

#include "decision.h"
#include "decision-spec.h"
#include "clef-prompt.h"

#include "llama.h"
#include "../../src/llama-ext.h" // llama_set_decision_order (staging API)

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {

struct engine_clef : decision_engine {
    llama_model *       model = nullptr;
    llama_context *     ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    ggml_backend_dev_t  dev   = nullptr; // nullptr: the CPU
    int32_t             n_threads  = 1;
    int32_t             max_tokens = CLEF_DEFAULT_MAX_TOKENS;
    bool                use_mmap   = true;

    const std::atomic<bool> * cancel = nullptr; // read by the abort callback during a forward pass

    ~engine_clef() override {
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }

    decision_caps caps() const override {
        decision_caps c;
        c.layout           = "clef";
        c.format           = "clef-v1";
        c.plan             = "joint";
        c.plan_independent = "joint";
        c.plans            = { "joint" };
        c.question_types   = { "noul", "choice", "score" };
        c.max_options      = DECISION_CLEF_MAX_OPTIONS;
        c.max_tokens       = max_tokens;
        c.n_threads        = n_threads;
        c.device           = dev ? ggml_backend_dev_name(dev) : "cpu";
        c.device_description = dev ? ggml_backend_dev_description(dev) : "";
        c.placement        = nullptr;
        if (dev) {
            std::string name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) { return (char) std::tolower(ch); });
            c.kernels = name == "mtl" ? "metal" : name; // the Metal registry is called "MTL"
        } else {
            c.kernels = "cpu";
        }
        c.precision        = "default";
        const size_t size  = llama_model_size(model);
        if (dev) {
            c.weights_device = size;
        } else if (use_mmap) {
            c.weights_mapped = size;
        } else {
            c.weights_loaded = size;
        }
        return c;
    }

    std::vector<int32_t> tokenize(const std::string & text) const {
        // HF tokenizer(text, add_special_tokens=False): no BOS, special-token text is parsed
        int32_t n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, false, true);
        std::vector<int32_t> ids(std::max(n, 0));
        if (n > 0 && llama_tokenize(vocab, text.data(), (int32_t) text.size(), ids.data(), n, false, true) != n) {
            throw decision_error(DECISION_REASON_INTERNAL, "tokenization failed");
        }
        return ids;
    }

    int32_t n_tokens(const std::string & text, bool) const override {
        return (int32_t) tokenize(text).size();
    }

    clef_prompt build(const std::vector<const decision_question *> & qs, const json & state) const {
        return clef_build_prompt(qs, state, max_tokens, [this](const std::string & t) { return tokenize(t); });
    }

    static decision_output describe(const clef_prompt & p, bool first) {
        decision_output out;
        out.n_state     = p.n_state;
        out.n_state_cut = p.n_state_cut;
        // one prompt for all questions: its tokens count once, on its first question
        if (first) {
            out.n_tokens = (int32_t) p.tokens.size();
            out.tokens   = p.tokens;
        }
        return out;
    }

    decision_output render(const decision_item & item) const override {
        return describe(build({ &item.q }, item.state), true);
    }

    static bool cancelled(const std::atomic<bool> * c) {
        return c && c->load();
    }

    // the head scores of a prompt, one per option span; false when cancelled
    bool compute(const clef_prompt & p, const std::atomic<bool> * c, std::vector<float> & scores) {
        if (cancelled(c)) {
            return false;
        }
        cancel = c;
        llama_set_decision_order(ctx, p.order.data(), (int32_t) p.order.size());
        llama_batch batch = llama_batch_get_one(const_cast<int32_t *>(p.tokens.data()), (int32_t) p.tokens.size());
        const int32_t rc = llama_decode(ctx, batch);
        cancel = nullptr;
        if (rc == 2 && cancelled(c)) {
            return false;
        }
        if (rc != 0) {
            throw decision_error(DECISION_REASON_INTERNAL, "clef forward pass failed (" + std::to_string(rc) + ")");
        }
        const float * emb = llama_get_embeddings(ctx);
        if (!emb) {
            throw decision_error(DECISION_REASON_INTERNAL, "clef forward pass returned no scores");
        }
        scores.assign(emb, emb + p.n_options);
        for (float s : scores) {
            // the head answers NaN when the batch order has no usable spans
            if (!std::isfinite(s)) {
                throw decision_error(DECISION_REASON_INTERNAL, "clef head returned a non-finite score");
            }
        }
        return true;
    }

    bool evaluate(
            const std::vector<decision_item> & items,
            bool independent,
            const std::atomic<bool> * c,
            const decision_render_check & check,
            std::vector<decision_output> & out) override {
        out.assign(items.size(), decision_output());

        // groups of items in one prompt: all items of one state, unless they must not see each other
        std::vector<std::pair<size_t, size_t>> groups;
        for (size_t i = 0; i < items.size(); ++i) {
            if (!independent && !groups.empty() && items[i].state == items[groups.back().first].state) {
                groups.back().second = i + 1;
            } else {
                groups.emplace_back(i, i + 1);
            }
        }

        std::vector<clef_prompt> prompts;
        for (const auto & g : groups) {
            if (cancelled(c)) {
                return false;
            }
            std::vector<const decision_question *> qs;
            for (size_t i = g.first; i < g.second; ++i) {
                qs.push_back(&items[i].q);
            }
            prompts.push_back(build(qs, items[g.first].state));
            for (size_t i = g.first; i < g.second; ++i) {
                out[i] = describe(prompts.back(), i == g.first);
                if (check) {
                    check(i, out[i]);
                }
            }
        }

        std::vector<float> scores;
        for (size_t k = 0; k < groups.size(); ++k) {
            const clef_prompt & p = prompts[k];
            if (!compute(p, c, scores)) {
                return false;
            }
            for (size_t i = groups[k].first; i < groups[k].second; ++i) {
                const std::vector<int32_t> & index = p.option_index[i - groups[k].first];
                out[i].logits.resize(index.size());
                for (size_t j = 0; j < index.size(); ++j) {
                    out[i].logits[j] = scores[index[j]];
                }
                out[i].n_evaluated = out[i].n_tokens;
            }
        }
        return true;
    }
};

bool engine_clef_abort(void * data) {
    const engine_clef * eng = (const engine_clef *) data;
    return eng->cancel && eng->cancel->load();
}

ggml_backend_dev_t clef_gpu_device(int32_t index) {
    int32_t n = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(d);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            if (n++ == index) {
                return d;
            }
        }
    }
    return nullptr;
}

} // namespace

std::unique_ptr<decision_engine> decision_engine_clef_init(const decision_spec & spec, const decision_engine_params & params) {
    std::string plan = params.plan;
    if (plan.empty() && spec.plan.contains("name") && spec.plan.at("name").is_string()) {
        plan = spec.plan.at("name").get<std::string>();
    }
    if (!plan.empty() && plan != "joint") {
        throw std::runtime_error("clef: unknown decision plan '" + plan + "' (joint)");
    }
    if (!params.kernels.empty() && params.kernels != "auto" && params.kernels != "default") {
        throw std::runtime_error("clef: kernels '" + params.kernels + "' are not supported (auto, default)");
    }
    if (!params.precision.empty() && params.precision != "default") {
        throw std::runtime_error("clef: precision '" + params.precision + "' is not supported (default)");
    }

    const std::string device_mode = params.device.empty() ? "cpu" : params.device;
    ggml_backend_dev_t dev = nullptr;
    if (device_mode == "gpu" || device_mode == "auto") {
        dev = clef_gpu_device(params.gpu_index);
        if (!dev && device_mode == "gpu") {
            throw std::runtime_error("clef: no GPU device " + std::to_string(params.gpu_index));
        }
    } else if (device_mode != "cpu") {
        throw std::runtime_error("clef: unknown decision device '" + device_mode + "' (cpu, gpu, auto)");
    }

    std::unique_ptr<engine_clef> eng(new engine_clef());
    eng->dev        = dev;
    eng->n_threads  = std::max(1, params.n_threads);
    eng->max_tokens = params.n_ctx > 0 ? params.n_ctx : CLEF_DEFAULT_MAX_TOKENS;
    eng->use_mmap   = params.use_mmap;

    ggml_backend_dev_t devices[2] = { dev, nullptr };
    llama_model_params mparams = llama_model_default_params();
    mparams.devices      = devices; // { nullptr }: no device, the CPU only
    mparams.n_gpu_layers = dev ? -1 : 0;
    mparams.load_mode    = params.use_mmap ? (params.use_mlock ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MMAP)
                                           : (params.use_mlock ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE);
    eng->model = llama_model_load_from_file(params.model_path.c_str(), mparams);
    if (!eng->model) {
        throw std::runtime_error("clef: failed to load " + params.model_path);
    }
    char arch[64] = {};
    llama_model_meta_val_str(eng->model, "general.architecture", arch, sizeof(arch));
    if (std::string(arch) != "clef") {
        throw std::runtime_error(std::string("clef: the GGUF architecture is '") + arch + "', not 'clef'");
    }
    eng->vocab = llama_model_get_vocab(eng->model);

    // no memory: the whole prompt is one ubatch
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) eng->max_tokens;
    cparams.n_batch         = (uint32_t) eng->max_tokens;
    cparams.n_ubatch        = (uint32_t) eng->max_tokens;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = eng->n_threads;
    cparams.n_threads_batch = eng->n_threads;
    cparams.embeddings      = true;
    cparams.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cparams.op_offload      = dev != nullptr;
    cparams.no_perf         = true;
    cparams.abort_callback      = engine_clef_abort;
    cparams.abort_callback_data = eng.get();
    eng->ctx = llama_init_from_model(eng->model, cparams);
    if (!eng->ctx) {
        throw std::runtime_error("clef: failed to create the context (" + std::to_string(eng->max_tokens) + " tokens)");
    }

    if (params.warmup) {
        decision_question q;
        q.id           = "warmup";
        q.type         = DECISION_QTYPE_NOUL;
        q.instructions = "warmup";
        q.keys         = { "false", "true" };
        std::vector<float> scores;
        if (!eng->compute(eng->build({ &q }, json("warmup")), nullptr, scores)) {
            throw std::runtime_error("clef: warm-up forward pass failed");
        }
    }
    return eng;
}
