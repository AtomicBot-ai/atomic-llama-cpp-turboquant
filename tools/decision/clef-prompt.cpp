#include "clef-prompt.h"

#include <algorithm>
#include <map>

static const char * CLEF_SYSTEM_PROMPT =
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options.";

std::string clef_render(const json & value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    return decision_py_dumps_compact_sorted(value);
}

std::vector<std::pair<std::string, json>> clef_question_options(const decision_question & q) {
    std::vector<std::pair<std::string, json>> out;
    switch (q.type) {
        case DECISION_QTYPE_NOUL: {
            json criteria = {
                {"true",  "The proposition is true or the answer is yes."},
                {"false", "The proposition is false or the answer is no."},
            };
            if (q.criteria.is_object()) {
                for (const char * key : { "true", "false" }) {
                    if (q.criteria.contains(key)) {
                        criteria[key] = q.criteria.at(key);
                    }
                }
            }
            out.emplace_back("true",  criteria.at("true"));
            out.emplace_back("false", criteria.at("false"));
            break;
        }
        case DECISION_QTYPE_CHOICE:
            for (auto it = q.criteria.begin(); it != q.criteria.end(); ++it) {
                out.emplace_back(it.key(), it.value());
            }
            // sorted() of str keys: code point order, the byte order of UTF-8
            std::sort(out.begin(), out.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
            break;
        case DECISION_QTYPE_SCORE:
            for (size_t i = 0; i < q.criteria.size(); ++i) {
                out.emplace_back(std::to_string(i), q.criteria.at(i));
            }
            break;
    }
    return out;
}

static int32_t clef_question_order(decision_qtype type) {
    switch (type) {
        case DECISION_QTYPE_NOUL:   return CLEF_ORDER_QUESTION_NOUL;
        case DECISION_QTYPE_CHOICE: return CLEF_ORDER_QUESTION_CHOICE;
        case DECISION_QTYPE_SCORE:  return CLEF_ORDER_QUESTION_SCORE;
    }
    return CLEF_ORDER_NONE;
}

clef_prompt clef_build_prompt(
        const std::vector<const decision_question *> & questions,
        const json & state,
        int32_t max_tokens,
        const clef_tokenize_fn & tokenize) {
    clef_prompt res;

    std::vector<int32_t> schema;
    std::vector<int32_t> schema_order;
    auto add = [&](const std::string & text, int32_t order) {
        const std::vector<int32_t> ids = tokenize(text);
        schema.insert(schema.end(), ids.begin(), ids.end());
        schema_order.resize(schema.size(), order);
        return ids.size();
    };

    add("\n\nSCHEMA FIELDS:\n", CLEF_ORDER_NONE);
    for (size_t i = 0; i < questions.size(); ++i) {
        const decision_question & q = *questions[i];
        add("\nFIELD " + std::to_string(i + 1) + "\nID: " + q.id + "\nTYPE: " + decision_qtype_name(q.type) + "\nINSTRUCTION: ", CLEF_ORDER_NONE);
        if (add(q.instructions, clef_question_order(q.type)) == 0) {
            throw decision_error(DECISION_REASON_EMPTY_INSTRUCTIONS, "the instructions of question '" + q.id + "' have no tokens",
                                 "questions." + q.id + ".instructions");
        }
        add("\nALLOWED OPTIONS:\n", CLEF_ORDER_NONE);

        std::map<std::string, int32_t> index;
        const auto options = clef_question_options(q);
        for (size_t j = 0; j < options.size(); ++j) {
            add("OPTION " + std::to_string(j + 1) + ": ", CLEF_ORDER_NONE);
            json semantics = {{"option_id", options[j].first}};
            if (!options[j].second.is_null()) {
                semantics["description"] = options[j].second;
            }
            add(clef_render(semantics), CLEF_ORDER_OPTION);
            add("\n", CLEF_ORDER_NONE);
            index[options[j].first] = res.n_options++;
        }
        add("END FIELD\n", CLEF_ORDER_NONE);

        std::vector<int32_t> keys;
        for (const auto & k : q.keys) {
            const auto it = index.find(k);
            if (it == index.end()) {
                throw decision_error(DECISION_REASON_INTERNAL, "option '" + k + "' of question '" + q.id + "' is not in the prompt");
            }
            keys.push_back(it->second);
        }
        res.option_index.push_back(std::move(keys));
    }

    const std::vector<int32_t> prefix = tokenize(
        std::string("<|im_start|>system\n") + CLEF_SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\nSTATE:\n");
    const std::vector<int32_t> suffix = tokenize(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");

    std::vector<int32_t> state_ids = tokenize(clef_render(state));
    const size_t fixed = prefix.size() + schema.size() + suffix.size();
    if (fixed > (size_t) max_tokens) {
        throw decision_error(DECISION_REASON_PROMPT_TOO_LONG,
                             "the questions need " + std::to_string(fixed) + " tokens before the state, the limit is " + std::to_string(max_tokens),
                             "questions");
    }
    res.n_state = (int32_t) state_ids.size();
    if (state_ids.size() > max_tokens - fixed) {
        res.n_state_cut = (int32_t) (state_ids.size() - (max_tokens - fixed));
        state_ids.resize(max_tokens - fixed);
    }

    res.tokens.reserve(fixed + state_ids.size());
    res.tokens.insert(res.tokens.end(), prefix.begin(), prefix.end());
    res.tokens.insert(res.tokens.end(), state_ids.begin(), state_ids.end());
    res.tokens.insert(res.tokens.end(), schema.begin(), schema.end());
    res.tokens.insert(res.tokens.end(), suffix.begin(), suffix.end());

    res.order.assign(prefix.size() + state_ids.size(), CLEF_ORDER_NONE);
    res.order.insert(res.order.end(), schema_order.begin(), schema_order.end());
    res.order.resize(res.tokens.size(), CLEF_ORDER_NONE);
    return res;
}
