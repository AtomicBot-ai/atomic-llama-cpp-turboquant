#pragma once

// Clef prompt: encode_record of joint_schema_model.py (Cloudflare/clef, Cloudflare/clef-flash).
// The pieces of the prompt are tokenized one by one, as the model was trained. The question text and
// every option text are spans that the joint head reads (llama_set_decision_order).

#include "decision.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

// same values as enum llama_decision_order in src/llama-ext.h
#define CLEF_ORDER_NONE            0
#define CLEF_ORDER_QUESTION_NOUL   1
#define CLEF_ORDER_QUESTION_CHOICE 2
#define CLEF_ORDER_QUESTION_SCORE  3
#define CLEF_ORDER_OPTION          4

// default max_length of encode_record
#define CLEF_DEFAULT_MAX_TOKENS 16384

// text -> token ids, as HF tokenizer(text, add_special_tokens=False): special-token text is parsed
using clef_tokenize_fn = std::function<std::vector<int32_t>(const std::string &)>;

struct clef_prompt {
    std::vector<int32_t> tokens;
    std::vector<int32_t> order;           // one CLEF_ORDER_* per token
    int32_t              n_options   = 0; // option spans in the prompt; the head returns one score per span
    int32_t              n_state     = 0; // state tokens before the cut
    int32_t              n_state_cut = 0; // state tokens dropped to fit max_tokens (from the end, as encode_record)
    // per question: for each key of q.keys, the index of its option span in the prompt
    std::vector<std::vector<int32_t>> option_index;
};

// render(value) of the reference: a string as is, anything else as json.dumps(sort_keys=True, compact)
std::string clef_render(const json & value);

// question_options of the reference: (option id, description) in prompt order; a null description has no
// "description" field. noul: true, false (built-in descriptions unless the criteria give one); choice:
// sorted by id; score: by level
std::vector<std::pair<std::string, json>> clef_question_options(const decision_question & q);

// the prompt of a state and its questions. Throws decision_error PROMPT_TOO_LONG when the prompt without
// the state does not fit max_tokens.
clef_prompt clef_build_prompt(
        const std::vector<const decision_question *> & questions,
        const json & state,
        int32_t max_tokens,
        const clef_tokenize_fn & tokenize);
