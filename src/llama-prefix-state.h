#pragma once

#include "llama.h"

// Internal-only restore flag for complete slot files written by the prefix
// checkpoint owner. Public slot restore always uses the ordinary flags.
constexpr llama_state_seq_flags LLAMA_STATE_SEQ_FLAGS_ATOMIC_PREFIX = 1u << 30;
