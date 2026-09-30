#pragma once

// Unicode steps of the byte-level BPE tokenizer (English laya: ModernBERT / OLMo BPE), with the
// tables of HF tokenizers (laya-unicode-data.inc, see gen-unicode-data.py). Internal to laya.cpp.

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// NFC of s into out, as the HF NFC normalizer computes it. Returns false (out untouched) when s is
// already NFC by a quick check (ASCII) or is not valid UTF-8.
bool laya_nfc(const std::string & s, std::string & out);

// The ByteLevel pre-tokenizer regex (GPT-2):
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// Appends the [begin, end) byte ranges of the pieces of s[begin, end) to out.
void laya_gpt2_split(const std::string & s, size_t begin, size_t end, std::vector<std::pair<size_t, size_t>> & out);
