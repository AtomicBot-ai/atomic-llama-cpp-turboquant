#include "laya-unicode.h"

#include <cstdint>

#include "laya-unicode-data.inc"

template <size_t N>
static bool laya_in_ranges(const uint32_t (&r)[N][2], uint32_t c) {
    size_t lo = 0;
    size_t hi = N;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (r[mid][1] < c) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < N && r[lo][0] <= c;
}

static uint32_t laya_ccc(uint32_t c) {
    if (c < 0x300) {
        return 0;
    }
    size_t lo = 0;
    size_t hi = sizeof(laya_ucd_ccc) / sizeof(laya_ucd_ccc[0]);
    const size_t n = hi;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (laya_ucd_ccc[mid][1] < c) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < n && laya_ucd_ccc[lo][0] <= c ? laya_ucd_ccc[lo][2] : 0;
}

static const uint32_t S_BASE = 0xAC00, L_BASE = 0x1100, V_BASE = 0x1161, T_BASE = 0x11A7;
static const uint32_t L_COUNT = 19, V_COUNT = 21, T_COUNT = 28, N_COUNT = V_COUNT * T_COUNT, S_COUNT = L_COUNT * N_COUNT;

static void laya_decompose(uint32_t c, std::vector<uint32_t> & out) {
    if (c >= S_BASE && c < S_BASE + S_COUNT) {
        const uint32_t i = c - S_BASE;
        out.push_back(L_BASE + i / N_COUNT);
        out.push_back(V_BASE + (i % N_COUNT) / T_COUNT);
        if (i % T_COUNT) {
            out.push_back(T_BASE + i % T_COUNT);
        }
        return;
    }
    if (c >= 0xC0) {
        size_t lo = 0;
        size_t hi = sizeof(laya_ucd_decomp) / sizeof(laya_ucd_decomp[0]);
        const size_t n = hi;
        while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if (laya_ucd_decomp[mid][0] < c) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < n && laya_ucd_decomp[lo][0] == c) {
            const uint32_t * p = laya_ucd_decomp_cps + laya_ucd_decomp[lo][1];
            out.insert(out.end(), p, p + laya_ucd_decomp[lo][2]);
            return;
        }
    }
    out.push_back(c);
}

// primary composite of (a, b), 0 if none
static uint32_t laya_compose(uint32_t a, uint32_t b) {
    if (a >= L_BASE && a < L_BASE + L_COUNT && b >= V_BASE && b < V_BASE + V_COUNT) {
        return S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT;
    }
    if (a >= S_BASE && a < S_BASE + S_COUNT && (a - S_BASE) % T_COUNT == 0 && b > T_BASE && b < T_BASE + T_COUNT) {
        return a + (b - T_BASE);
    }
    size_t lo = 0;
    size_t hi = sizeof(laya_ucd_comp) / sizeof(laya_ucd_comp[0]);
    const size_t n = hi;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        const uint32_t * e = laya_ucd_comp[mid];
        if (e[0] < a || (e[0] == a && e[1] < b)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < n && laya_ucd_comp[lo][0] == a && laya_ucd_comp[lo][1] == b ? laya_ucd_comp[lo][2] : 0;
}

// strict UTF-8 decode of s[i] (no overlongs, no surrogates); false on an invalid sequence (len 1)
static bool laya_utf8_next(const std::string & s, size_t i, size_t end, uint32_t & c, size_t & len) {
    const uint8_t * u = (const uint8_t *) s.data() + i;
    const size_t    n = end - i;
    len = 1;
    if (u[0] < 0x80) {
        c = u[0];
        return true;
    }
    size_t   need = 0;
    uint8_t  lo2  = 0x80;
    uint8_t  hi2  = 0xBF;
    if (u[0] >= 0xC2 && u[0] <= 0xDF) {
        need = 2; c = u[0] & 0x1F;
    } else if (u[0] >= 0xE0 && u[0] <= 0xEF) {
        need = 3; c = u[0] & 0x0F;
        lo2 = u[0] == 0xE0 ? 0xA0 : 0x80;
        hi2 = u[0] == 0xED ? 0x9F : 0xBF;
    } else if (u[0] >= 0xF0 && u[0] <= 0xF4) {
        need = 4; c = u[0] & 0x07;
        lo2 = u[0] == 0xF0 ? 0x90 : 0x80;
        hi2 = u[0] == 0xF4 ? 0x8F : 0xBF;
    } else {
        return false;
    }
    if (n < need || u[1] < lo2 || u[1] > hi2) {
        return false;
    }
    for (size_t k = 1; k < need; ++k) {
        if ((u[k] & 0xC0) != 0x80) {
            return false;
        }
        c = (c << 6) | (u[k] & 0x3F);
    }
    len = need;
    return true;
}

static void laya_utf8_append(uint32_t c, std::string & out) {
    if (c < 0x80) {
        out += (char) c;
    } else if (c < 0x800) {
        out += (char) (0xC0 | (c >> 6));
        out += (char) (0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        out += (char) (0xE0 | (c >> 12));
        out += (char) (0x80 | ((c >> 6) & 0x3F));
        out += (char) (0x80 | (c & 0x3F));
    } else {
        out += (char) (0xF0 | (c >> 18));
        out += (char) (0x80 | ((c >> 12) & 0x3F));
        out += (char) (0x80 | ((c >> 6) & 0x3F));
        out += (char) (0x80 | (c & 0x3F));
    }
}

bool laya_nfc(const std::string & s, std::string & out) {
    size_t i = 0;
    while (i < s.size() && (uint8_t) s[i] < 0x80) {
        ++i;
    }
    if (i == s.size()) {
        return false;
    }

    // canonical decomposition
    std::vector<uint32_t> d;
    d.reserve(s.size());
    for (size_t p = 0; p < s.size();) {
        uint32_t c   = 0;
        size_t   len = 0;
        if (!laya_utf8_next(s, p, s.size(), c, len)) {
            return false;
        }
        laya_decompose(c, d);
        p += len;
    }

    // canonical ordering: stable insertion sort of each run of non-starters by class
    std::vector<uint32_t> cls(d.size());
    for (size_t k = 0; k < d.size(); ++k) {
        cls[k] = laya_ccc(d[k]);
    }
    for (size_t k = 1; k < d.size(); ++k) {
        const uint32_t c  = d[k];
        const uint32_t cc = cls[k];
        size_t j = k;
        while (cc != 0 && j > 0 && cls[j - 1] > cc) {
            d[j]   = d[j - 1];
            cls[j] = cls[j - 1];
            --j;
        }
        d[j]   = c;
        cls[j] = cc;
    }

    // canonical composition (UAX #15)
    size_t   n_out   = 0;
    int64_t  starter = -1;
    uint32_t last    = 256;
    for (size_t k = 0; k < d.size(); ++k) {
        const uint32_t c  = d[k];
        const uint32_t cc = cls[k];
        if (starter >= 0 && (last < cc || last == 0)) {
            const uint32_t comp = laya_compose(d[starter], c);
            if (comp) {
                d[starter] = comp;
                continue;
            }
        }
        if (cc == 0) {
            starter = (int64_t) n_out;
        }
        last = cc;
        d[n_out++] = c;
    }

    out.clear();
    out.reserve(s.size());
    for (size_t k = 0; k < n_out; ++k) {
        laya_utf8_append(d[k], out);
    }
    return true;
}

enum laya_cclass { LAYA_CC_LETTER, LAYA_CC_NUMBER, LAYA_CC_SPACE, LAYA_CC_OTHER };

static laya_cclass laya_classify(uint32_t c) {
    if (c < 0x80) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            return LAYA_CC_LETTER;
        }
        if (c >= '0' && c <= '9') {
            return LAYA_CC_NUMBER;
        }
        if (c == ' ' || (c >= 0x09 && c <= 0x0D)) {
            return LAYA_CC_SPACE;
        }
        return LAYA_CC_OTHER;
    }
    if (laya_in_ranges(laya_ucd_letter, c)) {
        return LAYA_CC_LETTER;
    }
    if (laya_in_ranges(laya_ucd_number, c)) {
        return LAYA_CC_NUMBER;
    }
    if (laya_in_ranges(laya_ucd_space, c)) {
        return LAYA_CC_SPACE;
    }
    return LAYA_CC_OTHER;
}

void laya_gpt2_split(const std::string & s, size_t begin, size_t end, std::vector<std::pair<size_t, size_t>> & out) {
    struct cpt {
        uint32_t    c;
        laya_cclass cls;
        size_t      off;
    };
    std::vector<cpt> cps;
    cps.reserve(end - begin);
    for (size_t p = begin; p < end;) {
        uint32_t c   = 0xFFFD;
        size_t   len = 1;
        // an invalid byte is one "other" codepoint of its own
        const bool ok = laya_utf8_next(s, p, end, c, len);
        cps.push_back({ ok ? c : 0xFFFD, ok ? laya_classify(c) : LAYA_CC_OTHER, p });
        p += len;
    }
    const size_t n = cps.size();
    auto off = [&](size_t k) { return k < n ? cps[k].off : end; };
    auto emit = [&](size_t a, size_t b) { out.emplace_back(off(a), off(b)); };

    size_t pos = 0;
    while (pos < n) {
        const uint32_t c = cps[pos].c;

        // 's|'t|'re|'ve|'m|'ll|'d
        if (c == '\'' && pos + 1 < n) {
            const uint32_t c1 = cps[pos + 1].c;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                emit(pos, pos + 2);
                pos += 2;
                continue;
            }
            if (pos + 2 < n) {
                const uint32_t c2 = cps[pos + 2].c;
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    emit(pos, pos + 3);
                    pos += 3;
                    continue;
                }
            }
        }

        // ' ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+'
        const size_t q = c == ' ' ? pos + 1 : pos;
        if (q < n && cps[q].cls != LAYA_CC_SPACE) {
            const laya_cclass cls = cps[q].cls;
            size_t e = q + 1;
            while (e < n && cps[e].cls == cls) {
                ++e;
            }
            emit(pos, e);
            pos = e;
            continue;
        }

        // '\s+(?!\S)|\s+': a run followed by a non-space leaves its last space to that piece
        size_t e = pos;
        while (e < n && cps[e].cls == LAYA_CC_SPACE) {
            ++e;
        }
        if (e < n && e - pos >= 2) {
            --e;
        }
        emit(pos, e);
        pos = e;
    }
}
