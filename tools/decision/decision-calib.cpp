#include "decision-calib.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

bool decision_confidence_from_name(const std::string & name, decision_confidence_mode & mode) {
    if (name == "laya")     { mode = DECISION_CONFIDENCE_LAYA;     return true; }
    if (name == "typesafe") { mode = DECISION_CONFIDENCE_TYPESAFE; return true; }
    if (name == "max_p")    { mode = DECISION_CONFIDENCE_MAX_P;    return true; }
    return false;
}

// bucket match rank: 0 exact "N", 1 range "A-B", 2 open "N+", 3 "*", -1 no match
static int bucket_rank(const std::string & b, int32_t k) {
    if (b == "*") {
        return 3;
    }
    if (!b.empty() && b.back() == '+') {
        return k >= atoi(b.c_str()) ? 2 : -1;
    }
    const size_t dash = b.find('-');
    if (dash != std::string::npos) {
        const int lo = atoi(b.substr(0, dash).c_str());
        const int hi = atoi(b.substr(dash + 1).c_str());
        return k >= lo && k <= hi ? 1 : -1;
    }
    return atoi(b.c_str()) == k ? 0 : -1;
}

static const json * temperature_entry(const decision_calibration & cal, decision_qtype type) {
    if (cal.method != "temperature" || !cal.temperature.is_object()) {
        return nullptr;
    }
    const char * name = decision_qtype_name(type);
    if (cal.temperature.contains(name)) {
        return &cal.temperature.at(name);
    }
    if (cal.temperature.contains("all")) {
        return &cal.temperature.at("all");
    }
    return nullptr;
}

// value of entry for n_options; false when no bucket matches
static bool temperature_value(const json & entry, int32_t n_options, double & t) {
    if (entry.is_number()) {
        t = entry.get<double>();
        return true;
    }
    int best = 4;
    if (entry.is_object()) {
        for (auto it = entry.begin(); it != entry.end(); ++it) {
            const int r = bucket_rank(it.key(), n_options);
            if (r >= 0 && r < best && it->is_number()) {
                best = r;
                t = it->get<double>();
            }
        }
    }
    return best < 4;
}

double decision_temperature(const decision_calibration & cal, decision_qtype type, int32_t n_options) {
    // no entry or no bucket: T = 1 as is (only a required calibration rules this out at
    // load); the clamp applies to temperatures the calibration gives
    const json * entry = temperature_entry(cal, type);
    double t = 1.0;
    if (!entry || !temperature_value(*entry, n_options, t) || !std::isfinite(t)) {
        return 1.0;
    }
    return std::min(cal.clamp_max, std::max(cal.clamp_min, t));
}

bool decision_calibration_covers(const decision_calibration & cal, int32_t max_options, std::string & missing) {
    static const decision_qtype types[] = { DECISION_QTYPE_NOUL, DECISION_QTYPE_CHOICE, DECISION_QTYPE_SCORE };
    for (decision_qtype type : types) {
        const json * entry = temperature_entry(cal, type);
        const int32_t k_max = type == DECISION_QTYPE_NOUL ? 2 : max_options;
        for (int32_t k = 2; k <= k_max; ++k) {
            double t = 1.0;
            if (!entry || !temperature_value(*entry, k, t)) {
                missing = std::string(decision_qtype_name(type)) + " with " + std::to_string(k) + " options";
                return false;
            }
        }
    }
    return true;
}

std::vector<double> decision_softmax(const std::vector<double> & logits, double temperature) {
    std::vector<double> p(logits.size());
    if (logits.empty()) {
        return p;
    }
    double zmax = logits[0] / temperature;
    for (double z : logits) {
        zmax = std::max(zmax, z / temperature);
    }
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        p[i] = std::exp(logits[i] / temperature - zmax);
        sum += p[i];
    }
    for (double & v : p) {
        v /= sum;
    }
    return p;
}

static double normalized_entropy_confidence(const std::vector<double> & p) {
    const size_t k = p.size();
    if (k < 2) {
        return 1.0;
    }
    double ent = 0.0;
    for (double v : p) {
        if (v > 0.0) {
            ent -= v * std::log(v);
        }
    }
    return std::min(1.0, std::max(0.0, 1.0 - ent / std::log((double) k)));
}

double decision_confidence(decision_confidence_mode mode, decision_qtype type, const std::vector<double> & probs) {
    switch (mode) {
        case DECISION_CONFIDENCE_LAYA:
            if (type == DECISION_QTYPE_NOUL && probs.size() == 2) {
                return std::max(probs[1], 1.0 - probs[1]);
            }
            return normalized_entropy_confidence(probs);
        case DECISION_CONFIDENCE_TYPESAFE:
            return normalized_entropy_confidence(probs);
        case DECISION_CONFIDENCE_MAX_P:
        default:
            return probs.empty() ? 0.0 : *std::max_element(probs.begin(), probs.end());
    }
}

double decision_platt(double a, double b, double z) {
    const double x = a * z + b;
    if (x >= 0) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    const double e = std::exp(x);
    return e / (1.0 + e);
}

json decision_answer(const decision_question & q, const std::vector<double> & probs, decision_confidence_mode mode) {
    json ans = json::object();
    ans["type"] = decision_qtype_name(q.type);
    switch (q.type) {
        case DECISION_QTYPE_NOUL:
            ans["noul"] = probs[1];
            break;
        case DECISION_QTYPE_CHOICE: {
            size_t best = 0;
            for (size_t i = 1; i < probs.size(); ++i) {
                if (probs[i] > probs[best]) {
                    best = i;
                }
            }
            json p = json::object();
            for (size_t i = 0; i < probs.size(); ++i) {
                p[q.keys[i]] = probs[i];
            }
            ans["choice"]        = q.labels.size() == q.keys.size() ? q.labels[best] : json(q.keys[best]);
            ans["probabilities"] = p;
            break;
        }
        case DECISION_QTYPE_SCORE: {
            double score = 0.0;
            json p      = json::object();
            json legend = json::object();
            for (size_t i = 0; i < probs.size(); ++i) {
                score += (double) i * probs[i];
                p[q.keys[i]]      = probs[i];
                legend[q.keys[i]] = q.criteria[i];
            }
            ans["score"]         = score;
            ans["probabilities"] = p;
            ans["legend"]        = legend;
            break;
        }
    }
    ans["confidence"] = decision_confidence(mode, q.type, probs);
    return ans;
}
