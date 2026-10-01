#pragma once

// From raw per-option logits to calibrated probabilities and answers.
// All math is in double; probabilities are never rounded.

#include "decision.h"
#include "decision-spec.h"

#include <string>
#include <vector>

enum decision_confidence_mode {
    DECISION_CONFIDENCE_LAYA,      // noul: max(p, 1 - p); choice/score: 1 - H/ln K (laya reference)
    DECISION_CONFIDENCE_TYPESAFE,  // 1 - H/ln K for every type
    DECISION_CONFIDENCE_MAX_P,     // max p (arbiter/prompt.py answer)
};

bool decision_confidence_from_name(const std::string & name, decision_confidence_mode & mode);

// temperature for (type, option count): type entry, else "all"; inside an entry the
// bucket "N" wins over "A-B", then "N+", then "*". A temperature found this way is
// clamped to the spec clamp; with no entry or no matching bucket the result is T = 1,
// not clamped (a required calibration cannot get here: it covers every case at load).
double decision_temperature(const decision_calibration & cal, decision_qtype type, int32_t n_options);

// true when every question type has a temperature for every option count (noul: 2,
// choice/score: 2..max_options); else missing names the first gap
bool decision_calibration_covers(const decision_calibration & cal, int32_t max_options, std::string & missing);

// softmax(logits / temperature)
std::vector<double> decision_softmax(const std::vector<double> & logits, double temperature);

double decision_confidence(decision_confidence_mode mode, decision_qtype type, const std::vector<double> & probs);

// sigmoid(a * z + b)
double decision_platt(double a, double b, double z);

// TypeSafe answer: noul = P(true); choice = first argmax in criteria order (the label as
// given when q.labels is set: laya list criteria keep numbers and booleans);
// score = sum(i * p_i) with legend = the request criteria
json decision_answer(const decision_question & q, const std::vector<double> & probs, decision_confidence_mode mode);
