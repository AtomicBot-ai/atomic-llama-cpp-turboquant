#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

class common_atomic_dflash_cost {
public:
    static constexpr float prior = 0.50f;

    float row_cost() const { return slope_ratio; }

    void observe(int32_t depth, int64_t elapsed_us) {
        if (elapsed_us <= 0) {
            return;
        }
        double elapsed = elapsed_us;
        if (count != 0) {
            // A stalled read should inform the fit without dominating it.
            const double predicted = mean_us * (1.0 + slope_ratio * depth) / (1.0 + slope_ratio * mean_depth);
            elapsed = std::min(elapsed, 2.0 * std::max(mean_us, predicted));
        }
        samples[next] = { double(depth), elapsed };
        next = (next + 1) % samples.size();
        count = std::min(count + 1, samples.size());

        mean_depth = mean_us = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            mean_depth += samples[i].depth;
            mean_us += samples[i].us;
        }
        mean_depth /= count;
        mean_us /= count;

        double variance = 0.0;
        double covariance = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double centered_depth = samples[i].depth - mean_depth;
            variance += centered_depth * centered_depth;
            covariance += centered_depth * (samples[i].us - mean_us);
        }
        slope_ratio = prior;
        if (count < 4 || variance == 0.0) {
            return;
        }

        // Fit a line centered on the observed mean. Shrink its slope toward the
        // prior with the weight of four observations one row away from the mean.
        constexpr double prior_weight = 4.0;
        const double prior_slope = mean_us * prior / (1.0 + prior * mean_depth);
        const double slope = (covariance + prior_weight * prior_slope) / (variance + prior_weight);

        // Keep the anchor cost positive and allow verification rows to cost up to
        // 1.5 anchors; the normalized ratio is what the depth selector needs.
        constexpr double max_ratio = 1.5;
        const double max_slope = mean_us * max_ratio / (1.0 + max_ratio * mean_depth);
        const double bounded = std::max(0.0, std::min(slope, max_slope));
        slope_ratio = bounded / (mean_us - bounded * mean_depth);
    }

private:
    struct sample {
        double depth = 0.0;
        double us = 0.0;
    };

    std::array<sample, 32> samples{};
    std::size_t next = 0;
    std::size_t count = 0;
    double mean_us = 0.0;
    double mean_depth = 0.0;
    float slope_ratio = prior;
};

class common_atomic_dflash_depth {
public:
    // The budgeted DFlash2 profile admits at most six draft rows.
    static constexpr std::size_t capacity = 6;

private:
    struct observations {
        unsigned accepted = 0;
        unsigned reached = 0;
    };

    std::array<std::array<observations, 3>, capacity> by_position{};
    std::array<std::size_t, capacity> pending_bins{};
    std::size_t pending_depth = 0;
    unsigned rounds = 0;

public:
    std::size_t choose(const float * confidence, std::size_t offered, std::size_t minimum, float measured_row_cost) {
        pending_depth = 0;
        if (offered == 0) {
            return 0;
        }

        float survival = 1.0f;
        float expected_tokens = 1.0f; // the target also emits a correction or bonus token
        float best_rate = 1.0f;
        const float row_cost = measured_row_cost >= 0.0f ? measured_row_cost : common_atomic_dflash_cost::prior;
        for (std::size_t i = 0; i < offered; ++i) {
            const float p = confidence[i];
            const std::size_t bin = (p >= 0.80f) + (p >= 0.95f);
            pending_bins[i] = bin;
            const auto & obs = by_position[i][bin];

            // The peak over the selector's shortlist is a feature, not a calibrated probability.
            const float prior = 0.15f + 0.65f * p;
            const float conditional = (obs.accepted + 4.0f * prior) / (obs.reached + 4.0f);
            survival *= conditional;
            expected_tokens += survival;

            // Cost in units of one ordinary decode step. A verification batch shares some
            // target work, but another row still adds routed expert reads.
            const float rate = expected_tokens / (1.0f + row_cost * (i + 1));
            if (i + 1 >= minimum && rate > best_rate) {
                best_rate = rate;
                pending_depth = i + 1;
            }
        }

        // Occasionally observe positions that the policy would otherwise never send.
        if (++rounds % 8 == 0) {
            pending_depth = offered;
        }
        return pending_depth;
    }

    void observe(std::size_t accepted) {
        for (std::size_t i = 0; i < pending_depth && i <= accepted; ++i) {
            auto & obs = by_position[i][pending_bins[i]];
            ++obs.reached;
            obs.accepted += i < accepted;
        }
        pending_depth = 0;
    }
};
