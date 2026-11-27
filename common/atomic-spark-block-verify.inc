// Block verification (Sun et al., ICLR 2025, Algorithm 2).
// Score each proposed prefix on a private copy. Only the retained prefix and
// the correction token advance the live sampler.
llama_tokens common_sampler_sample_and_accept_spark_block(
        common_sampler * sampler, llama_context * ctx, const std::vector<int> & indices,
        const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & proposals) {
    auto ordinary = [&] {
        return common_sampler_sample_and_accept_n_rejection(sampler, ctx, indices, draft, proposals);
    };
    const auto & params = sampler->params;
    if (draft.empty() || draft.size() > 7 || sampler->grmr || params.backend_sampling ||
        params.mirostat != 0 || params.xtc_probability != 0 ||
        std::find(params.samplers.begin(), params.samplers.end(), COMMON_SAMPLER_TYPE_ADAPTIVE_P) != params.samplers.end()) {
        return ordinary();
    }
    GGML_ASSERT(indices.size() == draft.size() + 1 && proposals.size() == draft.size());

    common_sampler_ptr shadow(common_sampler_clone(sampler));
    auto rng = sampler->rng;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    size_t retained = 0;
    double weight = 1.0;
    double selected_mass = 0.0;
    std::vector<std::pair<llama_token, double>> selected, residual;

    for (size_t i = 0; i <= draft.size() && weight > 0.0; ++i) {
        const bool last = i == draft.size();
        common_sampler_sample(shadow.get(), ctx, indices[i]);
        const auto & p = *common_sampler_get_candidates(shadow.get(), false);
        double p_sum = 0.0;
        for (size_t j = 0; j < p.size; ++j) {
            p_sum += p.data[j].p;
        }
        double q_sum = 0.0;
        if (!last) {
            for (const auto & item : proposals[i]) {
                q_sum += item.p;
            }
        }
        if (!(p_sum > 0.0) || !std::isfinite(p_sum) || (!last && (!(q_sum > 0.0) || !std::isfinite(q_sum)))) {
            return ordinary();
        }

        double mass = 0.0;
        double next_p = 0.0;
        residual.clear();
        for (size_t j = 0; j < p.size; ++j) {
            const auto & item = p.data[j];
            const double target = item.p / p_sum;
            const double proposal = last ? 0.0 : prob_of(proposals[i].data(), proposals[i].size(), item.id) / q_sum;
            const double remaining = last ? target : std::max(0.0, weight * target - proposal);
            if (remaining > 0.0) {
                residual.emplace_back(item.id, remaining);
                mass += remaining;
            }
            if (!last && item.id == draft[i]) {
                next_p = target;
            }
        }
        // A prefix with no residual mass cannot supply the correction.
        const double keep = last ? weight : (mass > 0.0 ? mass / (mass + (1.0 - weight)) : 0.0);
        if (i == 0 || uniform(rng) < keep) {
            retained = i;
            selected.swap(residual);
            selected_mass = mass;
        }
        if (!last) {
            const double next_q = prob_of(proposals[i].data(), proposals[i].size(), draft[i]) / q_sum;
            if (!(next_q > 0.0)) {
                return ordinary();
            }
            weight = std::min(1.0, weight * next_p / next_q);
            if (weight > 0.0) {
                common_sampler_accept(shadow.get(), draft[i], true);
            }
        }
    }

    if (selected.empty() || !(selected_mass > 0.0)) {
        return ordinary();
    }
    llama_token correction = selected.back().first;
    double draw = uniform(rng) * selected_mass;
    for (const auto & item : selected) {
        draw -= item.second;
        if (draw < 0.0) {
            correction = item.first;
            break;
        }
    }
    sampler->rng = rng;
    llama_tokens result(draft.begin(), draft.begin() + retained);
    result.push_back(correction);
    const auto * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const auto eos = std::find_if(result.begin(), result.end(), [vocab](llama_token id) { return llama_vocab_is_eog(vocab, id); });
    if (eos != result.end()) {
        result.erase(eos + 1, result.end());
    }
    for (llama_token id : result) {
        common_sampler_accept(sampler, id, true);
    }
    return result;
}
