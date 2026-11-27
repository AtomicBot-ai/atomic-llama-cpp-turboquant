    // A wide prompt may span all experts even when only a fraction of one
    // layer fits beside attention. Pack routed rows by bank, then scatter
    // each result back to its original (token, route) without a reduction.
    bool compute_spark_grouped_prompt(const projection & p, ggml_tensor * dst) {
        const auto * input = dst->src[0];
        const auto * ids = dst->src[1];
        if (!spark_alloc_shared || ids->ne[0] != 8 || input->type != GGML_TYPE_F32 ||
                (input->ne[1] != 1 && input->ne[1] != ids->ne[0]) ||
                dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) { return false; }
        constexpr int experts = 16, rows = 1024;
        const size_t meta = 4 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false);
        auto ticket = budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, meta);
        if (!ticket) { return false; }
        ggml_context_ptr local(ggml_init({meta, nullptr, true}));
        if (!local) { return false; }
        ticket.commit();
        const auto & info = spark_prompt_source->tensor(p.name);
        auto * weights = ggml_new_tensor_3d(local.get(), info.type, p.input, p.output, experts);
        auto * x = ggml_new_tensor_3d(local.get(), GGML_TYPE_F32, p.input, 1, rows);
        auto * index = ggml_new_tensor_2d(local.get(), GGML_TYPE_I32, 1, rows);
        auto * y = ggml_mul_mat_id(local.get(), weights, x, index);
        if (weights->nb[2] != info.expert_bytes() || !ggml_backend_supports_op(backend.get(), y)) {
            return false;
        }
        auto * graph = ggml_new_graph_custom(local.get(), 8, false);
        ggml_build_forward_expand(graph, y);
        llama_allocation_scope scope(budget, llama_memory_category::scratch, {}, discrete,
                {[](void * user, llama_memory_domain, size_t size) {
                    static_cast<llama_expert_store *>(user)->reclaim_to_fit(llama_memory_domain::device, size);
                }, store.get()});
        const size_t bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(local.get(), discrete.device);
        if (!bytes) { return false; }
        if (!spark_shared_weights || ggml_backend_buffer_get_size(spark_shared_weights.get()) < bytes) {
            spark_shared_weights.reset();
            spark_shared_cpu = nullptr;
            spark_shared_weights.reset(spark_alloc_shared(backend.get(), bytes, &spark_shared_cpu));
        }
        if (!spark_shared_weights) { return false; }
        auto allocator = ggml_tallocr_new(spark_shared_weights.get());
        for (auto * t : {weights, x, index, y}) {
            if (ggml_tallocr_alloc(&allocator, t) != GGML_STATUS_SUCCESS) { return false; }
        }
        auto mapped = [&](const ggml_tensor * t) {
            auto offset = reinterpret_cast<uintptr_t>(t->data) -
                          reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(spark_shared_weights.get()));
            return static_cast<char *>(spark_shared_cpu) + offset;
        };
        char * w = mapped(weights);
        char * batch_x = mapped(x);
        auto * batch_ids = reinterpret_cast<int32_t *>(mapped(index));
        char * batch_y = mapped(y);
        std::array<int32_t, rows> destinations;
        std::memset(w + ggml_nbytes(weights), 0,
                    ggml_backend_buft_get_alloc_size(discrete.device, weights) - ggml_nbytes(weights));
        for (int first = 0; first < p.experts; first += experts) {
            int count = 0;
            bool loaded = false;
            auto submit = [&]() {
                if (!loaded) {
                    if (!spark_prompt_source->read_spark_expert_range(p.name, first, first + experts, w, ggml_nbytes(weights))) {
                        spark_prompt_source->read_expert_range(p.name, first, first + experts, w, ggml_nbytes(weights));
                    }
                    prompt_matrix_read_bytes += ggml_nbytes(weights);
                    loaded = true;
                }
                if (count < rows) {
                    std::memset(batch_x + count*x->nb[2], 0, (rows-count)*x->nb[2]);
                    std::fill(batch_ids + count, batch_ids + rows, 0);
                }
                const auto status = ggml_backend_graph_compute(backend.get(), graph);
                if (status == GGML_STATUS_ALLOC_FAILED) { return false; }
                if (status != GGML_STATUS_SUCCESS) { throw std::runtime_error("Spark grouped prompt projection failed"); }
                for (int j = 0; j < count; ++j) {
                    std::memcpy(static_cast<char *>(dst->data) + size_t(destinations[j])*dst->nb[1],
                                batch_y + j*y->nb[2], dst->nb[1]);
                }
                ++n_matmuls;
                ++n_submissions;
                count = 0;
                return true;
            };
            for (int64_t row = 0; row < ids->ne[1]; ++row) {
                for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                    const int id = id_at(ids, slot, row);
                    if (id < first || id >= first + experts) { continue; }
                    const size_t offset = row*input->nb[2] + (slot % input->ne[1])*input->nb[1];
                    std::memcpy(batch_x + count*x->nb[2], static_cast<const char *>(input->data) + offset, x->nb[1]);
                    batch_ids[count] = id - first;
                    destinations[count++] = int32_t(row*ids->ne[0] + slot);
                    if (count == rows && !submit()) { return false; }
                }
            }
            if (count && !submit()) { return false; }
        }
        return true;
    }

    // The CPU callback owns its inputs until return. Upload packed activations
    // and original GGUF slices. Short NVFP4 batches index only the selected experts.
    bool compute_spark_prompt_matrix(const projection & p, ggml_tensor * dst, bool compact, bool tiled) {
        const auto * input = dst->src[0];
        const auto * ids = dst->src[1];
        std::array<bool, 128> selected{};
        if (p.experts != int64_t(selected.size()) || input->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
                input->nb[0] != sizeof(float) || ids->nb[0] != sizeof(int32_t) || !ggml_is_contiguous(dst)) {
            return false;
        }
        for (int64_t token = 0; token < ids->ne[1]; ++token) {
            for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                selected[id_at(ids, slot, token)] = true;
            }
        }
        const auto & info = spark_prompt_source->tensor(p.name);
        const size_t stride = info.expert_bytes();
        std::array<int32_t, 128> weight_slot;
        int64_t num_weights = 0;
        for (size_t i = 0; i < selected.size(); ++i) {
            if (!compact || selected[i]) { weight_slot[i] = int32_t(num_weights++); }
        }
        auto * hot = inkling_core ? inkling_core->hot_projection(p.name) : nullptr;
        const bool direct = compact && hot && hot->plan_direct(selected, weight_slot);
        if (direct) { num_weights = hot->cold_slot + hot->cold_capacity; }
        const size_t chunk_experts = std::min(size_t(p.experts), std::max(size_t(1), (size_t(16) << 20) / stride));
        // Mapped weights can be filled in place. Keep just one expert of fallback
        // staging for NVFP4 so the large prefill matrix still fits at tight caps.
        const size_t staging_bytes = (spark_nvfp4_matrix ? 1 : chunk_experts) * stride;
        const size_t meta = 4 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false);
        if (!store->reclaim_to_fit(llama_memory_domain::device, meta + staging_bytes)) { return false; }
        auto metadata = budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, meta);
        auto staging = budget->reserve(llama_memory_domain::host, llama_memory_category::io, staging_bytes);
        if (!metadata || !staging) { return false; }
        ggml_context_ptr local(ggml_init({meta, nullptr, true}));
        std::unique_ptr<uint8_t[]> block(new (std::nothrow) uint8_t[staging_bytes]);
        if (!local || !block) { return false; }
        metadata.commit();
        staging.commit();
        auto * weights = ggml_new_tensor_3d(local.get(), info.type, p.input, p.output, num_weights);
        // Keep one copy of the weights while processing equal-sized row tiles.
        // This bounds activation, output and CUDA MMQ scratch independently
        // of the number of experts selected across the original microbatch.
        int64_t rows = ids->ne[1];
        if (spark_nvfp4_matrix && rows > 1024) {
            rows = 256;
        } else {
            while (tiled && rows > 256 && rows % 2 == 0) { rows /= 2; }
        }
        auto * x = ggml_new_tensor_3d(local.get(), input->type, input->ne[0], input->ne[1], rows);
        auto * index = ggml_new_tensor_2d(local.get(), ids->type, ids->ne[0], rows);
        auto * y = ggml_mul_mat_id(local.get(), weights, x, index);
        if (staging_bytes < std::max(x->nb[1], index->nb[1]) ||
                stride != weights->nb[2] || dst->ne[0] != y->ne[0] || dst->ne[1] != y->ne[1] ||
                dst->ne[2] != ids->ne[1] || dst->ne[3] != 1 || !ggml_backend_supports_op(backend.get(), y)) {
            return false;
        }
        auto * graph = ggml_new_graph_custom(local.get(), 8, false);
        ggml_build_forward_expand(graph, y);
        llama_allocation_scope scope(budget, llama_memory_category::scratch, {}, discrete,
                {[](void * user, llama_memory_domain, size_t size) {
                    static_cast<llama_expert_store *>(user)->reclaim_to_fit(llama_memory_domain::device, size);
                }, store.get()});
        const size_t padded = ggml_backend_buft_get_alloc_size(discrete.device, weights);
        if (direct) {
            if (hot->offset > ggml_backend_buffer_get_size(hot->mapped) ||
                    padded > ggml_backend_buffer_get_size(hot->mapped) - hot->offset) { return false; }
            auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(hot->mapped));
            if (ggml_backend_tensor_alloc(hot->mapped, weights, base + hot->offset) != GGML_STATUS_SUCCESS) { return false; }
        } else if (spark_alloc_shared) {
            if (!spark_shared_weights || ggml_backend_buffer_get_size(spark_shared_weights.get()) < padded) {
                spark_shared_weights.reset();
                spark_shared_cpu = nullptr;
                spark_shared_weights.reset(spark_alloc_shared(backend.get(), padded, &spark_shared_cpu));
            }
            if (spark_shared_weights && ggml_backend_tensor_alloc(spark_shared_weights.get(), weights,
                    ggml_backend_buffer_get_base(spark_shared_weights.get())) != GGML_STATUS_SUCCESS) { return false; }
        }
        auto * shared = direct ? hot->data : static_cast<uint8_t *>(spark_shared_cpu);
        const size_t bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(local.get(), discrete.device);
        if (!bytes || bytes > ggml_backend_buft_get_max_size(discrete.device)) { return false; }
        if (!prompt_matrix_workspace || ggml_backend_buffer_get_size(prompt_matrix_workspace.get()) < bytes) {
            prompt_matrix_workspace.reset();
            prompt_matrix_workspace.reset(ggml_backend_buft_alloc_buffer(discrete.device, bytes));
            if (!prompt_matrix_workspace) { return false; }
        }
        auto allocator = ggml_tallocr_new(prompt_matrix_workspace.get());
        for (auto * tensor : {weights, x, index, y}) {
            if (!tensor->buffer && ggml_tallocr_alloc(&allocator, tensor) != GGML_STATUS_SUCCESS) { return false; }
        }
        // A quantized CUDA tile can read past the last row of one expert into
        // the adjacent weight bytes. Zero unused ranges so those padded loads are finite.
        const size_t max_chunk = shared ? chunk_experts : staging_bytes / stride;
        const size_t payload = ggml_nbytes(weights);
        if (direct) { prompt_matrix_read_bytes += hot->populate_tail(*spark_prompt_source, p.name, selected); }
        const bool hot_read = direct || (shared && hot && hot->ready &&
                hot->fill(*spark_prompt_source, p.name, selected, weight_slot, shared, payload, compact, prompt_matrix_read_bytes));
        const bool packed_read = !hot_read && compact && shared &&
                spark_prompt_source->read_spark_selected_experts(p.name, selected, shared, payload);
        if (packed_read) { prompt_matrix_read_bytes += payload; }
        for (size_t first = 0; !packed_read && !hot_read && first < size_t(p.experts);) {
            if (compact && !selected[first]) { ++first; continue; }
            size_t end = first + 1;
            while (end < selected.size() && end - first < max_chunk && selected[end] == selected[first]) { ++end; }
            const size_t size = (end - first) * stride;
            const size_t offset = size_t(weight_slot[first]) * stride;
            if (selected[first]) {
                auto * target = shared ? shared + offset : block.get();
                if ((!spark_iq2_parallel_reads && !(spark_nvfp4_matrix && !compact)) ||
                        !spark_prompt_source->read_spark_expert_range(p.name, first, end, target, size)) {
                    spark_prompt_source->read_expert_range(p.name, first, end, target, size);
                }
                prompt_matrix_read_bytes += size;
                if (!shared) { ggml_backend_tensor_set(weights, block.get(), offset, size); }
            } else if (shared) {
                std::memset(shared + first * stride, 0, size);
            } else {
                ggml_backend_tensor_memset(weights, 0, first * stride, size);
            }
            first = end;
        }
        if (shared) { std::memset(shared + payload, 0, padded - payload); }
        if (shared && hot && !hot->ready && !compact) {
            std::array<unsigned, 128> frequency{};
            for (int64_t row = 0; row < ids->ne[1]; ++row) {
                for (int64_t lane = 0; lane < ids->ne[0]; ++lane) { ++frequency[id_at(ids, lane, row)]; }
            }
            hot->populate(frequency, shared);
        }
        const auto upload = [&](const ggml_tensor * src, ggml_tensor * target) {
            if (ggml_is_contiguous(src)) {
                ggml_backend_tensor_set(target, src->data, 0, ggml_nbytes(src));
                return;
            }
            const size_t row_bytes = target->nb[1];
            const size_t rows = ggml_nrows(src);
            const size_t per_chunk = staging_bytes / row_bytes;
            for (size_t first = 0; first < rows; first += per_chunk) {
                const size_t count = std::min(per_chunk, rows - first);
                for (size_t j = 0; j < count; ++j) {
                    const size_t row = first + j;
                    const size_t offset = row % src->ne[1] * src->nb[1] + row / src->ne[1] * src->nb[2];
                    std::memcpy(block.get() + j * row_bytes, static_cast<const char *>(src->data) + offset, row_bytes);
                }
                ggml_backend_tensor_set(target, block.get(), first * row_bytes, count * row_bytes);
            }
        };
        ggml_tensor input_tile = *input;
        ggml_tensor ids_tile = *ids;
        for (int64_t start = 0; start < ids->ne[1]; start += rows) {
            // The tail leaves valid input and expert IDs from the previous
            // full tile in the unused lanes. Copy back only the live outputs.
            const int64_t active = std::min(rows, ids->ne[1] - start);
            input_tile.ne[2] = ids_tile.ne[1] = active;
            input_tile.data = static_cast<char *>(input->data) + start * input->nb[2];
            ids_tile.data = static_cast<char *>(ids->data) + start * ids->nb[1];
            upload(&input_tile, x);
            if (compact) {
                std::array<int32_t, 8 * 32> slots;
                for (int64_t row = 0; row < rows; ++row) {
                    for (int64_t lane = 0; lane < ids->ne[0]; ++lane) {
                        slots[row * ids->ne[0] + lane] = weight_slot[id_at(&ids_tile, lane, row)];
                    }
                }
                ggml_backend_tensor_set(index, slots.data(), 0, ggml_nbytes(index));
            } else {
                upload(&ids_tile, index);
            }
            const auto status = ggml_backend_graph_compute(backend.get(), graph);
            if (status == GGML_STATUS_ALLOC_FAILED) { return false; }
            if (status != GGML_STATUS_SUCCESS) { throw std::runtime_error("Spark Inkling matrix projection failed"); }
            ggml_backend_tensor_get(y, static_cast<char *>(dst->data) + start * dst->nb[2], 0, active * y->nb[2]);
            ++n_matmuls;
            ++n_submissions;
        }
        return true;
    }
