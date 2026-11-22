    // The route index has validated every ID. Each selected GGUF slice is read
    // directly into its original tensor position for a single Metal MUL_MAT_ID.
    bool compute_inkling_prompt_matrix(const projection & p, ggml_tensor * dst, const llama_expert_route_index & routes) {
        const auto * input = dst->src[0];
        const auto * ids = dst->src[1];
        const bool use_staged = prompt_down_ready == &p;
        prompt_down_ready = nullptr;
        if (!routes.count || input->nb[0] != sizeof(float) || ids->nb[0] != sizeof(int32_t) ||
                input->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 || !ggml_is_contiguous(dst)) {
            return false;
        }
        llama_expert_timer timer(profile ? &profile->graph_setup_ns : nullptr);
        const auto & info = prompt_matrix_source->tensor(p.name);
        const size_t meta = (p.prompt_down ? 5 : 4) * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false);
        auto charge = budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, meta);
        if (!charge) { return false; }
        ggml_context_ptr local(ggml_init({meta, nullptr, true}));
        if (!local) { return false; }
        charge.commit();
        auto * weights = ggml_new_tensor_3d(local.get(), info.type, p.input, p.output, p.experts);
        const auto shared_input = [&](const ggml_tensor * t) {
            return prompt_host_base(t->buffer) && (!t->view_src || prompt_host_base(t->view_src->buffer));
        };
        const bool direct = balanced_prompt && shared_input(input) && shared_input(ids);
        auto * x = direct ? ggml_view_tensor(local.get(), dst->src[0]) : ggml_dup_tensor(local.get(), input);
        auto * index = direct ? ggml_view_tensor(local.get(), dst->src[1]) : ggml_dup_tensor(local.get(), ids);
        if (direct && (ggml_backend_view_init(x) != GGML_STATUS_SUCCESS ||
                       ggml_backend_view_init(index) != GGML_STATUS_SUCCESS)) { return false; }
        auto * y = ggml_mul_mat_id(local.get(), weights, x, index);
        const size_t stride = info.expert_bytes();
        if (stride != weights->nb[2] || info.bytes != ggml_nbytes(weights) ||
                !ggml_are_same_shape(y, dst) || !ggml_backend_supports_op(backend.get(), y)) { return false; }
        auto * graph = ggml_new_graph_custom(local.get(), 8, false);
        ggml_build_forward_expand(graph, y);

        const auto buft = ggml_backend_get_default_buffer_type(backend.get());
        const uint8_t * valid = nullptr;
        if (use_staged) {
            auto staged = ggml_tallocr_new(prompt_down_workspace.get());
            if (ggml_tallocr_alloc(&staged, weights) != GGML_STATUS_SUCCESS) { return false; }
            valid = prompt_down_valid(p.experts);
        }
        const size_t bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(local.get(), buft);
        if (!bytes || bytes > ggml_backend_buft_get_max_size(buft)) { return false; }
        if (!prompt_matrix_workspace || ggml_backend_buffer_get_size(prompt_matrix_workspace.get()) < bytes) {
            // The preceding projection has completed. Drop its buffer before
            // admitting a larger one against the same ledger.
            prompt_matrix_workspace.reset();
            llama_allocation_scope scope(budget, llama_memory_category::scratch, {}, {},
                    {[](void * user, llama_memory_domain domain, size_t size) {
                        static_cast<llama_expert_store *>(user)->reclaim_to_fit(domain, size);
                    }, store.get()});
            prompt_matrix_workspace.reset(ggml_backend_buft_alloc_buffer(buft, bytes));
            if (!prompt_matrix_workspace || !prompt_host_base(prompt_matrix_workspace.get()) ||
                    prompt_host_base(prompt_matrix_workspace.get()) != ggml_backend_buffer_get_base(prompt_matrix_workspace.get())) {
                return false;
            }
        }
        auto allocator = ggml_tallocr_new(prompt_matrix_workspace.get());
        for (auto * tensor : {weights, x, index, y}) {
            if (!tensor->buffer && ggml_tallocr_alloc(&allocator, tensor) != GGML_STATUS_SUCCESS) { return false; }
        }

        bool seed_decode = cache_prompt_tail;
        if (seed_decode) {
            const auto memory = budget->snapshot(llama_memory_domain::host);
            const size_t fixed = memory.used() - store->storage_bytes();
            seed_decode = memory.applied > fixed && memory.applied - fixed >= max_expert_allocation + max_expert_payload;
        }
        // Routes are linked in token/slot order. Only the last real prompt
        // row seeds the cache, and its original bytes serve both consumers.
        const size_t tail_begin = (ids->ne[1] - 1) * ids->ne[0];
        timer.next(nullptr);
        for (size_t i = 0; i < routes.count;) {
            std::array<llama_expert_source::read_request, 16> batch;
            size_t count = 0;
            while (count < batch.size() && i < routes.count) {
                const auto expert = routes.active[i++];
                auto * destination = static_cast<char *>(weights->data) + expert * stride;
                if (seed_decode && routes.last[expert] >= tail_begin) {
                    auto lease = store->acquire(p.name, expert);
                    ggml_backend_tensor_get(lease.tensor(), destination, 0, stride);
                } else if (!valid || !valid[expert]) {
                    batch[count++] = {&info, expert, destination};
                }
            }
            read_prompt_batch(batch.data(), count, stride);
        }
        timer.next(profile ? &profile->input_ns : nullptr);
        if (!direct && ggml_is_contiguous(input)) {
            std::memcpy(x->data, input->data, ggml_nbytes(x));
        } else if (!direct) {
            for (int64_t t = 0; t < input->ne[2]; ++t) {
                for (int64_t k = 0; k < input->ne[1]; ++k) {
                    std::memcpy(static_cast<char *>(x->data) + t*x->nb[2] + k*x->nb[1],
                            static_cast<const char *>(input->data) + t*input->nb[2] + k*input->nb[1], p.input*sizeof(float));
                }
            }
        }
        if (!direct) {
            for (int64_t t = 0; t < ids->ne[1]; ++t) {
                std::memcpy(static_cast<char *>(index->data) + t*index->nb[1],
                        static_cast<const char *>(ids->data) + t*ids->nb[1], ids->ne[0]*sizeof(int32_t));
            }
        }
        timer.next(profile ? &profile->compute_ns : nullptr);
        auto * next = direct && p.prompt_down ? prepare_prompt_down(local.get(), *p.prompt_down) : nullptr;
        if (next) {
            if (ggml_backend_graph_compute_async(backend.get(), graph) != GGML_STATUS_SUCCESS) {
                ggml_backend_synchronize(backend.get());
                throw std::runtime_error("failed to submit Inkling prompt matrix");
            }
            // Drain on read failure as well. The GPU references local tensors
            // and the outer graph's input views until this fence completes.
            try {
                const auto & down = prompt_matrix_source->tensor(p.prompt_down->name);
                const size_t down_stride = down.expert_bytes();
                auto * filled = prompt_down_valid(p.experts);
                std::fill_n(filled, p.experts, uint8_t(0));
                for (size_t i = 0; i < routes.count;) {
                    std::array<llama_expert_source::read_request, 16> batch;
                    size_t count = 0;
                    while (count < batch.size() && i < routes.count) {
                        const auto expert = routes.active[i++];
                        if (cache_prompt_tail && routes.last[expert] >= tail_begin) { continue; }
                        batch[count++] = {&down, expert, static_cast<char *>(next->data) + expert * down_stride};
                        filled[expert] = 1;
                    }
                    read_prompt_batch(batch.data(), count, down_stride);
                }
                prompt_down_ready = p.prompt_down;
            } catch (...) {
                ggml_backend_synchronize(backend.get());
                throw;
            }
            ggml_backend_synchronize(backend.get());
        } else if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("failed to compute Inkling prompt matrix");
        }
        timer.next(profile ? &profile->readback_ns : nullptr);
        std::memcpy(dst->data, y->data, ggml_nbytes(y));
        ++n_matmuls;
        ++n_submissions;
        return true;
    }
    ggml_tensor * prepare_prompt_down(ggml_context * local, const projection & down) {
        const auto & info = prompt_matrix_source->tensor(down.name);
        auto * tensor = ggml_new_tensor_3d(local, info.type, down.input, down.output, down.experts);
        if (tensor->nb[2] != info.expert_bytes() || ggml_nbytes(tensor) != info.bytes) { return nullptr; }
        const auto buft = ggml_backend_get_default_buffer_type(backend.get());
        // Keep a validity byte per expert after the aligned weight allocation.
        const size_t bytes = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, tensor), ggml_backend_buft_get_alignment(buft)) + down.experts;
        if (bytes > ggml_backend_buft_get_max_size(buft)) { return nullptr; }
        if (!prompt_down_workspace || ggml_backend_buffer_get_size(prompt_down_workspace.get()) < bytes) {
            prompt_down_workspace.reset();
            const auto memory = budget->snapshot(llama_memory_domain::host);
            const size_t fixed = memory.used() - store->storage_bytes();
            if (memory.applied <= fixed || bytes + max_expert_allocation + max_expert_payload > memory.applied - fixed) { return nullptr; }
            llama_allocation_scope scope(budget, llama_memory_category::scratch, {}, {},
                    {[](void * user, llama_memory_domain domain, size_t size) {
                        static_cast<llama_expert_store *>(user)->reclaim_to_fit(domain, size);
                    }, store.get()});
            prompt_down_workspace.reset(ggml_backend_buft_alloc_buffer(buft, bytes));
        }
        if (!prompt_down_workspace || !prompt_host_base(prompt_down_workspace.get()) ||
                prompt_host_base(prompt_down_workspace.get()) != ggml_backend_buffer_get_base(prompt_down_workspace.get())) {
            prompt_down_workspace.reset();
            return nullptr;
        }
        auto allocator = ggml_tallocr_new(prompt_down_workspace.get());
        return ggml_tallocr_alloc(&allocator, tensor) == GGML_STATUS_SUCCESS ? tensor : nullptr;
    }

    uint8_t * prompt_down_valid(int64_t experts) const {
        return static_cast<uint8_t *>(ggml_backend_buffer_get_base(prompt_down_workspace.get())) +
                ggml_backend_buffer_get_size(prompt_down_workspace.get()) - experts;
    }

    void read_prompt_batch(const llama_expert_source::read_request * batch, size_t count, size_t stride) {
        if (!count) { return; }
        llama_expert_timer reads(profile ? &profile->read_ns : nullptr);
        prompt_matrix_source->read_experts(batch, count, false, false, llama_expert_read_layout::parallel_slices);
        prompt_matrix_read_bytes += count * stride;
        if (profile) { profile->direct_read_bytes += count * stride; }
    }
