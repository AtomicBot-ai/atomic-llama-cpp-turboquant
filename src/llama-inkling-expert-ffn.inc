    struct inkling_ffn_params {
        const projection * down;
        float beta, linear_beta;
    };
    static constexpr size_t inkling_ffn_max_routes = 64;
    // ggml's CPU custom op reads only the leading callback parameters.
    // Keep these graph-owned settings in the trailing bytes copied by the scheduler.
    static_assert(sizeof(ggml_custom_op_params) + sizeof(inkling_ffn_params) + sizeof(int32_t) <= GGML_MAX_OP_PARAMS);

    using inkling_leases = std::array<llama_expert_store::lease, inkling_ffn_max_routes>;
    struct inkling_window {
        llama_memory_budget::ticket charge;
        inkling_leases gates, downs;
    };
    std::unique_ptr<inkling_window> inkling_deferred;

    void prepare_inkling_window() {
        if (!inkling_window_active || !inkling_window_append || inkling_deferred) { return; }
        auto charge = budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, sizeof(inkling_window));
        if (!charge) { return; }
        auto state = std::unique_ptr<inkling_window>(new (std::nothrow) inkling_window);
        if (!state) { return; }
        charge.commit();
        state->charge = std::move(charge);
        inkling_deferred = std::move(state);
    }

    void compute_inkling_ffn(const projection & gate_up, const inkling_ffn_params & params, ggml_tensor * dst) {
        llama_expert_timer projection_timer(profile ? &profile->projection_ns : nullptr);
        llama_allocation_scope explicit_admission;
        const auto & down = *params.down;
        const auto * input = dst->src[0];
        const auto * ids = dst->src[1];
        auto & routes = *route_index;
        routes.build(ids, gate_up.experts);
        if (profile) { profile->route_id_reads += ggml_nelements(ids); }
        std::array<int64_t, inkling_ffn_max_routes> experts;
        std::copy(routes.active.begin(), routes.active.begin() + routes.count, experts.begin());

        const bool join = inkling_deferred && inkling_window_backend &&
            ggml_is_contiguous(dst) && inkling_window_active(inkling_window_backend);
        // Keep the existing two-tile workspace geometry. A fully resident
        // first tile can finish its FFN while the host admits the other tile.
        size_t ready = 0;
        if (inkling_resident_first && join && ids->ne[1] == 4 &&
                routes.count > max_wave_groups && routes.count <= 2 * max_wave_groups) {
            std::array<int64_t, inkling_ffn_max_routes> ordered;
            size_t tail = routes.count;
            size_t down_misses = 0;
            for (size_t i = 0; i < routes.count; ++i) {
                const auto e = experts[i];
                const bool down_resident = store->contains(down.name, e);
                if (!down_resident) { ++down_misses; }
                const bool resident = down_resident && store->contains(gate_up.name, e);
                if (resident && ready < max_wave_groups) { ordered[ready++] = e; }
                else { ordered[--tail] = e; }
            }
            // An extra submission for a lightly loaded layer costs more than
            // it hides. Use the same miss threshold as the gate-up pipeline.
            if (down_misses >= 3 && ready && routes.count - ready <= max_wave_groups) {
                std::reverse(ordered.begin() + ready, ordered.begin() + routes.count);
                std::copy_n(ordered.begin(), routes.count, experts.begin());
            } else {
                ready = 0;
            }
        }

        // Advance cache history once per logical projection, even if its
        // leases are acquired in two groups. Hold both sets through completion.
        inkling_leases gates, downs;
        const auto read_layout = join && ids->ne[1] == 4 ?
            llama_expert_read_layout::parallel_slices : llama_expert_read_layout::adjacent;
        const auto acquire = [&](const projection & p, llama_expert_store::lease * leases, size_t begin, size_t end) {
            llama_expert_timer timer(profile ? &profile->acquire_ns : nullptr);
            for (size_t first = begin; first < end; first += max_wave_groups) {
                const size_t count = std::min(end - first, max_wave_groups);
                store->acquire_many(p.name, experts.data() + first, count, leases + first, nullptr, read_layout);
            }
        };
        store->begin_projection(gate_up.name, ids->ne[1]);
        acquire(gate_up, gates.data(), 0, ready ? ready : routes.count);
        // Only split a graph if several known down-projection misses can
        // overlap the gate-up dispatch. Fully cached layers submit once.
        size_t misses = 0;
        if (inkling_pipeline && !ready) {
            for (size_t i = 0; i < routes.count && misses < 3; ++i) {
                if (!store->contains(down.name, experts[i])) { ++misses; }
            }
        }
        const bool pipeline = misses >= 3;
        if (!pipeline) {
            if (!ready) { store->begin_projection(down.name, ids->ne[1]); }
            acquire(down, downs.data(), 0, ready ? ready : routes.count);
        }

        ggml_reset(ctx.get());
        auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
        const auto buft = ggml_backend_buffer_get_type(buffer.get());
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        const size_t capacity = ggml_backend_buffer_get_size(buffer.get());
        auto * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer.get()));
        size_t cursor = 0;
        const auto bind = [&](ggml_tensor * tensor) {
            if (tensor->view_src) {
                if (ggml_backend_view_init(tensor) == GGML_STATUS_SUCCESS) { return; }
            } else {
                const size_t bytes = rounded(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
                if (bytes <= capacity - cursor && ggml_backend_tensor_alloc(buffer.get(), tensor, base + cursor) == GGML_STATUS_SUCCESS) {
                    cursor += bytes;
                    return;
                }
            }
            throw std::runtime_error("cannot bind Inkling expert FFN");
        };
        ggml_tensor * destination_rows = nullptr;
        if (join) {
            // Make a leaf in the expert graph over the scheduler's shared
            // output. Expanding dst itself would visit the primary graph.
            destination_rows = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, down.output, ggml_nrows(dst));
            if (ggml_backend_tensor_alloc(dst->buffer, destination_rows, dst->data) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("cannot bind shared Inkling expert output");
            }
        }
        struct tile {
            ggml_tensor * indices;
            ggml_tensor * gate_up;
            ggml_tensor * down;
            size_t first_row;
        };
        std::array<tile, inkling_ffn_max_routes / max_wave_groups> tiles;
        std::array<int32_t, inkling_ffn_max_routes> route_order;
        size_t n_routes = 0;
        const size_t n_slots = ids->ne[0];
        const size_t input_bytes = gate_up.input * sizeof(float);
        const size_t output_bytes = down.output * sizeof(float);
        const auto project = [&](ggml_tensor * src, ggml_tensor * indices, const llama_expert_store::lease * weights, size_t count) {
            std::array<uint64_t, max_wave_groups> refs;
            for (size_t i = 0; i < count; ++i) { refs[i] = uintptr_t(weights[i].tensor()); }
            auto * table = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, count);
            bind(table);
            ggml_backend_tensor_set(table, refs.data(), 0, count * sizeof(uint64_t));
            auto * op = ggml_mul_mat_id(ctx.get(), weights[0].tensor(), src, indices);
            if (!inkling_addresses(op, table) || !ggml_backend_supports_op(backend.get(), op)) {
                throw std::runtime_error("Metal rejected the Inkling expert FFN");
            }
            bind(op);
            return op;
        };
        const auto build_down = [&](tile & part, size_t first, size_t count) {
            auto * fused = part.gate_up;
            const int64_t rows = fused->ne[1];
            ggml_tensor * act = inkling_situ ? inkling_situ(ctx.get(), fused, params.beta, params.linear_beta) : nullptr;
            if (act) {
                bind(act);
            } else {
                auto * gate = ggml_view_2d(ctx.get(), fused, down.input, rows, fused->nb[1], 0);
                auto * up = ggml_view_2d(ctx.get(), fused, down.input, rows, fused->nb[1], down.input * sizeof(float));
                // Match the separate SiTU operations and their F32 rounding.
                act = ggml_scale(ctx.get(), ggml_tanh(ctx.get(), ggml_scale(ctx.get(), gate, 1.0f / params.beta)), params.beta);
                act = ggml_mul(ctx.get(), act, ggml_sigmoid(ctx.get(), gate));
                if (params.linear_beta > 0.0f) {
                    up = ggml_scale(ctx.get(), ggml_tanh(ctx.get(), ggml_scale(ctx.get(), up, 1.0f / params.linear_beta)),
                            params.linear_beta);
                }
                act = ggml_mul(ctx.get(), act, up);
                for (auto * tensor = gate; tensor; tensor = ggml_get_next_tensor(ctx.get(), tensor)) { bind(tensor); }
            }
            part.down = project(act, part.indices, downs.data() + first, count);
            if (destination_rows) {
                auto * scatter = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, rows);
                bind(scatter);
                ggml_backend_tensor_set(scatter, route_order.data() + part.first_row, 0, rows * sizeof(int32_t));
                auto * output = ggml_set_rows(ctx.get(), destination_rows, part.down, scatter);
                bind(output);
                ggml_build_forward_expand(graph, output);
            } else {
                ggml_build_forward_expand(graph, part.down);
            }
        };
        const auto build_tile = [&](tile & part, size_t first, size_t count) {
            std::array<int32_t, inkling_ffn_max_routes> local_ids;
            size_t rows = 0;
            for (size_t i = 0; i < count; ++i) {
                const auto expert = experts[first + i];
                for (auto r = routes.first[expert]; r != llama_expert_route_index::end; r = routes.next[r]) {
                    route_order[n_routes + rows] = r;
                    local_ids[rows++] = i;
                }
            }
            auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, gate_up.input, rows, 1);
            auto * indices = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, rows, 1);
            bind(x);
            bind(indices);
            for (size_t i = 0; i < rows; ++i) {
                const size_t token = route_order[n_routes + i] / n_slots;
                ggml_backend_tensor_set(x, static_cast<const char *>(input->data) + token * input->nb[2], i * input_bytes, input_bytes);
            }
            ggml_backend_tensor_set(indices, local_ids.data(), 0, rows * sizeof(int32_t));
            part = {indices, project(x, indices, gates.data() + first, count), nullptr, n_routes};
            if (pipeline) { ggml_build_forward_expand(graph, part.gate_up); }
            else { build_down(part, first, count); }
            n_routes += rows;
            if (profile) { ++profile->moe_batch_groups; }
        };
        const auto compute = [&](ggml_cgraph * segment, bool load_down) {
            llama_expert_timer timer(profile ? &profile->submit_ns : nullptr);
            const auto status = ggml_backend_graph_compute_async(backend.get(), segment);
            timer.next(nullptr);
            if (load_down && status == GGML_STATUS_SUCCESS) {
                try {
                    llama_expert_timer overlap(profile ? &profile->overlap_acquire_ns : nullptr);
                    const auto before = profile ? profile->direct_read_bytes : 0;
                    if (ready) { acquire(gate_up, gates.data(), ready, routes.count); }
                    store->begin_projection(down.name, ids->ne[1]);
                    acquire(down, downs.data(), ready, routes.count);
                    if (profile) { profile->overlap_read_bytes += profile->direct_read_bytes - before; }
                } catch (...) {
                    // Finish the queued work before unwinding its leases.
                    ggml_backend_synchronize(backend.get());
                    throw;
                }
            }
            timer.next(profile ? &profile->wait_ns : nullptr);
            ggml_backend_synchronize(backend.get());
            timer.next(nullptr);
            if (status != GGML_STATUS_SUCCESS) { throw std::runtime_error("Inkling expert FFN graph failed"); }
            if (profile && graph_timing) {
                uint64_t elapsed = 0, buffers = 0;
                if (graph_timing(backend.get(), &elapsed, &buffers)) {
                    profile->gpu_command_ns += elapsed;
                    profile->gpu_command_buffers += buffers;
                    ++profile->gpu_timed_submissions;
                }
            }
            ++n_submissions;
        };
        const auto submit_output = [&](ggml_cgraph * segment) {
            if (!join) {
                compute(segment, false);
                return;
            }
            const auto release = +[](void * user) {
                auto & self = *static_cast<impl *>(user);
                ggml_reset(self.ctx.get());
                for (auto & lease : self.inkling_deferred->gates) { lease.reset(); }
                for (auto & lease : self.inkling_deferred->downs) { lease.reset(); }
            };
            llama_expert_timer timer(profile ? &profile->submit_ns : nullptr);
            if (inkling_window_append(inkling_window_backend, segment, release, this) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("cannot append Inkling experts to the target Metal window");
            }
            // Encoding is synchronous. The target backend submits later and
            // releases every lease after completion or discarded encoding.
            inkling_deferred->gates = std::move(gates);
            inkling_deferred->downs = std::move(downs);
            if (profile) { ++profile->inkling_joined_graphs; }
        };
        if (ready) {
            if (profile) { ++profile->inkling_resident_splits; }
            build_tile(tiles[0], 0, ready);
            const int prefix = ggml_graph_n_nodes(graph);
            compute(graph, true);
            build_tile(tiles[1], ready, routes.count - ready);
            auto suffix = ggml_graph_view(graph, prefix, ggml_graph_n_nodes(graph));
            submit_output(&suffix);
        } else {
            for (size_t first = 0; first < routes.count; first += max_wave_groups) {
                build_tile(tiles[first / max_wave_groups], first, std::min(size_t(routes.count) - first, max_wave_groups));
            }
            if (pipeline) {
                const int prefix = ggml_graph_n_nodes(graph);
                compute(graph, true);
                for (size_t first = 0; first < routes.count; first += max_wave_groups) {
                    const size_t count = std::min(size_t(routes.count) - first, max_wave_groups);
                    build_down(tiles[first / max_wave_groups], first, count);
                }
                auto suffix = ggml_graph_view(graph, prefix, ggml_graph_n_nodes(graph));
                submit_output(&suffix);
            } else {
                submit_output(graph);
            }
        }
        n_projections += 2;
        n_matmuls += 2 * n_routes;
        if (join) { return; }
        llama_expert_timer readback_timer(profile ? &profile->readback_ns : nullptr);
        size_t r = 0;
        for (size_t first = 0; first < routes.count; first += max_wave_groups) {
            const auto * output = tiles[first / max_wave_groups].down;
            for (int64_t i = 0; i < output->ne[1]; ++i, ++r) {
                const size_t route = route_order[r];
                auto * destination = static_cast<char *>(dst->data) + route / n_slots * dst->nb[2] + route % n_slots * dst->nb[1];
                ggml_backend_tensor_get(output, destination, i * output_bytes, output_bytes);
            }
        }
        ggml_reset(ctx.get());
    }

    static void inkling_ffn_callback(ggml_tensor * dst, int ith, int, void * userdata) noexcept {
        if (ith != 0) { return; }
        const auto & gate_up = *static_cast<const projection *>(userdata);
        auto & owner = *gate_up.owner;
        if (!owner.failed.load()) {
            inkling_ffn_params params;
            std::memcpy(&params, reinterpret_cast<const char *>(dst->op_params) + sizeof(ggml_custom_op_params), sizeof(params));
            try { owner.compute_inkling_ffn(gate_up, params, dst); return; }
            catch (const std::exception & e) { std::snprintf(owner.failure, sizeof(owner.failure), "%s", e.what()); }
            catch (...) { std::snprintf(owner.failure, sizeof(owner.failure), "unknown Inkling expert FFN failure"); }
            owner.failed.store(true);
        }
        std::memset(dst->data, 0, ggml_nbytes(dst));
    }
