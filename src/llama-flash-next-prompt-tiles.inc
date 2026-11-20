#if defined(__APPLE__)
    // Run a prompt projection against a tile of the original GGUF weights.
    // Route rows are independent. Scatter each result to its original token/slot.
    void compute_prompt_tiles(const projection & p, ggml_tensor * dst) {
        constexpr int experts_per_tile = 32;
        constexpr int batch_rows = 256;
        const auto * input = dst->src[0];
        const auto * ids = dst->src[1];
        const auto & info = row_source->tensor(p.name);
        const auto backend_type = ggml_backend_get_default_buffer_type(backend.get());
        const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        using host_base_fn = void * (*)(ggml_backend_buffer_t);
        const auto host_base = reinterpret_cast<host_base_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_buffer_get_host_base"));
        if (!host_base || p.experts != 512 || input->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
            throw std::runtime_error("unsupported prompt tile layout");
        }
        const size_t meta = 4 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false);
        auto metadata = admitted(*budget, llama_memory_category::scratch, meta + 2 * batch_rows * sizeof(int32_t));
        ggml_context_ptr local(ggml_init({meta, nullptr, true}));
        if (!local) { throw std::bad_alloc(); }
        metadata.commit();
        auto * weights = ggml_new_tensor_3d(local.get(), info.type, p.input, p.output, experts_per_tile);
        auto * x = ggml_new_tensor_3d(local.get(), GGML_TYPE_F32, p.input, 1, batch_rows);
        auto * index = ggml_new_tensor_2d(local.get(), GGML_TYPE_I32, 1, batch_rows);
        auto * y = ggml_mul_mat_id(local.get(), weights, x, index);
        if (!ggml_backend_supports_op(backend.get(), y)) { throw std::runtime_error("unsupported prompt tile matmul"); }
        auto * graph = ggml_new_graph_custom(local.get(), 8, false);
        ggml_build_forward_expand(graph, y);
        llama_allocation_scope admission(budget, llama_memory_category::scratch, {}, {},
                {[](void * user, llama_memory_domain domain, size_t bytes) {
                    static_cast<llama_expert_store *>(user)->reclaim_to_fit(domain, bytes);
                }, store.get()});
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(local.get(), backend_type));
        if (!buf || !host_base(buf.get()) || host_base(buf.get()) != ggml_backend_buffer_get_base(buf.get())) {
            throw std::runtime_error("cannot admit shared Metal prompt tile");
        }
        ggml_backend_buffer_clear(buf.get(), 0);
        std::array<int32_t, batch_rows> rows, slots;
        const size_t stride = info.expert_bytes();
        for (int first = 0; first < p.experts; first += experts_per_tile) {
            int packed = 0;
            bool loaded = false;
            auto flush_tile = [&]() {
                if (!packed) { return; }
                if (!loaded) {
                    row_source->read_expert_range(p.name, first, first + experts_per_tile,
                            weights->data, stride * experts_per_tile);
                    loaded = true;
                }
                // The unused rows carry valid expert IDs; their results are discarded.
                std::fill_n(static_cast<int32_t *>(index->data) + packed, batch_rows - packed, 0);
                if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("Metal prompt tile failed");
                }
                for (int i = 0; i < packed; ++i) {
                    std::memcpy(static_cast<char *>(dst->data) + rows[i]*dst->nb[2] + slots[i]*dst->nb[1],
                            static_cast<const char *>(y->data) + i*y->nb[2], p.output*sizeof(float));
                }
                packed = 0;
            };
            for (int64_t token = 0; token < ids->ne[1]; ++token) {
                for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                    const auto expert = id_at(ids, slot, token);
                    if (expert < 0 || expert >= p.experts) { throw std::runtime_error("invalid prompt tile route"); }
                    if (expert < first || expert >= first + experts_per_tile) { continue; }
                    rows[packed] = token;
                    slots[packed] = slot;
                    static_cast<int32_t *>(index->data)[packed] = expert - first;
                    const auto input_slot = input->ne[1] == 1 ? 0 : slot;
                    std::memcpy(static_cast<char *>(x->data) + packed*x->nb[2],
                            static_cast<const char *>(input->data) + token*input->nb[2] + input_slot*input->nb[1],
                            p.input*sizeof(float));
                    if (++packed == batch_rows) { flush_tile(); }
                }
            }
            flush_tile();
        }
    }
#endif
