// Fork-only buffer type; ordinary buffer allocation and release stay unchanged.
// Reuse the backend's tensor access/layout callbacks, with separate ownership.
static void ggml_backend_metal_expert_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_expert_buffer_free((ggml_metal_buffer_t) buffer->context);
}

static const char * ggml_backend_metal_expert_buffer_name(ggml_backend_buffer_type_t) {
    return "Metal_Expert_Grouped";
}

static ggml_backend_buffer_t ggml_backend_metal_expert_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto buffer = ggml_metal_expert_buffer_new((ggml_metal_expert_pool_t) buft->context, size);
    if (!buffer) { return nullptr; }
    auto iface = ggml_backend_metal_buffer_shared_i;
    iface.free_buffer = ggml_backend_metal_expert_free_buffer;
    return ggml_backend_buffer_init(buft, iface, buffer, size);
}

static ggml_backend_buffer_type_t ggml_backend_metal_create_expert_buffer_type(ggml_backend_dev_t dev) {
    const auto ordinary = ggml_backend_dev_buffer_type(dev);
    if (ordinary->iface.get_name != ggml_backend_metal_buffer_type_shared_get_name) { return nullptr; }
    auto result = new (std::nothrow) ggml_backend_buffer_type(*ordinary);
    if (!result) { return nullptr; }
    auto pool = ggml_metal_expert_pool_new((ggml_metal_device_t) dev->context);
    if (!pool) { delete result; return nullptr; }
    result->context = pool;
    result->iface.get_name = ggml_backend_metal_expert_buffer_name;
    result->iface.alloc_buffer = ggml_backend_metal_expert_alloc_buffer;
    return result;
}

static void ggml_backend_metal_free_expert_buffer_type(ggml_backend_buffer_type_t buft) {
    if (!buft) { return; }
    GGML_ASSERT(buft->iface.get_name == ggml_backend_metal_expert_buffer_name);
    ggml_metal_expert_pool_free((ggml_metal_expert_pool_t) buft->context);
    delete buft;
}
