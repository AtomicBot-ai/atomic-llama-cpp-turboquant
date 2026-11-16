#include "ggml-metal-expert.h"
#include "ggml-metal-ops.h"

#include <cstdint>
#include <cstring>

// Caller owns the bound output and IDs exclusively, with no GPU work in flight.
// IDs, tensor layouts/bindings and the prepared tail stay unchanged until completion.
// Activation values may still be populated before submission; they do not affect the map.
// This is an explicit executor contract, never inferred from arbitrary GPU IDs.
bool ggml_backend_metal_prepare_single_expert(ggml_tensor * op) {
    if (!op || op->op != GGML_OP_MUL_MAT_ID) { return false; }
    op->op_params[0] = 0;
    const auto * weights = op->src[0];
    const auto * input = op->src[1];
    const auto * ids = op->src[2];
    if (!weights || !input || !ids || weights->ne[0] < 64 || weights->ne[2] != 1 || weights->ne[3] != 1 ||
        input->ne[1] != 1 || input->ne[2] != 32 || input->ne[3] != 1 ||
        ids->type != GGML_TYPE_I32 || ids->op != GGML_OP_NONE || ids->view_src ||
        ids->ne[0] != 1 || ids->ne[1] != 32 || ids->ne[2] != 1 || ids->ne[3] != 1 || !ggml_is_contiguous(ids) ||
        op->type != GGML_TYPE_F32 || op->view_src || !ggml_is_contiguous(op) ||
        op->ne[0] != weights->ne[1] || op->ne[1] != 1 || op->ne[2] != 32 || op->ne[3] != 1) { return false; }
    auto host_range = [](const ggml_tensor * tensor, size_t bytes) {
        void * base = ggml_backend_metal_buffer_get_host_base(tensor->buffer);
        if (!base || !tensor->data) { return false; }
        const auto start = reinterpret_cast<uintptr_t>(base);
        const auto at = reinterpret_cast<uintptr_t>(tensor->data);
        const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
        return at >= start && at - start <= size && bytes <= size - (at - start);
    };
    const size_t tpe_bytes = ggml_metal_op_mul_mat_id_extra_tpe(op);
    const size_t ids_bytes = ggml_metal_op_mul_mat_id_extra_ids(op);
    if (tpe_bytes != 4 * sizeof(int32_t) || ids_bytes != 33 * sizeof(int32_t) ||
        !host_range(ids, ggml_nbytes(ids)) || !host_range(op, ggml_nbytes(op) + tpe_bytes + ids_bytes)) { return false; }
    for (int i = 0; i < 32; ++i) {
        int32_t id;
        memcpy(&id, static_cast<const char *>(ids->data) + i * sizeof(id), sizeof(id));
        if (id != 0) { return false; }
    }
    // Identical to map0's one tile descriptor and compact row list. The final
    // padding slot is unused by the unchanged matrix kernel.
    const int32_t tpe[] = {1, 0, 0, 32};
    int32_t rows[33] = {};
    for (int32_t i = 0; i < 32; ++i) { rows[i] = i; }
    char * tail = static_cast<char *>(op->data) + ggml_nbytes(op);
    memcpy(tail, tpe, sizeof(tpe));
    memcpy(tail + tpe_bytes, rows, sizeof(rows));
    op->op_params[0] = GGML_METAL_PREPARED_SINGLE_EXPERT;
    return true;
}
