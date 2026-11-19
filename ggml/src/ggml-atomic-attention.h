#pragma once

// Private FLASH_ATTN_EXT hint shared by the graph builder and Metal dispatch.
// Slots 0..4 are ggml attention parameters. Zero selects the ordinary pipeline.
enum { GGML_ATOMIC_FA_VARIANT = 5, GGML_ATOMIC_FA_QWEN35_GQA8 = 0x47514138, GGML_ATOMIC_FA_LFM_GQA4 = 0x47514134 };
