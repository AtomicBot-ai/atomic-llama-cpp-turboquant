#import "Inkling.h"
#include "llama-atomic.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

@implementation Inkling {
    ggml_backend_dev_t _devices[2];
    llama_model *_model;
    llama_context *_context;
}

- (NSString *)loadFromPath:(NSString *)path {
    static dispatch_once_t once;
    dispatch_once(&once, ^{ llama_backend_init(); });
    _devices[0] = ggml_backend_dev_by_name("MTL0");
    if (!_devices[0]) return @"Metal недоступен";
    if (![NSFileManager.defaultManager isReadableFileAtPath:path]) return @"Файл модели отсутствует в Documents";
    auto params = llama_model_default_params();
    params.devices = _devices;
    params.memory_budget_size = size_t{4096} << 20;
    params.n_gpu_layers = -1;
    _model = llama_model_load_from_file(path.fileSystemRepresentation, params);
    if (!_model) return @"Не удалось загрузить модель";

    auto ctx = llama_context_default_params();
    ctx.n_ctx = ctx.n_batch = 1024;
    ctx.n_ubatch = 256;
    ctx.n_seq_max = 1;
    ctx.n_threads = ctx.n_threads_batch = 8;
    ctx.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx.type_k = ctx.type_v = GGML_TYPE_Q4_0;
    _context = llama_init_from_model(_model, ctx);
    return _context ? nil : @"Не удалось создать контекст модели";
}

- (void)dealloc {
    if (_context) llama_free(_context);
    if (_model) llama_model_free(_model);
}

static bool add(llama_batch_ext *batch, llama_token tok, llama_pos pos, bool logits) {
    int i = llama_batch_ext_add_token(batch, 0, tok);
    if (i < 0) return false;
    llama_batch_ext_set_pos(batch, i, &pos);
    llama_batch_ext_set_output_logits(batch, i, logits);
    return true;
}

- (NSString *)replyTo:(NSString *)message
                since:(NSTimeInterval)start
               update:(void (^)(NSString *, NSString *))update {
    const llama_vocab *vocab = llama_model_get_vocab(_model);
    const int capacity = llama_n_ctx(_context);
    std::vector<llama_token> tokens(capacity);
    int count = 0;
    auto append = [&](NSString *s, bool special) {
        NSData *data = [s dataUsingEncoding:NSUTF8StringEncoding];
        if (!data || data.length > INT32_MAX) return false;
        int n = llama_tokenize(vocab, (const char *)data.bytes, (int)data.length, tokens.data() + count,
                               capacity - count, count == 0, special);
        if (n < 0) return false;
        count += n;
        return true;
    };
    // Inkling's text-only, single-turn template. User input never parses role markers.
    if (!append(@"<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>", true) ||
        !append(message, false)) return @"Сообщение слишком длинное для контекста в 1024 токена";
    int endIndex = count;
    if (!append(@"<|end_message|><|message_model|><|content_text|>", true) || count >= capacity) {
        return @"В контексте недостаточно места для ответа";
    }
    llama_memory_clear(llama_get_memory(_context), false);
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(_context), llama_batch_ext_free);
    if (!batch) return @"Не удалось подготовить запрос";
    for (int i = 0; i < count; ++i) {
        if (!add(batch.get(), tokens[i], i, i + 1 == count)) return @"Не удалось подготовить запрос";
    }
    if (llama_process(_context, LLAMA_PROCESS_TYPE_DECODE, batch.get())) return @"Ошибка обработки запроса";

    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
    std::string bytes;
    NSString *answer = @"";
    NSTimeInterval first = 0;
    int limit = std::min(128, capacity - count);
    for (int i = 0; i < limit; ++i) {
        llama_token tok = llama_sampler_sample(sampler.get(), _context, -1);
        if (tok == tokens[endIndex] || llama_vocab_is_eog(vocab, tok) ||
            (llama_vocab_get_attr(vocab, tok) & LLAMA_TOKEN_ATTR_CONTROL)) {
            return i == 0 ? @"Модель завершила ответ без текста" : nil;
        }
        NSTimeInterval now = NSProcessInfo.processInfo.systemUptime;
        if (i == 0) first = now;
        char local[256];
        int size = llama_token_to_piece(vocab, tok, local, sizeof(local), 0, false);
        if (size < 0) {
            std::vector<char> piece(-size);
            size = llama_token_to_piece(vocab, tok, piece.data(), piece.size(), 0, false);
            if (size < 0) return @"Ошибка чтения токена";
            bytes.append(piece.data(), size);
        } else {
            bytes.append(local, size);
        }
        // A UTF-8 character can span multiple tokens.
        NSString *decoded = [[NSString alloc] initWithBytes:bytes.data() length:bytes.size() encoding:NSUTF8StringEncoding];
        if (decoded) answer = decoded;
        NSString *rate = i && now > first ? [NSString stringWithFormat:@"%.2f ток/с", i / (now - first)] : @"— ток/с";
        NSString *metrics = [NSString stringWithFormat:@"%@   ·   TTFT %.2f с   ·   %d ток.", rate, first - start, i + 1];
        update(answer, metrics);
        if (i + 1 == limit) return @"Достигнут лимит ответа";
        llama_batch_ext_clear(batch.get());
        if (!add(batch.get(), tok, count + i, true) ||
            llama_process(_context, LLAMA_PROCESS_TYPE_DECODE, batch.get())) return @"Ошибка генерации";
    }
    return nil;
}
@end
