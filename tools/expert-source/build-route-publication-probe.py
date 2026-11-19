from pathlib import Path
import json,shlex,subprocess,shutil,sys
import argparse
parser=argparse.ArgumentParser(description="Build a diagnostic server without modifying repository sources.")
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--build',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.source.resolve();build=args.build.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
source=root/'tools/server/server-context.cpp';text=source.read_text()
helper=r'''
#include "llama-expert-profile.h"
#include "llama-atomic.h"
static void route_hot_probe(llama_context * ctx, int n) {
    static uint64_t reads = 0, calls = 0;
    llama_expert_streaming_data stats{};
    GGML_ASSERT(llama_get_expert_streaming_stats(ctx, &stats));
    if (n == 1) { reads = stats.logical_read_bytes; calls = stats.combined_moe; llama_expert_profile_set(ctx, true); }
    if (n == 2) {
        llama_expert_profile_data p{}; GGML_ASSERT(llama_expert_profile_get(ctx, &p));
        FILE * f = std::fopen(std::getenv("ATOMIC_ROUTE_HOT"), "a"); GGML_ASSERT(f);
        std::fprintf(f, "{\"logical_read_bytes\":%llu,\"moe_calls\":%llu,\"prefetch_bytes\":%llu,\"prefetch_used_bytes\":%llu}\n",
            (unsigned long long)(stats.logical_read_bytes-reads), (unsigned long long)(stats.combined_moe-calls),
            (unsigned long long)p.next_prefetch_bytes, (unsigned long long)p.next_prefetch_used_bytes);
        std::fclose(f);
    }
}

static void window_logit_probe(llama_context * ctx, int index, int n) {
    const char * path = std::getenv("ATOMIC_WINDOW_LOGITS");
    GGML_ASSERT(path);
    const auto vocab = llama_model_get_vocab(llama_get_model(ctx));
    const size_t count = llama_vocab_n_tokens(vocab);
    FILE * file = std::fopen(path, n == 1 ? "wb" : "ab");
    GGML_ASSERT(file);
    GGML_ASSERT(std::fwrite(llama_get_logits_ith(ctx, index), sizeof(float), count, file) == count);
    GGML_ASSERT(std::fclose(file) == 0);
    if (std::getenv("ATOMIC_ROUTE_PROFILE") && n == 1) llama_expert_profile_set(ctx, true);
    if (std::getenv("ATOMIC_ROUTE_PROFILE") && n == 64) {
        llama_expert_profile_data p{};
        GGML_ASSERT(llama_expert_profile_get(ctx, &p));
        const std::string output = std::string(path) + ".profile.json";
        FILE * f = std::fopen(output.c_str(), "w"); GGML_ASSERT(f);
        std::fprintf(f, "{");
#define FIELD(name) std::fprintf(f, "\"" #name "\":%llu,", (unsigned long long) p.name)
        FIELD(route_to_read_ns); FIELD(route_read_windows); FIELD(route_hidden_io_ns); FIELD(route_ready_bytes); FIELD(route_publications); FIELD(route_publication_lead_ns); FIELD(route_preparation_ns);
        FIELD(demand_queue_wait_ns); FIELD(demand_queue_batches); FIELD(submit_ns); FIELD(wait_ns);
        FIELD(gpu_command_ns); FIELD(gpu_command_buffers); FIELD(next_prefetch_bytes); FIELD(next_prefetch_used_bytes);
        FIELD(next_prefetch_hidden_ns); FIELD(next_prefetch_wait_ns); FIELD(next_prefetch_busy_skips);
        FIELD(direct_read_bytes); FIELD(moe_batch_groups); FIELD(moe_address_setup_ns);
#undef FIELD
        std::fprintf(f, "\"complete\":true}\n"); std::fclose(f);
    }
    if (n == 1 || n == 64) {
        llama_expert_streaming_data stats {};
        GGML_ASSERT(llama_get_expert_streaming_stats(ctx, &stats));
        std::fprintf(stderr, "WINDOW_MOE_PROBE %d %llu %llu %zu\n", n,
            (unsigned long long) stats.combined_moe, (unsigned long long) stats.logical_read_bytes, stats.high_water);
    }
}
'''
marker='constexpr int HTTP_POLLING_SECONDS';assert text.count(marker)==1
text=text.replace(marker,helper+'\n'+marker,1)
start=text.index('const int tok_idx = slot.i_batch - off;')
sample='            slot.stats.n_gen += 1;';pos=text.index(sample,start)
text=text[:pos]+text[pos:].replace(sample,sample+'''
            if (std::getenv("ATOMIC_ROUTE_HOT") && slot.task->params.n_predict == 2) route_hot_probe(ctx_tgt, slot.stats.n_gen);
            if (params_base.engine_budget && (slot.task->params.n_predict == 64 || slot.task->params.n_predict == 16)) {
                window_logit_probe(ctx_tgt, tok_idx, slot.stats.n_gen);
            }
''',1)
# Capture accepted verify rows as well as ordinary one-token sampling.
needle = "                slot.stats.n_gen += 1;"
assert text.count(needle) == 1
text = text.replace(needle, needle + "\n                if (params_base.engine_budget && (slot.task->params.n_predict == 64 || slot.task->params.n_predict == 16)) window_logit_probe(ctx_tgt, spec_i_batch[i], slot.stats.n_gen);", 1)
needle = "                if (n_rollback > 0) {"
assert text.count(needle) == 1
text = text.replace(needle, needle + "\n                    if (std::getenv(\"ATOMIC_ROUTE_PROFILE\")) std::fprintf(stderr, \"ROUTE_ROLLBACK %u %d\\n\", n_rollback, int(use_ckpt_tgt));", 1)
probe=out/'server-context.cpp';probe.write_text(text)
e=next(e for e in json.loads((build/'compile_commands.json').read_text()) if e['file']==str(source))
a=shlex.split(e['command']);a[a.index('-c')+1]=str(probe);a[a.index('-o')+1]=str(out/'server-context.cpp.o')
subprocess.run(a+['-I'+str(source.parent),'-I'+str(root/'src')],cwd=e['directory'],check=True)
archive=out/'libserver-probe.a';shutil.copy2(build/'tools/server/libserver-context.a',archive)
subprocess.run(['ar','r',str(archive),str(out/'server-context.cpp.o')],check=True)
subprocess.run(['ranlib',str(archive)],check=True)
d=build/'tools/server';a=shlex.split((d/'CMakeFiles/llama-server.dir/link.txt').read_text())
r={'libserver-context.a':str(archive),'../../bin/llama-server':str(out/'server-logits-probe')}
subprocess.run([r.get(x,x) for x in a],cwd=d,check=True)
