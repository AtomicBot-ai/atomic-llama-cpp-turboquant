// llama-laya-convert: laya HF checkpoint directory -> GGUF, without Python (tools/laya/laya-convert.h).
//
// Usage:
//   llama-laya-convert <ckpt-dir> -o out.gguf [--outtype f32|f16|q8_0] [--model-name NAME]
//                      [--verify-against ref.gguf] [-v]
//   llama-laya-convert --compare a.gguf b.gguf
//
// The output matches `convert_hf_to_gguf.py <ckpt-dir> --outfile out.gguf --outtype T [--model-name NAME]`
// byte for byte. --verify-against compares the result with another GGUF (KVs, tensor infos,
// tensor data, then raw bytes) and prints the first difference.
// Exit codes: 0 done (identical), 1 error (conversion failed, a file cannot be read), 2 the files differ.

#include "decision.h" // decision_utf8_args
#include "laya-convert.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s <ckpt-dir> -o out.gguf [--outtype f32|f16|q8_0] [--model-name NAME] [--verify-against ref.gguf] [-v]\n"
            "       %s --compare a.gguf b.gguf\n",
            argv0, argv0);
}

int main(int argc, char ** argv) {
    // Windows: argv is in the ANSI code page; paths go on as UTF-8 (laya_convert_fopen, gguf)
    const std::vector<std::string> args = decision_utf8_args(argc, argv);
    std::string dir, out, verify;
    std::vector<std::string> compare;
    bool verbose = false;
    laya_convert_params params;
    for (size_t i = 1; i < args.size(); i++) {
        const std::string & a = args[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= args.size()) {
                fprintf(stderr, "error: %s needs a value\n", what);
                exit(1);
            }
            return args[++i];
        };
        if (a == "-o" || a == "--outfile") {
            out = next("-o");
        } else if (a == "--outtype") {
            const std::string t = next("--outtype");
            if (!laya_convert_parse_outtype(t, params.outtype)) {
                fprintf(stderr, "error: unknown --outtype '%s' (f32, f16, q8_0)\n", t.c_str());
                return 1;
            }
        } else if (a == "--model-name") {
            params.has_model_name = true;
            params.model_name = next("--model-name");
        } else if (a == "--verify-against") {
            verify = next("--verify-against");
        } else if (a == "--compare") {
            compare.push_back(next("--compare"));
            compare.push_back(next("--compare"));
        } else if (a == "-v" || a == "--verbose") {
            verbose = true;
        } else if (a == "-h" || a == "--help") {
            print_usage(args[0].c_str());
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            print_usage(args[0].c_str());
            return 1;
        } else if (dir.empty()) {
            dir = a;
        } else {
            fprintf(stderr, "error: unexpected argument '%s'\n", a.c_str());
            return 1;
        }
    }

    if (!compare.empty()) {
        bool failed = false;
        const std::string diff = laya_convert_compare(compare[0], compare[1], &failed);
        if (failed) {
            fprintf(stderr, "error: %s\n", diff.c_str());
            return 1;
        }
        if (!diff.empty()) {
            printf("DIFFERENT: %s\n", diff.c_str());
            return 2;
        }
        printf("IDENTICAL\n");
        return 0;
    }
    if (dir.empty() || out.empty()) {
        print_usage(args[0].c_str());
        return 1;
    }

    params.log = [verbose](const std::string & msg) {
        if (verbose || msg.rfind("warning", 0) == 0) {
            fprintf(stderr, "%s\n", msg.c_str());
        }
    };
    const auto t0 = std::chrono::steady_clock::now();
    const std::string err = laya_convert(dir, out, params);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!err.empty()) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "wrote %s in %.2f s\n", out.c_str(), s);

    if (!verify.empty()) {
        bool failed = false;
        const std::string diff = laya_convert_compare(out, verify, &failed);
        if (failed) {
            fprintf(stderr, "error: --verify-against: %s\n", diff.c_str());
            return 1;
        }
        if (!diff.empty()) {
            printf("DIFFERENT from %s: %s\n", verify.c_str(), diff.c_str());
            return 2;
        }
        printf("IDENTICAL to %s\n", verify.c_str());
    }
    return 0;
}
