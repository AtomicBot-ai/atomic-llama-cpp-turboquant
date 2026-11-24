# Inkling Chat for iPhone

These reference apps run Inkling 2 Mini in the iOS app process on Metal. Both
targets build the Atomic Engine and ggml sources from this checkout. Each target
has its own bundle ID and Documents directory, so the apps can coexist on one
iPhone.

- `InklingIQ1S` uses the sources in [`iq1/`](iq1) and loads
  `Inkling-2-Mini-AD-IQ1_S.gguf` (6.88 GB). Its bundle ID is
  `com.atomicengine.inklingiq1`.
- `InklingIQ2M` uses the sources in this directory and loads
  `Inkling-2-Mini-AD-IQ2_M.gguf` (8.67 GB). Its bundle ID is
  `com.atomicengine.inklingchat`.

## Build and run a reference app

Use Xcode, CMake 3.21 or later, and an arm64 iPhone running iOS 18 or later.
From the repository root, generate the device project and build both targets:

```sh
cmake -S examples/inkling-ios-chat -B build-inkling-ios -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0
cmake --build build-inkling-ios --config Release --target InklingIQ1S InklingIQ2M \
  -- CODE_SIGNING_ALLOWED=NO
```

The unsigned device apps are in `build-inkling-ios/Release-iphoneos/`.
To install one, open
`build-inkling-ios/InklingChat.xcodeproj`, select the `InklingIQ1S` or
`InklingIQ2M` scheme, choose your signing team and connected iPhone, and run.
If your team requires a different bundle ID, pass
`-DINKLING_IQ1S_BUNDLE_ID=your.iq1.bundle` or
`-DINKLING_IQ2M_BUNDLE_ID=your.iq2.bundle` at configure time and use that same
ID when copying the model.

Download the matching file from the
[pinned AtomicChat GGUF revision](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF/tree/309509848b3151419aa26d069eb03db57fd29a00).
This example transfers IQ1_S into the installed app's data container:

```sh
hf download AtomicChat/Inkling-2-Mini-GGUF Inkling-2-Mini-AD-IQ1_S.gguf \
  --revision 309509848b3151419aa26d069eb03db57fd29a00 --local-dir models
xcrun devicectl list devices
xcrun devicectl device copy to --device '<device-id>' \
  --source models/Inkling-2-Mini-AD-IQ1_S.gguf \
  --domain-type appDataContainer --domain-identifier com.atomicengine.inklingiq1 \
  --destination Documents/Inkling-2-Mini-AD-IQ1_S.gguf
```

For IQ2_M, use `Inkling-2-Mini-AD-IQ2_M.gguf` and
`com.atomicengine.inklingchat`. Relaunch once the entire file has copied. Wait
for the ready status, then send a short question and check for a streamed
response, time to first token, and generation speed.

## Embed in a separate iOS app

1. Pin Atomic Engine to a commit from this branch. For a CMake-based app, use
   [`CMakeLists.txt`](CMakeLists.txt) as the template for linking the `llama`
   target and enabling the static Metal backend with embedded shaders. For an
   existing Xcode-only app, build the `llama` target with the same CMake
   configuration above and collect the static archives under
   `build-inkling-ios/engine/`. Link them in this order:
   `libllama.a`, `libggml.a`, `libggml-metal.a`, `libggml-cpu.a`, and
   `libggml-base.a`. Link Foundation, Metal, MetalKit, and Accelerate. The
   reference targets also link UIKit for their UI.
2. Add one copy of [`Inkling.h`](iq1/Inkling.h) and
   [`Inkling.mm`](iq1/Inkling.mm) to the host target with Objective-C++17 and ARC.
   Add the engine's `include`, `ggml/include`, and `src` to the header search
   paths. A Swift target can expose `Inkling.h` through its bridging header.
   [`main.mm`](iq1/main.mm) shows the reference UI, model lookup, and queue usage.
   The host app supplies its own entry point, UI, signing, and model location.
3. Keep a complete GGUF on disk at a stable, readable path for the lifetime of
   its model instance. Pass the path to `loadFromPath:` once on a serial worker
   queue. A `nil` return means it is ready. On a load failure or model switch,
   release the old `Inkling` instance on that queue and create a new one. Both
   quants use the same wrapper. The selected GGUF path determines the quant.
4. Call `replyTo:since:update:` on the same queue after loading succeeds. Pass a
   start time from `NSProcessInfo.processInfo.systemUptime`. The callback runs
   on the worker queue and receives the full answer so far plus a metrics
   string. Dispatch UI updates to the main queue. The return value is a final
   status string or `nil` on normal completion. Release the wrapper on its
   worker queue after outstanding generation finishes.

Start with the reference profile. It uses Metal `MTL0`, a 4 GiB engine budget,
a 1,024-token context, a 256-token microbatch, Q4_0 KV cache, and greedy decoding.
The budget covers engine allocations tracked by Atomic Engine. Measure total
iOS process memory separately. The wrapper prepares a text-only, single-turn
prompt, clears the KV cache for each question, and emits up to 128 tokens per
answer. The examples run with one context and one active request.

Both models previously generated on a physical iPhone 15 with a 4 GiB engine
budget. Those experimental builds used a local engine worktree, and the IQ1_S
profiling build also enabled experimental Metal settings. The reference targets
use the engine defaults for those settings. Validate loading, streamed
generation, total process memory, and throughput on the target devices with
the host app's own build. The simulator scripts require a separate simulator
engine setup.
