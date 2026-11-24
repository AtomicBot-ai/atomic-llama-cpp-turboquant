#!/bin/bash
set -euo pipefail
if (( $# < 2 )); then
    echo "Usage: $0 <engine simulator build directory> <output directory> [engine object overrides...]" >&2
    exit 2
fi
build=$1
app=$2/InklingChat.app
shift 2
src=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$src/../../.." && pwd)
sdk=$(xcrun --sdk iphonesimulator --show-sdk-path)
mkdir -p "$app"
cp "$src/Info.plist" "$app/Info.plist"
xcrun --sdk iphonesimulator clang++ -target arm64-apple-ios18.0-simulator -isysroot "$sdk" \
    -std=c++17 -fobjc-arc -O2 -I"$repo/include" -I"$repo/ggml/include" -I"$repo/src" \
    "$src/main.mm" "$src/Inkling.mm" "$@" \
    "$build/src/libllama.a" "$build/ggml/src/libggml.a" "$build/ggml/src/ggml-metal/libggml-metal.a" \
    "$build/ggml/src/libggml-cpu.a" "$build/ggml/src/libggml-base.a" \
    -framework UIKit -framework Foundation -framework Metal -framework MetalKit -framework Accelerate -o "$app/InklingChat"
codesign --force --sign - "$app"
