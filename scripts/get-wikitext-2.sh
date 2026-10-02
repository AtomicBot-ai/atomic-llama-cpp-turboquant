#!/bin/sh
# vim: set ts=4 sw=4 et:

ZIP="wikitext-2-raw-v1.zip"
FILE="wikitext-2-raw/wiki.test.raw"
URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/$ZIP"

die() {
    printf "%s\n" "$@" >&2
    exit 1
}

have_cmd() {
    for cmd; do
        command -v "$cmd" >/dev/null || return
    done
}

dl() {
    if have_cmd wget; then
        wget -q "$1" -O "$2" || return 1
    elif have_cmd curl; then
        # -f: fail on HTTP errors (avoid HTML error pages named .zip)
        curl -fsSL "$1" -o "$2" || return 1
    else
        die "Please install wget or curl"
    fi
}

have_cmd unzip || die "Please install unzip"

if [ ! -f "$FILE" ]; then
    if [ -f "$ZIP" ] && ! unzip -t "$ZIP" >/dev/null 2>&1; then
        rm -f -- "$ZIP"
    fi
    if [ ! -f "$ZIP" ]; then
        dl "$URL" "$ZIP" || exit 1
    fi
    unzip -o "$ZIP" || exit 1
    rm -f -- "$ZIP"
fi

cat <<EOF
Usage:

  llama-perplexity -m model.gguf -f $FILE [other params]

EOF
