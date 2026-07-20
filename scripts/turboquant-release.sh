#!/usr/bin/env bash
# Cut a stable TurboQuant release from master.
#
#   ./scripts/turboquant-release.sh patch|minor|major|X.Y.Z
#
# Version format: <upstream-base>-<fork-semver>, e.g. b10018-1.2.0
#   - b10018 is the upstream llama.cpp build the fork is based on
#     (`git rev-list --count $(git merge-base master upstream)`); it changes
#     only on upstream syncs and is edited by hand in TURBOQUANT_VERSION as
#     part of the sync PR.
#   - 1.2.0 is the fork's own semver: major for breaking changes, minor for
#     features (e.g. new turbo-ops backends), patch for fixes.
#
# This script bumps the fork-semver part, commits "release: <version>", tags
# `<version>` and pushes branch + tag. The tag push triggers
# .github/workflows/release-turboquant.yml which builds every backend and
# publishes a single GitHub release with all archives.
set -euo pipefail

cd "$(dirname "$0")/.."

RELEASE_BRANCH="${RELEASE_BRANCH:-master}"

if [ $# -ne 1 ]; then
    echo "usage: $0 patch|minor|major|X.Y.Z" >&2
    exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "$RELEASE_BRANCH" ]; then
    echo "error: releases are cut from '$RELEASE_BRANCH' (currently on '$BRANCH')" >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "error: working tree is not clean" >&2
    exit 1
fi

git fetch origin "$RELEASE_BRANCH"
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$RELEASE_BRANCH")" ]; then
    echo "error: local $RELEASE_BRANCH is not in sync with origin/$RELEASE_BRANCH" >&2
    exit 1
fi

CURRENT=$(tr -d ' \n' < TURBOQUANT_VERSION)
if [[ ! "$CURRENT" =~ ^(b[0-9]+)-([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "error: TURBOQUANT_VERSION '$CURRENT' does not match b<N>-X.Y.Z" >&2
    exit 1
fi
BASE="${BASH_REMATCH[1]}"
MAJOR="${BASH_REMATCH[2]}"
MINOR="${BASH_REMATCH[3]}"
PATCH="${BASH_REMATCH[4]}"

case "$1" in
    patch) SEMVER="$MAJOR.$MINOR.$((PATCH + 1))" ;;
    minor) SEMVER="$MAJOR.$((MINOR + 1)).0" ;;
    major) SEMVER="$((MAJOR + 1)).0.0" ;;
    *)
        if [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            SEMVER="$1"
        else
            echo "error: '$1' is not patch|minor|major|X.Y.Z" >&2
            exit 1
        fi
        ;;
esac

NEW="$BASE-$SEMVER"
TAG="$NEW"
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    echo "error: tag $TAG already exists" >&2
    exit 1
fi

echo "turboquant $CURRENT -> $NEW"
printf '%s\n' "$NEW" > TURBOQUANT_VERSION

git add TURBOQUANT_VERSION
git commit -m "release: $NEW"
git tag -a "$TAG" -m "TurboQuant $NEW"

git push origin "$RELEASE_BRANCH"
git push origin "$TAG"

echo "Pushed $TAG — release workflow: https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant/actions/workflows/release-turboquant.yml"
