#!/usr/bin/env bash
# Cut a stable TurboQuant release from master.
#
#   ./scripts/turboquant-release.sh patch|minor|major|X.Y.Z
#
# Bumps TURBOQUANT_VERSION, commits "release: turboquant-vX.Y.Z", tags
# turboquant-vX.Y.Z and pushes branch + tag. The tag push triggers
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
IFS=. read -r MAJOR MINOR PATCH <<< "$CURRENT"

case "$1" in
    patch) NEW="$MAJOR.$MINOR.$((PATCH + 1))" ;;
    minor) NEW="$MAJOR.$((MINOR + 1)).0" ;;
    major) NEW="$((MAJOR + 1)).0.0" ;;
    *)
        if [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            NEW="$1"
        else
            echo "error: '$1' is not patch|minor|major|X.Y.Z" >&2
            exit 1
        fi
        ;;
esac

TAG="turboquant-v$NEW"
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    echo "error: tag $TAG already exists" >&2
    exit 1
fi

echo "turboquant v$CURRENT -> v$NEW"
printf '%s\n' "$NEW" > TURBOQUANT_VERSION

git add TURBOQUANT_VERSION
git commit -m "release: $TAG"
git tag -a "$TAG" -m "TurboQuant v$NEW"

git push origin "$RELEASE_BRANCH"
git push origin "$TAG"

echo "Pushed $TAG — release workflow: https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant/actions/workflows/release-turboquant.yml"
