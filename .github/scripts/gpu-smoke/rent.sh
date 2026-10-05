#!/usr/bin/env bash
# Rent ONE vast.ai GPU box for the smoke test: walk the cheapest offers,
# create sequentially, first to reach `running` + answer ssh wins.
# Losers/failures are destroyed. Simpler sibling of atomic-forge's
# rent_race.sh (we need one short-lived box, not a race).
#
# env in : VAST_API_KEY, GPU_QUERY, SSH_KEY_FILE, [DISK_GB=40] [MAX_OFFERS=5]
# out    : iid/host/port appended to $GITHUB_OUTPUT
set -euo pipefail

DISK_GB="${DISK_GB:-40}"
MAX_OFFERS="${MAX_OFFERS:-5}"
IMAGE="nvidia/cuda:12.8.0-runtime-ubuntu22.04"

# a trailing newline in the secret breaks the API auth header
VAST_API_KEY="$(printf '%s' "$VAST_API_KEY" | tr -d '[:space:]')"
vastai set api-key "$VAST_API_KEY" >/dev/null

# which account the key belongs to, without leaking it into public logs: email hash prefix and a credit range
vastai show user --raw > user.json 2>&1 || true
if ! jq -e '.email' user.json >/dev/null 2>&1; then
  echo "::error::vast rejected VAST_API_KEY: $(head -c 300 user.json)"; exit 1
fi
EMAIL_HASH=$(jq -r '.email' user.json | tr -d '\n' | sha256sum | cut -c1-8)
CREDIT_RANGE=$(jq -r '(.credit // .balance // null) as $c | if $c == null then "unknown"
  elif $c <= 0 then "<= 0" elif $c < 1 then "0..1" elif $c < 5 then "1..5" elif $c < 20 then "5..20" else ">= 20" end' user.json)
echo "vast account: email sha256 prefix=$EMAIL_HASH, credit range=$CREDIT_RANGE"

vastai search offers \
  "$GPU_QUERY disk_space>=$DISK_GB inet_down>=500 reliability>0.98 rentable=true" \
  -o dph --raw > offers.json 2>&1 || true
jq -e 'type == "array"' offers.json >/dev/null 2>&1 || { echo "::error::vast search failed: $(head -c 300 offers.json)"; exit 1; }
N=$(jq 'length' offers.json)
[ "$N" -gt 0 ] || { echo "::error::no vast offers match: $GPU_QUERY"; exit 1; }
echo "offers found: $N, trying up to $MAX_OFFERS cheapest"

IID=""
RENTED=""
cleanup() { [ -n "$IID" ] && vastai destroy instance "$IID" >/dev/null 2>&1 || true; }
# the workflow only learns the instance id on success, so any failure after create must destroy it here
trap '[ -n "$RENTED" ] || cleanup' EXIT

for i in $(seq 0 $((MAX_OFFERS - 1))); do
  [ "$i" -lt "$N" ] || break
  OFFER=$(jq -r ".[$i].id" offers.json)
  DPH=$(jq -r ".[$i].dph_total" offers.json)
  echo "--- offer $OFFER (\$${DPH}/h)"
  if ! vastai create instance "$OFFER" --image "$IMAGE" \
       --disk "$DISK_GB" --ssh --direct --raw > create.json 2>&1; then
    echo "create failed, next offer"; continue
  fi
  IID=$(jq -r '.new_contract // empty' create.json)
  [ -n "$IID" ] || { echo "no contract id, next offer: $(jq -c '{error, msg}' create.json 2>/dev/null || head -c 200 create.json)"; continue; }

  # created -> loading (image pull) -> running; give it 10 minutes
  for tick in $(seq 1 40); do
    ST=$(vastai show instance "$IID" --raw 2>/dev/null | jq -r '.actual_status // "?"')
    [ "$ST" = "running" ] && break
    sleep 15
  done
  if [ "$ST" != "running" ]; then
    echo "offer $OFFER never reached running ($ST), destroying"
    cleanup; IID=""; continue
  fi

  URL=$(vastai ssh-url "$IID")
  HOST=$(echo "$URL" | sed -E 's|ssh://root@([^:]+):.*|\1|')
  PORT=$(echo "$URL" | sed -E 's|.*:([0-9]+)$|\1|')

  # ssh может подняться на минуту позже статуса running — пробуем с ретраями
  OK=""
  for try in $(seq 1 8); do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
           -i "$SSH_KEY_FILE" -p "$PORT" root@"$HOST" 'echo ssh-ok' 2>/dev/null | grep -q ssh-ok; then
      OK=1; break
    fi
    sleep 15
  done
  if [ -z "$OK" ]; then
    echo "offer $OFFER: ssh unreachable, destroying"
    cleanup; IID=""; continue
  fi

  echo "rented: iid=$IID $HOST:$PORT"
  { echo "iid=$IID"; echo "host=$HOST"; echo "port=$PORT"; } >> "$GITHUB_OUTPUT"
  RENTED=1
  exit 0
done

echo "::error::all offers failed"
exit 1
