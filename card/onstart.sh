#!/usr/bin/env bash
# Runs at every start of the container, from the last line of the rental's own onstart, and at the end of
# bootstrap.sh for the first start (README, "The card"). It keeps the instance's id and key for SSH sessions, starts
# the service's launcher and returns at once. It downloads and installs nothing, and it arms no guard.
# SIMPLE_SERVING_CARD_DIR and SIMPLE_SERVING_CARD_ROOT move the state and /root, for tests.
set -uo pipefail
umask 077
code=$(cd "$(dirname "$0")/.." && pwd)
state=${SIMPLE_SERVING_CARD_DIR:-/workspace/simple-serving-card}
root=${SIMPLE_SERVING_CARD_ROOT:-/root}

# The launcher reads the instance's id and key from these files in an SSH session, which lacks Vast's environment.
# They are the files of simple-story-chat's gpu/trial-onstart.sh.
key=$root/.simple-chat-instance-api-key id=$root/.simple-chat-instance-id
if [[ ${CONTAINER_ID:-} =~ ^[1-9][0-9]*$ && -n ${CONTAINER_API_KEY:-} ]]; then
  # umask leaves the mode of an older file as it was, so each is made 0600 before the key goes in.
  if touch "$key" "$id" && chmod 600 "$key" "$id"; then
    printf '%s' "$CONTAINER_API_KEY" > "$key"
    printf '%s' "$CONTAINER_ID" > "$id"
  fi
fi

cd "$code" && exec "$state/gateway/bin/python" -m simple_serving.card
