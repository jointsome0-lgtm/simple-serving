#!/usr/bin/env bash
# The picture lane's start at every start of the container (docs/pictures.md, section 9), as card/onstart.sh is the text
# lane's: simple-story-chat's gpu/trial-onstart.sh runs it before the text lane's, and bootstrap.sh at its end for the
# first start. It keeps the instance's id and key for SSH sessions and starts the picture launcher, and returns at once.
# Where the lane is not prepared, as on a text card, it starts nothing and exits 0. It downloads and installs nothing,
# and it arms no guard. SIMPLE_SERVING_PICTURES_CARD_DIR and SIMPLE_SERVING_CARD_ROOT move the state and /root, for
# tests.
set -uo pipefail
umask 077
code=$(cd "$(dirname "$0")/../.." && pwd)
state=${SIMPLE_SERVING_PICTURES_CARD_DIR:-/workspace/simple-serving-pictures-card}
root=${SIMPLE_SERVING_CARD_ROOT:-/root}

# The launcher reads the instance's id and key from these files in an SSH session, which lacks Vast's environment.
# They are the files of simple-story-chat's gpu/trial-onstart.sh and of the text lane's onstart.sh.
key=$root/.simple-chat-instance-api-key id=$root/.simple-chat-instance-id
if [[ ${CONTAINER_ID:-} =~ ^[1-9][0-9]*$ && -n ${CONTAINER_API_KEY:-} ]]; then
  # umask leaves the mode of an older file as it was, so each is made 0600 before the key goes in.
  if touch "$key" "$id" && chmod 600 "$key" "$id"; then
    printf '%s' "$CONTAINER_API_KEY" > "$key"
    printf '%s' "$CONTAINER_ID" > "$id"
  fi
fi

[[ -f $state/prepared && -x $state/gateway/bin/python ]] || exit 0
cd "$code" && exec "$state/gateway/bin/python" -m simple_serving.pictures.card
