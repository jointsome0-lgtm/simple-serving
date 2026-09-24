#!/usr/bin/env bash
# Runs at every start of the container, from the line bootstrap.sh adds to /root/onstart.sh. It arms the trial guard
# apart from the service, starts the service's launcher and returns at once. It downloads and installs nothing.
# SIMPLE_SERVING_CARD_DIR and SIMPLE_SERVING_CARD_ROOT move the state and /root, for tests.
set -uo pipefail
umask 077
code=$(cd "$(dirname "$0")/.." && pwd)
state=${SIMPLE_SERVING_CARD_DIR:-/workspace/simple-serving-card}
root=${SIMPLE_SERVING_CARD_ROOT:-/root}

# The trial guard of simple-story-chat (gpu/trial-onstart.sh), with the same files, so that the rental's own onstart
# and this one arm one guard between them. It deletes the instance three hours after the first start. The deadline is
# written once and never moved, and the lock keeps one guard running. The launcher reads the instance's id and key
# from the same files in an SSH session, which lacks Vast's environment.
if [[ ${CONTAINER_ID:-} =~ ^[1-9][0-9]*$ && -n ${CONTAINER_API_KEY:-} ]]; then
  printf '%s' "$CONTAINER_API_KEY" > "$root/.simple-chat-instance-api-key"
  printf '%s' "$CONTAINER_ID" > "$root/.simple-chat-instance-id"
fi
deadline=$root/.simple-chat-trial-deadline guard=$root/.simple-chat-trial-guard
[[ -f $deadline ]] || printf '%s' "$(( $(date +%s) + 10800 ))" > "$deadline"
[[ -f $guard.sh ]] || cat > "$guard.sh" <<'GUARD'
#!/usr/bin/env bash
set -euo pipefail
deadline="$(cat /root/.simple-chat-trial-deadline)"
instance="$(cat /root/.simple-chat-instance-id)"
[[ "$deadline" =~ ^[0-9]+$ && "$instance" =~ ^[1-9][0-9]*$ ]] || exit 1
while (( $(date +%s) < deadline )); do sleep 10; done
while true; do
  if printf 'header = "Authorization: Bearer %s"\n' "$(cat /root/.simple-chat-instance-api-key)" \
    | curl --config - --silent --fail --output /dev/null --connect-timeout 10 --max-time 20 \
      --request DELETE "https://console.vast.ai/api/v0/instances/$instance/"; then
    exit 0
  fi
  sleep 30
done
GUARD
nohup flock -n "$guard.lock" bash "$guard.sh" </dev/null >/dev/null 2>&1 &

cd "$code" && exec "$state/gateway/bin/python" -m simple_serving.card
