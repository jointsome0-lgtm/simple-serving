#!/usr/bin/env bash
# The picture card's preparation (docs/pictures.md, section 9), once per rental and never at a resume, as
# card/bootstrap.sh is the text card's. Run it over SSH with the SHA-256 of the client key and of the control key on
# stdin, one per line, or with nothing once the card holds them. The keys are the text lane's two:
#
#   uv run python -m simple_serving.cli keys | ssh <card> bash /workspace/simple-serving/card/pictures/bootstrap.sh
#
# It checks each copy of simple-story-chat's scripts against SOURCE, keeps the key hashes in keys.json, and installs
# the gateway's lock into a venv of its own with every hash checked. Then ComfyUI: an install that the old scripts left
# whole is adopted, and nothing of it is touched, since the copied image-bootstrap.sh would take its ready mark away,
# upgrade pip and resolve ComfyUI's requirements again. Whole is .venv-cu130 with its ready mark, ComfyUI at the pinned
# revision, image-verified.txt naming that revision and the three Qwen files with the manifest's hashes, those files at
# their pinned sizes, and the two Qwen graphs. Otherwise the copied image-bootstrap.sh prepares it with the manifest's
# settings, Viggle and the prompt enhancer off: the same downloads checked by size and SHA-256, the same hashed wheels,
# the same torch pins, the same check of sm_120 and the same image-verified.txt. Then it stamps what it prepared and runs
# onstart.sh for the first start. At every later start simple-story-chat's gpu/trial-onstart.sh runs onstart.sh.
# A second run changes nothing that is in place, and no run prints a hash. Exit codes: 0 prepared and started; 1
# image-bootstrap.sh failed, or left no whole install; 3 a parameter, a lock or a copy is missing or not SOURCE's; 4 no
# keys, or not two different SHA-256 digests; 5 the card holds other keys, which stay. The last step is the launcher's
# start, with its own codes (simple_serving/card.py). SIMPLE_SERVING_PICTURES_CARD_DIR and SIMPLE_SERVING_CARD_ROOT
# move the state and /root, for tests.
set -euo pipefail
umask 077
here=$(cd "$(dirname "$0")" && pwd)
state=${SIMPLE_SERVING_PICTURES_CARD_DIR:-/workspace/simple-serving-pictures-card}
source "$here/manifest.env"
copies=(image-bootstrap.sh image-serve.sh image-sweeper.py image-manifest.env image-workflow.json
        image-workflow-qwen.json image-workflow-qwen-edit.json)
stamped=("$here/manifest.env" "$here/gateway-requirements.txt" "$here/SOURCE" "${copies[@]/#/$here/}")

for pin in MODEL_ALIAS LOAD_DEADLINE_S PUBLIC_PORT CONTROL_PORT ENGINE_PORT GPU_DIR READER_NODES; do
  [[ -n ${!pin:-} ]] || exit 3
done
# The bot card's engine and no other: the other Qwen and torch lines are the experiments'.
[[ ${IMAGE_QWEN:-} == only && ${IMAGE_TORCH:-} == cu130 && ${IMAGE_TRITON:-} =~ ^[01]$ && ${IMAGE_GPU:-} =~ ^[0-9]$ ]] ||
  exit 3
[[ $GPU_DIR == /* && $GPU_DIR != / ]] || exit 3
for file in "${stamped[@]}"; do [[ -s $file ]] || exit 3; done
for name in "${copies[@]}"; do
  grep -Fqx "$(sha256sum < "$here/$name" | cut -d ' ' -f 1)  $name" "$here/SOURCE" || exit 3
done
mkdir -p "$state"

read -r client || true
read -r control || true
if [[ -n $client$control ]]; then
  [[ $client =~ ^[0-9a-f]{64}$ && $control =~ ^[0-9a-f]{64}$ && $client != "$control" ]] || exit 4
  keys="{\"client\": \"$client\", \"control\": \"$control\"}"
  [[ -e $state/keys.json ]] || printf '%s\n' "$keys" > "$state/keys.json"
  [[ $(< "$state/keys.json") == "$keys" ]] || exit 5
fi
[[ -s $state/keys.json ]] || exit 4
chmod 600 "$state/keys.json"  # umask leaves the mode of an older file as it was

[[ -x $state/gateway/bin/pip ]] || python3 -m venv "$state/gateway"
"$state/gateway/bin/pip" install --quiet --no-cache-dir --no-deps --require-hashes --only-binary :all: \
  -r "$here/gateway-requirements.txt"
"$state/gateway/bin/pip" check  # --no-deps trusts the lock to hold every requirement; this checks that it does

# Whether the install serves this manifest whole, in a shell of its own, since image-manifest.env sets many names.
whole() (
  source "$here/image-manifest.env"
  comfy=$GPU_DIR/ComfyUI verified=$GPU_DIR/image-verified.txt
  [[ -f $comfy/.venv-cu130/simple-chat-ready && -f $verified ]] || exit 1
  [[ $(git -C "$comfy" rev-parse HEAD 2> /dev/null) == "$COMFYUI_REVISION" ]] || exit 1
  grep -Fqx "revision $COMFYUI_REVISION" "$verified" || exit 1
  for entry in "diffusion_models $IMAGE_QWEN_MODEL_FILE $IMAGE_QWEN_MODEL_SHA256 $IMAGE_QWEN_MODEL_BYTES" \
               "text_encoders $IMAGE_QWEN_ENCODER_FILE $IMAGE_QWEN_ENCODER_SHA256 $IMAGE_QWEN_ENCODER_BYTES" \
               "vae $IMAGE_QWEN_VAE_FILE $IMAGE_QWEN_VAE_SHA256 $IMAGE_QWEN_VAE_BYTES"; do
    read -r folder file digest bytes <<< "$entry"
    grep -Fqx "$digest  $file" "$verified" || exit 1
    [[ $(stat -c %s -- "$comfy/models/$folder/$file" 2> /dev/null) == "$bytes" ]] || exit 1
  done
  [[ -f $GPU_DIR/$IMAGE_QWEN_WORKFLOW && -f $GPU_DIR/$IMAGE_QWEN_EDIT_WORKFLOW ]]
)
if ! whole; then
  SIMPLE_CHAT_GPU_DIR=$GPU_DIR SIMPLE_CHAT_IMAGE_QWEN=$IMAGE_QWEN SIMPLE_CHAT_IMAGE_TORCH=$IMAGE_TORCH \
    SIMPLE_CHAT_IMAGE_VIGGLE=false SIMPLE_CHAT_IMAGE_QWEN_PE=false bash "$here/image-bootstrap.sh" < /dev/null || exit 1
  whole || exit 1
fi

cat "${stamped[@]}" | sha256sum | cut -d ' ' -f 1 > "$state/prepared"
exec bash "$here/onstart.sh"
