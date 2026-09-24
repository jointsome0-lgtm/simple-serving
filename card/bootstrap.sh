#!/usr/bin/env bash
# The card's preparation, once per rental and never at a resume (contract section 12). Run it over SSH with the
# SHA-256 of the client key and of the control key on stdin, one per line, or with nothing once the card holds them:
#
#   ssh <card> bash /workspace/simple-serving/card/bootstrap.sh < key-hashes
#
# It installs each lock into a venv of its own with every hash checked, fetches the weights and the tokenizer files
# at their pinned revisions and checks their hashes, keeps the key hashes in keys.json, adds card/onstart.sh to the
# container's onstart, and then runs it for the first start. A second run changes nothing that is in place, and no run
# prints a hash. Exit codes: 0 prepared and started; 1 a download or a check failed; 3 a pin or a lock is missing;
# 4 no keys, or not two different SHA-256 digests; 5 the card holds other keys, which stay.
# SIMPLE_SERVING_CARD_DIR and SIMPLE_SERVING_CARD_ROOT move the state and /root, for tests.
set -euo pipefail
umask 077
code=$(cd "$(dirname "$0")/.." && pwd)
state=${SIMPLE_SERVING_CARD_DIR:-/workspace/simple-serving-card}
root=${SIMPLE_SERVING_CARD_ROOT:-/root}
source "$code/card/manifest.env"
stamped=("$code/card/manifest.env" "$code/card/gateway-requirements.txt" "$code/card/vllm-requirements.txt")

for pin in VLLM_VERSION MODEL_SHA256 TOKENIZER_REPO TOKENIZER_REVISION TOKENIZER_FILES; do
  [[ -n ${!pin} ]] || exit 3
done
for file in "${stamped[@]}"; do [[ -s $file ]] || exit 3; done
grep -Fq "vllm==$VLLM_VERSION " "$code/card/vllm-requirements.txt" || exit 3
mkdir -p "$state/models" "$state/tokenizer"

read -r client || true
read -r control || true
if [[ -n $client$control ]]; then
  [[ $client =~ ^[0-9a-f]{64}$ && $control =~ ^[0-9a-f]{64}$ && $client != "$control" ]] || exit 4
  keys="{\"client\": \"$client\", \"control\": \"$control\"}"
  [[ -e $state/keys.json ]] || printf '%s\n' "$keys" > "$state/keys.json"
  [[ $(< "$state/keys.json") == "$keys" ]] || exit 5
fi
[[ -s $state/keys.json ]] || exit 4

for venv in gateway vllm; do
  [[ -x $state/$venv/bin/pip ]] || python3 -m venv "$state/$venv"
  "$state/$venv/bin/pip" install --quiet --no-cache-dir --no-deps --require-hashes \
    -r "$code/card/$venv-requirements.txt"
done

# fetch <repo> <revision> <file> <sha256> <path>: a file of the Hugging Face hub, kept only with its hash.
fetch() {
  [[ $(sha256sum 2>/dev/null < "$5") == "$4  -" ]] && return
  curl --fail --location --silent --show-error --continue-at - --output "$5" "https://huggingface.co/$1/resolve/$2/$3"
  [[ $(sha256sum < "$5") == "$4  -" ]] || { rm -f "$5"; exit 1; }
}
fetch "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_FILE" "$MODEL_SHA256" "$state/models/$MODEL_FILE"
IFS=, read -ra files <<< "$TOKENIZER_FILES"
for entry in "${files[@]}"; do
  fetch "$TOKENIZER_REPO" "$TOKENIZER_REVISION" "${entry%%:*}" "${entry#*:}" "$state/tokenizer/${entry%%:*}"
done

printf -v hook 'bash %q' "$code/card/onstart.sh"
touch "$root/onstart.sh"
grep -Fqx -- "$hook" "$root/onstart.sh" || printf '%s\n' "$hook" >> "$root/onstart.sh"
cat "${stamped[@]}" | sha256sum | cut -d ' ' -f 1 > "$state/prepared"
exec bash "$code/card/onstart.sh"
