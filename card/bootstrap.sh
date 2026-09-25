#!/usr/bin/env bash
# The card's preparation, once per rental and never at a resume (contract section 12). Run it over SSH with the
# SHA-256 of the client key and of the control key on stdin, one per line, or with nothing once the card holds them:
#
#   uv run python -m simple_serving.cli keys | ssh <card> bash /workspace/simple-serving/card/bootstrap.sh
#
# It installs each lock into a venv of its own with every hash checked, fetches the weights and the tokenizer files
# at their pinned revisions and checks their hashes, keeps the key hashes in keys.json, and then runs card/onstart.sh
# for the first start. At every later start the rental's own onstart runs it (README, "The card"): the preparation
# changes no onstart, which the platform may restore at a start. A second run changes nothing that is in place, and no
# run prints a hash. Exit codes: 0 prepared and started; 1 a download or a check failed; 3 a pin or a lock is missing;
# 4 no keys, or not two different SHA-256 digests; 5 the card holds other keys, which stay. The last step is the
# launcher's start, with its own codes (simple_serving/card.py): 3 also when the card lacks the instance's credential.
# SIMPLE_SERVING_CARD_DIR and SIMPLE_SERVING_CARD_ROOT move the state and /root, for tests.
set -euo pipefail
umask 077
code=$(cd "$(dirname "$0")/.." && pwd)
state=${SIMPLE_SERVING_CARD_DIR:-/workspace/simple-serving-card}
source "$code/card/manifest.env"
stamped=("$code/card/manifest.env" "$code/card/gateway-requirements.txt" "$code/card/vllm-requirements.txt")

for pin in VLLM_VERSION MODEL_REPO MODEL_REVISION MODEL_FILES TOKENIZER_REPO TOKENIZER_REVISION TOKENIZER_FILES; do
  [[ -n ${!pin} ]] || exit 3
done
for file in "${stamped[@]}"; do [[ -s $file ]] || exit 3; done
grep -Fq "vllm==$VLLM_VERSION " "$code/card/vllm-requirements.txt" || exit 3
model=$state/models/$MODEL_REVISION
mkdir -p "$model" "$state/tokenizer"

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

for venv in gateway vllm; do
  [[ -x $state/$venv/bin/pip ]] || python3 -m venv "$state/$venv"
  "$state/$venv/bin/pip" install --quiet --no-cache-dir --no-deps --require-hashes --only-binary :all: \
    -r "$code/card/$venv-requirements.txt"
  "$state/$venv/bin/pip" check  # --no-deps trusts the lock to hold every requirement; this checks that it does
done

# fetch <repo> <revision> <files> <directory>: each name:sha256 of the comma-separated files, from the Hugging Face
# hub, kept only with its hash. A file arrives under a hidden name, with aria2's control file beside it, and takes its
# own name once its hash is checked. It comes over 16 connections: one gave the 5090 of 2026-09-25 about 10 MB/s, and
# 16 gave it 110 MiB/s. aria2 is the distribution's own, unpinned, as the hash checks what it fetched; without it the
# file comes over one connection.
fetch() {
  local entries entry path part url
  IFS=, read -ra entries <<< "$3"
  for entry in "${entries[@]}"; do
    path=$4/${entry%%:*}
    [[ $(sha256sum 2>/dev/null < "$path") == "${entry#*:}  -" ]] && continue
    part=$4/.${entry%%:*}.part
    url=https://huggingface.co/$1/resolve/$2/${entry%%:*}
    command -v aria2c > /dev/null ||
      { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq aria2; } > /dev/null 2>&1 || true
    if command -v aria2c > /dev/null; then
      aria2c --continue=true --max-connection-per-server=16 --split=16 --min-split-size=64M --file-allocation=none \
        --max-tries=5 --retry-wait=5 --console-log-level=error --summary-interval=0 --show-console-readout=false \
        --download-result=hide --dir="$4" --out="${part##*/}" "$url" || exit 1
    else
      curl --fail --location --silent --show-error --retry 5 --retry-all-errors --continue-at - --output "$part" \
        "$url" || exit 1
    fi
    [[ $(sha256sum < "$part") == "${entry#*:}  -" ]] || { rm -f "$part" "$part.aria2"; exit 1; }
    mv "$part" "$path"
  done
}
# vLLM loads every *.safetensors in the model's directory, so the files there that are not pinned go. Hidden ones
# stay: this glob skips them, and so does vLLM's.
for path in "$model"/*; do [[ ,$MODEL_FILES == *,"${path##*/}":* ]] || rm -f "$path"; done
fetch "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_FILES" "$model"
fetch "$TOKENIZER_REPO" "$TOKENIZER_REVISION" "$TOKENIZER_FILES" "$state/tokenizer"

cat "${stamped[@]}" | sha256sum | cut -d ' ' -f 1 > "$state/prepared"
exec bash "$code/card/onstart.sh"
