#!/usr/bin/env bash
# Prepare the picture card inside the rented CUDA container, never on the bot host: pinned ComfyUI, pinned torch,
# weights verified by size and SHA256. Meant to run beside bootstrap.sh, which builds llama.cpp on the language
# card; both lanes share one link, so the speed floor below reads that link's own byte counter once, here, while
# the compiler and the other lane's download are busy on it.
#
# Tokens: the CivitAI file and the gated Krea repository need one each. They arrive in SIMPLE_CHAT_CIVITAI_TOKEN /
# SIMPLE_CHAT_HF_TOKEN, or as `civitai=...` / `hf=...` lines on stdin with --tokens-stdin. They are never written to
# disk, never echoed and never passed as an argument: curl reads the Authorization header from its stdin config, so
# `ps` shows nothing. A token is only ever sent to the host in the pinned URL; curl drops it across a redirect to a
# CDN, which is what we want, because the signed CDN link needs no credential.
set -euo pipefail
umask 077
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$task_dir/image-manifest.env"
gpu_dir="${SIMPLE_CHAT_GPU_DIR:-/workspace/simple-chat-gpu}"
[[ "$gpu_dir" = /* && "$gpu_dir" != / ]] || { echo 'Use an absolute SIMPLE_CHAT_GPU_DIR.' >&2; exit 1; }
comfy_dir="$gpu_dir/ComfyUI"
models_dir="$comfy_dir/models"
# `comfy` takes the encoder and the VAE from the public ComfyUI repackaging, `official` from the gated Krea
# repository (see image-manifest.env: its hashes have to be filled in first).
image_source="${SIMPLE_CHAT_IMAGE_SOURCE:-comfy}"
[[ "$image_source" = comfy || "$image_source" = official ]] || { echo 'Use SIMPLE_CHAT_IMAGE_SOURCE=comfy or official.' >&2; exit 1; }
# The comparison checkpoint is 13 GB and only earns its place if the fine-tune disappoints; it is still on by
# default, because deciding that at minute 50 of a rental is too late to download it.
turbo="${SIMPLE_CHAT_IMAGE_TURBO:-true}"
[[ "$turbo" = true || "$turbo" = false ]] || { echo 'Use SIMPLE_CHAT_IMAGE_TURBO=true or false.' >&2; exit 1; }
# Qwen-Image 2.1 is the opt-in third checkpoint: 17.28 GB that local/rent-plan.ts does not price, so a session that
# wants it says so before it rents. Off by default, unlike Turbo, because it is a comparison somebody chose to make.
# `only` fetches Qwen's three files and nothing of Krea's, Turbo included, and so needs no token: the box of the
# identity measurement (docs/identity-experiment.md#one-hour, docs/gpu.md#qwen-image), which draws nothing else and
# pays for every minute of 48.7 GB.
qwen="${SIMPLE_CHAT_IMAGE_QWEN:-false}"
[[ "$qwen" = true || "$qwen" = false || "$qwen" = only ]] || { echo 'Use SIMPLE_CHAT_IMAGE_QWEN=true, false or only.' >&2; exit 1; }
# Viggle's few-step LoRA for Qwen and its node (docs/gpu.md#viggle), a test: off by default, and only beside Qwen.
viggle="${SIMPLE_CHAT_IMAGE_VIGGLE:-false}"
[[ "$viggle" = true || "$viggle" = false ]] || { echo 'Use SIMPLE_CHAT_IMAGE_VIGGLE=true or false.' >&2; exit 1; }
[[ "$viggle" = false || "$qwen" != false ]] || { echo 'Viggle'\''s LoRA is for Qwen-Image 2.1: set SIMPLE_CHAT_IMAGE_QWEN too.' >&2; exit 1; }
# Qwen's text-to-image prompt enhancer (docs/action-experiment.md#prompt-arms), for the prompt arms probe alone: off by
# default, and only beside Qwen. One more text encoder, 9.47 GB, which the bot never loads.
qwen_pe="${SIMPLE_CHAT_IMAGE_QWEN_PE:-false}"
[[ "$qwen_pe" = true || "$qwen_pe" = false ]] || { echo 'Use SIMPLE_CHAT_IMAGE_QWEN_PE=true or false.' >&2; exit 1; }
[[ "$qwen_pe" = false || "$qwen" != false ]] || { echo 'The prompt enhancer is Qwen-Image 2.1'\''s: set SIMPLE_CHAT_IMAGE_QWEN too.' >&2; exit 1; }
# Which torch this run installs (docs/gpu.md#cu130): cu128, the default, into ComfyUI/.venv, or cu130 into
# ComfyUI/.venv-cu130, for comfy-kitchen's CUDA backend: the bot's card installs cu130 alone (docs/gpu.md#bot-card), and
# the experiments keep cu128. A cu130 run leaves the default environment as it is, so it can run in the background
# beside a server started from that one, and image-serve.sh switches between them with a restart.
torch_line="${SIMPLE_CHAT_IMAGE_TORCH:-cu128}"
case "$torch_line" in
  cu128) venv="$comfy_dir/.venv"; torch_index="$TORCH_INDEX_URL"
    torch_pins=("torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION") ;;
  cu130) venv="$comfy_dir/.venv-cu130"; torch_index="$TORCH_CU130_INDEX_URL"
    torch_pins=("torch==$TORCH_CU130_VERSION" "torchvision==$TORCHVISION_CU130_VERSION" "torchaudio==$TORCHAUDIO_CU130_VERSION") ;;
  *) echo 'Use SIMPLE_CHAT_IMAGE_TORCH=cu128 or cu130.' >&2; exit 1 ;;
esac
# An offer that advertises 1171 Mbit/s has delivered 115 (docs/knowledge/gpu-measurements.md#costs-and-downloads).
# Below the floor the answer is to destroy the machine and take the next candidate, not to wait: 22 GB at 100 Mbit/s
# is half the session.
min_mbit="${SIMPLE_CHAT_IMAGE_MIN_MBIT:-200}"
window="${SIMPLE_CHAT_IMAGE_SPEED_WINDOW:-60}"
[[ "$min_mbit" =~ ^[0-9]+$ && "$window" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid SIMPLE_CHAT_IMAGE_MIN_MBIT/WINDOW.' >&2; exit 1; }
# The verdict is about the machine's link, so it is read where the link is: the byte counter of the interface that
# carries the default route. What lands in models/ is a part of that traffic and often the smaller part — the git
# fetch and the torch wheels below use the same wire, and bootstrap.sh is pulling ~24 GB over 16 connections beside
# this run. SIMPLE_CHAT_IMAGE_LINK_IF names another interface when the default route is not the one that matters.
link_if="${SIMPLE_CHAT_IMAGE_LINK_IF:-}"
if [[ -z "$link_if" ]]; then
  link_if="$(awk '$2 == "00000000" && $8 == "00000000" { print $1; exit }' /proc/net/route 2>/dev/null || true)"
fi
for command in git curl python3 flock; do
  command -v "$command" >/dev/null || { echo "Missing dependency: $command (use a CUDA development image)." >&2; exit 1; }
done

dry_run=false
tokens_stdin=false
print_workflow=false
for argument in "$@"; do
  case "$argument" in
    # Everything that costs nothing: the pins, the tokens, the leftovers and the plan, without a byte downloaded.
    --dry-run) dry_run=true ;;
    --tokens-stdin) tokens_stdin=true ;;
    # The graph this run would post, with the file names of the chosen source filled in.
    --print-workflow) print_workflow=true ;;
    *) echo 'Usage: image-bootstrap.sh [--dry-run] [--tokens-stdin] [--print-workflow]' >&2; exit 1 ;;
  esac
done

civitai_token="${SIMPLE_CHAT_CIVITAI_TOKEN:-}"
hf_token="${SIMPLE_CHAT_HF_TOKEN:-}"
if [[ "$tokens_stdin" = true ]]; then
  # `|| [[ -n "$line" ]]`: the natural way to pipe a secret is `printf 'civitai=%s' "$t"`, without a closing
  # newline, and a plain `read` loop would drop that last line and report a missing token instead.
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      civitai=*) civitai_token="${line#civitai=}" ;;
      hf=*) hf_token="${line#hf=}" ;;
      '') ;;
      # The offending line is not repeated: it holds a secret.
      *) echo 'On stdin use lines "civitai=<token>" and "hf=<token>", one per line.' >&2; exit 1 ;;
    esac
  done
fi

# One record per file: URL, destination, SHA256, size, which token to send.
records=()
add_record() { records+=("$1"$'\t'"$2"$'\t'"$3"$'\t'"$4"$'\t'"$5"); }
hf_url() { printf 'https://huggingface.co/%s/resolve/%s/%s' "$1" "$2" "$3"; }
[[ "$qwen" = only ]] || add_record "$IMAGE_MODEL_URL" "$models_dir/diffusion_models/$IMAGE_MODEL_FILE" "$IMAGE_MODEL_SHA256" "$IMAGE_MODEL_BYTES" civitai
# The two file names the graph's loaders have to name: `official` installs different ones, and a graph that kept
# the `comfy` names would fail inside CLIPLoader on a box that passed every check here.
if [[ "$qwen" = only ]]; then
  encoder_file=''; vae_file=''
elif [[ "$image_source" = official ]]; then
  encoder_file="$OFFICIAL_ENCODER_FILE"; vae_file="$OFFICIAL_VAE_FILE"
  add_record "$(hf_url "$OFFICIAL_REPO" "$OFFICIAL_REVISION" "$OFFICIAL_ENCODER_PATH")" \
    "$models_dir/text_encoders/$OFFICIAL_ENCODER_FILE" "$OFFICIAL_ENCODER_SHA256" "$OFFICIAL_ENCODER_BYTES" hf
  add_record "$(hf_url "$OFFICIAL_REPO" "$OFFICIAL_REVISION" "$OFFICIAL_VAE_PATH")" \
    "$models_dir/vae/$OFFICIAL_VAE_FILE" "$OFFICIAL_VAE_SHA256" "$OFFICIAL_VAE_BYTES" hf
  [[ "$turbo" = false ]] || add_record "$(hf_url "$OFFICIAL_REPO" "$OFFICIAL_REVISION" "$OFFICIAL_TURBO_PATH")" \
    "$models_dir/diffusion_models/$OFFICIAL_TURBO_FILE" "$OFFICIAL_TURBO_SHA256" "$OFFICIAL_TURBO_BYTES" hf
else
  encoder_file="$IMAGE_ENCODER_FILE"; vae_file="$IMAGE_VAE_FILE"
  add_record "$(hf_url "$IMAGE_ENCODER_REPO" "$IMAGE_ENCODER_REVISION" "$IMAGE_ENCODER_PATH")" \
    "$models_dir/text_encoders/$IMAGE_ENCODER_FILE" "$IMAGE_ENCODER_SHA256" "$IMAGE_ENCODER_BYTES" none
  add_record "$(hf_url "$IMAGE_VAE_REPO" "$IMAGE_VAE_REVISION" "$IMAGE_VAE_PATH")" \
    "$models_dir/vae/$IMAGE_VAE_FILE" "$IMAGE_VAE_SHA256" "$IMAGE_VAE_BYTES" none
  [[ "$turbo" = false ]] || add_record "$(hf_url "$IMAGE_TURBO_REPO" "$IMAGE_TURBO_REVISION" "$IMAGE_TURBO_PATH")" \
    "$models_dir/diffusion_models/$IMAGE_TURBO_FILE" "$IMAGE_TURBO_SHA256" "$IMAGE_TURBO_BYTES" none
fi
# Qwen brings its own transformer, its own text encoder and its own VAE; nothing of Krea's is shared with it, and
# nothing of Krea's is replaced, so both checkpoints stay drawable from the same box. Public repository, no token.
if [[ "$qwen" != false ]]; then
  add_record "$(hf_url "$IMAGE_QWEN_REPO" "$IMAGE_QWEN_REVISION" "$IMAGE_QWEN_MODEL_PATH")" \
    "$models_dir/diffusion_models/$IMAGE_QWEN_MODEL_FILE" "$IMAGE_QWEN_MODEL_SHA256" "$IMAGE_QWEN_MODEL_BYTES" none
  add_record "$(hf_url "$IMAGE_QWEN_REPO" "$IMAGE_QWEN_REVISION" "$IMAGE_QWEN_ENCODER_PATH")" \
    "$models_dir/text_encoders/$IMAGE_QWEN_ENCODER_FILE" "$IMAGE_QWEN_ENCODER_SHA256" "$IMAGE_QWEN_ENCODER_BYTES" none
  add_record "$(hf_url "$IMAGE_QWEN_REPO" "$IMAGE_QWEN_REVISION" "$IMAGE_QWEN_VAE_PATH")" \
    "$models_dir/vae/$IMAGE_QWEN_VAE_FILE" "$IMAGE_QWEN_VAE_SHA256" "$IMAGE_QWEN_VAE_BYTES" none
fi
# The enhancer, verified as the weights are and named in image-verified.txt, which the probe reads before it rewrites.
[[ "$qwen_pe" = false ]] || add_record "$(hf_url "$IMAGE_QWEN_REPO" "$IMAGE_QWEN_REVISION" "$IMAGE_QWEN_PE_T2I_PATH")" \
  "$models_dir/text_encoders/$IMAGE_QWEN_PE_T2I_FILE" "$IMAGE_QWEN_PE_T2I_SHA256" "$IMAGE_QWEN_PE_T2I_BYTES" none
# Viggle's LoRA, and its node, which is code the server runs: each fetched at the pinned revision and verified as the
# weights are, before it is given its name. image-serve.sh loads the node only when asked to.
if [[ "$viggle" = true ]]; then
  add_record "$(hf_url "$IMAGE_VIGGLE_REPO" "$IMAGE_VIGGLE_REVISION" "$IMAGE_VIGGLE_LORA_PATH")" \
    "$models_dir/loras/$IMAGE_VIGGLE_LORA_FILE" "$IMAGE_VIGGLE_LORA_SHA256" "$IMAGE_VIGGLE_LORA_BYTES" none
  add_record "$(hf_url "$IMAGE_VIGGLE_REPO" "$IMAGE_VIGGLE_REVISION" "$IMAGE_VIGGLE_NODE_PATH")" \
    "$comfy_dir/custom_nodes/$IMAGE_VIGGLE_NODE_FILE" "$IMAGE_VIGGLE_NODE_SHA256" "$IMAGE_VIGGLE_NODE_BYTES" none
fi

# The pinned graph names the `comfy` files; this fills in what this run installs, so `official` is posted with the
# file names it downloaded. Whether ComfyUI's loaders read the official diffusers layout at all is still unverified.
render_workflow() {
  python3 - "$task_dir/$1" "$2" "$3" "$4" <<'PY'
import json,sys
graph=json.load(open(sys.argv[1]))
widgets={'UNETLoader':('unet_name',sys.argv[2]),'CLIPLoader':('clip_name',sys.argv[3]),'VAELoader':('vae_name',sys.argv[4])}
for node in graph.values():
    widget=widgets.get(node['class_type'])
    if widget: node['inputs'][widget[0]]=widget[1]
json.dump(graph,sys.stdout,indent=2)
print()
PY
}
# The Qwen graphs have no second source to choose between — the manifest names their three files and nothing else
# installs them — so rendering them is not substitution but the same guarantee run twice: what is written into
# $gpu_dir names the files this run verified.
render_qwen() {
  for graph in "$IMAGE_QWEN_WORKFLOW" "$IMAGE_QWEN_EDIT_WORKFLOW"; do
    render_workflow "$graph" "$IMAGE_QWEN_MODEL_FILE" "$IMAGE_QWEN_ENCODER_FILE" "$IMAGE_QWEN_VAE_FILE" >"$gpu_dir/$graph"
  done
}
if [[ "$print_workflow" = true ]]; then
  [[ "$qwen" != only ]] || { echo 'A Qwen-only box installs no Krea graph to print; it writes the Qwen graphs once verified.' >&2; exit 1; }
  render_workflow "$IMAGE_WORKFLOW" "$IMAGE_MODEL_FILE" "$encoder_file" "$vae_file"
  exit 0
fi

# One run at a time on this machine. Two would resume the same .part from two ends and the corruption would only
# show in the SHA256, after the whole file has been paid for; a dry run would meanwhile throw away a leftover the
# other run is writing. gpu/measure-profile.sh holds its lock on a file descriptor the same way.
mkdir -p "$gpu_dir" || { echo "Cannot create $gpu_dir; set SIMPLE_CHAT_GPU_DIR to a directory this user owns." >&2; exit 1; }
exec 9>"$gpu_dir/image-bootstrap.lock"
flock -n 9 || { echo "Another image-bootstrap.sh is working in $gpu_dir; this run did nothing." >&2; exit 1; }
# The lock belongs to this shell alone: the background fetchers and the guard are started with `9>&-`, or a curl or
# a sleep of a run that has just ended would hold it for as long as it takes to die and refuse the operator's rerun.

total_bytes=0
pending=0
for record in "${records[@]}"; do
  IFS=$'\t' read -r url destination digest size auth <<<"$record"
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "No pinned SHA256 for $(basename -- "$destination"); fill it into image-manifest.env." >&2; exit 1; }
  [[ "$size" =~ ^[1-9][0-9]*$ ]] || { echo "No pinned size for $(basename -- "$destination")." >&2; exit 1; }
  [[ "$auth" != civitai || -n "$civitai_token" ]] || { echo 'A CivitAI API token is required: SIMPLE_CHAT_CIVITAI_TOKEN or --tokens-stdin.' >&2; exit 1; }
  [[ "$auth" != hf || -n "$hf_token" ]] || { echo 'A Hugging Face token with access to the gate is required: SIMPLE_CHAT_HF_TOKEN or --tokens-stdin.' >&2; exit 1; }
  # A leftover longer than the pinned file is not this file and can only be thrown away. It happens here, under the
  # lock and before any fetcher exists, so that nothing is deleted from under a curl that is writing it.
  python3 - "$destination.part" "$size" <<'PY'
import pathlib,sys
part=pathlib.Path(sys.argv[1])
if part.exists() and part.stat().st_size > int(sys.argv[2]):
    part.unlink(); print(f'{part.name}: a leftover longer than the pinned file was discarded.')
PY
  # A file already on disk needs no room for itself; a partial one is counted whole, on the safe side.
  [[ -f "$destination" ]] || { total_bytes=$(( total_bytes + size )); pending=$(( pending + 1 )); }
done
# The tokens travel inside a quoted curl config line; a quote or a backslash in one would end the quoting early.
for token in "$civitai_token" "$hf_token"; do
  [[ "$token" != *[\"\\]* ]] || { echo 'A token contains a quote or a backslash and cannot be passed this way.' >&2; exit 1; }
done
if [[ "$dry_run" = true ]]; then
  echo "$([[ "$qwen" = only ]] && echo 'Qwen only' || echo "Source $image_source"), $(( total_bytes / 1024**3 )) GiB to fetch:"
  for record in "${records[@]}"; do
    IFS=$'\t' read -r url destination digest size auth <<<"$record"
    printf '%s\t%s bytes\t%s\n' "$(basename -- "$destination")" "$size" "$([[ -f "$destination" ]] && echo present || echo 'to fetch')"
  done
  [[ "$qwen" = only ]] || echo "The graph would load $IMAGE_MODEL_FILE, $encoder_file and $vae_file; --print-workflow prints it."
  [[ "$qwen" = false ]] || echo "Qwen is on: $IMAGE_QWEN_WORKFLOW and $IMAGE_QWEN_EDIT_WORKFLOW would load $IMAGE_QWEN_MODEL_FILE, $IMAGE_QWEN_ENCODER_FILE and $IMAGE_QWEN_VAE_FILE."
  [[ "$viggle" = false ]] || echo "Viggle is on: its LoRA into models/loras and $IMAGE_VIGGLE_NODE_FILE into custom_nodes, which image-serve.sh loads with SIMPLE_CHAT_IMAGE_VIGGLE=true."
  [[ "$qwen_pe" = false ]] || echo "The prompt enhancer is on: $IMAGE_QWEN_PE_T2I_FILE into models/text_encoders, for the prompt arms probe."
  [[ "$torch_line" = cu128 ]] || echo "Torch is $torch_line: ${torch_pins[*]} into $venv; the default environment, if the box has one, stays as it is."
  exit 0
fi

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
mkdir -p "$models_dir/diffusion_models" "$models_dir/text_encoders" "$models_dir/vae"
[[ "$viggle" = false ]] || mkdir -p "$models_dir/loras" "$comfy_dir/custom_nodes"
# The weights plus the virtual environment: torch with its CUDA libraries is about 12 GB on disk.
python3 - "$gpu_dir" "$total_bytes" <<'PY'
import pathlib,shutil,sys
required=int(sys.argv[2])+13*1024**3
free=shutil.disk_usage(pathlib.Path(sys.argv[1])).free
if free < required:
    raise SystemExit(f'At least {required // 1024**3} GiB free disk is required; {free // 1024**3} GiB is free.')
PY

downloaded_bytes() {
  python3 - "$models_dir" <<'PY'
import pathlib,sys
total=0
for path in pathlib.Path(sys.argv[1]).rglob('*'):
    if path.is_file(): total+=path.stat().st_size
print(total)
PY
}

# Everything this machine has pulled over the network, weights and wheels and the language model alike.
link_bytes() {
  local counter="/sys/class/net/$link_if/statistics/rx_bytes"
  [[ -n "$link_if" && -r "$counter" ]] || return 1
  cat "$counter"
}

fetch_one() {
  local url="$1" destination="$2" size="$3" auth="$4"
  local part="$destination.part" token='' needed attempt curl_pid status=0
  # A finished download can survive an interrupted verification, and a partial file can only be resumed from its
  # own end. A leftover too long for the pinned file is already gone, discarded in the pre-flight before the guard
  # started; the only file thrown away here is one this host has just refused to resume.
  needed="$(python3 - "$part" "$destination" "$size" <<'PY'
import pathlib,sys
part=pathlib.Path(sys.argv[1]); target=pathlib.Path(sys.argv[2]); expected=int(sys.argv[3])
done=target.exists() or (part.exists() and part.stat().st_size == expected)
print('no' if done else 'yes')
PY
)"
  [[ "$needed" = yes ]] || return 0
  case "$auth" in civitai) token="$civitai_token" ;; hf) token="$hf_token" ;; esac
  for attempt in 1 2; do
    # The URL, the output path and the credential go to curl over stdin, so none of them can be read out of `ps`.
    {
      printf 'url = "%s"\n' "$url"
      printf 'output = "%s"\n' "$part"
      if [[ -n "$token" ]]; then printf 'header = "Authorization: Bearer %s"\n' "$token"; fi
      printf '\n'
    } | curl --config - --continue-at - --fail --location --silent --show-error --retry 5 --retry-delay 5 &
    curl_pid=$!
    # The speed guard kills this subshell; pass that on to curl instead of orphaning a download that keeps the link busy.
    trap 'kill "$curl_pid" 2>/dev/null || true' TERM
    status=0
    wait "$curl_pid" || status=$?
    (( status != 0 )) || return 0
    # curl 33 is a host that answered a ranged request with the whole file: this leftover can never be resumed, and
    # every later run would ask for the same impossible thing, so the box could not make progress on its own. Start
    # once from zero instead. Any other failure keeps the partial file — curl has already retried five times, and
    # throwing away eleven arrived gigabytes because a connection dropped costs more than stopping does.
    (( attempt == 1 && status == 33 )) || break
    echo "$(basename -- "$part") cannot be resumed by this host; it was discarded and the download restarted." >&2
    rm -f -- "$part"
  done
  echo "$(basename -- "$destination"): the download failed (curl exit $status)." >&2
  return "$status"
}

# Measures the shared link once the downloads are running and ends them if the machine cannot deliver. Judging by
# what reached models/ would condemn a fast machine whose link is busy with the other lane: the floor asks for
# 1.5 GB a minute, and this script's own pip install can take most of the wire while it is measured.
speed_guard() {
  local before after link_before link_after mbit weights pid alive=false
  link_before="$(link_bytes)" || {
    echo "No byte counter for this machine's link${link_if:+ ($link_if)}; its speed was not judged." >&2
    return 0
  }
  before="$(downloaded_bytes)"
  sleep "$window"
  link_after="$(link_bytes)"
  after="$(downloaded_bytes)"
  for pid in "${download_pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=true; done
  # Everything arrived inside the window: an average over a link that was then idle says nothing, and a printed
  # "0 Mbit/s" reads exactly like the dead link this guard exists to catch. There is also nothing left to end.
  if [[ "$alive" = false ]]; then
    echo 'Every pinned file arrived inside the measurement window; there was nothing left to measure.'
    return 0
  fi
  # A counter that went backwards is an interface that came and went, not a reading, and a negative rate is never a
  # reason to destroy a machine. The weights' share is read the same way: a discarded leftover can shrink models/.
  if (( link_after < link_before )); then
    echo "The byte counter of $link_if went backwards; the link was not judged." >&2
    return 0
  fi
  mbit=$(( (link_after - link_before) * 8 / window / 1000000 ))
  weights=$(( after > before ? (after - before) * 8 / window / 1000000 : 0 ))
  if (( mbit >= min_mbit )); then
    echo "The link is carrying about ${mbit} Mbit/s, ${weights} of it into models/."
    return 0
  fi
  echo "Only ${mbit} Mbit/s over ${window}s on ${link_if}, ${weights} of it weights: below the ${min_mbit} Mbit/s floor, so destroy this machine and take the next offer." >&2
  kill "${download_pids[@]}" 2>/dev/null || true
}

download_pids=()
guard_pid=''
# Whatever ends this run takes the downloads with it: the repository check below, a failing `git fetch` or pip under
# `set -e`, the sm_120 abort, Ctrl-C. A curl that outlives the script keeps pulling into a .part that the next run
# then resumes from the wrong end, and that corruption only shows in the SHA256 once the whole file has been paid for.
cleanup() {
  [[ ${#download_pids[@]} -eq 0 ]] || kill "${download_pids[@]}" 2>/dev/null || true
  [[ -z "$guard_pid" ]] || kill "$guard_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
for record in "${records[@]}"; do
  IFS=$'\t' read -r url destination digest size auth <<<"$record"
  fetch_one "$url" "$destination" "$size" "$auth" 9>&- &
  download_pids+=($!)
done
# On a rerun where every pinned file is already here there is nothing to measure, and the guard would hold the run
# for the whole window to say so.
if (( pending > 0 )); then
  speed_guard 9>&- &
  guard_pid=$!
fi

# Meanwhile the environment is built, so the link and the installer do not wait for each other.
if [[ ! -d "$comfy_dir/.git" ]]; then
  git init -q "$comfy_dir"
  git -C "$comfy_dir" remote add origin "$COMFYUI_REPO"
fi
[[ "$(git -C "$comfy_dir" remote get-url origin)" = "$COMFYUI_REPO" ]] || { echo 'Unexpected ComfyUI repository.' >&2; exit 1; }
git -C "$comfy_dir" fetch --depth 1 origin "$COMFYUI_REVISION"
git -C "$comfy_dir" checkout --detach "$COMFYUI_REVISION"
[[ "$(git -C "$comfy_dir" rev-parse HEAD)" = "$COMFYUI_REVISION" ]]
[[ -d "$venv" ]] || python3 -m venv "$venv"
python="$venv/bin/python"
# cu130's mark of a whole environment, which image-serve.sh starts it on: gone while this run changes it, and written
# at the end, once the sm_120 check has passed and the weights are verified. The runbook waits for it.
ready="$venv/simple-chat-ready"
[[ "$torch_line" = cu128 ]] || rm -f -- "$ready"
# cu130 goes beside an environment and the weights already on the box, and a Qwen-only box has 60 GB: pip keeps no
# second copy of the wheels.
pip_flags=(--quiet)
[[ "$torch_line" = cu128 ]] || pip_flags+=(--no-cache-dir)
"$python" -m pip install "${pip_flags[@]}" --upgrade pip
# Torch first and pinned, from the CUDA index: ComfyUI's requirements.txt asks for a bare `torch` (and a bare
# torchvision and torchaudio, which the pinned ones satisfy), and the default index would serve a build without sm_120
# kernels. On cu130 its four wheels go on by their sha256 first (image-manifest.env), alone, and the line after them
# adds what they pull.
if [[ "$torch_line" = cu130 ]]; then
  lock="$(mktemp)"
  printf '%s --hash=sha256:%s\n' "torch==$TORCH_CU130_VERSION" "$TORCH_CU130_SHA256" "torchvision==$TORCHVISION_CU130_VERSION" "$TORCHVISION_CU130_SHA256" \
    "torchaudio==$TORCHAUDIO_CU130_VERSION" "$TORCHAUDIO_CU130_SHA256" "triton==$TRITON_CU130_VERSION" "$TRITON_CU130_SHA256" > "$lock"
  "$python" -m pip install "${pip_flags[@]}" --require-hashes --no-deps --index-url "$torch_index" -r "$lock"
  rm -f -- "$lock"
fi
"$python" -m pip install "${pip_flags[@]}" --index-url "$torch_index" "${torch_pins[@]}"
"$python" -m pip install "${pip_flags[@]}" -r "$comfy_dir/requirements.txt"
torch_said="$("$python" - "$TORCH_ARCH" <<'PY'
import sys,torch
archs=torch.cuda.get_arch_list()
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, architectures: {" ".join(archs)}')
if sys.argv[1] not in archs:
    raise SystemExit(f'This torch build has no {sys.argv[1]} kernels; the card would fall back or fail.')
PY
)"
echo "$torch_said"
# Nothing here installs a custom node but Viggle's one file when asked, and image-serve.sh disables the folder anyway,
# that file aside when asked to load it; say so if a node's directory appeared.
if compgen -G "$comfy_dir/custom_nodes/*/" >/dev/null; then
  echo 'Third-party custom nodes are present in ComfyUI/custom_nodes; they stay disabled at run time.' >&2
fi

failed=0
for pid in "${download_pids[@]}"; do wait "$pid" || failed=1; done
# The guard may still be inside its window; nothing is left for it to measure.
if [[ -n "$guard_pid" ]]; then
  kill "$guard_pid" 2>/dev/null || true
  wait "$guard_pid" 2>/dev/null || true
fi
(( failed == 0 )) || { echo 'A download failed or was ended; nothing is verified.' >&2; exit 1; }

# What this box was verified to run, the ComfyUI revision and each file's SHA256 as computed here, one
# `<sha256>  <file>` line apiece: the identity runbook copies it home before the first job and local/image-identity.ts
# pins the run to it. It is written whole or not at all, and a record from an earlier run does not outlive a
# verification that failed.
verified="$gpu_dir/image-verified.txt"
rm -f -- "$verified"
echo "revision $(git -C "$comfy_dir" rev-parse HEAD)" >"$verified.part"
for record in "${records[@]}"; do
  IFS=$'\t' read -r url destination digest size auth <<<"$record"
  python3 - "$destination" "$digest" "$size" "$verified.part" <<'PY'
import hashlib,pathlib,sys
target=pathlib.Path(sys.argv[1]); current=target if target.exists() else pathlib.Path(str(target)+'.part')
def mismatch(message):
    if current != target: current.unlink()
    raise SystemExit(f'{target.name}: {message}')
if current.stat().st_size != int(sys.argv[3]): mismatch('size mismatch; not starting.')
with current.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
if digest != sys.argv[2]: mismatch('SHA256 mismatch; not starting.')
if current != target: current.rename(target)
with open(sys.argv[4],'a') as record: record.write(f'{digest}  {target.name}\n')
print(f'{target.name}: SHA256 verified.')
PY
done
mv -- "$verified.part" "$verified"
# The graph is written only once the weights it names are verified, so its presence means the box can render.
[[ "$qwen" = only ]] || render_workflow "$IMAGE_WORKFLOW" "$IMAGE_MODEL_FILE" "$encoder_file" "$vae_file" >"$gpu_dir/$IMAGE_WORKFLOW"
[[ "$qwen" = false ]] || render_qwen
torch_env=''
if [[ "$torch_line" != cu128 ]]; then
  printf '%s\n' "$torch_said" >"$ready"
  torch_env="SIMPLE_CHAT_IMAGE_TORCH=$torch_line "
fi
if [[ "$qwen" = only ]]; then
  echo "Prepared, Qwen only: $gpu_dir/$IMAGE_QWEN_WORKFLOW draws frames and portraits, $gpu_dir/$IMAGE_QWEN_EDIT_WORKFLOW takes reference portraits; $verified says what was verified. Start with: ${torch_env}SIMPLE_CHAT_IMAGE_QWEN=only bash $task_dir/image-serve.sh"
else
  echo "Prepared; post $gpu_dir/$IMAGE_WORKFLOW. Start with: ${torch_env}bash $task_dir/image-serve.sh"
  [[ "$qwen" = false ]] || echo "Qwen is on: $gpu_dir/$IMAGE_QWEN_WORKFLOW draws frames, $gpu_dir/$IMAGE_QWEN_EDIT_WORKFLOW takes reference portraits."
fi
[[ "$viggle" = false ]] || echo "Viggle's LoRA and node are verified; a server started with SIMPLE_CHAT_IMAGE_VIGGLE=true loads the node, for its graphs alone."
[[ "$qwen_pe" = false ]] || echo "The prompt enhancer is verified; the prompt arms probe loads it through CLIPLoader, and the bot never does."
