#!/usr/bin/env bash
# ComfyUI for the picture lane: headless, loopback only, one chosen card, no custom nodes (one pinned file when a test
# asks for it), no SageAttention, and every picture in RAM for seconds (image-sweeper.py).
set -euo pipefail
umask 077
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$task_dir/image-manifest.env"
gpu_dir="${SIMPLE_CHAT_GPU_DIR:-/workspace/simple-chat-gpu}"
comfy_dir="$gpu_dir/ComfyUI"
port="${SIMPLE_CHAT_IMAGE_PORT:-8188}"
# Which card the pictures use. The language server takes the whole of its own card (gpu/serve.sh --gpu-layers 99),
# so a picture that lands there means an OOM in the middle of somebody's scene. CUDA_VISIBLE_DEVICES is the only
# switch used here: at the pinned revision `--cuda-device N` is a plain assignment to that same variable
# (main.py:99), so passing both would say the same thing twice, and the choice stays in one place, the environment.
device="${SIMPLE_CHAT_IMAGE_GPU:-1}"
[[ "$port" =~ ^[0-9]+$ && "$device" =~ ^[0-9]$ ]] || { echo 'Use a numeric SIMPLE_CHAT_IMAGE_PORT and a single-digit SIMPLE_CHAT_IMAGE_GPU index.' >&2; exit 1; }
(( port > 0 && port <= 65535 )) || exit 1
# Which torch the server runs on (docs/gpu.md#cu130): cu128 from ComfyUI/.venv, the default, which the experiments
# keep, or cu130 from ComfyUI/.venv-cu130, the bot's (docs/gpu.md#bot-card), which image-bootstrap.sh builds when asked,
# alone or beside the other, so that a switch is a restart and no install. That one may be built beside a running
# server, and it starts only once its install has ended whole, with the mark the bootstrap writes last. A run pins the
# torch /system_stats reports (local/image-batch.ts `serverPins`).
torch_line="${SIMPLE_CHAT_IMAGE_TORCH:-cu128}"
case "$torch_line" in
  cu128) venv="$comfy_dir/.venv" ;;
  cu130) venv="$comfy_dir/.venv-cu130"
    [[ -f "$venv/simple-chat-ready" ]] || { echo 'No whole cu130 environment; run image-bootstrap.sh with SIMPLE_CHAT_IMAGE_TORCH=cu130 first.' >&2; exit 1; } ;;
  *) echo 'Use SIMPLE_CHAT_IMAGE_TORCH=cu128 or cu130.' >&2; exit 1 ;;
esac
python="$venv/bin/python"
[[ -x "$python" ]] || { echo 'Run image-bootstrap.sh first.' >&2; exit 1; }
[[ "$(git -C "$comfy_dir" rev-parse HEAD)" = "$COMFYUI_REVISION" ]] || { echo 'Unexpected ComfyUI revision; rerun image-bootstrap.sh.' >&2; exit 1; }
# The Qwen checkpoint is additive (`true`): the same server serves it, from its own three files and its own two
# graphs. `only` is a box image-bootstrap.sh gave Qwen's files and nothing of Krea's, and it serves Qwen alone.
qwen="${SIMPLE_CHAT_IMAGE_QWEN:-false}"
[[ "$qwen" = true || "$qwen" = false || "$qwen" = only ]] || { echo 'Use SIMPLE_CHAT_IMAGE_QWEN=true, false or only.' >&2; exit 1; }
if [[ "$qwen" != only ]]; then
  image_source="${SIMPLE_CHAT_IMAGE_SOURCE:-comfy}"
  if [[ "$image_source" = official ]]; then
    encoder="$OFFICIAL_ENCODER_FILE"; vae="$OFFICIAL_VAE_FILE"
  else
    encoder="$IMAGE_ENCODER_FILE"; vae="$IMAGE_VAE_FILE"
  fi
  for file in "models/diffusion_models/$IMAGE_MODEL_FILE" "models/text_encoders/$encoder" "models/vae/$vae"; do
    [[ -f "$comfy_dir/$file" ]] || { echo "Missing $file; rerun image-bootstrap.sh." >&2; exit 1; }
  done
  # image-bootstrap.sh writes this graph with the file names of the source it installed, after verifying them; the
  # bot posts it. Without it the server would answer /prompt with loaders pointing at files nobody fetched.
  workflow="$gpu_dir/$IMAGE_WORKFLOW"
  [[ -f "$workflow" ]] || { echo "Missing $workflow; rerun image-bootstrap.sh." >&2; exit 1; }
  grep -q "\"$encoder\"" "$workflow" || { echo "$workflow names another encoder than $image_source installed." >&2; exit 1; }
fi
# This refuses to start with Qwen on and its files absent, rather than let the first cell of a timed run find out.
if [[ "$qwen" != false ]]; then
  for file in "models/diffusion_models/$IMAGE_QWEN_MODEL_FILE" "models/text_encoders/$IMAGE_QWEN_ENCODER_FILE" "models/vae/$IMAGE_QWEN_VAE_FILE"; do
    [[ -f "$comfy_dir/$file" ]] || { echo "Missing $file; rerun image-bootstrap.sh with SIMPLE_CHAT_IMAGE_QWEN=$qwen." >&2; exit 1; }
  done
  for graph in "$IMAGE_QWEN_WORKFLOW" "$IMAGE_QWEN_EDIT_WORKFLOW"; do
    [[ -f "$gpu_dir/$graph" ]] || { echo "Missing $gpu_dir/$graph; rerun image-bootstrap.sh with SIMPLE_CHAT_IMAGE_QWEN=$qwen." >&2; exit 1; }
  done
fi
# comfy-kitchen's Triton backend, off unless SIMPLE_CHAT_IMAGE_TRITON=1 (docs/action-experiment.md#pilot). On a torch
# built for CUDA below 13 the pinned server runs every int8 layer on the kitchen's eager path (comfy/quant_ops.py:22-43);
# the flag lets it use Triton's fused kernels instead, which may round differently. A run records it in its pins
# (local/image-batch.ts `serverPins`, from the command line /system_stats reports), and a resume across it is refused.
triton="${SIMPLE_CHAT_IMAGE_TRITON:-0}"
[[ "$triton" = 0 || "$triton" = 1 ]] || { echo 'Use SIMPLE_CHAT_IMAGE_TRITON=0 or 1.' >&2; exit 1; }
flags=()
if [[ "$triton" = 1 ]]; then flags+=(--enable-triton-backend); fi
# Viggle's few-step LoRA for Qwen (docs/gpu.md#viggle), off unless SIMPLE_CHAT_IMAGE_VIGGLE=true: its node, one file
# image-bootstrap.sh fetched and verified, is loaded by its name beside --disable-all-custom-nodes, which at the pinned
# revision still skips every other entry of custom_nodes (nodes.py:2371, main.py:215, main.py:513). The node is code the
# server runs, so its bytes are checked again here; the LoRA, weights, by its size. A server with the node draws the
# LoRA's graphs and no others (local/image-pilot.ts `turbo`).
viggle="${SIMPLE_CHAT_IMAGE_VIGGLE:-false}"
[[ "$viggle" = true || "$viggle" = false ]] || { echo 'Use SIMPLE_CHAT_IMAGE_VIGGLE=true or false.' >&2; exit 1; }
if [[ "$viggle" = true ]]; then
  lora="$comfy_dir/models/loras/$IMAGE_VIGGLE_LORA_FILE" node="$comfy_dir/custom_nodes/$IMAGE_VIGGLE_NODE_FILE"
  [[ -f "$lora" && "$(stat -c %s -- "$lora")" = "$IMAGE_VIGGLE_LORA_BYTES" ]] || { echo "Missing models/loras/$IMAGE_VIGGLE_LORA_FILE; rerun image-bootstrap.sh with SIMPLE_CHAT_IMAGE_VIGGLE=true." >&2; exit 1; }
  [[ -f "$node" && "$(sha256sum -- "$node" | cut -d ' ' -f 1)" = "$IMAGE_VIGGLE_NODE_SHA256" ]] || { echo "custom_nodes/$IMAGE_VIGGLE_NODE_FILE is missing or not the pinned file; rerun image-bootstrap.sh with SIMPLE_CHAT_IMAGE_VIGGLE=true." >&2; exit 1; }
  flags+=(--whitelist-custom-nodes "$IMAGE_VIGGLE_NODE_FILE")
fi
# Krea 2 produces garbage under SageAttention, and several rented ComfyUI templates turn it on through their own
# launcher. This script is the launcher: the flag is absent, and an inherited request for it is refused rather than
# silently ignored, because a bad picture would otherwise be blamed on the fine-tune.
if [[ "${COMFYUI_ARGS:-}${CLI_ARGS:-}${COMFYUI_EXTRA_ARGS:-}" = *sage* ]]; then
  echo 'SageAttention is requested in the environment; Krea 2 breaks under it. Clear COMFYUI_ARGS/CLI_ARGS first.' >&2
  exit 1
fi
# The picture of a reader's scene is written to ComfyUI's temp directory before /view hands it to the bot, and no
# route of the HTTP API deletes that file: the server empties the directory only when it starts. So the directory is
# on a tmpfs, which a stopped instance's disk never holds, and image-sweeper.py beside the server deletes a picture a
# few seconds after the bot has deleted its job record (local/image-batch.ts `drawOne`), and every picture and record
# after ten minutes whatever happened. A /dev/shm that is not a writable tmpfs would put the pictures back on the
# disk, so the server is refused rather than started without it. SIMPLE_CHAT_IMAGE_TEMP_ROOT moves the directory,
# to another tmpfs only. The pinned main.py appends `temp` to --temp-directory and empties that at startup
# (start_comfyui, cleanup_temp_filesystem), which is why the sweeper is pointed at `$temp_root/temp`.
temp_root="${SIMPLE_CHAT_IMAGE_TEMP_ROOT:-/dev/shm/simple-chat-comfy}"
on_tmpfs() { [[ "$(stat -f -c %T -- "$1" 2>/dev/null)" = tmpfs ]]; }
temp_parent="$(dirname -- "$temp_root")"
if ! on_tmpfs "$temp_parent" || [[ ! -w "$temp_parent" ]]; then
  echo "$temp_parent is not a writable tmpfs, and the pictures of scenes would land on the disk. Refusing to start." >&2
  exit 1
fi
mkdir -p -- "$temp_root"
if [[ -L "$temp_root" || ! -d "$temp_root" || ! -O "$temp_root" ]] || ! on_tmpfs "$temp_root"; then
  echo "$temp_root is not a directory of this user on a tmpfs. Refusing to start." >&2
  exit 1
fi
chmod 700 -- "$temp_root"
sweeper="$task_dir/image-sweeper.py"
[[ -f "$sweeper" ]] || { echo "Missing $sweeper; copy it beside this script." >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$device"
ulimit -c 0
if [[ "$qwen" = only ]]; then
  echo "Starting ComfyUI $COMFYUI_VERSION for $IMAGE_QWEN_NAME alone on GPU $device, loopback port $port, temp in $temp_root; post $gpu_dir/$IMAGE_QWEN_WORKFLOW or $gpu_dir/$IMAGE_QWEN_EDIT_WORKFLOW."
else
  echo "Starting ComfyUI $COMFYUI_VERSION for $IMAGE_MODEL_NAME on GPU $device, loopback port $port, temp in $temp_root; post $workflow."
fi
if [[ "$triton" = 1 ]]; then echo "With comfy-kitchen's Triton backend (SIMPLE_CHAT_IMAGE_TRITON=1)."; fi
if [[ "$viggle" = true ]]; then echo "With Viggle's node, $IMAGE_VIGGLE_NODE_FILE, for the LoRA's graphs alone (SIMPLE_CHAT_IMAGE_VIGGLE=true)."; fi
if [[ "$torch_line" != cu128 ]]; then echo "On $(<"$venv/simple-chat-ready") (SIMPLE_CHAT_IMAGE_TORCH=$torch_line)."; fi
# The sweeper starts before the exec, with this shell's PID, which the exec hands to ComfyUI, and it stops by itself
# once that PID is gone. Its rows are counts and codes, in this script's log beside the server's own lines.
"$python" "$sweeper" --pid "$$" --temp "$temp_root/temp" --port "$port" &
# --disable-metadata: ComfyUI writes the whole prompt into the PNG by default, and a picture leaves the card.
# --disable-all-custom-nodes and --disable-api-nodes: only the pinned core runs, Viggle's one file aside when asked,
# and nothing calls a paid endpoint.
# --preview-method none: previews cost VRAM on the card the language model does not share.
# The attention implementation is left at the pinned build's default, which is the same on every run of this commit;
# ComfyUI prints which one it chose at startup, and that line belongs with the seconds-per-picture number.
# /history holds a prompt until the bot or the sweeper deletes its record, and output/ holds what the batch harness
# drew — do not copy either home.
exec "$python" "$comfy_dir/main.py" \
  --listen 127.0.0.1 --port "$port" --disable-auto-launch --temp-directory "$temp_root" \
  --disable-metadata --disable-all-custom-nodes --disable-api-nodes --preview-method none ${flags[@]+"${flags[@]}"}
