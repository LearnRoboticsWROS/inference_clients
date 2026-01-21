#!/usr/bin/env bash
set -Eeuo pipefail

IC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
SESSION="big400"

# chiudi sessione precedente se esiste
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

# crea sessione + pane
tmux new-session -d -s "$SESSION" -c "$IC_DIR" \
  "bash --login -i scripts/run_lip_inference.sh"

tmux split-window -v -t "$SESSION:0.0" -c "$IC_DIR" \
  "bash --login -i scripts/run_body_inference.sh"

tmux split-window -h -t "$SESSION:0.1" -c "$IC_DIR" \
  "bash --login -i scripts/run_bodycl_inference.sh"

tmux split-window -h -t "$SESSION:0.0" -c "$IC_DIR" \
  "bash --login -i scripts/run_orchestrator.sh"

# layout + titoli (opzionale ma utile)
tmux select-layout -t "$SESSION" tiled
tmux select-pane -t "$SESSION:0.0" -T "LIP"
tmux select-pane -t "$SESSION:0.1" -T "BODY"
tmux select-pane -t "$SESSION:0.2" -T "BODYCL"
tmux select-pane -t "$SESSION:0.3" -T "ORCH"

tmux attach -t "$SESSION"
