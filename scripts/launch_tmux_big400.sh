#!/usr/bin/env bash
set -Eeuo pipefail

# === Config ===
IC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
SESSION="big400"

# tmux installato?
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux non trovato. Installa con: sudo apt-get update && sudo apt-get install -y tmux"
  exit 1
fi

# --- chiedi sudo una volta e tienilo vivo mentre lanci tutto ---
echo "[sudo] verrà richiesta UNA volta ora, poi non più."
sudo -v

sudo_keepalive() {
  while true; do
    sleep 30
    sudo -n true 2>/dev/null || exit 0
  done
}
sudo_keepalive &
KEEPALIVE_PID=$!

cleanup() {
  kill "$KEEPALIVE_PID" 2>/dev/null || true
  sudo -k || true
}
trap cleanup EXIT

# Chiudi eventuale sessione precedente
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

# Setup una tantum MVS (se ti serve davvero farlo ad ogni avvio)
sudo bash -lc "cd '$IC_DIR' && bash scripts/env_mvs_setup_once.sh"

# === Pane 1: LIP client ===
tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd \"$IC_DIR\" && \
  source .venv/bin/activate && \
  source scripts/env_mvs_user.sh && \
  python lip_client.py --daemon --cam-index 2 \
    --api-url http://localhost:9001 \
    --api-key EC9puzE6crcRm7buAF1S \
    --model big400-lip-insp-before-cleaning-da2vm/9 \
    --exposure-us 10000 --gain 0 --fps 25 \
    --th 0.10 \
    --trigger-file /tmp/lip_go --result-file /tmp/lip_res; \
  echo; echo \"[LIP chiuso] Premere INVIO per chiudere il pannello\"; read'"

# === Pane 2: BODY client ===
tmux split-window -v -t "$SESSION" \
  "bash -lc 'cd \"$IC_DIR\" && \
  source .venv/bin/activate && \
  source scripts/env_mvs_user.sh && \
  python body_client.py --daemon --cam-index 0 \
    --api-url http://localhost:9001 \
    --api-key EC9puzE6crcRm7buAF1S \
    --model big400-body-insp-before-cleaning-7cr8v/8 \
    --th 0.90 \
    --trigger-file /tmp/body_go --result-file /tmp/body_res; \
  echo; echo \"[BODY chiuso] Premere INVIO per chiudere il pannello\"; read'"

# === Pane 3: BODY-CL client ===
tmux split-window -h -t "$SESSION:0.1" \
  "bash -lc 'cd \"$IC_DIR\" && \
  source .venv/bin/activate && \
  source scripts/env_mvs_user.sh && \
  python body_cl_client.py --daemon --cam-index 1 \
    --api-url http://localhost:9001 \
    --api-key EC9puzE6crcRm7buAF1S \
    --model big400-lip-insp-before-cleaning-da2vm/8 \
    --th 0.95 \
    --trigger-file /tmp/bodycl_go --result-file /tmp/bodycl_res; \
  echo; echo \"[BODYCL chiuso] Premere INVIO per chiudere il pannello\"; read'"

# === Pane 4: ORCHESTRATOR ===
tmux split-window -h -t "$SESSION:0.0" \
  "bash -lc 'cd \"$IC_DIR\" && \
  source .venv-orch/bin/activate && \
  python orchestrator_modbus_deamon_hr.py \
    --bind-ip 0.0.0.0 --bind-port 5020 \
    --timeout 3.5 --ack-timeout 1.5 \
    --lip-trigger /tmp/lip_go     --lip-result /tmp/lip_res \
    --body-trigger /tmp/body_go   --body-result /tmp/body_res \
    --bodycl-trigger /tmp/bodycl_go --bodycl-result /tmp/bodycl_res; \
  echo; echo \"[ORCH chiuso] Premere INVIO per chiudere il pannello\"; read'"

# Layout e titoli
tmux select-layout -t "$SESSION" tiled
tmux select-pane -t "$SESSION:0.0" -T "LIP client"
tmux select-pane -t "$SESSION:0.1" -T "BODY client"
tmux select-pane -t "$SESSION:0.2" -T "ORCHESTRATOR"
tmux select-pane -t "$SESSION:0.3" -T "BODYCL client"

tmux attach -t "$SESSION"

