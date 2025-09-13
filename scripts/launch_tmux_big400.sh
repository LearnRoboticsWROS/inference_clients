#!/usr/bin/env bash
set -euo pipefail

# === Config ===
IC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
SESSION="big400"

# tmux installato?
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux non trovato. Installa con: sudo apt-get update && sudo apt-get install -y tmux"
  exit 1
fi

# --- NEW: chiedi sudo una volta e tienilo vivo mentre lanci tutto ---
echo "[sudo] verrà richiesta UNA volta ora, poi non più."
sudo -v   # chiede password UNA volta nel terminale corrente

# keep-alive finché gira questo script
# (evita che scada il ticket sudo mentre i pannelli partono)
sudo_keepalive() {
  while true; do
    sleep 30
    sudo -n true 2>/dev/null || exit 0
  done
}
sudo_keepalive &            # background
KEEPALIVE_PID=$!
cleanup() {
  # uccidi keepalive e invalida sudo quando esci
  kill $KEEPALIVE_PID 2>/dev/null || true
  sudo -k || true
}
trap cleanup EXIT
# --- END NEW ---

# Chiudi eventuale sessione precedente
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

sudo bash -lc "cd '$IC_DIR' && bash scripts/env_mvs_setup_once.sh"


# === Panello 1: LIP client ===
tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd \"$IC_DIR\" && source .venv/bin/activate && source scripts/env_mvs_user.sh && \
python lip_client.py --daemon --cam-index 0 --exposure-us 10000 --gain 0 --fps 25 \
  --trigger-file /tmp/lip_go --result-file /tmp/lip_res; \
echo; echo \"[LIP chiuso] Premere INVIO per chiudere il pannello\"; read'"

# === Panello 2: BODY client ===
tmux split-window -v -t "$SESSION" \
  "bash -lc 'cd \"$IC_DIR\" && source .venv/bin/activate && source scripts/env_mvs_user.sh && \
python body_client.py --daemon --cam-index 2 \
  --trigger-file /tmp/body_go --result-file /tmp/body_res; \
echo; echo \"[BODY chiuso] Premere INVIO per chiudere il pannello\"; read'"

# === Panello 3: ORCHESTRATORE ===
tmux split-window -h -t "$SESSION:0.0" \
  "bash -lc 'cd \"$IC_DIR\" && source .venv-orch/bin/activate && \
python orchestrator_modbus_deamon_hr.py \
  --bind-ip 0.0.0.0 --bind-port 5020 \
  --timeout 4 --ack-timeout 2 \
  --lip-trigger /tmp/lip_go   --lip-result /tmp/lip_res \
  --body-trigger /tmp/body_go --body-result /tmp/body_res \
  --bodycl-trigger /tmp/bodycl_go --bodycl-result /tmp/bodycl_res; \
echo; echo \"[ORCH chiuso] Premere INVIO per chiudere il pannello\"; read'"

tmux select-layout -t "$SESSION" tiled
tmux select-pane -t "$SESSION:.0" -T "LIP client"
tmux select-pane -t "$SESSION:.1" -T "BODY client"
tmux select-pane -t "$SESSION:.2" -T "ORCHESTRATOR"
tmux attach -t "$SESSION"
