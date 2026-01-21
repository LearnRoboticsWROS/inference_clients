#!/usr/bin/env bash
set -e

cd /home/nvidia/big400_bottle_inspection/inference_clients

# simula esattamente GNOME Terminal
source ~/.bashrc

source .venv-orch/bin/activate
source scripts/env_mvs_user.sh

exec python -u orchestrator_modbus_deamon_hr.py \
  --bind-ip 0.0.0.0 --bind-port 5020 \
  --timeout 3.5 --ack-timeout 1.5 \
  --lip-trigger /tmp/lip_go   --lip-result /tmp/lip_res \
  --body-trigger /tmp/body_go --body-result /tmp/body_res \
  --bodycl-trigger /tmp/bodycl_go --bodycl-result /tmp/bodycl_res