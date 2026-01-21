#!/usr/bin/env bash
set -e

cd /home/nvidia/big400_bottle_inspection/inference_clients

# simula esattamente GNOME Terminal
source ~/.bashrc

source .venv/bin/activate
source scripts/env_mvs_user.sh

exec python -u body_client.py \
  --daemon \
  --cam-index 0 \
  --api-url http://localhost:9001 \
  --api-key EC9puzE6crcRm7buAF1S \
  --model big400-body-insp-before-cleaning-7cr8v/8 \
  --th 0.70 \
  --trigger-file /tmp/body_go \
  --result-file /tmp/body_res \
  --save-on-defect \
  --bucket bewtr-bottle-defect-validation \
  --site nice \
  --station body \
  # --headless