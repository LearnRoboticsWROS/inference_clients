#!/usr/bin/env bash
set -e

cd /home/nvidia/big400_bottle_inspection/inference_clients

# simula esattamente GNOME Terminal
source ~/.bashrc

source .venv/bin/activate
source scripts/env_mvs_user.sh

exec python -u body_cl_client.py \
  --daemon \
  --cam-index 1 \
  --api-url http://localhost:9001 \
  --api-key EC9puzE6crcRm7buAF1S \
  --model big400-body-insp-before-cleaning-7cr8v/8 \
  --th 0.95 \
  --trigger-file /tmp/bodycl_go \
  --result-file /tmp/bodycl_res \
  --save-on-defect \
  --bucket bewtr-bottle-defect-validation \
  --site nice \
  --station body_cl \
  # --headless