#!/usr/bin/env bash
set -e

cd /home/nvidia/big400_bottle_inspection/inference_clients

# simula esattamente GNOME Terminal
source ~/.bashrc

source .venv/bin/activate
source scripts/env_mvs_user.sh

exec python -u lip_client.py \
  --daemon \
  --cam-index 2 \
  --api-url http://localhost:9001 \
  --api-key EC9puzE6crcRm7buAF1S \
  --model big400-lip-insp-before-cleaning-da2vm/9 \
  --th 0.90 \
  --trigger-file /tmp/lip_go \
  --result-file /tmp/lip_res \
  --save-on-defect \
  --bucket bewtr-bottle-defect-validation \
  --site nice \
  --station lip \
  # --headless



# if you want to test you should add
  # --save-on-defect \
  # --force-upload \ if you want to push every inferenced photo on s3
