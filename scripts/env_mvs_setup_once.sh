#!/usr/bin/env bash
# scripts/env_mvs_setup_once.sh
set -euo pipefail

# Imposta usbfs_memory_size (richiede root). Sicuro da rilanciare.
if [[ -x /opt/MVS/bin/set_usbfs_memory_size.sh ]]; then
  /opt/MVS/bin/set_usbfs_memory_size.sh
else
  echo "WARN: /opt/MVS/bin/set_usbfs_memory_size.sh non trovato"
fi

# Mostra il valore corrente (debug)
if [[ -r /sys/module/usbcore/parameters/usbfs_memory_mb ]]; then
  echo "usbfs_memory_mb=$(cat /sys/module/usbcore/parameters/usbfs_memory_mb)"
fi

echo "[env_mvs_setup_once] OK"
