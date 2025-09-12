# --- scripts/env_mvs.sh ---
export MVS_HOME=/opt/MVS
export MVCAM_COMMON_RUNENV="$MVS_HOME/lib"
# Jetson = aarch64: the SDK put the .so file in lib/aarch64
export LD_LIBRARY_PATH="$MVS_HOME/lib:$MVS_HOME/lib/aarch64:${LD_LIBRARY_PATH}"

# Per importing binaries of python
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python:/opt/MVS/Samples/aarch64/Python/MvImport:${PYTHONPATH}"

# (opzionale ma utile con più USB3)
if [ -x /opt/MVS/bin/set_usbfs_memory_size.sh ]; then
  sudo /opt/MVS/bin/set_usbfs_memory_size.sh
fi
