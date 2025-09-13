#!/usr/bin/env bash
# scripts/env_mvs_user.sh
export MVS_HOME=/opt/MVS
export MVCAM_COMMON_RUNENV="$MVS_HOME/lib"
export LD_LIBRARY_PATH="$MVS_HOME/lib:$MVS_HOME/lib/aarch64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python:/opt/MVS/Samples/aarch64/Python/MvImport:${PYTHONPATH:-}"
