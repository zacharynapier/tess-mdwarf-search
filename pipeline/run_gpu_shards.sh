#!/bin/bash
# run_gpu_shards.sh [start|stop] [WORKERS]: one search shard per GPU
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if [ "$1" = stop ]; then pkill -f "python search.py"; sleep 3; pkill -9 -f "python search.py"; rm -rf /tmp/exo_dl; exit 0; fi
NG=$(nvidia-smi -L | wc -l); NW=${2:-24}
for g in $(seq 0 $((NG - 1))); do CUDA_VISIBLE_DEVICES=$g SHARD=$g/$NG setsid nohup python search.py allsky_targets.csv results $NW >> run_gpu$g.log 2>&1 < /dev/null & done
