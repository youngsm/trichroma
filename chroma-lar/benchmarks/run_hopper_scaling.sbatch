#!/bin/bash
#SBATCH --job-name=chroma-hopper-smoke
#SBATCH --partition=hopper
#SBATCH --account=neutrino:ml-dev
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/triton_chroma/chroma-lar/benchmarks/hopper-smoke-%j.out

set -euo pipefail

workspace=/sdf/group/neutrino/youngsam/sim/triton_chroma
python_bin=/sdf/home/y/youngsam/sw/dune/.conda/envs/py310_torch/bin/python
cache_root=/tmp/chroma-hopper-${SLURM_JOB_ID}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${workspace}/chroma-lite:${workspace}/chroma-lar"
export XDG_CACHE_HOME="${cache_root}/xdg"
export TRITON_CACHE_DIR="${cache_root}/triton"

mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}"
cd "${workspace}/chroma-lar"

nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
"${python_bin}" benchmarks/benchmark_triton_scaling.py \
  --counts 500000 1000000 2000000 5000000 \
  --replicates 3 \
  --warmup-photons 131072 \
  --json benchmarks/throughput_scaling_hopper_smoke.json
