#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH --job-name=gpu_smoketest
#SBATCH --output=slurm-gpu-smoketest-%j.out

set -euo pipefail

module load anaconda/3
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
CONDA_ENV="${CONDA_ENV:-researchlab}"
conda activate "$CONDA_ENV"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
