#!/bin/bash
#SBATCH --partition=gpu-a100,gpu-h100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_audio_dataset
#SBATCH --account=punim2907
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=padmanabhk@student.unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mem=100000
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=0-22:59:00

# Set the working directory to root of repository
cd /data/gpfs/projects/punim2907/paddy/code3/VisualCortex
export PYTHONPATH=$PWD

# Define variables
script="scripts/train.py"
data_dir="/data/gpfs/projects/punim2907/paddy/data"
output_dir="outputs/$(date +'%y_%m_%d')_${SLURM_JOB_ID}/"

# For now, don't need output dir 
# Check if the output directory exists, if not create it
if [ ! -d "${output_dir}" ]; then
    mkdir -p ${output_dir}
fi

echo "========== JOB METADATA =========="
echo "Date: $(date --iso-8601=seconds)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Working Dir: $PWD"
echo "Output Dir: ${output_dir}"
echo "Script: ${script}"
echo "Data Dir: ${data_dir}"
echo

echo "========== GIT INFO =========="
echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Git status:"
git status --short || true
echo

echo "========== ENVIRONMENT =========="
module load GCCcore/13.3.0
module load CUDA
module load Python/3.12.3
source .venv/bin/activate
# Specify port for wandb that is free
. /apps/resplat/helpers/spartan_helpers
PORT=$(find_port 35000 36000)
wandb server start --port "${PORT}"
echo "Wandb service port: ${PORT}"
echo

echo "========== RUN SCRIPT =========="
python3 "${script}" "${data_dir}" "${output_dir}"
echo

echo "========== RESOURCE USAGE =========="
my-job-stats -a -n -s
