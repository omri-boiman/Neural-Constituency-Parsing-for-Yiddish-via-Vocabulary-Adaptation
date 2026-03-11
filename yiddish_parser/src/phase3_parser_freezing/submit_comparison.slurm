#!/bin/bash
#SBATCH --job-name=kulick_baseline
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --partition=studentkillable   
#SBATCH --gpus=1
#SBATCH --time=12:00:00               
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=USR1@90              

# --- 1. TAU STORAGE SETUP ---
BASE_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/omriboiman"
export HF_HOME="$BASE_DIR/cache"
export XDG_CACHE_HOME="$BASE_DIR/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 2. ENVIRONMENT ---
source "$BASE_DIR/anaconda3/etc/profile.d/conda.sh"
conda activate yiddish

echo "🚀 Starting Kulick Baseline Job on $(hostname)"
echo "📅 Date: $(date)"

# --- 3. RUN THE PYTHON SCRIPT ---
# This is the only line that changes from your parser script
python src/train_parser_base.py

echo "🎉 Job Finished."