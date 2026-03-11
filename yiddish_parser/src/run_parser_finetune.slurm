#!/bin/bash
#SBATCH --job-name=yiddish_ft
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=studentkillable   
#SBATCH --gpus=1
#SBATCH --time=4:00:00               
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# --- SETUP ---
BASE_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/omriboiman"
export HF_HOME="$BASE_DIR/cache"
export TORCH_HOME="$BASE_DIR/cache"

# Enable memory fragmentation fix (Helps avoid OOM on fine-tuning)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$BASE_DIR/anaconda3/etc/profile.d/conda.sh"
conda activate yiddish

# --- RUN ---
echo "🚀 Running Partial Fine-Tuning Strategy (Top 6 Layers)..."
python src/train_parser_finetune.py

echo "✅ Job Finished."