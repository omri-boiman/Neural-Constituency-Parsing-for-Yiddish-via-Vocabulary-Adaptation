import os
import sys
import torch
import transformers

# =========================================================
# 🩹 PATCH: Inject AdamW
# =========================================================
transformers.AdamW = torch.optim.AdamW
try:
    from transformers import optimization
    if not hasattr(optimization, 'AdamW'):
        optimization.AdamW = torch.optim.AdamW
except ImportError:
    pass
# =========================================================

import supar
try:
    from supar.utils import Config
except ImportError:
    from supar.config import Config

from supar import CRFConstituencyParser

# ================= CONFIGURATION =================
# 🔴 CONTROL GROUP: The Raw Model (No Adaptation)
ENCODER_PATH = 'skulick/xlmb-ybc-ck05' 

# Save to a separate folder for comparison
OUTPUT_DIR = './output/kulick_raw_baseline'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'kulick_raw.pt')

# Make the folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = 'data/processed/supar_ready'
# =================================================

def freeze_recursive(module):
    frozen_count = 0
    name = module.__class__.__name__
    if 'Embeddings' in name and 'Bert' in name: 
        print(f"      Found Candidate: {name}")
        for param in module.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        return frozen_count

    for child in module.children():
        frozen_count += freeze_recursive(child)
    return frozen_count

def apply_freeze_patch():
    print("\n🐒 APPLYING FREEZE PATCH...")
    OriginalModel = CRFConstituencyParser.MODEL
    
    class FrozenModel(OriginalModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            print(f"❄️  INTERCEPTED: Searching for embeddings in {self.__class__.__name__}...")
            
            count = 0
            if hasattr(self, 'encoder'):
                count = freeze_recursive(self.encoder)
            
            if count == 0:
                print("   ⚠️ Encoder scan empty. Trying generic scan...")
                for m in self.modules():
                    if 'Embeddings' in m.__class__.__name__:
                         for p in m.parameters():
                             p.requires_grad = False
                             count += p.numel()

            if count > 0:
                print(f"   ✅ SUCCESS: Frozen {count/1e6:.2f}M parameters.")
            else:
                print("   ❌ ERROR: Could not find any embeddings to freeze!")

    CRFConstituencyParser.MODEL = FrozenModel
    print("✅ Patch Applied.\n")

def train():
    apply_freeze_patch()

    args = {
        'train': os.path.join(DATA_DIR, 'train.txt'),
        'dev': os.path.join(DATA_DIR, 'dev.txt'),
        'test': os.path.join(DATA_DIR, 'test.txt'),
        'path': MODEL_FILE,
        'mode': 'train',
        'build': True,
        'checkpoint': False, 
        'encoder': 'bert',
        'bert': ENCODER_PATH,
        'finetune': False,
        'n_bert_layers': 4,
        'bert_pooling': 'mean',
        'mix_dropout': 0.0,
        'feat': ['char'],
        'n_embed': 100,
        'n_char_embed': 50,
        'n_char_hidden': 100,
        'n_feat_embed': 100,
        'n_encoder_hidden': 800,
        'n_encoder_layers': 3,
        'encoder_dropout': 0.33,
        'n_span_mlp': 500,
        'n_label_mlp': 100,
        'mlp_dropout': 0.33,
        'embed_dropout': 0.33,
        'lr': 5e-5,
        'lr_rate': 20,
        'batch_size': 2000,
        'epochs': 100,
        'patience': 10,
        'update_steps': 1,
        'warmup': 0.1,
        'clip': 5.0,
        'decay': 0.75,
        'decay_steps': 5000,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'eps': 1e-8,
        'mu': 0.9,
        'nu': 0.9,
        'structure': 'joint',
        'mbr': True,
        'delete': {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
        'equal': {'ADVP': 'PRT'},
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 1,
        'amp': False,
        'cache': False,
        'verbose': True,
        'punct': False,
        'buckets': 32,
        'workers': 0 
    }
    
    print(f"🚀 Starting BASELINE Training (Raw Kulick Model): {ENCODER_PATH}")
    print(f"💾 Saving model to: {MODEL_FILE}")
    
    config = Config(**args)
    print("   -> Building Parser instance...")
    parser = CRFConstituencyParser.build(**config)
    
    print("   -> Starting Training Loop...")
    parser.train(**config)
    print("🎉 Training Complete.")

if __name__ == "__main__":
    train()