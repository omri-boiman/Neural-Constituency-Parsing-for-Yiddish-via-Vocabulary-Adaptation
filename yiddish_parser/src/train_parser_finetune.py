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
OUTPUT_DIR = './output/phase3_parser_finetune_12000'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'yiddish_parser_ft.pt')

os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = 'data/processed/supar_ready'
ENCODER_PATH = './output/phase2_trained/checkpoint-12000' 
# =================================================

def apply_smart_freeze_patch(n_layers_to_unfreeze=6):
    """
    Logic:
    1. Freeze EVERYTHING initially.
    2. Find the layer list using the correct 'supar' path.
    3. UNFREEZE the top N layers explicitly.
    """
    print(f"\n🐒 APPLYING SMART PARTIAL-UNFREEZE PATCH (Top {n_layers_to_unfreeze} Layers)...")
    OriginalModel = CRFConstituencyParser.MODEL
    
    class SmartFrozenModel(OriginalModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            print(f"❄️  INTERCEPTED: Configuring layers in {self.__class__.__name__}...")
            
            # 1. Freeze EVERYTHING first (Safety first!)
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # 2. Identify the Encoder Layers
            # SUPAR stores the HF model in 'self.encoder.bert'
            layers = None
            
            # Attempt 1: Standard SUPAR + XLM-R/RoBERTa structure
            if hasattr(self.encoder, 'bert') and hasattr(self.encoder.bert, 'encoder') and hasattr(self.encoder.bert.encoder, 'layer'):
                layers = self.encoder.bert.encoder.layer
            # Attempt 2: Direct path (some versions)
            elif hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
                layers = self.encoder.encoder.layer
            # Attempt 3: BERT structure
            elif hasattr(self.encoder, 'bert') and hasattr(self.encoder.bert, 'encoder') and hasattr(self.encoder.bert.encoder, 'bert_layer_groups'):
                 layers = self.encoder.bert.encoder.bert_layer_groups
                
            if layers is not None:
                total_layers = len(layers)
                start_unfreeze = total_layers - n_layers_to_unfreeze
                
                print(f"   -> Found {total_layers} layers. Unfreezing from layer {start_unfreeze} to {total_layers-1}.")
                
                # 3. Unfreeze the specific top layers
                unfrozen_params = 0
                for i in range(start_unfreeze, total_layers):
                    print(f"      🔓 Unfreezing Layer {i}")
                    for param in layers[i].parameters():
                        param.requires_grad = True
                        unfrozen_params += param.numel()
                
                print(f"   ✅ SUCCESS: {unfrozen_params/1e6:.2f}M Encoder parameters are TRAINABLE.")
                print(f"   🔒 Embeddings and Bottom layers remain FROZEN.")
            else:
                # DEBUGGING: Print structure if it fails
                print("   ❌ ERROR: Could not locate transformer layers structure.")
                print("   ⚠️ DEBUG: Available attributes in encoder:", dir(self.encoder))
                if hasattr(self.encoder, 'bert'):
                    print("   ⚠️ DEBUG: Available attributes in encoder.bert:", dir(self.encoder.bert))

    CRFConstituencyParser.MODEL = SmartFrozenModel
    print("✅ Patch Applied.\n")

def train():
    # 🔓 UNFREEZE TOP 6 LAYERS
    apply_smart_freeze_patch(n_layers_to_unfreeze=6)

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
        
        # 🟢 STRATEGY 3: PARTIAL FINE-TUNE
        'finetune': True,             
        'n_bert_layers': 12,
        'bert_pooling': 'scalar_mix', 
        
        # 🟢 AGGRESSIVE DROPOUT
        'encoder_dropout': 0.5,
        'mlp_dropout': 0.5,
        'embed_dropout': 0.33,
        
        # 🟢 TRAINING DYNAMICS
        'lr': 5e-5,
        'lr_rate': 20,
        'batch_size': 1500,           
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
    
    print(f"🚀 Starting PARTIAL FINE-TUNE Training for: {ENCODER_PATH}")
    print(f"💾 Saving model to: {MODEL_FILE}")
    
    config = Config(**args)
    parser = CRFConstituencyParser.build(**config)
    parser.train(**config)
    print("🎉 Training Complete.")

if __name__ == "__main__":
    train()