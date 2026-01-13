import os
import shutil
from transformers import AutoTokenizer, AutoModelForMaskedLM

# --- CONFIGURATION ---
ORIGINAL_MODEL = "skulick/xlmb-ybc-ck05"
VOCAB_FILE = "./data/processed/yiddish_vocab.txt"
OUTPUT_DIR = "./output/phase1_expanded_model"
TARGET_NEW_TOKENS = 2000 

# 🛡️ SAFETY: Force the download to go to the Big Drive
# We create a specific folder for cache so it never touches your Home quota
CACHE_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/omriboiman/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def inject_vocab():
    print(f" Loading YBC model: {ORIGINAL_MODEL}")
    print(f"   (Downloading to safe storage: {CACHE_DIR})")
    
    try:
        # We explicitly tell it WHERE to download
        tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL, cache_dir=CACHE_DIR)
        model = AutoModelForMaskedLM.from_pretrained(ORIGINAL_MODEL, cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 1. Load the Gold Standard Vocab
    if not os.path.exists(VOCAB_FILE):
        print(f"❌ Error: Vocab file not found at {VOCAB_FILE}")
        return

    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        jochre_words = [line.strip() for line in f if line.strip()]
    
    print(f"   Loaded {len(jochre_words)} candidate words from Jochre.")

    # 2. Find Missing Tokens
    new_tokens_to_add = []
    print("   Scanning for fragmented tokens...")
    
    for word in jochre_words:
        tokens = tokenizer.tokenize(word)
        if len(tokens) > 1:
            new_tokens_to_add.append(word)
        if len(new_tokens_to_add) >= TARGET_NEW_TOKENS:
            break
            
    print(f"   Found {len(new_tokens_to_add)} important words to inject.")
    
    # 3. Inject into Tokenizer
    if new_tokens_to_add:
        num_added = tokenizer.add_tokens(new_tokens_to_add)
        print(f"✅ Added {num_added} new tokens to the tokenizer.")
        
        # 4. Resize Model Embeddings
        model.resize_token_embeddings(len(tokenizer))
        
        # 5. Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"💾 Saved expanded model to: {OUTPUT_DIR}")
    else:
        print("⚠️ No new tokens needed!")

if __name__ == "__main__":
    inject_vocab()