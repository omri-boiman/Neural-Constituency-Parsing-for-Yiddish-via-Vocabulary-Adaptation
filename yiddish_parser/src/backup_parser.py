import os
from huggingface_hub import login, HfApi

# --- 1. CONFIGURATION ---
HF_USERNAME = "Omriboiman"
REPO_NAME = f"{HF_USERNAME}/yiddish-constituency-parser"  # New Repo Name for the Parser
MY_TOKEN = "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" 

# Local Paths
PARSER_FILE = "./output/phase3_parser_model/yiddish_parser.pt"
TOKENIZER_DIR = "./output/phase2_trained/checkpoint-6000"

# --- 2. FORCE CACHE (Safety) ---
CACHE_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/omriboiman/cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

def backup():
    print("🔑 Logging in...")
    login(token=MY_TOKEN)
    api = HfApi()

    # 1. Validation
    if not os.path.exists(PARSER_FILE):
        print(f"❌ Error: Could not find parser file at {PARSER_FILE}")
        return
    if not os.path.exists(TOKENIZER_DIR):
        print(f"❌ Error: Could not find tokenizer folder at {TOKENIZER_DIR}")
        return

    # 2. Create Repo
    print(f"☁️  Creating Repository: {REPO_NAME}...")
    try:
        api.create_repo(repo_id=REPO_NAME, private=True, exist_ok=True)
    except Exception as e:
        print(f"   ⚠️ Repo might already exist or error: {e}")

    # 3. Upload Parser Model (.pt file)
    print("🚀 Uploading Parser Model (This might take a minute)...")
    try:
        api.upload_file(
            path_or_fileobj=PARSER_FILE,
            path_in_repo="yiddish_parser.pt",
            repo_id=REPO_NAME
        )
        print("   ✅ Parser Uploaded.")
    except Exception as e:
        print(f"   ❌ Failed to upload parser: {e}")

    # 4. Upload Tokenizer (Folder)
    # We upload the tokenizer contents to a subfolder called 'tokenizer'
    # so it doesn't clutter the main repo.
    print("🚀 Uploading Tokenizer files...")
    try:
        api.upload_folder(
            folder_path=TOKENIZER_DIR,
            path_in_repo="tokenizer", 
            repo_id=REPO_NAME
        )
        print("   ✅ Tokenizer Uploaded.")
    except Exception as e:
        print(f"   ❌ Failed to upload tokenizer: {e}")

    print("\n🎉 BACKUP COMPLETE!")
    print(f"👉 View your model here: https://huggingface.co/{REPO_NAME}")

if __name__ == "__main__":
    backup()