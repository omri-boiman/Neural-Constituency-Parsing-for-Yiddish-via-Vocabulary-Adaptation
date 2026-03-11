import os
import random
from collections import defaultdict
import nltk
from nltk import Tree

# ================= CONFIGURATION =================
# Let's run this on your ORIGINAL train file to keep it clean
INPUT_FILE  = 'data/processed/supar_ready/train.txt'
OUTPUT_FILE = 'data/processed/supar_ready/train_lexical.txt'

# How many new sentences to generate? (e.g., 0.5 = 50% more data)
AUGMENT_PCT = 0.5  
# If we augment a tree, what is the chance we swap a specific word?
SWAP_PROB   = 0.3  

# The POS tags we want to swap (Based on standard Yiddish/Penn tags)
TARGET_TAGS = ['N', 'NPR', 'ADJ', 'VBF', 'VBN'] 
# =================================================

def load_trees(path):
    trees = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    trees.append(Tree.fromstring(line.strip()))
                except ValueError:
                    pass
    return trees

def harvest_vocabulary(trees):
    """Collects all words categorized by their POS tag."""
    vocab = defaultdict(list)
    for tree in trees:
        # tree.pos() returns a list of (word, pos_tag) tuples
        for word, pos_tag in tree.pos():
            # Clean up tags that might have extra info (like N-NOM -> N)
            clean_tag = pos_tag.split('-')[0] if '-' in pos_tag else pos_tag
            if clean_tag in TARGET_TAGS:
                vocab[clean_tag].append(word)
    
    # Remove duplicates to make random choice faster
    for tag in vocab:
        vocab[tag] = list(set(vocab[tag]))
    return vocab

def lexical_swap(tree, vocab):
    """Swaps words at the leaf level based on their POS tag."""
    new_tree = tree.copy(deep=True)
    
    # treepositions('leaves') gets the exact index path to every word
    for leaf_pos in new_tree.treepositions('leaves'):
        parent_pos = leaf_pos[:-1]     # Go up one level to the POS tag node
        parent_node = new_tree[parent_pos]
        pos_tag = parent_node.label()
        
        clean_tag = pos_tag.split('-')[0] if '-' in pos_tag else pos_tag
        
        # If it's a target tag and we roll a success
        if clean_tag in TARGET_TAGS and random.random() < SWAP_PROB:
            if vocab[clean_tag]:
                # Pick a random word of the same type
                new_word = random.choice(vocab[clean_tag])
                # Overwrite the original word
                new_tree[leaf_pos] = new_word
                
    return new_tree

def main():
    print("🔀 Starting Lexical Substitution Augmentation...")
    
    gold_trees = load_trees(INPUT_FILE)
    print(f"   Loaded {len(gold_trees)} original trees.")
    
    print("   Harvesting vocabulary...")
    vocab = harvest_vocabulary(gold_trees)
    for tag in TARGET_TAGS:
        print(f"      Found {len(vocab.get(tag, []))} unique words for {tag}")

    aug_trees = []
    print(f"   Generating augmented trees...")
    
    for tree in gold_trees:
        aug_trees.append(tree) # Keep original
        
        if random.random() < AUGMENT_PCT:
            mutant = lexical_swap(tree, vocab)
            aug_trees.append(mutant)
            
    print(f"   Saving {len(aug_trees)} trees to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for tree in aug_trees:
            f.write(tree.pformat(margin=float("inf")) + "\n")
            
    print("✅ Done! Your lexically augmented dataset is ready.")

if __name__ == "__main__":
    main()