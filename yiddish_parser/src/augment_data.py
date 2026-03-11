import os
import random
from collections import defaultdict
import nltk
from nltk import Tree

# ================= CONFIGURATION =================
INPUT_FILE  = 'data/processed/supar_ready/train.txt'
OUTPUT_FILE = 'data/processed/supar_ready/train_aug.txt'
AUGMENT_PCT = 0.3  # Augment 30% of the trees
SWAP_PROB   = 0.2  # Inside an augmented tree, swap 20% of phrases
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

def get_subtrees_by_label(trees):
    """Harvests valid phrases from the Gold Data."""
    buckets = defaultdict(list)
    for tree in trees:
        for subtree in tree.subtrees():
            # SAFETY FILTER: Only swap short phrases (1-3 words)
            if isinstance(subtree, Tree):
                # Count leaves safely
                num_leaves = len(subtree.leaves())
                if num_leaves <= 3 and subtree.height() <= 3:
                    label = subtree.label()
                    buckets[label].append(subtree)
    return buckets

def augment_tree(tree, buckets):
    """Creates a Frankenstein tree."""
    # Work on a copy so we don't destroy the original
    new_tree = tree.copy(deep=True)
    
    # Traverse and potentially swap
    for subtree in new_tree.subtrees():
        # Skip leaf nodes (which are strings, not Trees)
        if not isinstance(subtree, Tree):
            continue

        num_leaves = len(subtree.leaves())
        
        if num_leaves <= 3 and subtree.height() <= 3:
            if random.random() < SWAP_PROB:
                label = subtree.label()
                if buckets[label]:
                    # Pick a donor organ
                    donor = random.choice(buckets[label])
                    
                    # Perform the transplant
                    subtree.clear()
                    
                    # 🩹 THE FIX: Check if child is Tree or String before copying
                    new_children = []
                    for child in donor:
                        if isinstance(child, Tree):
                            new_children.append(child.copy(deep=True))
                        else:
                            new_children.append(child) # Strings don't need copying
                    
                    subtree.extend(new_children)
                    
    return new_tree

def main():
    print("🧟 Starting Frankenstein Augmentation...")
    print(f"   Input: {INPUT_FILE}")
    
    # 1. Load Gold Data
    gold_trees = load_trees(INPUT_FILE)
    print(f"   Loaded {len(gold_trees)} trees.")
    
    # 2. Harvest Organs (Subtrees)
    print("   Harvesting valid subtrees...")
    buckets = get_subtrees_by_label(gold_trees)
    
    # Check if we found anything
    count = len(buckets.get('NP', []))
    print(f"   Found {count} noun phrases to swap.")

    # 3. Create Augmented Data
    aug_trees = []
    print(f"   Generating augmented trees (Target: {int(len(gold_trees) * AUGMENT_PCT)} new trees)...")
    
    for tree in gold_trees:
        # Always keep the original!
        aug_trees.append(tree)
        
        # Sometimes create a mutant twin
        if random.random() < AUGMENT_PCT:
            try:
                mutant = augment_tree(tree, buckets)
                aug_trees.append(mutant)
            except Exception as e:
                # If a specific tree fails, just skip it and move on
                continue
            
    # 4. Save
    print(f"   Saving {len(aug_trees)} trees to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for tree in aug_trees:
            # Convert back to one-line bracketed string
            # _pformat_flat is internal, using standard format
            f.write(tree.pformat(margin=float("inf")) + "\n")
            
    print("✅ Done. New dataset is ready.")

if __name__ == "__main__":
    main()