# ============================================================
#   FULL SALESFORCE LOCALIZATION DATASET ANALYSIS
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer

# ------------------------------------------------------------
# 1) CONFIGURATION
# ------------------------------------------------------------
DATA_ROOT = "localization-xml-mt/data"  # root folder of Salesforce JSONs
LANG_PAIRS = ["ende", "enfr", "ennl", "enfi", "enru"]
SMOKE_TEST = False  # toggle True for quick test
SMOKE_SAMPLES = 100
MODEL_ID = "google/gemma-3-270m-it"  # tokenizer for stats

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ------------------------------------------------------------
# 2) HELPER FUNCTIONS
# ------------------------------------------------------------
def load_split(root, lang_pair, split):
    """Load Salesforce Localization dataset JSON split."""
    base = os.path.join(root, lang_pair)
    src_file = os.path.join(base, f"{lang_pair}_en_{split}.json")
    tgt_file = os.path.join(base, f"{lang_pair}_{lang_pair[2:]}_{split}.json")

    # Check for file existence before attempting to open
    if not (os.path.exists(src_file) and os.path.exists(tgt_file)):
        print(f"Warning: Skipping split '{split}' for '{lang_pair}' as one or both files are missing: {src_file}, {tgt_file}")
        return [], [] # Return empty lists if files not found

    with open(src_file, "r", encoding="utf-8") as f:
        src_json = json.load(f)
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_json = json.load(f)

    # Corrected: Extract only the actual text content from the 'text' key
    src_texts = list(src_json['text'].values())
    tgt_texts = list(tgt_json['text'].values())

    if SMOKE_TEST:
        src_texts = src_texts[:SMOKE_SAMPLES]
        tgt_texts = tgt_texts[:SMOKE_SAMPLES]

    return src_texts, tgt_texts

def collect_lengths(src_texts, tgt_texts, split_name):
    src_lengths = []
    tgt_lengths = []

    print(f"\n  Processing {split_name} split ({len(src_texts):,} samples)...")

    for src, tgt in tqdm(zip(src_texts, tgt_texts), total=len(src_texts)):
        src_ids = tokenizer(src, add_special_tokens=True, truncation=False)["input_ids"]
        tgt_ids = tokenizer(tgt, add_special_tokens=True, truncation=False)["input_ids"]

        src_lengths.append(len(src_ids))
        tgt_lengths.append(len(tgt_ids))

    return np.array(src_lengths), np.array(tgt_lengths)

def summarize(arr):
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": round(arr.mean(), 2),
        "median": int(np.median(arr)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
    }

def plot_histogram(src, tgt, title_prefix, filename):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(src, bins=120)
    plt.title(f"{title_prefix} – Source Token Lengths")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(tgt, bins=120)
    plt.title(f"{title_prefix} – Target Token Lengths")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def truncation_rate(lengths, max_len):
    return np.mean(lengths > max_len) * 100

# ------------------------------------------------------------
# 3) MAIN LOOP: ALL LANG_PAIRS AND SPLITS
# ------------------------------------------------------------
SPLITS = ["train", "dev", "test"]
all_stats = {}

for lang_pair in LANG_PAIRS:
    print(f"\n=== Processing LANG PAIR: {lang_pair} ===")
    all_stats[lang_pair] = {}

    for split in SPLITS:
        src_texts, tgt_texts = load_split(DATA_ROOT, lang_pair, split)
        # Only proceed to collect_lengths if texts were actually loaded
        if src_texts and tgt_texts:
            src_lengths, tgt_lengths = collect_lengths(src_texts, tgt_texts, f"{split.upper()}-{lang_pair}")

            # Summarize
            stats = {
                "src": summarize(src_lengths),
                "tgt": summarize(tgt_lengths)
            }
            all_stats[lang_pair][split] = stats

            print(f"\n  {split.upper()}-{lang_pair} STATS")
            print(f"Source tokens: {stats['src']}")
            print(f"Target tokens: {stats['tgt']}")

            # Plot histograms
            plot_histogram(src_lengths, tgt_lengths, f"{split.upper()}-{lang_pair}", f"{split}_{lang_pair}_seq_len_hist.png")

            # Truncation analysis using p95
            max_input_len = stats['src']['p95']
            max_output_len = stats['tgt']['p95']
            print("\n✂  TRUNCATION ANALYSIS (95th percentile cutoff)")
            print(f"Max input length (p95): {max_input_len}, truncation rate: {truncation_rate(src_lengths, max_input_len):.2f}%")
            print(f"Max output length (p95): {max_output_len}, truncation rate: {truncation_rate(tgt_lengths, max_output_len):.2f}%")
        else:
            # If no texts loaded, log that this split was skipped and continue
            print(f"\n  Skipping stats for {split.upper()}-{lang_pair} due to missing files.")

# ------------------------------------------------------------
# 4) SAVE ALL STATS
# ------------------------------------------------------------
with open("salesforce_token_stats.json", "w") as f:
    json.dump(all_stats, f, indent=4)

print("\n✅ Complete analysis for all language pairs done.")
print("Generated files: histograms PNGs + salesforce_token_stats.json")
