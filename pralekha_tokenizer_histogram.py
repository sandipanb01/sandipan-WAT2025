# ============================================================
# 📊 FULL PRALAKHA DATASET ANALYSIS (NO SAMPLING)
# Train / Dev / Test statistics + tokenizer length histograms
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json

# ------------------------------------------------------------
# 1) DATASET CONFIGURATION (OFFICIAL)
# ------------------------------------------------------------
DATASET_NAME = "ai4bharat/Pralekha"
LANG_PAIR = "eng_hin"

print("Loading Pralekha dataset (this may take time)...")

train = load_dataset(DATASET_NAME, "train", split=LANG_PAIR)
dev   = load_dataset(DATASET_NAME, "dev",   split=LANG_PAIR)
test  = load_dataset(DATASET_NAME, "test",  split=LANG_PAIR)

print("\n📦 DATASET SIZES")
print(f"Train size: {len(train):,}")
print(f"Dev   size: {len(dev):,}")
print(f"Test  size: {len(test):,}")

# ------------------------------------------------------------
# 2) TOKENIZER (MODEL UNDER STUDY)
# ------------------------------------------------------------
MODEL_ID = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ------------------------------------------------------------
# 3) TOKEN LENGTH COLLECTION
# ------------------------------------------------------------
def collect_lengths(dataset, split_name):
    src_lengths = []
    tgt_lengths = []

    print(f"\n🔍 Processing {split_name} split ({len(dataset):,} samples)...")

    for ex in tqdm(dataset):
        src_ids = tokenizer(
            ex["src_txt"],
            add_special_tokens=True,
            truncation=False
        )["input_ids"]

        tgt_ids = tokenizer(
            ex["tgt_txt"],
            add_special_tokens=True,
            truncation=False
        )["input_ids"]

        src_lengths.append(len(src_ids))
        tgt_lengths.append(len(tgt_ids))

    return np.array(src_lengths), np.array(tgt_lengths)

train_src, train_tgt = collect_lengths(train, "TRAIN")
dev_src,   dev_tgt   = collect_lengths(dev,   "DEV")
test_src,  test_tgt  = collect_lengths(test,  "TEST")

# ------------------------------------------------------------
# 4) STATISTICS
# ------------------------------------------------------------
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

stats = {
    "train_src": summarize(train_src),
    "train_tgt": summarize(train_tgt),
    "dev_src":   summarize(dev_src),
    "dev_tgt":   summarize(dev_tgt),
    "test_src":  summarize(test_src),
    "test_tgt":  summarize(test_tgt),
}

print("\n📐 SEQUENCE LENGTH STATISTICS (TOKENS)")
for k, v in stats.items():
    print(f"\n{k.upper()}")
    for kk, vv in v.items():
        print(f"  {kk:>6}: {vv}")

# Save stats for reproducibility
with open("pralekha_sequence_length_stats.json", "w") as f:
    json.dump(stats, f, indent=4)

# ------------------------------------------------------------
# 5) HISTOGRAMS (FULL DATA)
# ------------------------------------------------------------
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

plot_histogram(train_src, train_tgt, "TRAIN", "train_seq_len_hist.png")
plot_histogram(dev_src,   dev_tgt,   "DEV",   "dev_seq_len_hist.png")
plot_histogram(test_src,  test_tgt,  "TEST",  "test_seq_len_hist.png")

# ------------------------------------------------------------
# 6) TRUNCATION ANALYSIS (ADVISOR-IMPORTANT)
# ------------------------------------------------------------
def truncation_rate(lengths, max_len):
    return np.mean(lengths > max_len) * 100

MAX_INPUT_LEN  = stats["train_src"]["p95"]
MAX_OUTPUT_LEN = stats["train_tgt"]["p95"]

print("\n✂️ TRUNCATION ANALYSIS (using 95th percentile cutoffs)")
print(f"Max input length  (p95): {MAX_INPUT_LEN}")
print(f"Max output length (p95): {MAX_OUTPUT_LEN}")

print(f"Train SRC truncation rate: {truncation_rate(train_src, MAX_INPUT_LEN):.2f}%")
print(f"Train TGT truncation rate: {truncation_rate(train_tgt, MAX_OUTPUT_LEN):.2f}%")

print("\n✅ Analysis complete.")
print("Generated files:")
print("- pralekha_sequence_length_stats.json")
print("- train_seq_len_hist.png")
print("- dev_seq_len_hist.png")
print("- test_seq_len_hist.png")
print("\n🧠 PRACTICAL GUIDELINES (INTERPRETATION)")
print(f"- 95% of input sequences are ≤ {stats['train_src']['p95']} tokens")
print(f"- 95% of output sequences are ≤ {stats['train_tgt']['p95']} tokens")
print("- Recommended max_input_length  ≈ p95 (+ safety margin if needed)")
print("- Recommended max_output_length ≈ p95 (+ safety margin if needed)")
print("- Truncation beyond p95 may remove semantic content and affect BLEU/chrF")
