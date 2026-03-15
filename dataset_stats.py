# ============================================================
# DATASET STATISTICS (NO TOKENIZER) + COLOR TABLE + DOWNLOAD
# ============================================================

import pandas as pd
import numpy as np
from IPython.display import display

rows = []

def word_count(text):
    return len(text.split())

def char_count(text):
    return len(text)

for lang_pair in LANG_PAIRS:
    for split in SPLITS:

        src_texts, tgt_texts = load_split(DATA_ROOT, lang_pair, split)

        if not src_texts:
            continue

        src_chars = [char_count(t) for t in src_texts]
        tgt_chars = [char_count(t) for t in tgt_texts]

        src_words = [word_count(t) for t in src_texts]
        tgt_words = [word_count(t) for t in tgt_texts]

        rows.append({
            "Language Pair": lang_pair,
            "Split": split,
            "Sentence Pairs": len(src_texts),

            "Avg SRC Words": round(np.mean(src_words),2),
            "Avg TGT Words": round(np.mean(tgt_words),2),

            "Avg SRC Characters": round(np.mean(src_chars),2),
            "Avg TGT Characters": round(np.mean(tgt_chars),2),

            "Max SRC Characters": max(src_chars),
            "Max TGT Characters": max(tgt_chars)
        })

df_dataset = pd.DataFrame(rows)

# ============================================================
# COLORFUL TABLE
# ============================================================

styled_table = (
    df_dataset.style
    .set_caption("Salesforce Localization XML MT Dataset Statistics")
    .background_gradient(cmap="plasma")
    .set_properties(**{
        "text-align": "center",
        "border-color": "black"
    })
)

display(styled_table)

# ============================================================
# SAVE FILES
# ============================================================

csv_file = "salesforce_dataset_statistics.csv"
excel_file = "salesforce_dataset_statistics.xlsx"

df_dataset.to_csv(csv_file, index=False)
df_dataset.to_excel(excel_file, index=False)

print("✅ Saved dataset statistics files:")
print(csv_file)
print(excel_file)

# ============================================================
# AUTO DOWNLOAD (COLAB)
# ============================================================

try:
    from google.colab import files
    files.download(csv_file)
    files.download(excel_file)
except:
    print("📥 Files saved locally in working directory.")
