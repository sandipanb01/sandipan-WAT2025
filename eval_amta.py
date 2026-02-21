!pip install -q evaluate sacrebleu
import json
from pathlib import Path
import pandas as pd
import sacrebleu

# ======================================
# CONFIG
# ======================================
JSONL_FILES = [
    # Gemma 270M
    "ende_gemma_270m_it.jsonl",
    "enfi_gemma_270m_it.jsonl",
    "enfr_gemma_270m_it.jsonl",
    "ennl_gemma_270m_it.jsonl",
    "enru_gemma_270m_it.jsonl",

    # Gemma 3-4B
    "ende_gemma_3-4B_it.jsonl",
    "enfi_gemma_3-4B_it.jsonl",
    "enfr_gemma_3-4B_it.jsonl",
    "ennl_gemma_3-4B_it.jsonl",
    "enru_gemma_3-4B_it.jsonl",

    # Sarvam
    "ende_sarvam_translate.jsonl",
    "enfi_sarvam_translate.jsonl",
    "enfr_sarvam_translate.jsonl",
    "ennl_sarvam_translate.jsonl",
    "enru_sarvam_translate.jsonl",
]

OUTPUT_CSV = "xml_mt_results.csv"

# ======================================
# HELPERS
# ======================================
def load_preds_refs(jsonl_path):
    preds, refs = [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            preds.append(obj["pred"])
            refs.append(obj["ref"])
    return preds, refs


def compute_scores(preds, refs):
    refs = [refs]  # sacrebleu expects list of reference lists

    bleu = sacrebleu.corpus_bleu(preds, refs).score
    chrf = sacrebleu.corpus_chrf(preds, refs, beta=1).score
    chrf_pp = sacrebleu.corpus_chrf(preds, refs, beta=2).score

    # XML-chrF == chrF on raw XML strings (paper-faithful)
    xml_chrf = chrf
    xml_chrf_pp = chrf_pp

    return bleu, chrf, chrf_pp, xml_chrf, xml_chrf_pp


# ======================================
# RUN EVALUATION
# ======================================
rows = []

for file in JSONL_FILES:
    path = Path(file)
    if not path.exists():
        print(f"⚠️ Missing file, skipped: {file}")
        continue

    preds, refs = load_preds_refs(path)
    bleu, chrf, chrfpp, xmlc, xmlcpp = compute_scores(preds, refs)

    rows.append({
        "System": path.stem,
        "BLEU": bleu,
        "chrF": chrf,
        "chrF++": chrfpp,
        "XML-chrF": xmlc,
        "XML-chrF++": xmlcpp,
    })

df = pd.DataFrame(rows)

# ======================================
# SAVE CSV
# ======================================
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved results to {OUTPUT_CSV}")

# ======================================
# DISPLAY (COLORFUL TABLE)
# ======================================
numeric_cols = ["BLEU", "chrF", "chrF++", "XML-chrF", "XML-chrF++"]
styled = (
    df.style
    .background_gradient(
        cmap="viridis",
        subset=numeric_cols
    )
    .format("{:.2f}", subset=numeric_cols) # Apply format only to numeric columns
)

styled
