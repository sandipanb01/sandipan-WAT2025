# ============================================================
# 0. IMPORTS
# ============================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
import evaluate
from lxml import etree
from pathlib import Path
import shutil
from datetime import datetime

# ============================================================
# 1. SEED & ENV
# ============================================================
set_seed(42)
torch.set_grad_enabled(False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 2. CONFIG
# ============================================================
BASE_MODELS = {
    "translategemma_4b_it": "google/translategemma-4b-it",
}

DATA_ROOT  = "localization-xml-mt"
LANG_PAIRS = ["ende", "enfr", "ennl", "enfi", "enru"]

# TranslateGemma requires ISO 639-1 language codes
LANG_CODE_MAP = {
    "ende": "de",
    "enfr": "fr",
    "ennl": "nl",
    "enfi": "fi",
    "enru": "ru",
}

LANG_NAME_MAP = {
    "ende": "German",
    "enfr": "French",
    "ennl": "Dutch",
    "enfi": "Finnish",
    "enru": "Russian",
}

BATCH_SIZE     = 1             # process one at a time (required for TranslateGemma)
MAX_NEW_TOKENS = 512
OUTPUT_FOLDER  = "salesforce_eval_outputs_translategemma"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# =======================
# SANITY TEST TOGGLE
# =======================
SANITY_TEST    = True    # <- SET False FOR FULL RUN
SANITY_SAMPLES = 10      # <- fast smoke test size

# ============================================================
# 3. LOGGING — output goes to terminal AND log file
# ============================================================
log_file_path = f"{OUTPUT_FOLDER}/run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile  = open(filepath, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

sys.stdout = Logger(log_file_path)
print(f"Logging to  : {log_file_path}")
print(f"Run started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================
# 4. LOAD SALESFORCE DATA
# ============================================================
def normalize_salesforce_entry(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if "text" in v:
            return v["text"]
        if "segments" in v:
            return "".join(seg.get("text", "") for seg in v["segments"])
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def load_dev_as_test(root, lang_pair):
    """
    Loads the dev set as the test set (paper Section 4.1):
    'We use the development set of 2,000 sentence pairs as the test set
     because the test set is hidden.'
    """
    base     = os.path.join(root, "data", lang_pair)
    src_file = os.path.join(base, f"{lang_pair}_en_dev.json")
    tgt_file = os.path.join(base, f"{lang_pair}_{lang_pair[2:]}_dev.json")

    with open(src_file, encoding="utf-8") as f:
        src_json = json.load(f)
    with open(tgt_file, encoding="utf-8") as f:
        tgt_json = json.load(f)

    src_texts = [normalize_salesforce_entry(v) for v in src_json["text"].values()]
    tgt_texts = [normalize_salesforce_entry(v) for v in tgt_json["text"].values()]
    return src_texts, tgt_texts

# ============================================================
# 5. PROMPT
# TranslateGemma official structured message format.
# Uses apply_chat_template with tokenize=True + return_dict=True
# in ONE step — this is the only approach that works correctly.
# ============================================================
def build_messages(src_text, tgt_lang_code):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "en",
                    "target_lang_code": tgt_lang_code,
                    "text": src_text,
                }
            ],
        }
    ]

# ============================================================
# 6. LOAD METRICS
# Paper uses sacrebleu (not multi-bleu), chrF, and XML-specific metrics.
# ============================================================
bleu_metric  = evaluate.load("bleu")
chrf_metric  = evaluate.load("chrf")
chrf2_metric = evaluate.load("chrf")

# ============================================================
# 7. XML METRICS (paper Section 4.2 and Appendix B)
# XML-Match: % predictions with exactly same XML structure as reference
# XML-chrF:  chrF if structures match, else 0 (hard penalty)
# ============================================================
def get_xml_structure(text):
    try:
        root = etree.fromstring(f"<root>{text}</root>")
        def structure(el):
            return (el.tag, [structure(c) for c in el])
        return structure(root)
    except Exception:
        return None

def compute_xml_match(predictions, references):
    matches = 0
    for pred, ref in zip(predictions, references):
        if get_xml_structure(pred) == get_xml_structure(ref):
            matches += 1
    return matches / len(predictions) if predictions else 0.0

def compute_xml_chrf(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        if get_xml_structure(pred) != get_xml_structure(ref):
            scores.append(0.0)
        else:
            score = chrf_metric.compute(
                predictions=[pred], references=[ref], beta=1
            )["score"]
            scores.append(score)
    return float(np.mean(scores)) if scores else 0.0

# ============================================================
# 8. MAIN LOOP
# ============================================================
all_results = []

for lang_pair in LANG_PAIRS:
    print(f"\n{'='*60}")
    print(f"  Language pair: {lang_pair.upper()}")
    print(f"{'='*60}")

    src_texts, tgt_texts = load_dev_as_test(DATA_ROOT, lang_pair)

    # Sanity slice
    if SANITY_TEST:
        src_texts = src_texts[:SANITY_SAMPLES]
        tgt_texts = tgt_texts[:SANITY_SAMPLES]
        print(f"SANITY MODE: {SANITY_SAMPLES} samples")
    else:
        print(f"Full run: {len(src_texts)} samples")

    tgt_lang_code = LANG_CODE_MAP[lang_pair]
    dataset = Dataset.from_dict({"src": src_texts, "ref": tgt_texts})

    for model_key, model_name in BASE_MODELS.items():
        print(f"\n-> Model: {model_key}  ({model_name})")

        # ---- Load processor ----
        processor = AutoProcessor.from_pretrained(model_name)

        # ---- Load model ----
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map={"": 0},
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model.eval()

        predictions, references = [], []

        for i in tqdm(range(0, len(src_texts)), desc=f"{lang_pair}"):
            src = src_texts[i]
            ref = tgt_texts[i]

            # Official single-step tokenization (tokenize=True, return_dict=True)
            # This is the ONLY method that works for TranslateGemma
            inputs = processor.apply_chat_template(
                build_messages(src, tgt_lang_code),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,          # greedy (paper Section 4.2)
                )

            # Decode only the newly generated tokens (not the prompt)
            input_lengths = inputs["input_ids"].shape[1]
            new_tokens    = outputs[:, input_lengths:]
            decoded       = processor.decode(new_tokens[0], skip_special_tokens=True)

            predictions.append(decoded.strip())
            references.append(ref)

            torch.cuda.empty_cache()

        # ---- Print first 3 examples for sanity check ----
        print("\n--- Sample predictions ---")
        for k in range(min(3, len(predictions))):
            print(f"  SRC : {src_texts[k]}")
            print(f"  REF : {references[k]}")
            print(f"  PRED: {predictions[k]}")
            print()

        # ====================================================
        # METRICS
        # ====================================================
        predictions_safe = [p if p.strip() else "EMPTY" for p in predictions]

        # Standard BLEU (sacrebleu via evaluate)
        bleu_score = bleu_metric.compute(
            predictions=predictions_safe,
            references=references,
        )["bleu"]

        # Plain chrF (beta=1) and chrF++ (beta=2) - extra diagnostics
        chrf_score = chrf_metric.compute(
            predictions=predictions_safe,
            references=references,
            beta=1,
        )["score"]

        chrf2_score = chrf2_metric.compute(
            predictions=predictions_safe,
            references=references,
            beta=2,
        )["score"]

        # Paper primary metrics (Appendix B)
        xml_match = compute_xml_match(predictions_safe, references)
        xml_chrf  = compute_xml_chrf(predictions_safe, references)

        print(f"{'─'*40}")
        print(f"  BLEU        : {bleu_score * 100:.2f}")
        print(f"  chrF (b=1)  : {chrf_score:.2f}")
        print(f"  chrF++ (b=2): {chrf2_score:.2f}")
        print(f"  XML-chrF    : {xml_chrf:.2f}   <- paper primary metric")
        print(f"  XML-Match   : {xml_match * 100:.2f}%  <- paper primary metric")
        print(f"{'─'*40}")

        # ====================================================
        # SAVE per-sample predictions
        # ====================================================
        out_jsonl = f"{OUTPUT_FOLDER}/{lang_pair}_{model_key}.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for s, r, p in zip(src_texts, references, predictions):
                json.dump({"src": s, "ref": r, "pred": p}, f, ensure_ascii=False)
                f.write("\n")

        # ====================================================
        # SAVE metrics CSV
        # ====================================================
        row = {
            "lang_pair"  : lang_pair,
            "model"      : model_key,
            "BLEU"       : round(bleu_score * 100, 2),
            "chrF"       : round(chrf_score, 2),
            "chrF++"     : round(chrf2_score, 2),
            "XML_chrF"   : round(xml_chrf, 2),
            "XML_Match"  : round(xml_match * 100, 2),
            "sanity_mode": SANITY_TEST,
            "n_samples"  : len(predictions),
        }
        all_results.append(row)

        pd.DataFrame([row]).to_csv(
            f"{OUTPUT_FOLDER}/{lang_pair}_{model_key}_metrics.csv",
            index=False,
        )

        # Free GPU memory before next language pair
        del model
        torch.cuda.empty_cache()

# ============================================================
# 9. COMBINED RESULTS TABLE
# ============================================================
if all_results:
    combined_df = pd.DataFrame(all_results)
    combined_csv = f"{OUTPUT_FOLDER}/ALL_metrics_summary.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"\n{'='*60}")
    print("COMBINED RESULTS:")
    print(combined_df.to_string(index=False))
    print(f"{'='*60}")

print(f"\nRun finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# 10. ZIP RESULTS
# ============================================================
shutil.make_archive(OUTPUT_FOLDER, "zip", OUTPUT_FOLDER)
print(f"\nDONE -> {OUTPUT_FOLDER}.zip")
print(f"Full log saved -> {log_file_path}")