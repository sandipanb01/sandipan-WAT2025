# ============================================================
# 0. IMPORTS
# ============================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import evaluate
import re
from lxml import etree
from collections import Counter
from pathlib import Path
import shutil

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
    "gemma_3-4B_it": "google/gemma-3-4b-it",
}

DATA_ROOT = "localization-xml-mt"
LANG_PAIRS = ["ende", "enfr", "ennl", "enfi", "enru"]

# Full language names for the prompt
LANG_MAP = {
    "ende": "German",
    "enfr": "French",
    "ennl": "Dutch",
    "enfi": "Finnish",
    "enru": "Russian",
}

BATCH_SIZE = 2
MAX_NEW_TOKENS = 512
OUTPUT_FOLDER = "salesforce_eval_outputs_4B_V1"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# =======================
# SANITY TEST TOGGLE
# =======================
SANITY_TEST = True           # <- SET False FOR FULL RUN
SANITY_SAMPLES = 100         # <- fast smoke test size

# ============================================================
# 3. LOAD SALESFORCE DATA
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
    base = os.path.join(root, "data", lang_pair)
    src_file = os.path.join(base, f"{lang_pair}_en_dev.json")
    tgt_file = os.path.join(base, f"{lang_pair}_{lang_pair[2:]}_dev.json")

    with open(src_file, encoding="utf-8") as f:
        src_json = json.load(f)
    with open(tgt_file, encoding="utf-8") as f:
        tgt_json = json.load(f)

    # Data lives under the "text" key as {id: sentence} dict
    src_texts = [normalize_salesforce_entry(v) for v in src_json["text"].values()]
    tgt_texts = [normalize_salesforce_entry(v) for v in tgt_json["text"].values()]
    return src_texts, tgt_texts

# ============================================================
# 4. PROMPT
# Gemma-3 instruction-tuned models use the chat template:
#   <start_of_turn>user\n{message}<end_of_turn>\n
#   <start_of_turn>model\n
# Paper (Appendix A.3) IFT template (0-shot):
#   "Translate the following sentence from E to F.
#    The translation should be in F and no other language.
#    E: [ S ]
#    F: [ T ]"
# We follow the paper's instruction wording inside Gemma's chat format.
# ============================================================
def build_prompt(src, tgt_lang):
    instruction = (
        f"Translate the following sentence from English to {tgt_lang}. "
        f"The translation should be in {tgt_lang} and no other language.\n\n"
        f"English: {src}\n"
        f"{tgt_lang}:"
    )
    return (
        f"<start_of_turn>user\n"
        f"{instruction}"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

# ============================================================
# 5. LOAD METRICS
# Paper uses sacrebleu (not multi-bleu), chrF, and XML-specific metrics.
# ============================================================
bleu_metric  = evaluate.load("bleu")
chrf_metric  = evaluate.load("chrf")
chrf2_metric = evaluate.load("chrf")

# ============================================================
# 6. XML METRICS (paper Section 4.2 and Appendix B)
#
# XML-Match: percentage of predictions whose XML structure
#            exactly matches the reference structure.
#
# XML-chrF:  chrF computed only on structure-matching pairs;
#            score is set to 0 for mismatched pairs (hard penalty).
# ============================================================

def get_xml_structure(text):
    """
    Parse XML structure of a string.
    Returns a nested tuple of (tag, [children]) or None if invalid XML.
    Text content is ignored — only tag hierarchy is captured.
    """
    try:
        root = etree.fromstring(f"<root>{text}</root>")
        def structure(el):
            return (el.tag, [structure(c) for c in el])
        return structure(root)
    except Exception:
        return None

def compute_xml_match(predictions, references):
    """
    XML-Match (paper Appendix B):
    Percentage of outputs that have exactly the same XML structure
    as their references. Returns a float in [0, 1].
    """
    matches = 0
    for pred, ref in zip(predictions, references):
        pred_struct = get_xml_structure(pred)
        ref_struct  = get_xml_structure(ref)
        if pred_struct == ref_struct:
            matches += 1
    return matches / len(predictions) if predictions else 0.0

def compute_xml_chrf(predictions, references):
    """
    XML-chrF (paper Appendix B):
    - If XML structures match: compute chrF(beta=1) on the full strings.
    - If XML structures do NOT match: score = 0 (hard penalty).
    Returns the mean score across all samples.
    """
    scores = []
    for pred, ref in zip(predictions, references):
        pred_struct = get_xml_structure(pred)
        ref_struct  = get_xml_structure(ref)
        if pred_struct != ref_struct:
            scores.append(0.0)
        else:
            score = chrf_metric.compute(
                predictions=[pred],
                references=[ref],
                beta=1
            )["score"]
            scores.append(score)
    return float(np.mean(scores)) if scores else 0.0

# ============================================================
# 7. DECODE HELPER
# Strips the prompt prefix from the model output so we only
# keep the generated translation.
# ============================================================
def strip_prompt(decoded_str, prompt_str):
    """Remove the prompt from decoded output, return only the generation."""
    if decoded_str.startswith(prompt_str):
        return decoded_str[len(prompt_str):]
    # Fallback: try to find 'model\n' turn marker
    marker = "<start_of_turn>model\n"
    idx = decoded_str.rfind(marker)
    if idx != -1:
        return decoded_str[idx + len(marker):]
    return decoded_str

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
        print(f"⚡ SANITY MODE: {SANITY_SAMPLES} samples")
    else:
        print(f"Full run: {len(src_texts)} samples")

    tgt_lang = LANG_MAP[lang_pair]

    dataset = Dataset.from_dict({"src": src_texts, "ref": tgt_texts})

    for model_key, model_name in BASE_MODELS.items():
        print(f"\n→ Model: {model_key}  ({model_name})")

        # ---- Load tokenizer ----
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"   # left-pad for decoder-only generation

        # ---- Load model ----
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="balanced",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model.eval()

        predictions, references = [], []

        for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc=f"{lang_pair}"):
            batch   = dataset[i : i + BATCH_SIZE]
            prompts = [build_prompt(s, tgt_lang) for s in batch["src"]]

            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,          # greedy (paper Section 4.2)
                    repetition_penalty=1.1,
                )

            # Decode only the newly generated tokens (not the prompt)
            input_lengths = inputs["input_ids"].shape[1]
            new_tokens    = outputs[:, input_lengths:]
            decoded       = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            predictions.extend([d.strip() for d in decoded])
            references.extend(batch["ref"])

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

        # Standard BLEU (sacrebleu via evaluate)
        bleu_score = bleu_metric.compute(
            predictions=predictions,
            references=references,
        )["bleu"]

        # Plain chrF (beta=1) and chrF++ (beta=2) — extra diagnostics
        chrf_score = chrf_metric.compute(
            predictions=predictions,
            references=references,
            beta=1,
        )["score"]

        chrf2_score = chrf2_metric.compute(
            predictions=predictions,
            references=references,
            beta=2,
        )["score"]

        # Paper primary metrics (Appendix B)
        xml_match = compute_xml_match(predictions, references)
        xml_chrf  = compute_xml_chrf(predictions, references)

        print(f"{'─'*40}")
        print(f"  BLEU        : {bleu_score * 100:.2f}")
        print(f"  chrF (β=1)  : {chrf_score:.2f}")
        print(f"  chrF++ (β=2): {chrf2_score:.2f}")
        print(f"  XML-chrF    : {xml_chrf:.2f}   ← paper primary metric")
        print(f"  XML-Match   : {xml_match * 100:.2f}%  ← paper primary metric")
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

# ============================================================
# 10. ZIP RESULTS
# ============================================================
shutil.make_archive(OUTPUT_FOLDER, "zip", OUTPUT_FOLDER)
print(f"\n✅ DONE → {OUTPUT_FOLDER}.zip")
