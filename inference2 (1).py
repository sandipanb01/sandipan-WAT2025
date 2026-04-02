# ============================================================
# XML MT EVALUATION — STABLE VERSION (FIXED)
# ============================================================

import os
import json
import re
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from lxml import etree

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT   = "localization-xml-mt"
OUTPUT_DIR  = "xml_mt_eval_fixed"
BASE_MODEL  = "google/gemma-3-4b-it"
FINAL_MODEL = "./xml_mt_lora/checkpoint-15506"

LANG_PAIRS = ["ende", "enfr", "ennl", "enfi", "enru"]

LANG_NAME_MAP = {
    "ende": "German",
    "enfr": "French",
    "ennl": "Dutch",
    "enfi": "Finnish",
    "enru": "Russian",
}

BATCH_SIZE     = 4
MAX_NEW_TOKENS = 512

SANITY_TEST    = True
SANITY_SAMPLES = 100

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ============================================================
# GPU CLEANUP
# ============================================================

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# METRICS
# ============================================================

bleu_metric  = evaluate.load("bleu")
chrf_metric  = evaluate.load("chrf")
chrf2_metric = evaluate.load("chrf")

# ============================================================
# PROMPT (FIXED — NO CHAT TEMPLATE)
# ============================================================

def build_prompt(src, tgt_lang):
    instruction = (
        f"Translate the following XML document from English to {tgt_lang}.\n\n"
        f"English:\n{src}\n"
        f"{tgt_lang}:"
    )

    return (
        f"<bos><start_of_turn>user\n"
        f"{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

# ============================================================
# XML UTILITIES
# ============================================================

def extract_xml_tags(text):
    return re.findall(r"</?[^>]+>", text)

def fix_xml_tags(src, pred):
    """Non-destructive: ensure all source tags exist in prediction"""
    src_tags = extract_xml_tags(src)

    for tag in src_tags:
        if tag not in pred:
            pred += tag

    return pred

def normalize_xml(text):
    text = re.sub(r">\s+<", "><", text)
    return text.strip()

def get_xml_structure(text):
    try:
        text = normalize_xml(text)
        root = etree.fromstring(f"<root>{text}</root>")

        def structure(el):
            return (el.tag, [structure(c) for c in el])

        return structure(root)
    except Exception:
        return None

def compute_xml_match(predictions, references):
    matches = sum(
        1 for p, r in zip(predictions, references)
        if get_xml_structure(p) == get_xml_structure(r)
    )
    return matches / len(predictions) if predictions else 0.0

def compute_xml_chrf(predictions, references):
    scores = []
    for p, r in zip(predictions, references):
        if get_xml_structure(p) != get_xml_structure(r):
            scores.append(0.0)
        else:
            score = chrf_metric.compute(
                predictions=[p], references=[r], beta=1
            )["score"]
            scores.append(score)
    return float(np.mean(scores)) if scores else 0.0

def compute_xml_retention(src_texts, predictions):
    scores = []
    for s, p in zip(src_texts, predictions):
        s_tags = extract_xml_tags(s)
        if not s_tags:
            scores.append(1.0)
            continue

        p_tags = extract_xml_tags(p)
        sc, pc = Counter(s_tags), Counter(p_tags)
        retained = sum(min(sc[t], pc.get(t, 0)) for t in sc)
        scores.append(retained / sum(sc.values()))

    return np.mean(scores) * 100

# ============================================================
# DATA LOADER
# ============================================================

def normalize_entry(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if "text" in v:
            return v["text"]
        if "segments" in v:
            return "".join(s.get("text", "") for s in v["segments"])
        return json.dumps(v)
    return str(v)

def load_dev(lang_pair):
    base = os.path.join(DATA_ROOT, "data", lang_pair)

    with open(os.path.join(base, f"{lang_pair}_en_dev.json")) as f:
        src_json = json.load(f)

    with open(os.path.join(base, f"{lang_pair}_{lang_pair[2:]}_dev.json")) as f:
        tgt_json = json.load(f)

    src = [normalize_entry(v) for v in src_json["text"].values()]
    tgt = [normalize_entry(v) for v in tgt_json["text"].values()]

    return src, tgt

# ============================================================
# INFERENCE
# ============================================================

def evaluate_model(model, tokenizer, src_texts, tgt_texts, tgt_lang):
    preds_raw, preds_fixed = [], []

    for i in tqdm(range(0, len(src_texts), BATCH_SIZE)):
        batch_src = src_texts[i:i+BATCH_SIZE]

        prompts = [build_prompt(s, tgt_lang) for s in batch_src]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
                length_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        new_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        raw = [d.strip() for d in decoded]
        fixed = [fix_xml_tags(s, d) for s, d in zip(batch_src, raw)]

        preds_raw.extend(raw)
        preds_fixed.extend(fixed)

    return preds_raw, preds_fixed

# ============================================================
# METRICS
# ============================================================

def compute_metrics(preds, refs, srcs):
    preds_safe = [p if p.strip() else "EMPTY" for p in preds]

    return {
        "BLEU": round(bleu_metric.compute(predictions=preds_safe, references=refs)["bleu"] * 100, 2),
        "chrF": round(chrf_metric.compute(predictions=preds_safe, references=refs, beta=1)["score"], 2),
        "chrF++": round(chrf2_metric.compute(predictions=preds_safe, references=refs, beta=2)["score"], 2),
        "XML_Match": round(compute_xml_match(preds_safe, refs) * 100, 2),
        "XML_chrF": round(compute_xml_chrf(preds_safe, refs), 2),
        "XML_Retention": round(compute_xml_retention(srcs, preds_safe), 2),
    }

# ============================================================
# MAIN
# ============================================================

def main():
    free_gpu()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, FINAL_MODEL)
    model = model.merge_and_unload()
    model.eval()

    for lp in LANG_PAIRS:
        print(f"\n=== {lp} ===")

        src, tgt = load_dev(lp)

        if SANITY_TEST:
            src = src[:SANITY_SAMPLES]
            tgt = tgt[:SANITY_SAMPLES]

        preds_raw, preds_fixed = evaluate_model(
            model, tokenizer, src, tgt, LANG_NAME_MAP[lp]
        )

        print("\nRAW:", compute_metrics(preds_raw, tgt, src))
        print("FIX:", compute_metrics(preds_fixed, tgt, src))

if __name__ == "__main__":
    main()