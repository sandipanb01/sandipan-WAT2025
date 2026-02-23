# ============================================================
# TRANSLATEGEMMA XML-MT EVALUATION — PAPER-STRICT
# ============================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
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

# ============================================================
# 1. SEED
# ============================================================
set_seed(42)

# ============================================================
# 2. CONFIG
# ============================================================
MODEL_NAME = "google/translategemma-4b-it"
DATA_ROOT = "localization-xml-mt"

LANG_PAIRS = ["ende", "enfr", "ennl", "enfi", "enru"]

SOURCE_LANG_CODE = "en"

LANG_CODE_MAP = {
    "ende": "de",
    "enfr": "fr",
    "ennl": "nl",
    "enfi": "fi",
    "enru": "ru",
}

LANG_NAME_SRC = {
    lp: "English" for lp in LANG_PAIRS
}

LANG_NAME_TGT = {
    "ende": "German",
    "enfr": "French",
    "ennl": "Dutch",
    "enfi": "Finnish",
    "enru": "Russian",
}

BATCH_SIZE = 1
MAX_NEW_TOKENS = 512

SANITY_TEST = True
SANITY_SAMPLES = 10

OUTPUT_FOLDER = "salesforce_eval_outputs_TRANSLATEGEMMA_PAPER"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# ============================================================
# 3. DATA LOADING
# ============================================================
def normalize_salesforce_entry(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if "text" in v:
            return v["text"]
        if "segments" in v:
            return "".join(seg.get("text", "") for seg in v["segments"])
    return str(v)

def load_dev_as_test(root, lang_pair):
    base = os.path.join(root, "data", lang_pair)
    with open(f"{base}/{lang_pair}_en_dev.json", encoding="utf-8") as f:
        src = json.load(f)
    with open(f"{base}/{lang_pair}_{lang_pair[2:]}_dev.json", encoding="utf-8") as f:
        tgt = json.load(f)

    src_texts = [normalize_salesforce_entry(v) for v in src["text"].values()]
    tgt_texts = [normalize_salesforce_entry(v) for v in tgt["text"].values()]
    return src_texts, tgt_texts

# ============================================================
# 4. PAPER-VERBATIM PROMPT
# ============================================================
def build_messages(src_text, lang_pair):
    source_lang = LANG_NAME_SRC[lang_pair]
    target_lang = LANG_NAME_TGT[lang_pair]
    tgt_lang_code = LANG_CODE_MAP[lang_pair]

    prompt = (
        f"You are a professional {source_lang} ({SOURCE_LANG_CODE}) to {target_lang} ({tgt_lang_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while adhering to "
        f"{target_lang} grammar, vocabulary, and cultural sensitivities. "
        f"Produce only the {target_lang} translation, without any additional explanations or commentary. "
        f"Please translate the following {source_lang} text into {target_lang}:\n\n\n"
        f"{src_text}"
    )

    return [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": SOURCE_LANG_CODE,
                "target_lang_code": tgt_lang_code,
                "text": prompt,
            }
        ],
    }]

# ============================================================
# 5. METRICS
# ============================================================
bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

# ============================================================
# 6. XML METRICS (Appendix B)
# ============================================================
def get_xml_structure(text):
    try:
        root = etree.fromstring(f"<root>{text}</root>")
        def rec(el):
            return (el.tag, [rec(c) for c in el])
        return rec(root)
    except Exception:
        return None

def compute_xml_match(preds, refs):
    return sum(get_xml_structure(p) == get_xml_structure(r) for p, r in zip(preds, refs)) / len(preds)

def compute_xml_chrf(preds, refs):
    scores = []
    for p, r in zip(preds, refs):
        if get_xml_structure(p) != get_xml_structure(r):
            scores.append(0.0)
        else:
            scores.append(chrf_metric.compute(predictions=[p], references=[r], beta=1)["score"])
    return float(np.mean(scores))

# ============================================================
# 7. MAIN LOOP
# ============================================================
all_results = []

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    device_map={"": 0},
    dtype=torch.bfloat16
).eval()

for lang_pair in LANG_PAIRS:
    print(f"\n=== {lang_pair.upper()} ===")

    src, tgt = load_dev_as_test(DATA_ROOT, lang_pair)

    if SANITY_TEST:
        src = src[:SANITY_SAMPLES]
        tgt = tgt[:SANITY_SAMPLES]

    dataset = Dataset.from_dict({"src": src, "ref": tgt})

    predictions, references = [], []

    for item in tqdm(dataset):
        messages = build_messages(item["src"], lang_pair)

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        gen_tokens = output[0][prompt_len:]
        decoded = processor.decode(gen_tokens, skip_special_tokens=True)

        predictions.append(decoded.strip())
        references.append(item["ref"])

    bleu = bleu_metric.compute(predictions=predictions, references=references)["bleu"] * 100
    chrf = chrf_metric.compute(predictions=predictions, references=references, beta=1)["score"]
    xml_m = compute_xml_match(predictions, references) * 100
    xml_c = compute_xml_chrf(predictions, references)

    print(f"BLEU={bleu:.2f} | chrF={chrf:.2f} | XML-chrF={xml_c:.2f} | XML-Match={xml_m:.2f}%")

    all_results.append({
        "lang_pair": lang_pair,
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2),
        "XML_chrF": round(xml_c, 2),
        "XML_Match": round(xml_m, 2),
    })

# ============================================================
# 8. SAVE
# ============================================================
df = pd.DataFrame(all_results)
df.to_csv(f"{OUTPUT_FOLDER}/ALL_metrics_summary.csv", index=False)

shutil.make_archive(OUTPUT_FOLDER, "zip", OUTPUT_FOLDER)
print("\n✅ DONE:", OUTPUT_FOLDER)
