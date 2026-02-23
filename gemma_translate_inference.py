# ============================================================
# TRANSLATEGEMMA EVALUATION SCRIPT — PAPER-STRICT
# ============================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
import evaluate
from lxml import etree
import shutil

# ============================================================
# 1. SEED (paper-consistent)
# ============================================================
set_seed(42)

# ============================================================
# 2. CONFIG
# ============================================================
BASE_MODELS = {
    "translategemma_4b_it": "google/translategemma-4b-it",
}

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
    "ende": "English",
    "enfr": "English",
    "ennl": "English",
    "enfi": "English",
    "enru": "English",
}

LANG_NAME_TGT = {
    "ende": "German",
    "enfr": "French",
    "ennl": "Dutch",
    "enfi": "Finnish",
    "enru": "Russian",
}

BATCH_SIZE = 1              # MUST be 1 (paper: no padding interaction)
MAX_NEW_TOKENS = 512

OUTPUT_FOLDER = "TranslateGemma_PAPER_STRICT_RESULTS"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

SANITY_TEST = True
SANITY_SAMPLES = 10

# ============================================================
# 3. DATA LOADING (Salesforce XML-MT)
# ============================================================
def normalize_entry(v):
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
    src_path = os.path.join(base, f"{lang_pair}_en_dev.json")
    tgt_path = os.path.join(base, f"{lang_pair}_{lang_pair[2:]}_dev.json")

    with open(src_path, encoding="utf-8") as f:
        src_json = json.load(f)
    with open(tgt_path, encoding="utf-8") as f:
        tgt_json = json.load(f)

    src_texts = [normalize_entry(v) for v in src_json["text"].values()]
    tgt_texts = [normalize_entry(v) for v in tgt_json["text"].values()]
    return src_texts, tgt_texts

# ============================================================
# 4. PAPER-STRICT VERBATIM PROMPT
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
        "content": {
            "source_lang": SOURCE_LANG_CODE,
            "target_lang": tgt_lang_code,
            "text": prompt
        }
    }]
# ============================================================
# 5. METRICS
# ============================================================
bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

# ============================================================
# 6. XML STRUCTURE METRICS (Appendix-faithful)
# ============================================================
def get_xml_structure(text):
    try:
        root = etree.fromstring(f"<root>{text}</root>")
        def rec(el):
            return (el.tag, [rec(c) for c in el])
        return rec(root)
    except Exception:
        return None

def xml_match(preds, refs):
    return sum(
        get_xml_structure(p) == get_xml_structure(r)
        for p, r in zip(preds, refs)
    ) / len(preds)

def xml_chrf(preds, refs):
    scores = []
    for p, r in zip(preds, refs):
        if get_xml_structure(p) != get_xml_structure(r):
            scores.append(0.0)
        else:
            scores.append(
                chrf_metric.compute(
                    predictions=[p],
                    references=[r],
                    beta=1
                )["score"]
            )
    return float(np.mean(scores))

# ============================================================
# 7. MAIN EVALUATION LOOP
# ============================================================
all_results = []

for lang_pair in LANG_PAIRS:
    print(f"\n=== {lang_pair.upper()} ===")

    src, ref = load_dev_as_test(DATA_ROOT, lang_pair)

    if SANITY_TEST:
        src = src[:SANITY_SAMPLES]
        ref = ref[:SANITY_SAMPLES]

    dataset = Dataset.from_dict({"src": src, "ref": ref})

    for model_key, model_name in BASE_MODELS.items():
        print(f"\n→ Model: {model_key}")

        processor = AutoProcessor.from_pretrained(
                    model_name,
                    use_fast=False)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        ).eval()

        preds, refs = [], []

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
                    use_cache=True,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            gen = output[0][inputs["input_ids"].shape[-1]:]
            text = processor.decode(gen, skip_special_tokens=True).strip()

            preds.append(text)
            refs.append(item["ref"])

            torch.cuda.empty_cache()

        bleu = bleu_metric.compute(predictions=preds, references=refs)["bleu"] * 100
        chrf = chrf_metric.compute(predictions=preds, references=refs, beta=1)["score"]
        xm = xml_match(preds, refs) * 100
        xc = xml_chrf(preds, refs)

        print(f"BLEU      : {bleu:.2f}")
        print(f"chrF      : {chrf:.2f}")
        print(f"XML-Match : {xm:.2f}%")
        print(f"XML-chrF  : {xc:.2f}")

        out_file = f"{OUTPUT_FOLDER}/{lang_pair}_{model_key}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for s, r, p in zip(src, refs, preds):
                json.dump({"src": s, "ref": r, "pred": p}, f, ensure_ascii=False)
                f.write("\n")

        all_results.append({
            "lang_pair": lang_pair,
            "model": model_key,
            "BLEU": round(bleu, 2),
            "chrF": round(chrf, 2),
            "XML_Match": round(xm, 2),
            "XML_chrF": round(xc, 2),
        })

        del model, processor
        torch.cuda.empty_cache()

# ============================================================
# 8. SAVE SUMMARY
# ============================================================
df = pd.DataFrame(all_results)
df.to_csv(f"{OUTPUT_FOLDER}/ALL_RESULTS.csv", index=False)

shutil.make_archive(OUTPUT_FOLDER, "zip", OUTPUT_FOLDER)
print("\nDONE →", f"{OUTPUT_FOLDER}.zip")
