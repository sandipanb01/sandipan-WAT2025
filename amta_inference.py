import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, apply_chat_template
from trl.data_utils import is_conversational
from pathlib import Path
import matplotlib.pyplot as plt
import evaluate

# ============================================================
# │ CONFIGURATION
# ============================================================
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
!git clone https://github.com/salesforce/localization-xml-mt.git

BASE_MODELS = {
    "gemma_270m_it": "google/gemma-3-270m-it",
    # Add more models here if needed
    "gemma_4b_it": "google/gemma-3-4b-it",
     "translategemma": "MedAIBase/TranslateGemma:4b-it",
     "sarvam_translate": "sarvamai/sarvam-translate"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "localization-xml-mt"
LANG_PAIRS = ["ende", "enfr", "ennl", "enfi", "enru"]
LANG_MAP = {"ende":"German", "enfr":"French", "ennl":"Dutch", "enfi":"Finnish", "enru":"Russian"}

SMOKE_TEST = False
SMOKE_SAMPLES = 100
BATCH_SIZE = 8
MAX_NEW_TOKENS = 512
OUTPUT_FOLDER = "salesforce_eval_outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# │ LOAD DEV SPLIT (used as test)
# ============================================================
def normalize_salesforce_entry(v):
    """
    Salesforce Localization entries can be:
    - str
    - dict with 'text'
    - dict with 'segments'
    This function converts everything into a clean string.
    """
    if isinstance(v, str):
        return v

    if isinstance(v, dict):
        # Case 1: {"text": "..."}
        if "text" in v:
            return v["text"]

        # Case 2: {"segments": [{"text": ...}, ...]}
        if "segments" in v:
            return "".join(seg.get("text", "") for seg in v["segments"])

        # Fallback (should not happen, but safe)
        return json.dumps(v, ensure_ascii=False)

    # Final fallback
    return str(v)


def load_dev_as_test(root, lang_pair):
    base = os.path.join(root, "data", lang_pair)
    split = "dev"  # dev is used as test (official test hidden)

    src_file = os.path.join(base, f"{lang_pair}_en_{split}.json")
    tgt_file = os.path.join(base, f"{lang_pair}_{lang_pair[2:]}_{split}.json")

    with open(src_file, "r", encoding="utf-8") as f:
        src_json = json.load(f)

    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_json = json.load(f)

    src_texts = [normalize_salesforce_entry(v) for v in src_json.values()]
    tgt_texts = [normalize_salesforce_entry(v) for v in tgt_json.values()]

    return src_texts, tgt_texts

#print(type(src_texts[0]), src_texts[0][:80])


# ============================================================
# │ PROMPT FUNCTION (ADVISOR-STYLE)
# ============================================================
def build_prompt(example, src_lang, tgt_lang, tokenizer):
    src = example["src_txt"]
    ref = example["tgt_txt"]

    prompt = (
        f"Translate the following text from {src_lang} to {tgt_lang}:\n"
        f"{src_lang}: {src}\n"
        f"{tgt_lang}: "
    )

    messages = [{"role": "user", "content": prompt}]

    # Apply chat template exactly as advisor did
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    tokens = tokenizer(
        prompt,
        truncation=True,
        padding=False,
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "reference": ref,
    }

# ============================================================
# │ METRICS
# ============================================================
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
chrf2 = evaluate.load("chrf", module_type="metric")  # beta=2
# xml_chrf = evaluate.load("xmlchrf", module_type="metric")  # Uncomment if installed

# ============================================================
# │ MAIN LOOP: LANGS + MODELS
# ============================================================
for lang_pair in LANG_PAIRS:
    print(f"\n=== Processing {lang_pair} ===")
    src_texts, tgt_texts = load_dev_as_test(DATA_ROOT, lang_pair)

    if SMOKE_TEST:
        src_texts = src_texts[:SMOKE_SAMPLES]
        tgt_texts = tgt_texts[:SMOKE_SAMPLES]

    src_lang = "English"
    tgt_lang = LANG_MAP[lang_pair]

    # Build HF dataset
    dataset = Dataset.from_dict({"src_txt": src_texts, "tgt_txt": tgt_texts})

    for model_key, model_name in BASE_MODELS.items():
        print(f"\n--> Running inference on model: {model_key}")

        # Tokenizer + model init
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        model.eval()

        # Build prompts
        dataset_prompted = dataset.map(lambda x: build_prompt(x, src_lang, tgt_lang, tokenizer))

        # Batched inference
        predictions = []
        references = []
        for i in tqdm(range(0, len(dataset_prompted), BATCH_SIZE)):
            batch = dataset_prompted[i:i+BATCH_SIZE]

            padded = tokenizer.pad(
                {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                },
                padding=True,
                return_tensors="pt",
            )

            input_ids = padded["input_ids"].to(model.device)
            attention_mask = padded["attention_mask"].to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    temperature=0.1,
                    repetition_penalty=1.1 
                )

            new_tokens = outputs[:, input_ids.shape[1]:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            decoded = [t.split(f"{tgt_lang}:")[-1].strip() for t in decoded]

            predictions.extend(decoded)
            references.extend(batch["reference"])

        # Compute metrics
        bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]
        chrf_score = chrf.compute(predictions=predictions, references=references, beta=1)["score"]
        chrf2_score = chrf2.compute(predictions=predictions, references=references, beta=2)["score"]
        # xml_chrf_score = xml_chrf.compute(predictions=predictions, references=references)["score"]

        print(f"\nModel: {model_key} — BLEU: {bleu_score:.2f}, chrF: {chrf_score:.2f}, chrF2: {chrf2_score:.2f}")
        print(f"Sample predictions: {predictions[:3]}")
        print(f"Sample references: {references[:3]}")

        # Save JSONL
        jsonl_file = os.path.join(OUTPUT_FOLDER, f"{lang_pair}_{model_key}_predictions.jsonl")
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for src, tgt, pred in zip(src_texts, tgt_texts, predictions):
                json.dump({"src": src, "ref": tgt, "pred": pred}, f, ensure_ascii=False)
                f.write("\n")
        print(f"Saved predictions to {jsonl_file}")

        # Save metrics CSV
        metrics_file = os.path.join(OUTPUT_FOLDER, f"{lang_pair}_{model_key}_metrics.csv")
        df = pd.DataFrame([{
            "model": model_key,
            "lang_pair": lang_pair,
            "BLEU": bleu_score,
            "chrF": chrf_score,
            "chrF2": chrf2_score,
            # "XML_chrF": xml_chrf_score
        }])
        df.to_csv(metrics_file, index=False)
        print(f"Saved metrics to {metrics_file}")

# ============================================================
# │ ZIP OUTPUT AND AUTO DOWNLOAD 
# ============================================================
import shutil
import sys

shutil.make_archive(OUTPUT_FOLDER, 'zip', OUTPUT_FOLDER)
print(f"ZIP created: {OUTPUT_FOLDER}.zip")

if "google.colab" in sys.modules:
    from google.colab import files
    files.download(f"{OUTPUT_FOLDER}.zip")
else:
    print("Not running in Colab — download the ZIP manually.")
# ============================================================
# │ XML MARKUP RETENTION CHECK
# ============================================================

import re
from collections import Counter

def extract_xml_tags(text):
    """
    Extract all XML-like tags from a string.
    Example: "<b>Hello</b>" -> ['<b>', '</b>']
    """
    return re.findall(r"</?[^>]+>", text)

def compute_xml_retention(src_texts, predictions):
    """
    Compute percentage of XML tags in source that are present in prediction.
    Returns a list of retention percentages for each sample.
    """
    retention_scores = []

    for src, pred in zip(src_texts, predictions):
        src_tags = extract_xml_tags(src)
        if not src_tags:
            retention_scores.append(1.0)  # no tags to preserve, consider 100%
            continue

        pred_tags = extract_xml_tags(pred)
        # Count tags in source and prediction
        src_count = Counter(src_tags)
        pred_count = Counter(pred_tags)

        # Compute proportion of tags in src that appear in pred
        retained = sum(min(src_count[tag], pred_count.get(tag, 0)) for tag in src_count)
        total = sum(src_count.values())
        retention_scores.append(retained / total)

    return retention_scores

# Run XML retention check
xml_retention_scores = compute_xml_retention(src_texts, predictions)
avg_xml_retention = np.mean(xml_retention_scores)

print(f"Average XML tag retention: {avg_xml_retention*100:.2f}%")

# Optionally, save per-sample retention
xml_csv_file = os.path.join(OUTPUT_FOLDER, f"xml_retention_{lang_pair}_{model_key}.csv")
pd.DataFrame({
    "src": src_texts,
    "pred": predictions,
    "xml_retention": xml_retention_scores
}).to_csv(xml_csv_file, index=False)
print(f"Saved XML retention per sample to {xml_csv_file}")
# ============================================================
# │ XML MARKUP RETENTION PER LANGUAGE DIRECTION
# ============================================================

import re
import numpy as np
import pandas as pd
import os

def extract_xml_tags(text):
    """Extract all XML-like tags from a string."""
    return re.findall(r"</?[^>]+>", text)

def compute_xml_retention(src_texts, predictions):
    """
    Compute percentage of XML tags in source preserved in prediction.
    Returns a list of retention percentages per sample.
    """
    retention_scores = []

    for src, pred in zip(src_texts, predictions):
        src_tags = extract_xml_tags(src)
        if not src_tags:
            retention_scores.append(1.0)  # no tags to preserve → 100%
            continue

        pred_tags = extract_xml_tags(pred)
        # Count tags in source and prediction
        from collections import Counter
        src_count = Counter(src_tags)
        pred_count = Counter(pred_tags)

        # Calculate proportion of source tags retained
        retained = sum(min(src_count[tag], pred_count.get(tag, 0)) for tag in src_count)
        total = sum(src_count.values())
        retention_scores.append(retained / total)

    return retention_scores

# =========================
# Run per language direction
# =========================

xml_retention_summary = {}

for lang_pair in LANG_PAIRS:
    # Load previous predictions and sources
    jsonl_file = os.path.join(OUTPUT_FOLDER, f"{lang_pair}_{model_key}_predictions.jsonl")
    if not os.path.exists(jsonl_file):
        print(f"Predictions file missing for {lang_pair}, skipping XML retention check.")
        continue

    src_texts_lang = []
    pred_texts_lang = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            src_texts_lang.append(item["src"])
            pred_texts_lang.append(item["pred"])

    # Compute retention
    xml_scores = compute_xml_retention(src_texts_lang, pred_texts_lang)
    avg_retention = np.mean(xml_scores)
    xml_retention_summary[lang_pair] = avg_retention

    # Save per-language CSV
    csv_file = os.path.join(OUTPUT_FOLDER, f"xml_retention_{lang_pair}_{model_key}.csv")
    pd.DataFrame({
        "src": src_texts_lang,
        "pred": pred_texts_lang,
        "xml_retention": xml_scores
    }).to_csv(csv_file, index=False)
    print(f"[{lang_pair}] Saved per-sample XML retention CSV: {csv_file}")
    print(f"[{lang_pair}] Average XML tag retention: {avg_retention*100:.2f}%\n")

# Summary table
summary_file = os.path.join(OUTPUT_FOLDER, f"xml_retention_summary_{model_key}.csv")
pd.DataFrame.from_dict(xml_retention_summary, orient="index", columns=["avg_xml_retention"]).to_csv(summary_file)
print(f"Saved XML retention summary across languages: {summary_file}")

