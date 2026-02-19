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
    "gemma_270m_it": "google/gemma-3-270m-it",
}

DATA_ROOT = "localization-xml-mt"
LANG_PAIRS = ["ende", "enfr", "ennl", "enfi", "enru"]
LANG_MAP = {
    "ende": "GERMAN",
    "enfr": "FRENCH",
    "ennl": "DUTCH",
    "enfi": "FINNISH",
    "enru": "RUSSIAN",
}

BATCH_SIZE = 2
MAX_NEW_TOKENS = 512
OUTPUT_FOLDER = "salesforce_eval_outputs"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# =======================
# 🔁 SANITY TEST TOGGLE
# =======================
SANITY_TEST = False          # ← SET False FOR FULL RUN
SANITY_SAMPLES = 100         # ← fast smoke test size

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
    base = os.path.join(root, "data", lang_pair)
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
# 4. PRALekha-STYLE PROMPT (STRICT)
# ============================================================
def build_prompt(src, tgt_lang):
    return (
        "<start_of_turn>user\n"
        f"Translate to {tgt_lang}:\n{src}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

# ============================================================
# 5. METRICS
# ============================================================
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
chrf2 = evaluate.load("chrf")

# ============================================================
# 6. XML RETENTION
# ============================================================
def extract_xml_tags(text):
    return re.findall(r"</?[^>]+>", text)

def compute_xml_retention(srcs, preds):
    scores = []
    for s, p in zip(srcs, preds):
        s_tags = extract_xml_tags(s)
        if not s_tags:
            scores.append(1.0)
            continue
        p_tags = extract_xml_tags(p)
        sc, pc = Counter(s_tags), Counter(p_tags)
        retained = sum(min(sc[t], pc.get(t, 0)) for t in sc)
        scores.append(retained / sum(sc.values()))
    return scores

# ============================================================
# 7. MAIN LOOP
# ============================================================
for lang_pair in LANG_PAIRS:
    print(f"\n=== {lang_pair.upper()} ===")
    src_texts, tgt_texts = load_dev_as_test(DATA_ROOT, lang_pair)

    # 🔹 SANITY SLICE
    if SANITY_TEST:
        src_texts = src_texts[:SANITY_SAMPLES]
        tgt_texts = tgt_texts[:SANITY_SAMPLES]
        print(f"⚡ SANITY MODE: {SANITY_SAMPLES} samples")

    tgt_lang = LANG_MAP[lang_pair]

    dataset = Dataset.from_dict({
        "src": src_texts,
        "ref": tgt_texts
    })

    for model_key, model_name in BASE_MODELS.items():
        print(f"\n→ Model: {model_key}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        #tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="balanced",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        model.eval()

        predictions, references = [], []

        for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
            batch = dataset[i:i + BATCH_SIZE]
            prompts = [build_prompt(s, tgt_lang) for s in batch["src"]]

            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=0.1,
                    repetition_penalty=1.1,
                )

            # ================================
            # ✅ SAFE MT DECODING (YOUR CHOICE)
            # ================================
            decoded = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )

            decoded = [
                d[len(tokenizer.decode(inputs.input_ids[i], skip_special_tokens=True)):]
                for i, d in enumerate(decoded)
            ]

            predictions.extend([d.strip() for d in decoded])
            references.extend(batch["ref"])

        # ====================================================
        # METRICS
        # ====================================================
        bleu_score = bleu.compute(
            predictions=predictions,
            references=references
        )["bleu"]

        chrf_score = chrf.compute(
            predictions=predictions,
            references=references,
            beta=1
        )["score"]

        chrf2_score = chrf2.compute(
            predictions=predictions,
            references=references,
            beta=2
        )["score"]

        print(f"BLEU   : {bleu_score:.2f}")
        print(f"chrF   : {chrf_score:.2f}")
        print(f"chrF++ : {chrf2_score:.2f}")

        # ====================================================
        # XML RETENTION
        # ====================================================
        xml_scores = compute_xml_retention(src_texts, predictions)
        avg_xml = np.mean(xml_scores)
        print(f"XML Retention: {avg_xml * 100:.2f}%")

        # ====================================================
        # SAVE OUTPUTS
        # ====================================================
        out_jsonl = f"{OUTPUT_FOLDER}/{lang_pair}_{model_key}.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for s, r, p in zip(src_texts, references, predictions):
                json.dump({"src": s, "ref": r, "pred": p}, f, ensure_ascii=False)
                f.write("\n")

        pd.DataFrame([{
            "lang_pair": lang_pair,
            "model": model_key,
            "BLEU": bleu_score,
            "chrF": chrf_score,
            "chrF++": chrf2_score,
            "XML_retention": avg_xml,
            "sanity_mode": SANITY_TEST
        }]).to_csv(
            f"{OUTPUT_FOLDER}/{lang_pair}_{model_key}_metrics.csv",
            index=False
        )

# ============================================================
# 8. ZIP RESULTS
# ============================================================
shutil.make_archive(OUTPUT_FOLDER, "zip", OUTPUT_FOLDER)
print(f"\n✅ DONE → {OUTPUT_FOLDER}.zip")