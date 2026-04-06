# ============================================================
# XML MT EVALUATION — TRANSLATEGEMMA MULTI-GPU (FINAL FIXED)
# ============================================================

import os
import json
import gc
import csv
import sys
import torch
import sacrebleu
import evaluate

from tqdm import tqdm
from pathlib import Path
from lxml import etree
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT   = "localization-xml-mt"
OUTPUT_DIR  = "xml_mt_eval_translategemma"

CHECKPOINT_DIR = "./xml_mt_translategemma_lora"
BASE_MODEL  = "google/translategemma-4b-it"

LANG_PAIRS = ["ende", "enfr", "ennl", "enfi", "enru"]

LANG_CODE_MAP = {
    "ende": "de",
    "enfr": "fr",
    "ennl": "nl",
    "enfi": "fi",
    "enru": "ru",
}

BATCH_SIZE     = 8
MAX_NEW_TOKENS = 512

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
# PROMPT (ALIGNED WITH TRAINING)
# ============================================================

def build_prompt(src, tgt_lang):
    return (
        f"<xml_translate>\n"
        f"<source_lang=en>\n"
        f"<target_lang={tgt_lang}>\n"
        f"<input>\n{src}\n</input>\n"
        f"<output>\n"
    )

# ============================================================
# 🔥 CRITICAL FIX: CLEAN OUTPUT
# ============================================================

def clean_output(text):
    if "<output>" in text:
        text = text.split("<output>")[-1]
    if "</output>" in text:
        text = text.split("</output>")[0]
    return text.strip()

# ============================================================
# XML METRICS (ROBUST PARSER)
# ============================================================

def get_xml_structure(text):
    try:
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(f"<root>{text}</root>", parser=parser)
    except:
        return None

    def structure(el):
        return (el.tag, tuple(structure(c) for c in el))

    return structure(root)

def extract_text_segments(text):
    try:
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(f"<root>{text}</root>", parser=parser)
    except:
        return []

    segments = []

    def collect(node):
        if node.text and node.text.strip():
            segments.append(node.text.strip())
        for child in node:
            collect(child)
        if node.tail and node.tail.strip():
            segments.append(node.tail.strip())

    collect(root)
    return segments

def compute_xml_metrics(preds, refs):

    match_count = 0
    chrf_scores = []

    for pred, ref in zip(preds, refs):

        pred_struct = get_xml_structure(pred)
        ref_struct  = get_xml_structure(ref)

        if pred_struct == ref_struct and pred_struct is not None:
            match_count += 1

            pred_segments = extract_text_segments(pred)
            ref_segments  = extract_text_segments(ref)

            n = min(len(pred_segments), len(ref_segments))

            if n == 0:
                chrf_scores.append(0.0)
                continue

            scores = []
            for i in range(n):
                score = sacrebleu.sentence_chrf(
                    pred_segments[i],
                    [ref_segments[i]]
                ).score
                scores.append(score)

            chrf_scores.append(sum(scores) / len(scores))
        else:
            chrf_scores.append(0.0)

    xml_match = match_count / len(preds) * 100
    xml_chrf  = sum(chrf_scores) / len(chrf_scores)

    return xml_match, xml_chrf

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
    return str(v)

def load_dev(lang_pair):

    base = os.path.join(DATA_ROOT, "data", lang_pair)

    with open(f"{base}/{lang_pair}_en_dev.json") as f:
        src_json = json.load(f)

    with open(f"{base}/{lang_pair}_{lang_pair[2:]}_dev.json") as f:
        tgt_json = json.load(f)

    src = [normalize_entry(v) for v in src_json["text"].values()]
    tgt = [normalize_entry(v) for v in tgt_json["text"].values()]

    return src, tgt

# ============================================================
# INFERENCE
# ============================================================

def evaluate_model(model, tokenizer, src_texts, tgt_lang):

    predictions = []

    for i in tqdm(range(0, len(src_texts), BATCH_SIZE)):

        batch_src = src_texts[i:i+BATCH_SIZE]
        prompts = [build_prompt(s, tgt_lang) for s in batch_src]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        new_tokens = outputs[:, input_len:]

        decoded = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True
        )

        preds = [clean_output(d) for d in decoded]  # 🔥 FIX APPLIED
        predictions.extend(preds)

    return predictions

# ============================================================
# METRICS
# ============================================================

def compute_metrics(preds, refs):

    preds_safe = [p if p.strip() else "EMPTY" for p in preds]

    bleu = bleu_metric.compute(predictions=preds_safe, references=refs)["bleu"] * 100

    chrf = chrf_metric.compute(predictions=preds_safe, references=refs, beta=1)["score"]

    chrf2 = chrf2_metric.compute(predictions=preds_safe, references=refs, beta=2)["score"]

    xml_match, xml_chrf = compute_xml_metrics(preds_safe, refs)

    return {
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2),
        "chrF++": round(chrf2, 2),
        "XML-Match": round(xml_match, 2),
        "XML-chrF": round(xml_chrf, 2)
    }

# ============================================================
# MAIN (MULTI-GPU SAME LOGIC)
# ============================================================

def main():

    free_gpu()

    part = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    checkpoints = sorted(os.listdir(CHECKPOINT_DIR))

    mid = len(checkpoints) // 2
    checkpoints = checkpoints[:mid] if part == 0 else checkpoints[mid:]

    csv_file = os.path.join(OUTPUT_DIR, f"results_part{part}.csv")

    with open(csv_file, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow([
            "checkpoint", "lang",
            "BLEU", "chrF", "chrF++",
            "XML-Match", "XML-chrF"
        ])

        for ckpt in checkpoints:

            if not ckpt.startswith("checkpoint-"):
                continue

            print(f"\nEvaluating {ckpt}")

            model_path = os.path.join(CHECKPOINT_DIR, ckpt)

            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
            model.eval()

            for lp in LANG_PAIRS:

                src, tgt = load_dev(lp)

                preds = evaluate_model(
                    model,
                    tokenizer,
                    src,
                    LANG_CODE_MAP[lp]
                )

                results = compute_metrics(preds, tgt)

                print(lp, results)

                writer.writerow([
                    ckpt,
                    lp,
                    results["BLEU"],
                    results["chrF"],
                    results["chrF++"],
                    results["XML-Match"],
                    results["XML-chrF"]
                ])

            del model
            del base_model
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n✅ Done GPU part {part}")


if __name__ == "__main__":
    main()