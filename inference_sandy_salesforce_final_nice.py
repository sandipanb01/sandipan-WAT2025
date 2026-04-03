# ============================================================
# XML MT EVALUATION — FINAL (ALL CHECKPOINTS + CSV SAVE)
# ============================================================

import os
import json
import re
import gc
import csv
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
OUTPUT_DIR  = "xml_mt_eval_final"
CHECKPOINT_DIR = "./xml_mt_lora"

BASE_MODEL  = "google/gemma-3-4b-it"

LANG_PAIRS = ["ende", "enfr", "ennl", "enfi", "enru"]

LANG_NAME_MAP = {
    "ende": "German",
    "enfr": "French",
    "ennl": "Dutch",
    "enfi": "Finnish",
    "enru": "Russian",
}

BATCH_SIZE     = 8   # start safe → increase later
MAX_NEW_TOKENS = 512

SANITY_TEST    = False
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
# PROMPT
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
# XML METRICS
# ============================================================

def get_xml_structure(text):
    try:
        root = etree.fromstring(f"<root>{text}</root>")
    except:
        return None

    def structure(el):
        return (el.tag, tuple(structure(c) for c in el))

    return structure(root)

def extract_text_segments(text):
    try:
        root = etree.fromstring(f"<root>{text}</root>")
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

        preds = [d.strip() for d in decoded]
        predictions.extend(preds)

    return predictions

# ============================================================
# METRICS
# ============================================================

def compute_metrics(preds, refs):

    preds_safe = [p if p.strip() else "EMPTY" for p in preds]

    bleu = bleu_metric.compute(
        predictions=preds_safe,
        references=refs
    )["bleu"] * 100

    chrf = chrf_metric.compute(
        predictions=preds_safe,
        references=refs,
        beta=1
    )["score"]

    chrf2 = chrf2_metric.compute(
        predictions=preds_safe,
        references=refs,
        beta=2
    )["score"]

    xml_match, xml_chrf = compute_xml_metrics(preds_safe, refs)

    return {
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2),
        "chrF++": round(chrf2, 2),
        "XML-Match": round(xml_match, 2),
        "XML-chrF": round(xml_chrf, 2)
    }

# ============================================================
# MAIN
# ============================================================

def main():

    free_gpu()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    torch.backends.cuda.matmul.allow_tf32 = True

    checkpoints = sorted(os.listdir(CHECKPOINT_DIR))

    csv_file = os.path.join(OUTPUT_DIR, "results.csv")

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

            print(f"\n==============================")
            print(f"Evaluating {ckpt}")
            print(f"==============================")

            model_path = os.path.join(CHECKPOINT_DIR, ckpt)

            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            )

            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
            model.eval()

            for lp in LANG_PAIRS:

                print(f"\n========== {lp} ==========")

                src, tgt = load_dev(lp)

                if SANITY_TEST:
                    src = src[:SANITY_SAMPLES]
                    tgt = tgt[:SANITY_SAMPLES]

                preds = evaluate_model(
                    model,
                    tokenizer,
                    src,
                    LANG_NAME_MAP[lp]
                )

                results = compute_metrics(preds, tgt)

                print(results)

                writer.writerow([
                    ckpt,
                    lp,
                    results["BLEU"],
                    results["chrF"],
                    results["chrF++"],
                    results["XML-Match"],
                    results["XML-chrF"]
                ])

            # free memory
            del model
            del base_model
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n✅ Results saved to: {csv_file}")


if __name__ == "__main__":
    main()
