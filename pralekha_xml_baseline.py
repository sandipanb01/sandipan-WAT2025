# ============================================================
# ZERO-SHOT XML MACHINE TRANSLATION BASELINE
# Pralekha Dataset (English → Hindi)
# Markdown XML Tag Protection + Strict XML Metrics
# ============================================================

import re
import gc
import torch
import sacrebleu
import evaluate

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from lxml import etree


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "google/gemma-3-4b-it"

SRC_LANG = "eng"
TGT_LANG = "hin"

TARGET_LANGUAGE_NAME = "Hindi"

BATCH_SIZE = 8
MAX_NEW_TOKENS = 4096

SANITY_TEST = False
SANITY_SAMPLES = 100


# ============================================================
# GPU CLEANUP
# ============================================================

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# LOAD PRALEKHA DATASET
# ============================================================

def load_pralekha():

    dataset = load_dataset("ai4bharat/Pralekha")

    dataset = dataset.filter(
        lambda x: x["src_lang"] == SRC_LANG and x["tgt_lang"] == TGT_LANG
    )

    src = dataset["test"]["src_txt"]
    tgt = dataset["test"]["tgt_txt"]

    return src, tgt


# ============================================================
# MARKDOWN XML TAG PROTECTION
# (AMTA paper style)
# ============================================================

TAG_PATTERN = re.compile(r"<[^>]+>")

def markdown_xml(text):

    def repl(match):
        tag = match.group(0)
        return f"`{tag}`"

    return TAG_PATTERN.sub(repl, text)


def restore_xml(text):
    return text.replace("`<", "<").replace(">`", ">")


# ============================================================
# PROMPT
# ============================================================

def build_prompt(src):

    instruction = (
        f"Translate the following XML document from English to Hindi.\n"
        f"Preserve ALL XML tags exactly.\n\n"
        f"English:\n{src}\n\n"
        f"Hindi:"
    )

    return (
        f"<bos><start_of_turn>user\n"
        f"{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


# ============================================================
# XML METRICS
# ============================================================

def normalize_xml_whitespace(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def get_xml_structure(text):

    try:
        root = etree.fromstring(f"<root>{text}</root>")
    except Exception:
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
        ref_struct = get_xml_structure(ref)

        if pred_struct == ref_struct and pred_struct is not None:

            match_count += 1

            pred_segments = extract_text_segments(pred)
            ref_segments = extract_text_segments(ref)

            score = sacrebleu.corpus_chrf(
                pred_segments,
                [ref_segments]
            ).score

            chrf_scores.append(score)

        else:
            chrf_scores.append(0.0)

    xml_match = match_count / len(preds) * 100
    xml_chrf = sum(chrf_scores) / len(chrf_scores)

    return {
        "XML-Match": xml_match,
        "XML-chrF": xml_chrf
    }


# ============================================================
# METRIC OBJECTS
# ============================================================

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")
chrf2_metric = evaluate.load("chrf")


# ============================================================
# INFERENCE
# ============================================================

def run_inference(model, tokenizer, src_texts):

    predictions = []

    for i in tqdm(range(0, len(src_texts), BATCH_SIZE)):

        batch = src_texts[i:i+BATCH_SIZE]

        prompts = [
            build_prompt(markdown_xml(s))
            for s in batch
        ]

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

        preds = [restore_xml(d.strip()) for d in decoded]

        predictions.extend(preds)

    return predictions


# ============================================================
# METRIC COMPUTATION
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

    xml_metrics = compute_xml_metrics(preds_safe, refs)

    return {
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2),
        "chrF++": round(chrf2, 2),
        "XML-Match": round(xml_metrics["XML-Match"], 2),
        "XML-chrF": round(xml_metrics["XML-chrF"], 2)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    free_gpu()

    print("Loading dataset...")

    src, tgt = load_pralekha()

    if SANITY_TEST:
        src = src[:SANITY_SAMPLES]
        tgt = tgt[:SANITY_SAMPLES]

    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model.eval()

    print("Running inference...")

    preds = run_inference(
        model,
        tokenizer,
        src
    )

    print("Computing metrics...")

    results = compute_metrics(preds, tgt)

    print("\n===================================")
    print("ZERO SHOT BASELINE RESULTS")
    print("===================================")

    for k, v in results.items():
        print(f"{k}: {v}")

    print("===================================")


if __name__ == "__main__":
    main()
