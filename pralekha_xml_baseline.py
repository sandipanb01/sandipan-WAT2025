# ============================================================
# AMTA-STYLE XML MT BASELINE
# Pralekha English → Hindi
# STRICT XML EVALUATION
# ============================================================

import gc
import re
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
SANITY_SAMPLES = 200


# ============================================================
# GPU CLEANUP
# ============================================================

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# WHITESPACE NORMALIZATION (Hashimoto style)
# ============================================================

def normalize_xml_whitespace(text):

    if text is None:
        return ""

    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# XML METRICS (VERBATIM IMPLEMENTATION)
# ============================================================

def get_xml_structure(text):

    try:
        root = etree.fromstring(f"<root>{text}</root>")
    except Exception:
        return None

    def structure(el):
        return (el.tag, tuple(structure(c) for c in el))

    return structure(root)


def extract_text_segments(text):

    root = etree.fromstring(f"<root>{text}</root>")

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
# LOAD PRALEKHA
# ============================================================

def load_pralekha():

    dataset = load_dataset("ai4bharat/Pralekha")

    dataset = dataset.filter(
        lambda x: x["src_lang"] == SRC_LANG
        and x["tgt_lang"] == TGT_LANG
        and x["src_txt"] != x["tgt_txt"]
    )

    src = dataset["test"]["src_txt"]
    tgt = dataset["test"]["tgt_txt"]

    return src, tgt


# ============================================================
# XML PARSING
# ============================================================

def parse_xml(text):

    try:
        root = etree.fromstring(f"<root>{text}</root>")
    except:
        return None

    return root


def extract_nodes(root):

    nodes = []

    def walk(node):

        if node.text and node.text.strip():
            nodes.append(("text", node))

        for child in node:
            walk(child)

        if node.tail and node.tail.strip():
            nodes.append(("tail", node))

    walk(root)

    return nodes


# ============================================================
# REBUILD XML
# ============================================================

def rebuild_xml(root):

    xml = "".join(
        etree.tostring(child, encoding="unicode")
        for child in root
    )

    return xml


# ============================================================
# PROMPT
# ============================================================

def build_prompt(text):

    instruction = (
        f"Translate the following text from English to Hindi.\n"
        f"Only translate the text content.\n\n"
        f"English:\n{text}\n\n"
        f"Hindi:"
    )

    return (
        f"<bos><start_of_turn>user\n"
        f"{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


# ============================================================
# SEGMENT TRANSLATION
# ============================================================

def translate_segments(model, tokenizer, segments):

    outputs = []

    for i in range(0, len(segments), BATCH_SIZE):

        batch = segments[i:i+BATCH_SIZE]

        prompts = [build_prompt(x) for x in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():

            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        new_tokens = generated[:, input_len:]

        decoded = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True
        )

        outputs.extend([d.strip() for d in decoded])

    return outputs


# ============================================================
# DOCUMENT TRANSLATION
# ============================================================

def translate_document(model, tokenizer, xml):

    root = parse_xml(xml)

    if root is None:
        return ""

    nodes = extract_nodes(root)

    segments = []

    for typ, node in nodes:

        if typ == "text":
            segments.append(node.text.strip())
        else:
            segments.append(node.tail.strip())

    if len(segments) == 0:
        return rebuild_xml(root)

    translated = translate_segments(
        model,
        tokenizer,
        segments
    )

    idx = 0

    for typ, node in nodes:

        if typ == "text":
            node.text = translated[idx]
        else:
            node.tail = translated[idx]

        idx += 1

    return rebuild_xml(root)


# ============================================================
# METRICS
# ============================================================

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")
chrf2_metric = evaluate.load("chrf")


def compute_metrics(preds, refs):

    preds_norm = [normalize_xml_whitespace(p) for p in preds]
    refs_norm = [normalize_xml_whitespace(r) for r in refs]

    bleu = bleu_metric.compute(
        predictions=preds_norm,
        references=refs_norm
    )["bleu"] * 100

    chrf = chrf_metric.compute(
        predictions=preds_norm,
        references=refs_norm,
        beta=1
    )["score"]

    chrf2 = chrf2_metric.compute(
        predictions=preds_norm,
        references=refs_norm,
        beta=2
    )["score"]

    xml_metrics = compute_xml_metrics(
        preds_norm,
        refs_norm
    )

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

    preds = []

    print("Running structure-aware translation...")

    for doc in tqdm(src):

        translated = translate_document(
            model,
            tokenizer,
            doc
        )

        preds.append(translated)

    print("Computing metrics...")

    results = compute_metrics(preds, tgt)

    print("\n==============================")
    print("ZERO-SHOT BASELINE RESULTS")
    print("==============================")

    for k, v in results.items():
        print(f"{k}: {v}")

    print("==============================")


if __name__ == "__main__":
    main()
