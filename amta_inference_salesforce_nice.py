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

import sacrebleu
import re
import unicodedata
import html
from lxml import etree
from pathlib import Path
import shutil

# ============================================================
# 1. SEED & DEVICE
# ============================================================

set_seed(42)
torch.set_grad_enabled(False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 2. CONFIG
# ============================================================

BASE_MODELS = {
    "gemma_3-1b_it": "google/gemma-3-1b-it",
}

DATA_ROOT = "localization-xml-mt-master"

LANG_PAIRS = ["ende","enfr","ennl","enfi","enru"]

LANG_MAP = {
    "ende":"German",
    "enfr":"French",
    "ennl":"Dutch",
    "enfi":"Finnish",
    "enru":"Russian"
}

BATCH_SIZE = 2
MAX_NEW_TOKENS = 512

OUTPUT_FOLDER = "salesforce_eval_outputs_RESEARCH"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

SANITY_TEST = True
SANITY_SAMPLES = 10

# ============================================================
# 3. EXTRACT DATASET
# ============================================================

if os.path.exists("localization-xml-mt-master.zip"):
    print("Extracting dataset...")
    shutil.unpack_archive("localization-xml-mt-master.zip",".")

# ============================================================
# 4. DATA LOADING
# ============================================================

def normalize_salesforce_entry(v):

    if isinstance(v,str):
        return v

    if isinstance(v,dict):

        if "text" in v:
            return v["text"]

        if "segments" in v:
            return "".join(seg.get("text","") for seg in v["segments"])

        return json.dumps(v,ensure_ascii=False)

    return str(v)


def find_json_file(root,pattern):

    for path in Path(root).rglob(pattern):
        return path

    return None


def load_dev_as_test(root,lang_pair):

    src_pattern = f"{lang_pair}_en_dev.json"
    tgt_pattern = f"{lang_pair}_{lang_pair[2:]}_dev.json"

    src_file = find_json_file(root,src_pattern)
    tgt_file = find_json_file(root,tgt_pattern)

    if src_file is None:
        raise FileNotFoundError(src_pattern)

    if tgt_file is None:
        raise FileNotFoundError(tgt_pattern)

    print("Found source:",src_file)
    print("Found target:",tgt_file)

    with open(src_file,encoding="utf-8") as f:
        src_json=json.load(f)

    with open(tgt_file,encoding="utf-8") as f:
        tgt_json=json.load(f)

    src_texts=[normalize_salesforce_entry(v) for v in src_json["text"].values()]
    tgt_texts=[normalize_salesforce_entry(v) for v in tgt_json["text"].values()]

    return src_texts,tgt_texts

# ============================================================
# 5. PROMPT
# ============================================================

def build_prompt(src,tgt_lang):

    instruction = (
        f"Translate the following sentence from English to {tgt_lang}. "
        f"The translation must preserve all XML tags.\n\n"
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
# 6. OUTPUT CLEANING
# ============================================================

def remove_markdown(text):

    text=re.sub(r"```[\s\S]*?```","",text)
    text=text.replace("```","")
    text=text.replace("`","")

    return text.strip()


def remove_explanations(text):

    match=re.search(r"<",text)

    if match:
        return text[match.start():]

    return text


def normalize_unicode(text):

    text=unicodedata.normalize("NFC",text)
    text=html.unescape(text)

    return text


def normalize_xml_spacing(text):

    text=re.sub(r">\s+<","><",text)
    text=re.sub(r"\s+>",">",text)
    text=re.sub(r"<\s+","<",text)
    text=re.sub(r"\s*=\s*","=",text)

    return text.strip()

# ============================================================
# 7. XML CANONICALIZATION
# ============================================================

def canonicalize_xml(text):

    try:

        parser = etree.XMLParser(remove_blank_text=True, recover=True)

        root = etree.fromstring(
            f"<root>{text}</root>".encode(),
            parser
        )

        return etree.tostring(root,method="c14n").decode()

    except Exception:
        return None


def normalize_xml(text):

    if text is None:
        return None

    text=remove_markdown(text)
    text=remove_explanations(text)
    text=normalize_unicode(text)
    text=normalize_xml_spacing(text)

    text=canonicalize_xml(text)

    return text

# ============================================================
# 8. XML STRUCTURE
# ============================================================

def get_structure(xml):

    if xml is None:
        return None

    try:

        root=etree.fromstring(xml.encode())

        def structure(el):
            return (el.tag,[structure(c) for c in el])

        return structure(root)

    except Exception:
        return None

# ============================================================
# 9. XML METRICS
# ============================================================

def compute_xml_match(preds,refs):

    matches=0
    total=len(preds)

    for p,r in zip(preds,refs):

        p_clean=normalize_xml(p)
        r_clean=normalize_xml(r)

        if get_structure(p_clean)==get_structure(r_clean):
            matches+=1

    return (matches/total)*100 if total>0 else 0


def compute_xml_chrf(preds,refs):

    scores=[]

    for p,r in zip(preds,refs):

        p_clean=normalize_xml(p)
        r_clean=normalize_xml(r)

        if p_clean is None or r_clean is None:
            scores.append(0)
            continue

        if get_structure(p_clean)!=get_structure(r_clean):
            scores.append(0)
            continue

        score=sacrebleu.corpus_chrf([p_clean],[[r_clean]]).score
        scores.append(score)

    return np.mean(scores)

# ============================================================
# 10. XML STRIPPING (PARSER VERSION)
# ============================================================

def strip_xml(text):

    try:
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(f"<root>{text}</root>".encode(), parser)
        return "".join(root.itertext())

    except Exception:
        return re.sub(r"<[^>]+>", "", text)

# ============================================================
# 11. MAIN LOOP
# ============================================================

all_results=[]

for lang_pair in LANG_PAIRS:

    print("\n================================================")
    print("Language:",lang_pair)
    print("================================================")

    src_texts,tgt_texts=load_dev_as_test(DATA_ROOT,lang_pair)

    if SANITY_TEST:

        src_texts=src_texts[:SANITY_SAMPLES]
        tgt_texts=tgt_texts[:SANITY_SAMPLES]

        print("SANITY MODE:",SANITY_SAMPLES)

    dataset=Dataset.from_dict({
        "src":src_texts,
        "ref":tgt_texts
    })

    tgt_lang=LANG_MAP[lang_pair]

    for model_key,model_name in BASE_MODELS.items():

        print("\nRunning:",model_key)

        tokenizer=AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token=tokenizer.eos_token

        tokenizer.padding_side="left"

        model=AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"":0},
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        model.eval()

        predictions=[]
        references=[]

        for i in tqdm(range(0,len(dataset),BATCH_SIZE)):

            batch=dataset[i:i+BATCH_SIZE]

            prompts=[build_prompt(x,tgt_lang) for x in batch["src"]]

            inputs=tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():

                outputs=model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            input_len=inputs["input_ids"].shape[1]

            new_tokens=outputs[:,input_len:]

            decoded=tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True
            )

            predictions.extend([d.strip() for d in decoded])
            references.extend(batch["ref"])

        plain_preds=[strip_xml(x) for x in predictions]
        plain_refs=[strip_xml(x) for x in references]

        bleu=sacrebleu.corpus_bleu(plain_preds,[plain_refs],tokenize="13a").score
        chrf=sacrebleu.corpus_chrf(plain_preds,[plain_refs],beta=1).score
        chrf2=sacrebleu.corpus_chrf(plain_preds,[plain_refs],beta=2).score

        xml_match=compute_xml_match(predictions,references)
        xml_chrf=compute_xml_chrf(predictions,references)

        print("BLEU:",bleu)
        print("chrF:",chrf)
        print("chrF++:",chrf2)
        print("XML chrF:",xml_chrf)
        print("XML Match (%):",xml_match)

        all_results.append({
            "lang_pair":lang_pair,
            "model":model_key,
            "BLEU":round(bleu,2),
            "chrF":round(chrf,2),
            "chrF++":round(chrf2,2),
            "XML_chrF":round(xml_chrf,2),
            "XML_Match":round(xml_match,2),
            "n_samples":len(predictions)
        })

        del model
        torch.cuda.empty_cache()

# ============================================================
# 12. SAVE RESULTS
# ============================================================

df=pd.DataFrame(all_results)

out_file=f"{OUTPUT_FOLDER}/ALL_metrics_summary.csv"

df.to_csv(out_file,index=False)

print("\nFINAL RESULTS")
print(df.to_string(index=False))

shutil.make_archive(OUTPUT_FOLDER,"zip",OUTPUT_FOLDER)

print("\nSaved to:",out_file)
print("\nDONE")
