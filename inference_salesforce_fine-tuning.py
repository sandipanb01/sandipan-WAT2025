# ============================================================
# XML MT CHECKPOINT EVALUATION 
# ============================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from lxml import etree

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

import evaluate


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "localization-xml-mt"
OUTPUT_DIR = "xml_mt_checkpoint_eval"
CHECKPOINT_DIR = "./xml_mt_lora"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

LANG_PAIRS = ["ende","enfr","ennl","enfi","enru"]

LANG_CODE_MAP = {
    "ende":"de",
    "enfr":"fr",
    "ennl":"nl",
    "enfi":"fi",
    "enru":"ru",
}

LANG_NAME_MAP = {
    "ende":"German",
    "enfr":"French",
    "ennl":"Dutch",
    "enfi":"Finnish",
    "enru":"Russian",
}

BATCH_SIZE = 8
MAX_NEW_TOKENS = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# GPU CLEANUP
# ============================================================

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================
# METRICS
# ============================================================

bleu_metric  = evaluate.load("bleu")
chrf_metric  = evaluate.load("chrf")
chrf2_metric = evaluate.load("chrf")


# ============================================================
# XML METRICS
# ============================================================
def normalize_xml_whitespace(text):

    if text is None:
        return ""

    # collapse multiple whitespace into single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()
    
def get_xml_structure(text):

    try:

        root = etree.fromstring(f"<root>{text}</root>")

        def structure(el):

            children = [structure(c) for c in el]

            text = normalize_xml_whitespace(el.text)
            tail = normalize_xml_whitespace(el.tail)

            return (
                el.tag,
                text,
                tail,
                children
            )

        return structure(root)

    except Exception:
        return None


def compute_xml_match(predictions,references):

    matches=0

    for p,r in zip(predictions,references):
        if get_xml_structure(p)==get_xml_structure(r):
            matches+=1

    return matches/len(predictions) if predictions else 0.0


def compute_xml_chrf(predictions,references):

    scores=[]

    for p,r in zip(predictions,references):

        if get_xml_structure(p)!=get_xml_structure(r):
            scores.append(0.0)

        else:

            score = chrf_metric.compute(
                predictions=[p],
                references=[r],
                beta=1
            )["score"]

            scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


# ============================================================
# DATA LOADER
# ============================================================

def normalize_salesforce_entry(v):

    if isinstance(v,str):
        return v

    if isinstance(v,dict):

        if "text" in v:
            return v["text"]

        if "segments" in v:
            return "".join(seg.get("text","") for seg in v["segments"])

        return json.dumps(v)

    return str(v)


def load_dev_as_test(lang_pair):

    base = os.path.join(DATA_ROOT,"data",lang_pair)

    src_file = os.path.join(base,f"{lang_pair}_en_dev.json")
    tgt_file = os.path.join(base,f"{lang_pair}_{lang_pair[2:]}_dev.json")

    with open(src_file,encoding="utf-8") as f:
        src_json=json.load(f)

    with open(tgt_file,encoding="utf-8") as f:
        tgt_json=json.load(f)

    src=[normalize_salesforce_entry(v) for v in src_json["text"].values()]
    tgt=[normalize_salesforce_entry(v) for v in tgt_json["text"].values()]

    return src,tgt


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt_wat(example, tokenizer, tgt_lang):

    src = example["src"]
    ref = example["ref"]

    prompt = (
        f"Translate the following XML document from English to {tgt_lang}.\n\n"
        f"English XML:\n{src}"
    )

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

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
        "ref": ref,
    }

# ============================================================
# INFERENCE (Advisor evaluate_wat)
# ============================================================

def evaluate_model(model,tokenizer,dataset,batch_size=8):

    predictions=[]
    references=[]

    print("Running inference...")

    for i in tqdm(range(0,len(dataset),batch_size)):

        batch=dataset[i:i+batch_size]

        padded = tokenizer.pad(
            {
                "input_ids":batch["input_ids"],
                "attention_mask":batch["attention_mask"],
            },
            padding=True,
            return_tensors="pt",
        )

        input_ids=padded["input_ids"].to(model.device)
        attention_mask=padded["attention_mask"].to(model.device)

        with torch.no_grad():

            outputs=model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        new_tokens = outputs[:, input_ids.shape[1]:]

        decoded = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True
        )

        predictions.extend([d.strip() for d in decoded])

        refs = dataset[i:i+batch_size]["ref"]
        references.extend(refs)

    return references,predictions


# ============================================================
# METRIC COMPUTATION
# ============================================================

def compute_metrics(predictions,references):

    predictions_safe=[p if p.strip() else "EMPTY" for p in predictions]

    bleu_score = bleu_metric.compute(
        predictions=predictions_safe,
        references=references
    )["bleu"]

    chrf_score = chrf_metric.compute(
        predictions=predictions_safe,
        references=references,
        beta=1
    )["score"]

    chrf2_score = chrf2_metric.compute(
        predictions=predictions_safe,
        references=references,
        beta=2
    )["score"]

    xml_match = compute_xml_match(predictions_safe,references)
    xml_chrf  = compute_xml_chrf(predictions_safe,references)

    return {
        "BLEU":bleu_score*100,
        "chrF":chrf_score,
        "chrF++":chrf2_score,
        "XML_chrF":xml_chrf,
        "XML_Match":xml_match*100
    }


# ============================================================
# MAIN
# ============================================================

def main():

    all_results=[]

    base_dir=CHECKPOINT_DIR

    checkpoints=sorted(os.listdir(base_dir))

    for name in checkpoints:

        path=os.path.join(base_dir,name)

        if not name.startswith("checkpoint-"):
            continue

        print("\n========================================")
        print("Evaluating:",name)
        print("========================================")

        free_gpu()

        # Load LoRA checkpoint

        model = AutoPeftModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        model=model.merge_and_unload()
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token=tokenizer.eos_token


        for lang_pair in LANG_PAIRS:

            print("\nLanguage:",lang_pair)

            src_texts,tgt_texts = load_dev_as_test(lang_pair)

            dataset = Dataset.from_dict({
                "src":src_texts,
                "ref":tgt_texts
            })

            tgt_lang = LANG_NAME_MAP[lang_pair]

            dataset = dataset.map(
                build_prompt_wat,
                fn_kwargs={
                    "tokenizer":tokenizer,
                    "tgt_lang":tgt_lang
                }
            )

            refs,preds = evaluate_model(
                model,
                tokenizer,
                dataset,
                batch_size=BATCH_SIZE
            )

            metrics = compute_metrics(preds,refs)

            print(metrics)

            row={
                "checkpoint":name,
                "lang_pair":lang_pair,
                **metrics
            }

            all_results.append(row)

            out_jsonl=f"{OUTPUT_DIR}/{name}_{lang_pair}_predictions.jsonl"

            with open(out_jsonl,"w",encoding="utf-8") as f:

                for s,r,p in zip(src_texts,refs,preds):
                    json.dump(
                        {"src":s,"ref":r,"pred":p},
                        f,
                        ensure_ascii=False
                    )
                    f.write("\n")


        del model
        torch.cuda.empty_cache()


    df=pd.DataFrame(all_results)

    df.to_csv(f"{OUTPUT_DIR}/ALL_checkpoint_metrics.csv",index=False)

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print(df)


# ============================================================
# ENTRY
# ============================================================

if __name__=="__main__":
    main()
