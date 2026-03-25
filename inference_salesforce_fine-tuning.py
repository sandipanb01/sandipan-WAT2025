# ============================================================
# XML MT CHECKPOINT EVALUATION
# ============================================================

import os
import json
import torch
import sacrebleu
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

DATA_ROOT = "localization-xml-mt"
MODEL_DIR = "./xml_mt_lora"

LANG_PAIRS = ["ende","enfr","ennl","enfi","enru"]

MAX_NEW_TOKENS = 512

LANG_CODE_MAP = {
    "ende":"German",
    "enfr":"French",
    "ennl":"Dutch",
    "enfi":"Finnish",
    "enru":"Russian",
}

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


def load_dev(lang_pair):

    base = os.path.join(DATA_ROOT,"data",lang_pair)

    src_file = os.path.join(base,f"{lang_pair}_en_dev.json")
    tgt_file = os.path.join(base,f"{lang_pair}_{lang_pair[2:]}_dev.json")

    with open(src_file) as f:
        src_json = json.load(f)

    with open(tgt_file) as f:
        tgt_json = json.load(f)

    src = [normalize_salesforce_entry(v) for v in src_json["text"].values()]
    tgt = [normalize_salesforce_entry(v) for v in tgt_json["text"].values()]

    return src,tgt


# ============================================================
# FIND CHECKPOINTS
# ============================================================

checkpoints = sorted([
    os.path.join(MODEL_DIR,d)
    for d in os.listdir(MODEL_DIR)
    if d.startswith("checkpoint")
])

print("Checkpoints found:",len(checkpoints))

results_jsonl = open("checkpoint_results.jsonl","w")

# ============================================================
# LOOP OVER CHECKPOINTS
# ============================================================

for ckpt in checkpoints:

    print("\nEvaluating",ckpt)

    model = AutoPeftModelForCausalLM.from_pretrained(
        ckpt,
        device_map="auto"
    )

    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    all_preds=[]
    all_refs=[]

    pred_file = open(
        f"{ckpt.replace('/','_')}_predictions.jsonl","w"
    )

    for lp in LANG_PAIRS:

        src,ref = load_dev(lp)

        tgt_lang = LANG_CODE_MAP[lp]

        for s,r in tqdm(list(zip(src,ref))):

            prompt = (
                f"Translate the following XML document from English to {tgt_lang}.\n\n"
                f"English XML:\n{s}\n\n"
                f"{tgt_lang} XML:"
            )

            inputs = tokenizer(prompt,return_tensors="pt").to(model.device)

            with torch.no_grad():

                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False
                )

            new_tokens = out[:,inputs["input_ids"].shape[1]:]

            pred = tokenizer.decode(new_tokens[0],skip_special_tokens=True).strip()

            all_preds.append(pred)
            all_refs.append(r)

            pred_file.write(json.dumps({
                "src":s,
                "ref":r,
                "pred":pred,
                "lang_pair":lp
            })+"\n")

    pred_file.close()

    bleu = sacrebleu.corpus_bleu(all_preds,[all_refs]).score
    chrf = sacrebleu.corpus_chrf(all_preds,[all_refs]).score

    result = {
        "checkpoint":ckpt,
        "BLEU":round(bleu,2),
        "chrF":round(chrf,2)
    }

    print(result)

    results_jsonl.write(json.dumps(result)+"\n")

results_jsonl.close()
