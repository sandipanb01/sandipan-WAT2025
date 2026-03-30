pip install indictrans2 sentencepiece sacrebleu datasets 
#-------Load IndicTrans2----------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# IndicTrans2 checkpoints
EN_HI_MODEL = "ai4bharat/indictrans2-en-indic-1B"
HI_EN_MODEL = "ai4bharat/indictrans2-indic-en-1B"

tokenizer_en_hi = AutoTokenizer.from_pretrained(EN_HI_MODEL, trust_remote_code=True)
model_en_hi = AutoModelForSeq2SeqLM.from_pretrained(
    EN_HI_MODEL, trust_remote_code=True, torch_dtype=torch.float16
).to(device)

tokenizer_hi_en = AutoTokenizer.from_pretrained(HI_EN_MODEL, trust_remote_code=True)
model_hi_en = AutoModelForSeq2SeqLM.from_pretrained(
    HI_EN_MODEL, trust_remote_code=True, torch_dtype=torch.float16
).to(device)

model_en_hi.eval()
model_hi_en.eval()
#---------Load REAL monolingual data------------
from datasets import load_dataset

# FLORES-200 monolingual splits
flores = load_dataset("facebook/flores", "eng_Latn-hin_Deva")

# Extract monolingual corpora
mono_en = [x["sentence_eng_Latn"] for x in flores["dev"]]
mono_hi = [x["sentence_hin_Deva"] for x in flores["dev"]]

print("English mono samples:", len(mono_en))
print("Hindi mono samples:", len(mono_hi))
from datasets import load_dataset

#-----------Generate synthetic translations using IndicTrans2----
def translate_indictrans(texts, model, tokenizer, src_lang_tag, tgt_lang_tag, batch_size=8, max_len=512):
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Prepend the language tags to each text in the batch
        formatted_batch = [f"{src_lang_tag} {tgt_lang_tag} {text}" for text in batch]
        inputs = tokenizer(
            formatted_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_len,
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.1,
                use_cache=False # Explicitly disable cache to avoid NoneType error
            )

        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs

# EN → HI
bt_hi = translate_indictrans(
    mono_en,
    model_en_hi,
    tokenizer_en_hi,
    src_lang_tag="eng_Latn",
    tgt_lang_tag="hin_Deva"
)
# HI → EN
bt_en = translate_indictrans(
    mono_hi,
    model_hi_en,
    tokenizer_hi_en,
    src_lang_tag="hin_Deva",
    tgt_lang_tag="eng_Latn"
)
#-----------Construct BT parallel corpus-----------------
# Round-trip EN -> HI -> EN
bt_en_round = translate_indictrans(
    bt_hi,
    model_hi_en,
    tokenizer_hi_en,
    src_lang_tag="hin_Deva",
    tgt_lang_tag="eng_Latn"
)

# Round-trip HI -> EN -> HI
bt_hi_round = translate_indictrans(
    bt_en,
    model_en_hi,
    tokenizer_en_hi,
    src_lang_tag="eng_Latn",
    tgt_lang_tag="hin_Deva"
)

bt_pairs = []

# ENG -> HIN -> ENG
for src, mid, back in zip(mono_en, bt_hi, bt_en_round):
    bt_pairs.append({
        "src": src,
        "tgt": mid,
        "bt_back": back,
        "direction": "ENG_to_HIN"
    })

# HIN -> ENG -> HIN
for src, mid, back in zip(mono_hi, bt_en, bt_hi_round):
    bt_pairs.append({
        "src": src,
        "tgt": mid,
        "bt_back": back,
        "direction": "HIN_to_ENG"
    })

print("Total BT pairs:", len(bt_pairs))

#---------------------Quality filtering---------------
import sacrebleu
import numpy as np

filtered_bt = []

for p in bt_pairs:
    score = sacrebleu.sentence_chrf(
        p["bt_back"],   # back-translated
        [p["src"]]      # original source
    ).score

    if score >= 30.0:   # realistic threshold
        p["bt_chrF"] = round(score, 2)
        filtered_bt.append(p)

print("Filtered BT pairs:", len(filtered_bt))
print("Avg chrF:", np.mean([p["bt_chrF"] for p in filtered_bt]))

#------------------Save JSONL--------------------

import json
from pathlib import Path
import shutil

out_dir = Path("bt_outputs")
out_dir.mkdir(exist_ok=True)

en_hi_path = out_dir / "bt_eng_to_hin.jsonl"
hi_en_path = out_dir / "bt_hin_to_eng.jsonl"

with open(en_hi_path, "w", encoding="utf-8") as fe, \
     open(hi_en_path, "w", encoding="utf-8") as fh:

    for p in filtered_bt:
        if p["direction"] == "ENG_to_HIN":
            fe.write(json.dumps(p, ensure_ascii=False) + "\n")
        else:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

shutil.make_archive("bt_jsonl_outputs", "zip", root_dir=out_dir)
print("✅ JSONL + ZIP saved")

#-------------------------Compute BT Chrf+BLEU----------
import pandas as pd

rows = []

for direction in ["ENG_to_HIN", "HIN_to_ENG"]:
    subset = [p for p in filtered_bt if p["direction"] == direction]
    bt_texts = [p["bt_back"] for p in subset]
    refs     = [p["src"] for p in subset]

    bleu = sacrebleu.corpus_bleu(bt_texts, [refs]).score
    chrf = sacrebleu.corpus_chrf(bt_texts, [refs]).score

    rows.append({
        "Direction": direction,
        "BT_BLEU": round(bleu, 2),
        "BT_chrF": round(chrf, 2),
        "Num_Samples": len(subset)
    })

df_bt_metrics = pd.DataFrame(rows)
display(df_bt_metrics)
#--------------------Convert BT data → Gemma SFT format----------------

def to_gemma_sft(p):
    return {
        "prompt": f"<start_of_turn>user\nTranslate:\n{p['src']}<end_of_turn>\n<start_of_turn>model\n",
        "completion": p["tgt"]
    }

bt_sft_data = [to_gemma_sft(p) for p in filtered_bt]

#----------------------------Merge with original parallel data----------------
from datasets import Dataset, concatenate_datasets

# Define original_parallel_sft from train_set using formatting_prompts_func
original_parallel_sft = train_set.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=train_set.column_names
).to_list()

orig_dataset = Dataset.from_list(original_parallel_sft)   # your real SFT data
bt_dataset   = Dataset.from_list(bt_sft_data)

final_train_dataset = concatenate_datasets([
    orig_dataset,
    bt_dataset.shuffle(seed=42).select(range(len(orig_dataset)))
]).shuffle(seed=42)

print("Final training size:", len(final_train_dataset))

#------------------------Fine-tune Gemma----------------
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    completion_only_loss=True,
    gradient_checkpointing=True,
    save_strategy="no",
    report_to="none",
    output_dir="./gemma_bt_stage2"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=final_train_dataset,
    args=training_args
)

trainer.train()
trainer.model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

#-------------------Load the BT-finetuned Gemma model------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./gemma_bt_stage2"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,
    device_map="auto"
)

model.eval()
print("✅ BT-finetuned Gemma loaded")

#------------------forward inference---------------

from tqdm import tqdm
import sacrebleu
import pandas as pd
import numpy as np

results = []
metrics = {
    "ENG_to_HIN": {"preds": [], "refs": []},
    "HIN_to_ENG": {"preds": [], "refs": []}
}

MAX_NEW_TOKENS = 512

for sample in tqdm(test_set):
    pairs = [
        ("ENG_to_HIN", "Translate to HINDI DEVANAGARI:", sample["src_txt"], sample["tgt_txt"]),
        ("HIN_to_ENG", "Translate to ENGLISH:", sample["tgt_txt"], sample["src_txt"]),
    ]

    for mode, instr, src, ref in pairs:
        prompt = (
            f"<start_of_turn>user\n{instr}\n{src}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1
            )

        pred_tokens = output[0][inputs.input_ids.shape[-1]:]
        pred = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

        results.append({
            "mode": mode,
            "source": src,
            "reference": ref,
            "prediction": pred
        })

        metrics[mode]["preds"].append(pred)
        metrics[mode]["refs"].append(ref)

#--------------------Forward BLEU+chrf-----------------
def calc_metrics(preds, refs):
    return {
        "BLEU": round(sacrebleu.corpus_bleu(preds, [refs]).score, 2),
        "chrF": round(sacrebleu.corpus_chrf(preds, [refs]).score, 2)
    }

forward_scores = {
    "ENG_to_HIN": calc_metrics(
        metrics["ENG_to_HIN"]["preds"],
        metrics["ENG_to_HIN"]["refs"]
    ),
    "HIN_to_ENG": calc_metrics(
        metrics["HIN_to_ENG"]["preds"],
        metrics["HIN_to_ENG"]["refs"]
    )
}

display(pd.DataFrame(forward_scores).T)
#----------------------Back-translation audit------------

def back_translate(results, model, tokenizer):
    audited = []

    for r in tqdm(results):
        if r["mode"] == "ENG_to_HIN":
            instr = "Translate to ENGLISH:"
            src = r["prediction"]
            ref = r["source"]
        else:
            instr = "Translate to HINDI DEVANAGARI:"
            src = r["prediction"]
            ref = r["source"]

        prompt = (
            f"<start_of_turn>user\n{instr}\n{src}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1
            )

        bt_tokens = output[0][inputs.input_ids.shape[-1]:]
        bt = tokenizer.decode(bt_tokens, skip_special_tokens=True).strip()

        r = dict(r)
        r["back_translation"] = bt
        r["bt_chrF"] = round(
            sacrebleu.sentence_chrf(bt, [ref]).score, 2
        )

        audited.append(r)

    return audited

audit_results = back_translate(results, model, tokenizer)

#---------------------Back translated BLEU+chrf-----------

bt_rows = []

for mode in ["ENG_to_HIN", "HIN_to_ENG"]:
    subset = [r for r in audit_results if r["mode"] == mode]

    bt_texts = [r["back_translation"] for r in subset]
    refs = [r["source"] for r in subset]

    bt_rows.append({
        "Direction": mode,
        "BT_BLEU": round(sacrebleu.corpus_bleu(bt_texts, [refs]).score, 2),
        "BT_chrF": round(sacrebleu.corpus_chrf(bt_texts, [refs]).score, 2),
        "Num_Samples": len(subset)
    })

df_bt = pd.DataFrame(bt_rows)
display(df_bt)

#-------------------JSNOL+ZIP---------------------

import json
from pathlib import Path
import shutil

out_dir = Path("final_inference_outputs")
out_dir.mkdir(exist_ok=True)

eng_path = out_dir / "eng_to_hin.jsonl"
hin_path = out_dir / "hin_to_eng.jsonl"

with open(eng_path, "w", encoding="utf-8") as fe, \
     open(hin_path, "w", encoding="utf-8") as fh:

    for r in audit_results:
        line = {
            "src": r["source"],
            "ref": r["reference"],
            "pred": r["prediction"]
        }

        if r["mode"] == "ENG_to_HIN":
            fe.write(json.dumps(line, ensure_ascii=False) + "\n")
        else:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

shutil.make_archive("final_translation_jsonl", "zip", out_dir)
print("✅ JSONL + ZIP saved")

#----------------------Scores saved as xlsx file---------------

xlsx_path = Path("final_metrics_bt.xlsx")

summary = []
for mode in ["ENG_to_HIN", "HIN_to_ENG"]:
    summary.append({
        "Direction": mode,
        "BLEU": forward_scores[mode]["BLEU"],
        "chrF": forward_scores[mode]["chrF"],
        "BT_BLEU": df_bt[df_bt.Direction == mode]["BT_BLEU"].values[0],
        "BT_chrF": df_bt[df_bt.Direction == mode]["BT_chrF"].values[0]
    })

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)
    pd.DataFrame(audit_results).to_excel(writer, sheet_name="Detailed", index=False)

print(f"✅ Metrics saved → {xlsx_path.resolve()}")










