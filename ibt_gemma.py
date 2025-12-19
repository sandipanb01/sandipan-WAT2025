# -*- coding: utf-8 -*-
# ======================================================
# ✅ FULL END-TO-END IBT + UNIVERSAL LoRA PIPELINE
# Bidirectional ENG↔HIN Translation
# Gemma LM + LoRA + FP32 + Streaming + Evaluation + Top-10 Preview
# Explicit N_MONO/N_TEST
# ======================================================

import os, gc, json, zipfile, random, re
from pathlib import Path
from itertools import islice
from functools import partial

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, apply_chat_template
from tqdm import tqdm
import sacrebleu

# ======================================================
# CONFIG
# ======================================================
MODEL_NAME = "google/gemma-3-270m-it"    # Change to 3-4b-it if needed
OUTPUT_DIR = Path("./ibt_universal_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 256
BATCH_SIZE = 1
GRAD_ACCUM = 2
MAX_STEPS = 100

N_MONO = 100   # Explicit number of monolingual samples
N_TEST = 100   # Explicit number of test samples

DIRECTIONS = ["eng_hin", "hin_eng"]

# ======================================================
# LANGUAGE FILTERS
# ======================================================
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")

def is_hindi(x): return bool(DEVANAGARI_RE.search(x))
def is_english(x): return bool(LATIN_RE.search(x))

def filter_lang_pairs(srcs, tgts, src_ok, tgt_ok):
    cs, ct = [], []
    for s, t in zip(srcs, tgts):
        if src_ok(s) and tgt_ok(t):
            cs.append(s)
            ct.append(t)
    return cs, ct

# ======================================================
# STREAM DATA
# ======================================================
def load_pralekha_split(split="eng_hin", n_samples=None):
    ds = load_dataset("ai4bharat/Pralekha", split="train", streaming=True, name="train")
    if n_samples:
        ds = list(islice(ds, n_samples))
    return ds

# ======================================================
# INDIC-TRANS ROUND-0
# ======================================================
from IndicTransToolkit.processor import IndicProcessor
from indicnlp.tokenize.sentence_tokenize import sentence_split

ip = IndicProcessor(inference=True)

def init_indic(ckpt):
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, torch_dtype=torch.float32).to(DEVICE).eval()
    return tok, model

def translate_docs(docs, src, tgt, lang, tok, model):
    outs = []
    for d in docs:
        sents = sentence_split(d, lang)
        inp = ip.preprocess_batch(sents, src_lang=src, tgt_lang=tgt)
        enc = tok(inp, return_tensors="pt", padding=True).to(DEVICE)
        out = model.generate(**enc, max_length=MAX_NEW_TOKENS, use_cache=False)
        dec = tok.batch_decode(out, skip_special_tokens=True)
        outs.append(" ".join(ip.postprocess_batch(dec, lang=tgt)))
    return outs

# ======================================================
# BUILD SFT DATASET
# ======================================================
def build_dataset(src1, tgt1, src2, tgt2):
    rows = []
    for s, t in zip(src1, tgt1):
        rows.append({"messages":[{"role":"user", "content":f"Translate this English text to Hindi:\n{s}"},{"role":"assistant","content":t}]})
    for s, t in zip(src2, tgt2):
        rows.append({"messages":[{"role":"user", "content":f"Translate this Hindi text to English:\n{s}"},{"role":"assistant","content":t}]})
    return Dataset.from_list(rows)

# ======================================================
# MODEL + LoRA INIT
# ======================================================
def prepare_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, target_modules="all-linear"))
    return model, tok

# ======================================================
# TRAINER
# ======================================================
def make_trainer(model, ds, outdir):
    cfg = SFTConfig(
        output_dir=str(outdir),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        max_steps=MAX_STEPS,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        completion_only_loss=True,
        packing=False
    )
    return SFTTrainer(model=model, args=cfg, train_dataset=ds)

# ======================================================
# SAFE GENERATION
# ======================================================
def safe_generate(texts, src, tgt, model, tok):
    outs = []
    for t in texts:
        msgs = [{"role": "user", "content": f"Translate this {src} text to {tgt}:\n{t}"}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        out = model.generate(torch.tensor([ids]).to(model.device), max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        outs.append(tok.decode(out[0][len(ids):], skip_special_tokens=True).strip())
    return outs

# ======================================================
# EVAL Dataset + Loader
# ======================================================
class EvalDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, src_lang, tgt_lang):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
    def __iter__(self):
        for ex in self.dataset:
            if self.src_lang=="eng" and self.tgt_lang=="hin":
                src_text = ex["src_txt"]; ref_text = ex["tgt_txt"]
            else:
                src_text = ex["tgt_txt"]; ref_text = ex["src_txt"]
            messages = [{"role":"user","content":f"Translate this {self.src_lang} text to {self.tgt_lang}:\n{src_text}"},
                        {"role":"assistant","content":""}]
            input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            yield {"input_ids": torch.tensor(input_ids,dtype=torch.long), "reference": ref_text.strip()}

def eval_collate_fn(batch, tokenizer):
    input_ids = [x["input_ids"] for x in batch]
    refs = [x["reference"] for x in batch]
    enc = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], refs

def generate_batch(model, tokenizer, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    preds = []
    for i in range(len(outputs)):
        prompt_len = attention_mask[i].sum().item()
        gen_ids = outputs[i][prompt_len:]
        preds.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    del outputs; torch.cuda.empty_cache()
    return preds

def evaluate_direction(model, tokenizer, src_lang, tgt_lang, max_samples=None, batch_size=1, topk=10):
    model.eval(); torch.cuda.empty_cache()
    raw_ds = load_pralekha_split(n_samples=max_samples)
    eval_ds = EvalDataset(raw_ds, tokenizer, src_lang, tgt_lang)
    collate = partial(eval_collate_fn, tokenizer=tokenizer)
    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collate, num_workers=0)
    preds, refs, processed = [], [], 0
    top_examples = []
    pbar = tqdm(desc=f"Evaluating {src_lang}→{tgt_lang}")
    for input_ids, attention_mask, batch_refs in loader:
        batch_preds = generate_batch(model, tokenizer, input_ids, attention_mask)
        preds.extend(batch_preds); refs.extend(batch_refs)
        for p,r in zip(batch_preds,batch_refs):
            if len(top_examples)<topk:
                top_examples.append({"prediction":p,"reference":r})
        processed += len(batch_refs)
        pbar.update(len(batch_refs))
        if max_samples and processed >= max_samples: break
    pbar.close()
    bleu = sacrebleu.corpus_bleu(preds,[refs]).score
    chrf = sacrebleu.metrics.CHRF(word_order=0).corpus_score(preds,[refs]).score

    # Top-10 preview
    print(f"\nTop-{topk} {src_lang}→{tgt_lang} preview:")
    for i, ex in enumerate(top_examples):
        print(f"[{i+1}] Prediction: {ex['prediction']}\n     Reference:  {ex['reference']}\n")

    print(f"{src_lang}→{tgt_lang} | BLEU={bleu:.2f} | chrF={chrf:.3f}")
    return bleu, chrf

# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__=="__main__":
    # 1️⃣ Prepare model
    model, tok = prepare_model()

    # 2️⃣ Load train/test data
    train_ds = load_pralekha_split(n_samples=N_MONO)
    test_ds = load_pralekha_split(n_samples=N_TEST)
    en_docs = [x["src_txt"] for x in train_ds]; hi_docs = [x["tgt_txt"] for x in train_ds]
    test_en = [x["src_txt"] for x in test_ds]; test_hi = [x["tgt_txt"] for x in test_ds]

    # 3️⃣ Round-0: IndicTrans BT
    tok_en2hin, model_en2hin = init_indic("ai4bharat/indictrans2-en-indic-1B")
    tok_hin2en, model_hin2en = init_indic("ai4bharat/indictrans2-indic-en-1B")
    bt_hi_r0 = translate_docs(en_docs,"eng_Latn","hin_Deva","eng",tok_en2hin,model_en2hin)
    bt_en_r0 = translate_docs(hi_docs,"hin_Deva","eng_Latn","hin",tok_hin2en,model_hin2en)
    del model_en2hin, model_hin2en, tok_en2hin, tok_hin2en; gc.collect(); torch.cuda.empty_cache()

    round0_ds = build_dataset(bt_en_r0, hi_docs, bt_hi_r0, en_docs)
    trainer = make_trainer(model, round0_ds, OUTPUT_DIR/"r0")
    trainer.train(); del trainer; gc.collect(); torch.cuda.empty_cache()

    # 4️⃣ IBT Rounds 1 & 2
    for r in [1,2]:
        gen_en = safe_generate(hi_docs,"Hindi","English",model,tok)
        gen_hi = safe_generate(en_docs,"English","Hindi",model,tok)
        gen_en, hi_f = filter_lang_pairs(gen_en, hi_docs, is_english, is_hindi)
        gen_hi, en_f = filter_lang_pairs(gen_hi, en_docs, is_hindi, is_english)
        ds = build_dataset(gen_en, hi_f, gen_hi, en_f)
        trainer = make_trainer(model, ds, OUTPUT_DIR/f"r{r}")
        trainer.train(); del trainer; gc.collect(); torch.cuda.empty_cache()

    # 5️⃣ Evaluation
    results = {}
    for split in DIRECTIONS:
        src, tgt = split.split("_")
        bleu, chrf = evaluate_direction(model, tok, src, tgt, max_samples=N_TEST, batch_size=1, topk=10)
        results[split] = {"BLEU":bleu,"chrF":chrf}
    print("\n✅ Final Results (ENG↔HIN):", results)

    # 6️⃣ JSONL Export
    jsonl_files = []
    for split in DIRECTIONS:
        src, tgt = split.split("_")
        raw_ds = load_pralekha_split(n_samples=N_TEST)
        eval_ds = EvalDataset(raw_ds, tok, src, tgt)
        loader = DataLoader(eval_ds,batch_size=1,collate_fn=partial(eval_collate_fn,tokenizer=tok))
        save_path = OUTPUT_DIR/f"{split}_pred_refs.jsonl"
        processed = 0
        with open(save_path,"w",encoding="utf-8") as f:
            for input_ids, attention_mask, refs in loader:
                preds = generate_batch(model,tok,input_ids,attention_mask)
                for p,r in zip(preds,refs):
                    f.write(json.dumps({"prediction":p,"reference":r},ensure_ascii=False)+"\n")
                processed += len(refs)
                if processed>=N_TEST: break
        jsonl_files.append(save_path)
        print(f"Saved {processed} examples to {save_path}")

    # 7️⃣ ZIP Export
    zip_path = OUTPUT_DIR/"ibt_universal_results.zip"
    with zipfile.ZipFile(zip_path,"w") as zipf:
        for f in jsonl_files: zipf.write(f, arcname=f.name)
    print(f"✅ ZIP saved at: {zip_path}")
