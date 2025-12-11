# ======================================================
# ✅ Universal Fine-tuning + Evaluation (Streaming / Low-Mem / Multi-GPU)
# Hydra-ready | LoRA | Benchmark Metrics | JSONL + ZIP
# ======================================================

import os, json, zipfile, warnings, gc
from pathlib import Path
from itertools import islice
import torch
from datasets import load_dataset, get_dataset_split_names
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu, evaluate
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
import wandb

# ======================================================
# -------------------- CONFIG -------------------------
# ------------------------------------------------------
DEFAULT_CONFIG = dict(
    model_name="google/gemma-3-4b-it",
    max_seq_len=382,
    max_new_tokens=256,
    batch_size=1,
    grad_accum=4,
    max_train_steps=50,
    eval_batch_size=8,
    full_dataset=False,
    max_colab_samples=50,
    beam_mode="A",  # A or B
    target_lang="hin",
    use_wandb=False,
    project_name="translation_benchmark"
)

# ======================================================
# ------------------- PROMPTS -------------------------
# ------------------------------------------------------
INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

def build_prompt(src, src_lang, tgt_lang, example=None, tokenizer=None):
    ex_src, ex_tgt = ("", "")
    if example: ex_src, ex_tgt = example
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        msgs = []
        if ex_src and ex_tgt:
            msgs.append({"role":"user","content":f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{ex_src}"})
            msgs.append({"role":"assistant","content":ex_tgt})
        msgs.append({"role":"user","content":f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"})
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if ex_src and ex_tgt:
        return f"Example translation ({LANG_MAP[src_lang]} → {LANG_MAP[tgt_lang]}):\n{ex_src} → {ex_tgt}\n\nTranslate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"
    else:
        return f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"

def eval_prompt(src, src_lang, tgt_lang):
    return f"Translate the following {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}\nTranslation: "

def extract_answer(full_output, prompt):
    if not full_output: return ""
    try:
        if prompt in full_output: return full_output.split(prompt, 1)[1].strip()
    except: pass
    markers = ["Translation:", "Output:", "Answer:", "Translation -", "Translation —"]
    for m in markers:
        if m in full_output: return full_output.split(m, 1)[1].strip()
    return full_output.strip()

# ======================================================
# ------------------ DATASET (Streaming) --------------
# ------------------------------------------------------
def stream_examples(tokenizer=None, max_samples=None, split_type="dev"):
    dataset_name = "ai4bharat/Pralekha"
    splits = get_dataset_split_names(dataset_name, split_type)
    for split in splits:
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS + ["eng"] or tl not in INDIAN_LANGS + ["eng"]: continue
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        ds = load_dataset(dataset_name, split=split, streaming=True, name=split_type)
        one_shot = ("","")
        for row in islice(ds,50):
            s=row.get("src_txt",""); t=row.get("tgt_txt","")
            if len(s.split())>5 and len(t.split())>5: one_shot=(s,t); break

        ds = load_dataset(dataset_name, split=split, streaming=True, name=split_type)
        count=0
        for row in ds:
            if max_samples and count>=max_samples: break
            s=row.get("src_txt",""); t=row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            use_example = one_shot if (one_shot[0] and one_shot[1] and "dev" not in split) else None
            for s_txt, t_txt, dirn in [(eng,indic,f"eng_{lang}"),(indic,eng,f"{lang}_eng")]:
                prompt = build_prompt(s_txt, dirn.split("_")[0], dirn.split("_")[1], use_example, tokenizer)
                if not prompt.strip(): continue
                yield {"input_text": prompt, "target_text": t_txt, "direction": dirn}
            count+=1

# ======================================================
# ------------------ MODEL PREP -----------------------
# ------------------------------------------------------
def detect_lora_modules(model):
    modules=[]
    for n,m in model.named_modules():
        n_lower=n.lower()
        if any(x in n_lower for x in ["q_proj","k_proj","gate_proj","v_proj","o_proj","up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo"]):
            modules.append(n.split(".")[-1])
    return list(set(modules))

def prepare_model(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    try: model.gradient_checkpointing_enable()
    except: pass
    target_modules = detect_lora_modules(model)
    if not target_modules:
        target_modules = ["q_proj","k_proj","gate_proj","v_proj","o_proj","up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo"]
    lora_cfg = LoraConfig(r=2, lora_alpha=4, target_modules=target_modules, lora_dropout=0.1, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    return model, tok

# ======================================================
# ------------------ TRAINING -------------------------
# ------------------------------------------------------
def train_model(cfg):
    model, tok = prepare_model(cfg)
    ds = IterableDataset.from_iterable(stream_examples(tok, max_samples=cfg.max_colab_samples, split_type="train"))
    sft_cfg = SFTConfig(
        output_dir=".",
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=cfg.max_train_steps,
        logging_steps=10,
        save_strategy="no",
        report_to="wandb" if cfg.use_wandb else "none",
        warmup_ratio=0.1,
        gradient_checkpointing=True
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds, tokenizer=tok)
    trainer.train()
    return model, tok, trainer

# ======================================================
# ------------------ LOW-MEM BATCH EVAL ----------------
# ------------------------------------------------------
def lowmem_eval(model, tok, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    preds, refs, inputs = {}, {}, {}
    target_lang = cfg.target_lang
    for d in [f"eng_{target_lang}", f"{target_lang}_eng"]: preds[d], refs[d], inputs[d] = [], [], []

    # Streaming evaluation
    beam_kwargs = dict(num_beams=3, early_stopping=True) if cfg.beam_mode=="A" else dict(num_beams=3, length_penalty=1.0)
    rouge = evaluate.load("rouge")
    chrf = evaluate.load("chrf")
    bertscore = evaluate.load("bertscore")

    batch_prompts, batch_refs, batch_dirs, batch_inputs, batch_rawlens = [], [], [], [], []

    for ex in stream_examples(tok, max_samples=None, split_type="dev"):
        batch_prompts.append(ex["input_text"])
        batch_refs.append(ex["target_text"])
        batch_dirs.append(ex["direction"])
        batch_inputs.append(ex["input_text"])
        batch_rawlens.append(len(tok(ex["input_text"], add_special_tokens=False)["input_ids"]))
        if len(batch_prompts) >= cfg.eval_batch_size:
            process_batch(model, tok, batch_prompts, batch_refs, batch_dirs, batch_inputs, batch_rawlens, preds, refs, inputs, device, cfg.max_new_tokens, beam_kwargs)
            batch_prompts, batch_refs, batch_dirs, batch_inputs, batch_rawlens = [], [], [], [], []
    if batch_prompts:
        process_batch(model, tok, batch_prompts, batch_refs, batch_dirs, batch_inputs, batch_rawlens, preds, refs, inputs, device, cfg.max_new_tokens, beam_kwargs)

    # Metrics
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True, parents=True)
    metrics = {}
    for d in preds:
        json_file = out_dir/f"{d}_pred_ref.jsonl"
        with open(json_file,"w",encoding="utf-8") as f:
            for inp,p,r in zip(inputs[d],preds[d],refs[d]):
                f.write(json.dumps({"input_text":inp,"pred":p,"ref":r},ensure_ascii=False)+"\n")
        try:
            metrics[d] = {
                "BLEU": sacrebleu.corpus_bleu(preds[d],[refs[d]]).score,
                "ROUGE": rouge.compute(predictions=preds[d], references=refs[d])["rouge1"],
                "chrF": chrf.compute(predictions=preds[d], references=refs[d])["score"],
                "BERTScore": bertscore.compute(predictions=preds[d], references=refs[d], lang="en")["f1"]
            }
        except: metrics[d] = {"BLEU":0.0,"ROUGE":0.0,"chrF":0.0,"BERTScore":0.0}

    with open(out_dir/"metrics.json","w",encoding="utf-8") as f: json.dump(metrics,f,ensure_ascii=False,indent=2)

    # ZIP
    zip_path = out_dir/"benchmark_outputs.zip"
    with zipfile.ZipFile(zip_path,"w") as zipf:
        for f in out_dir.glob("*"):
            zipf.write(f, arcname=f.name)
    print(f"✅ Outputs and metrics saved in {zip_path}")
    return metrics

# ======================================================
# ------------------ MAIN -----------------------------
# ------------------------------------------------------
@hydra.main(config_path=None, config_name=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.merge(DEFAULT_CONFIG, cfg)
    print(OmegaConf.to_yaml(cfg))
    if cfg.use_wandb: wandb.init(project=cfg.project_name, config=cfg)
    model, tok, trainer = train_model(cfg)
    metrics = lowmem_eval(model, tok, cfg)
    if cfg.use_wandb: wandb.log(metrics); wandb.finish()
    print("\n✅ Done. Metrics:", metrics)

if __name__=="__main__":
    main()
