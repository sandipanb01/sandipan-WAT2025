# ======================================================
# ✅ Universal Fine-tuning + Evaluation (Distributed / Streaming / Top-K)
# Hydra-ready | LoRA | JSONL + ZIP + Metrics + Plots + Top-K preview
# Multi-GPU / Multi-Node Compatible
# ======================================================

import os, json, zipfile, warnings, gc
from pathlib import Path
from itertools import islice
import torch
import torch.distributed as dist
from datasets import load_dataset, get_dataset_split_names
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu, evaluate
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
import hydra
from omegaconf import DictConfig, OmegaConf
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
    top_k=3,
    use_wandb=False,
    project_name="translation_benchmark",
    distributed_backend="nccl"
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

# ======================================================
# ------------------ DISTRIBUTED SETUP ----------------
# ------------------------------------------------------
def setup_distributed(cfg):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(cfg.distributed_backend)
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# ======================================================
# ------------------ DATASET --------------------------
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

class PralekhaDataset(IterableDataset):
    def __init__(self, tokenizer, max_samples=None, max_seq_len=382):
        self.max_samples = max_samples
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
    def __iter__(self):
        for ex in stream_examples(self.tok, self.max_samples):
            enc = self.tok(ex["input_text"], truncation=True, max_length=self.max_seq_len)
            if not enc.get("input_ids"): continue
            enc["labels"] = self.tok(ex["target_text"], truncation=True, max_length=self.max_seq_len)["input_ids"]
            yield enc

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

def prepare_model(cfg, device):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16).to(device)
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
def train_model(cfg, device):
    model, tok = prepare_model(cfg, device)
    ds = PralekhaDataset(tok, max_samples=cfg.max_colab_samples if not cfg.full_dataset else None, max_seq_len=cfg.max_seq_len)
    sft_cfg = SFTConfig(
        output_dir=".",
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=cfg.max_train_steps,
        logging_steps=10,
        save_strategy="steps",
        save_steps=20,
        report_to="wandb" if cfg.use_wandb else "none",
        warmup_ratio=0.1,
        gradient_checkpointing=True
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds, tokenizer=tok)
    trainer.train()
    return model, tok, trainer

# ======================================================
# ------------------ BATCH PROCESS --------------------
# ------------------------------------------------------
def process_batch(model, tok, batch_prompts, batch_refs, batch_dirs, batch_inputs, batch_rawlens, preds, refs, inputs, device, max_new_tokens, beam_kwargs, top_k=3):
    enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max(batch_rawlens)+max_new_tokens)
    for k in enc: enc[k] = enc[k].to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            num_return_sequences=top_k,
            do_sample=False,
            **beam_kwargs
        )
    # top-k decoding
    for i in range(len(batch_prompts)):
        prompt_len = batch_rawlens[i]
        sequences = out_ids[i*top_k:(i+1)*top_k].cpu().tolist()
        decoded_k = [tok.decode(seq[prompt_len:], skip_special_tokens=True).strip() for seq in sequences]
        dirn = batch_dirs[i]
        preds[dirn].append(decoded_k)
        refs[dirn].append(batch_refs[i])
        inputs[dirn].append(batch_inputs[i])

# ======================================================
# ------------------ EVALUATION -----------------------
# ------------------------------------------------------
def evaluate_model(model, tok, cfg, rank, world_size):
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    model.eval()
    preds, refs, inputs = {}, {}, {}
    target_lang = cfg.target_lang
    for d in [f"eng_{target_lang}", f"{target_lang}_eng"]:
        preds[d], refs[d], inputs[d] = [], [], []

    beam_kwargs = dict(num_beams=3, early_stopping=True) if cfg.beam_mode=="A" else dict(num_beams=3, length_penalty=1.0)
    batch_prompts, batch_refs, batch_dirs, batch_inputs, batch_rawlens = [], [], [], [], []

    for ex in stream_examples(tok, max_samples=None, split_type="dev"):
        batch_prompts.append(ex["input_text"])
        batch_refs.append(ex["target_text"])
        batch_dirs.append(ex["direction"])
        batch_inputs.append(ex["input_text"])
        batch_rawlens.append(len(tok(ex["input_text"], add_special_tokens=False)["input_ids"]))
        if len(batch_prompts) >= cfg.eval_batch_size:
            process_batch(model, tok, batch_prompts, batch_refs, batch_dirs, batch_inputs, batch_rawlens, preds, refs, inputs, device, cfg.max_new_tokens, beam_kwargs, cfg.top_k)
            batch_prompts, batch_refs, batch_dirs, batch_inputs, batch_rawlens = [], [], [], [], []
    if batch_prompts:
        process_batch(model, tok, batch_prompts, batch_refs, batch_dirs, batch_inputs, batch_rawlens, preds, refs, inputs, device, cfg.max_new_tokens, beam_kwargs, cfg.top_k)

    if rank==0:
        out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True, parents=True)
        metrics = {}
        for d in preds:
            json_file = out_dir/f"{d}_pred_ref.jsonl"
            with open(json_file,"w",encoding="utf-8") as f:
                for inp,p,r in zip(inputs[d],preds[d],refs[d]):
                    f.write(json.dumps({"input_text":inp,"pred_topk":p,"ref":r},ensure_ascii=False)+"\n")
            # Compute BLEU top-1 only
            top1 = [p[0] for p in preds[d]]
            try: metrics[d]=sacrebleu.corpus_bleu(top1,[refs[d]]).score
            except: metrics[d]=0.0
        # ASCII table
        table = [[k, v] for k,v in metrics.items()]
        print("\n📊 Evaluation Metrics:\n", tabulate(table, headers=["Direction","BLEU"]))
        # Save metrics + zip
        with open(out_dir/"metrics.json","w",encoding="utf-8") as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
        zip_path = out_dir/"benchmark_outputs.zip"
        with zipfile.ZipFile(zip_path,"w") as zipf:
            for f in out_dir.glob("*"): zipf.write(f, arcname=f.name)
        print(f"✅ Outputs saved: {zip_path}")
        # Training plots optional
        return metrics
    else:
        return {}

# ======================================================
# ------------------ MAIN -----------------------------
# ------------------------------------------------------
@hydra.main(config_path=None, config_name=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.merge(DEFAULT_CONFIG, cfg)
    rank, world_size = setup_distributed(cfg)
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if cfg.use_wandb and rank==0: wandb.init(project=cfg.project_name, config=cfg)
    model, tok, trainer = train_model(cfg, device)
    metrics = evaluate_model(model, tok, cfg, rank, world_size)
    if cfg.use_wandb and rank==0: wandb.log(metrics); wandb.finish()
    if rank==0: print("\n✅ Done. Metrics:", metrics)
    cleanup_distributed()

if __name__=="__main__":
    main()
