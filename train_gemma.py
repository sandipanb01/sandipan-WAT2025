# ======================================================
# ✅ Universal Fine-tuning + Evaluation for Hugging Face instruct/causal LM
# (Streaming, LoRA, Fast Evaluation, Metrics, Top-10 Preview)
# Patched: Top-10 ASCII Table + Scores + Plots + JSONL + ZIP
# Refactored: Manual tokenization removed for training
# Safe tokenization filters applied to avoid empty prompts
# ======================================================

import os
from pathlib import Path
import torch
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from trl import apply_chat_template


# ------------------------------ CONFIG
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./universal_output_best")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 256
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_TRAIN_STEPS = 200
EVAL_BATCH_SIZE = 8
FULL_DATASET = False
MAX_COLAB_SAMPLES = 300

# ------------------------------ BEAM SWITCH
#BEAM_MODE = "A"  # "A" or "B"
#BEAM_KWARGS = dict(num_beams=5, num_return_sequences=1, early_stopping=True) if BEAM_MODE=="A" else dict(num_beams=5, length_penalty=1.0)

INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}


def prepare_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    try: model.gradient_checkpointing_enable()
    except: pass

    return model, tok

# ------------------------------ STREAM + TOKENIZE
def stream_examples_list(max_samples=None, tokenizer=None):
    dataset_name = "ai4bharat/Pralekha"
    config_name = "train"
    splits = get_dataset_split_names(dataset_name, config_name)
    split = splits[0]
    

    def build_prompt(example):
        """Constructs a prompt with up to 5 retrieved passages."""
        prompt = f"Translate this {example['src_lang']} text to {example['tgt_lang']}:\n{example['src_txt']}"
        
        messages = {
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": example["tgt_txt"]}],
            }
        
        return messages

    parts = split.split("_")
    sl, tl = parts
    
    lang = tl if sl=="eng" else sl
    
    dataset = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
    print(dataset)
    dataset = dataset.map(build_prompt)
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    return dataset


# ------------------------------ TRAINING
def train_model(max_samples=None):
    model, tok = prepare_model()
    dataset = stream_examples_list(max_samples=max_samples, tokenizer=tok)
    
    peft_config = LoraConfig(r=256, lora_alpha=16, target_modules="all-linear")
    
    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=MAX_TRAIN_STEPS,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        completion_only_loss=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    # model.save_pretrained(OUTPUT_DIR)
    # tok.save_pretrained(OUTPUT_DIR)
    return model, tok, trainer
    
# ------------------------------ MAIN
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    max_samples = None if FULL_DATASET else MAX_COLAB_SAMPLES

    # 1️⃣ Train
    model, tok, trainer = train_model(max_samples=max_samples)

    
