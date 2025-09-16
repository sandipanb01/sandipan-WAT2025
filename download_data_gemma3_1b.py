# ===== Hugging Face login =====
from huggingface_hub import login
import os

HF_TOKEN = "YOUR HF_TOKEN"  # replace with your token
login(token=HF_TOKEN)
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# ===== Imports =====
import json
from pathlib import Path
from typing import List, Iterable
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from torch.utils.data import IterableDataset
import torch
from tqdm.auto import tqdm
from transformers import apply_chat_template

# ===== CONFIG =====
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = "outputs/finetuned_gemma_it"
PAIRS: List[str] = [
    "eng_ben","eng_guj","eng_hin","eng_kan","eng_mal",
    "eng_mar","eng_ori","eng_pan","eng_tam","eng_tel","eng_urd"
]
MAX_NEW_TOKENS = 4096
BATCH_SIZE = 1
ACCUM_STEPS = 8
DATA_DIR = Path("/content/pralekha_data")  # where Pralekha JSONL files are stored
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Helper: Build Chat Prompt =====
def build_chat_prompt(src_text, src_lang, tgt_lang):
    """Create chat-style instruction prompt for SFTTrainer"""
    return apply_chat_template(
        instruction=f"Translate from {src_lang} to {tgt_lang}.",
        input=src_text,
        response=None,
        add_eos_token=True
    )

# ===== IterableDataset for Streaming Tokenization =====
class PralekhaIterableDataset(IterableDataset):
    def __init__(self, pairs: List[str], tokenizer, max_length=MAX_NEW_TOKENS):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for pair in self.pairs:
            src_lang, tgt_lang = pair.split("_")
            src_path = DATA_DIR / "train" / pair / f"doc.{src_lang}.jsonl"
            tgt_path = DATA_DIR / "train" / pair / f"doc.{tgt_lang}.jsonl"

            if not src_path.exists() or not tgt_path.exists():
                print(f"⚠ Skipping missing files for {pair}")
                continue

            # ENG -> TGT
            with open(src_path, encoding="utf-8") as fsrc, open(tgt_path, encoding="utf-8") as ftgt:
                for s_line, t_line in zip(fsrc, ftgt):
                    try:
                        src_text = json.loads(s_line)[0]
                        prompt = build_chat_prompt(src_text, src_lang, tgt_lang)
                        tokenized = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length")
                        tokenized["labels"] = tokenized["input_ids"].copy()
                        yield tokenized
                    except Exception:
                        continue

            # TGT -> ENG
            with open(tgt_path, encoding="utf-8") as ftgt, open(src_path, encoding="utf-8") as fsrc:
                for t_line, s_line in zip(ftgt, fsrc):
                    try:
                        src_text = json.loads(t_line)[0]
                        prompt = build_chat_prompt(src_text, tgt_lang, src_lang)
                        tokenized = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length")
                        tokenized["labels"] = tokenized["input_ids"].copy()
                        yield tokenized
                    except Exception:
                        continue

# ===== TOKENIZER =====
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# ===== DATASET =====
print("Preparing streaming IterableDataset...")
train_dataset = PralekhaIterableDataset(PAIRS, tokenizer, MAX_NEW_TOKENS)

# ===== MODEL + LoRA =====
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
)
model.to(DEVICE)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("LoRA applied.")

# ===== TRAINING & SFTTrainer =====
trainer = SFTTrainer(
    model=model,
    dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    max_seq_length=MAX_NEW_TOKENS,
    batch_size=BATCH_SIZE,
    micro_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUM_STEPS,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=True
)

print("Starting fine-tuning...")
trainer.train()
print("Training finished!")

# ===== SAVE MODEL =====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
