# ============================================================
# 0. VARIABLES (MANDATORY — AS REQUIRED)
# ============================================================
MAX_TRAIN_SAMPLES = None
MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
MAX_SEQ_LEN = MAX_SRC_LEN + MAX_TGT_LEN


# ============================================================
# 1. IMPORTS
# ============================================================
import os
import json
import torch
import sacrebleu
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 3. LOAD + FILTER DATA (ENG → HIN)
# ============================================================
dataset = load_dataset("ai4bharat/pralekha", data_dir="train")

dataset = dataset.filter(
    lambda x: x["src_lang"] == "eng"
    and x["tgt_lang"] == "hin"
    and x["src_txt"] != x["tgt_txt"],
    num_proc=4
)

if MAX_TRAIN_SAMPLES is not None:
    dataset["train"] = dataset["train"].select(range(MAX_TRAIN_SAMPLES))


# ============================================================
# 4. TRAIN / VAL SPLIT
# ============================================================
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_raw = split["train"]
val_raw   = split["test"]


# ============================================================
# 5. FORMAT FOR CHAT TRAINING
# ============================================================
def format_example(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Translate the following sentence from English to Hindi (Devanagari script).\n\n"
                    f"English: {example['src_txt']}"
                ),
            },
            {
                "role": "assistant",
                "content": example["tgt_txt"],
            },
        ]
    }

train_chat = train_raw.map(format_example)
val_chat   = val_raw.map(format_example)


# ============================================================
# 6. MODEL + TOKENIZER
# ============================================================
model_name = "google/gemma-3-270m-it"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)


# ============================================================
# 7. TOKENIZATION WITH STRICT LENGTH CONTROL
# ============================================================
def tokenize_function(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

    return tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LEN
    )

train_tokenized = train_chat.map(tokenize_function, remove_columns=train_chat.column_names)
val_tokenized   = val_chat.map(tokenize_function, remove_columns=val_chat.column_names)


# ============================================================
# 8. LORA CONFIG
# ============================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ]
)


# ============================================================
# 9. TRAINER CONFIG (CHECKPOINTS + VALIDATION LOSS)
# ============================================================
training_args = SFTConfig(
    output_dir="./gemma_en_hi",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    logging_steps=10,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    max_length=MAX_SEQ_LEN,
    packing=False,
    report_to="none",
    completion_only_loss=True,
    load_best_model_at_end=True
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    peft_config=lora_config,
    args=training_args,
    processing_class=tokenizer
)


# ============================================================
# 10. TRAIN
# ============================================================
trainer.train()


# ============================================================
# 11. SAVE FINAL MODEL
# ============================================================
trainer.save_model("./gemma_en_hi/final_model")
tokenizer.save_pretrained("./gemma_en_hi/final_model")


# ============================================================
# 12. CHECKPOINT EVALUATION (VALIDATION SET ONLY)
# ============================================================
def evaluate_checkpoint(checkpoint_path):

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    preds, refs, srcs = [], [], []

    for sample in tqdm(val_raw):

        src = sample["src_txt"]
        ref = sample["tgt_txt"]

        prompt = (
            "Translate the following sentence from English to Hindi (Devanagari script).\n\n"
            f"English: {src}"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SRC_LEN
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_TGT_LEN,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1
            )

        pred = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        preds.append(pred)
        refs.append(ref)
        srcs.append(src)

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    # Save JSONL
    with open(f"{checkpoint_path}/validation_predictions.jsonl", "w", encoding="utf-8") as f:
        for s, r, p in zip(srcs, refs, preds):
            f.write(json.dumps(
                {"src": s, "ref": r, "pred": p},
                ensure_ascii=False
            ) + "\n")

    return bleu, chrf


# ============================================================
# 13. LOOP OVER CHECKPOINTS
# ============================================================
checkpoint_dir = "./gemma_en_hi"
checkpoints = sorted([
    os.path.join(checkpoint_dir, d)
    for d in os.listdir(checkpoint_dir)
    if "checkpoint" in d
])

bleu_scores = []
chrf_scores = []

for ckpt in checkpoints:
    bleu, chrf = evaluate_checkpoint(ckpt)
    bleu_scores.append(bleu)
    chrf_scores.append(chrf)
    print(f"{ckpt} → BLEU: {bleu:.2f}, chrF2: {chrf:.2f}")


# ============================================================
# 14. PLOT TRAINING & VALIDATION LOSS
# ============================================================
logs = trainer.state.log_history

train_loss = [x["loss"] for x in logs if "loss" in x]
eval_loss  = [x["eval_loss"] for x in logs if "eval_loss" in x]

plt.figure()
plt.plot(train_loss)
plt.title("Training Loss")
plt.savefig("training_loss.png")

plt.figure()
plt.plot(eval_loss)
plt.title("Validation Loss")
plt.savefig("validation_loss.png")


# ============================================================
# 15. PLOT BLEU + chrF2 OVER CHECKPOINTS
# ============================================================
plt.figure()
plt.plot(bleu_scores)
plt.title("BLEU over Checkpoints (Validation Set)")
plt.savefig("bleu_plot.png")

plt.figure()
plt.plot(chrf_scores)
plt.title("chrF2 over Checkpoints (Validation Set)")
plt.savefig("chrf_plot.png")

print("All validation-based evaluation artifacts saved successfully.")
