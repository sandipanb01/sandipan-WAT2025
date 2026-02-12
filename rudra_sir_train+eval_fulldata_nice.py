# ============================================================
# 0. VARIABLES
# ============================================================
MAX_TRAIN_SAMPLES = None
MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
TEST_SAMPLES = None
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
from trl import SFTTrainer, SFTConfig, apply_chat_template
from trl.data_utils import is_conversational


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 3. LOAD + FILTER TRAIN DATA
# ============================================================
dataset = load_dataset("ai4bharat/pralekha", data_dir="train")

dataset = dataset.filter(
    lambda x: x["src_lang"] == "eng"
    and x["tgt_lang"] == "hin"
    and x["src_txt"] != x["tgt_txt"],
    num_proc=4,
)

if MAX_TRAIN_SAMPLES is not None:
    dataset["train"] = dataset["train"].select(range(MAX_TRAIN_SAMPLES))

train_raw = dataset["train"]
print(train_raw)


# ============================================================
# 4. CHAT FORMATTING (FIXED)
# ============================================================
def format_example(example):
    messages = {"prompt": [{"role": "user", "content": "Translate the following sentence from English to HINDI DEVANAGARI.\n\n"
                  f"English: {example['src_txt']}"}],
    "completion": [{"role": "assistant", "content": example["tgt_txt"]}]}

    # messages = {
    #       "messages": [
    #       {
    #           "role": "user",
    #           "content": (
    #               "Translate the following sentence from English to Kannada.\n\n"
    #               f"English: {example['src_txt']}"
    #           ),
    #       },
    #       {
    #           "role": "assistant",
    #           "content": example["tgt_txt"],
    #       },
    #   ],
    # }


    return messages


train_chat = train_raw.map(
    format_example,
    remove_columns=train_raw.column_names,
)

print(train_chat[0])
print("Conversational (before template):", is_conversational(train_chat[0]))


# ============================================================
# 5. MODEL + TOKENIZER
# ============================================================
model_name = "google/gemma-3-270m-it"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32,
)


# ============================================================
# 6. APPLY CHAT TEMPLATE
# ============================================================
train_chat = train_chat.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=4,
    remove_columns=train_chat.column_names,
)

print(train_chat[0])
print("Conversational (after template):", is_conversational(train_chat[0]))


# ============================================================
# 7. LoRA CONFIG
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
    ],
)


# ============================================================
# 8. SFT CONFIG
# ============================================================
sft_config = SFTConfig(
    output_dir="./gemma-en-hi",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    logging_steps=10,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=20,
    packing=False,
    report_to="none",
    completion_only_loss=True,
    dataset_text_field="text",
)


# ============================================================
# 9. TRAINER
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_chat,
    peft_config=lora_config,
    args=sft_config,
    processing_class=tokenizer,
)


# ============================================================
# 10. SANITY CHECK (LOSS MASKING)
# ============================================================
batch = next(iter(trainer.get_train_dataloader()))
tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
labels = batch["labels"][0]

print(tokens)
print(labels)


# ============================================================
# 11. TRAIN
# ============================================================
trainer.train()


# ============================================================
# 12. SAVE FINAL MODEL
# ============================================================
trainer.save_model("./gemma-en-hi/final_model")
tokenizer.save_pretrained("./gemma-en-hi/final_model")


# ============================================================
# 13. TRL VERSION
# ============================================================
import trl
print("TRL version:", trl.__version__)


# ============================================================
# 11. LOAD OFFICIAL TEST SET
# ============================================================

test_dataset = load_dataset("ai4bharat/pralekha", data_dir="test")

test_dataset = test_dataset.filter(
    lambda x: x["src_lang"] == "eng"
    and x["tgt_lang"] == "hin"
    and x["src_txt"] != x["tgt_txt"],
    num_proc=4,
)

# Automatically detect split name
test_split_name = list(test_dataset.keys())[0]
test_raw = test_dataset[test_split_name]

# 🔁 Apply TEST_SAMPLES toggle
if TEST_SAMPLES is not None:
    test_raw = test_raw.select(range(TEST_SAMPLES))


# ============================================================
# 12. CHECKPOINT EVALUATION ON TEST SET
# ============================================================
def evaluate_checkpoint(checkpoint_path):

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    preds, refs, srcs = [], [], []

    for sample in tqdm(test_raw):

        src = sample["src_txt"]
        ref = sample["tgt_txt"]

        prompt = (
            "Translate the following sentence from English to Hindi (Devanagari script).\n\n"
            f"English: {src}"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            #max_length=MAX_SEQ_LEN
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                #max_new_tokens=MAX_TGT_LEN,
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

    with open(f"{checkpoint_path}/test_predictions.jsonl", "w", encoding="utf-8") as f:
        for s, r, p in zip(srcs, refs, preds):
            f.write(json.dumps(
                {"src": s, "ref": r, "pred": p},
                ensure_ascii=False
            ) + "\n")

    return bleu, chrf


# ============================================================
# 13. LOOP THROUGH CHECKPOINTS
# ============================================================
checkpoint_dir = "./gemma-en-hi"
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
# 14. PLOT TRAINING LOSS
# ============================================================
logs = trainer.state.log_history
train_loss = [x["loss"] for x in logs if "loss" in x]

plt.figure()
plt.plot(train_loss)
plt.title("Training Loss")
plt.savefig("training_loss.png")


# ============================================================
# 15. PLOT BLEU + chrF2 OVER CHECKPOINTS
# ============================================================
plt.figure()
plt.plot(bleu_scores)
plt.title("BLEU over Checkpoints (Test Set)")
plt.savefig("bleu_plot.png")

plt.figure()
plt.plot(chrf_scores)
plt.title("chrF2 over Checkpoints (Test Set)")
plt.savefig("chrf_plot.png")

print("All test-set evaluation artifacts saved successfully.")
