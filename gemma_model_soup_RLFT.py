!pip install flash-attn
# ==========================================================
# IMPORTS
# ==========================================================

import os
import gc
import random
import torch
import sacrebleu

from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)

from trl import SFTTrainer, SFTConfig

# ==========================================================
# SPEED SETTINGS
# ==========================================================

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ==========================================================
# CONFIG
# ==========================================================

MODEL_NAME = "google/gemma-3-270m-it"

OUTPUT_DIR = "./outputs"

MAX_LEN = 4800
GEN_LEN = 4096

SEEDS = [1, 2, 3]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RL_STEPS = 500
BATCH_SIZE = 4
KL_BETA = 0.02

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# UTILITIES
# ==========================================================


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)


# ==========================================================
# LANGUAGE SPLITS
# ==========================================================

language_splits = [
    "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal",
    "eng_mar", "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd"
]

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading Pralekha dataset")

train_sets = []
val_sets = []

for split in language_splits:

    train_sets.append(
        load_dataset("ai4bharat/Pralekha", "train", split=split)
    )

    val_sets.append(
        load_dataset("ai4bharat/Pralekha", "dev", split=split)
    )

train_dataset = concatenate_datasets(train_sets)
val_dataset = concatenate_datasets(val_sets)

# ==========================================================
# TOKENIZER
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

# ==========================================================
# PROMPT BUILDER
# ==========================================================


def build_prompt_wat(example):

    lang_map = {
        "ben": "Bengali", "guj": "Gujarati", "hin": "Hindi", "kan": "Kannada",
        "mal": "Malayalam", "mar": "Marathi", "ori": "Odiya", "pan": "Punjabi",
        "tam": "Tamil", "tel": "Telugu", "urd": "Urdu"
    }

    target_lang = lang_map[example["tgt_lang"]]

    return {
        "prompt": [{
            "role": "user",
            "content": f"Translate the following sentence from English to {target_lang}.\n\n"
                       f"English: {example['src_txt']}",
        }],
        "completion": [{
            "role": "assistant",
            "content": example["tgt_txt"]
        }]
    }


train_dataset = train_dataset.map(
    build_prompt_wat,
    remove_columns=train_dataset.column_names,
    num_proc=32
)

val_dataset = val_dataset.map(
    build_prompt_wat,
    remove_columns=val_dataset.column_names,
    num_proc=32
)

reward_dataset = train_dataset

# ==========================================================
# LORA
# ==========================================================


def build_lora(model):

    model.config.use_cache = False

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    return get_peft_model(model, config)


# ==========================================================
# TRAIN MULTIPLE SEEDS
# ==========================================================

ckpts = []

for seed in SEEDS:

    print("Training seed", seed)

    torch.manual_seed(seed)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.eos_token_id = tokenizer.eos_token_id

    model = build_lora(base)

    out = f"{OUTPUT_DIR}/seed_{seed}"

    config = SFTConfig(
        output_dir=out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=100,
        bf16=True,
        max_length=MAX_LEN,
        packing=False,
        completion_only_loss=True,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=config,
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model(out)

    ckpts.append(out)

    del trainer
    del model
    del base

    free_gpu()

# ==========================================================
# MODEL SOUP
# ==========================================================

print("Building model soup")

lora_weights = []

for path in ckpts:

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base = build_lora(base)

    model = PeftModel.from_pretrained(base, path)

    lora_state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if "lora_" in k
    }

    lora_weights.append(lora_state)

avg_lora = {}

for k in lora_weights[0]:

    avg_lora[k] = torch.stack(
        [w[k] for w in lora_weights]
    ).mean(0)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

base.config.pad_token_id = tokenizer.pad_token_id
base.config.eos_token_id = tokenizer.eos_token_id

base = build_lora(base)

base.load_state_dict(avg_lora, strict=False)

SOUP_DIR = f"{OUTPUT_DIR}/soup"
base.save_pretrained(SOUP_DIR)

free_gpu()

# ==========================================================
# BUILD PREFERENCE DATASET
# ==========================================================

print("Building preference dataset")

policy = AutoModelForCausalLM.from_pretrained(
    SOUP_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

policy.config.pad_token_id = tokenizer.pad_token_id

prefs = []

sample = reward_dataset.select(range(5000))

for i in tqdm(range(0, len(sample), BATCH_SIZE)):

    batch = sample.select(range(i, min(i + BATCH_SIZE, len(sample))))

    prompts = [p[0]["content"] for p in batch["prompt"]]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(DEVICE)

    with torch.no_grad():
        outputs = policy.generate(   
            **inputs,
            max_new_tokens=GEN_LEN,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    for j in range(len(prompts)):

        prompt_len = inputs["attention_mask"][j].sum()

        gen = outputs[j][prompt_len:]

        pred = tokenizer.decode(gen, skip_special_tokens=True)

        ref = batch["completion"][j][0]["content"]

        if pred != ref:

            prefs.append({
                "prompt": prompts[j],
                "chosen": ref,
                "rejected": pred
            })

free_gpu()

# ==========================================================
# REWARD MODEL
# ==========================================================


class RewardModel(torch.nn.Module):

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = torch.nn.Linear(base.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):

        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        lengths = attention_mask.sum(dim=1) - 1

        h = out.hidden_states[-1][
            torch.arange(len(lengths)), lengths
        ]

        # make dtype match the head
        h = h.to(self.head.weight.dtype)

        return self.head(h)

reward_base = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
reward_base.config.pad_token_id = tokenizer.pad_token_id

reward_model = RewardModel(reward_base).to(DEVICE)
reward_model.train()

opt = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)

print("Training reward model")

for i in tqdm(range(0, len(prefs), BATCH_SIZE)):

    batch = prefs[i:i + BATCH_SIZE]

    chosen = [b["prompt"] + b["chosen"] for b in batch]
    rejected = [b["prompt"] + b["rejected"] for b in batch]

    chosen = tokenizer(
        chosen,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(DEVICE)

    rejected = tokenizer(
        rejected,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(DEVICE)

    rc = reward_model(chosen["input_ids"], chosen["attention_mask"])
    rr = reward_model(rejected["input_ids"], rejected["attention_mask"])

    loss = -torch.nn.functional.logsigmoid(rc - rr).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

free_gpu()

# ==========================================================
# RLHF
# ==========================================================

print("Running RLHF")

policy = AutoModelForCausalLM.from_pretrained(
    SOUP_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

policy.config.pad_token_id = tokenizer.pad_token_id
policy.config.eos_token_id = tokenizer.eos_token_id

ref_model = AutoModelForCausalLM.from_pretrained(
    SOUP_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

ref_model.config.pad_token_id = tokenizer.pad_token_id
ref_model.config.eos_token_id = tokenizer.eos_token_id

ref_model.eval()

for p in ref_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)

for step in tqdm(range(RL_STEPS)):

    batch = random.sample(prefs, BATCH_SIZE)

    prompts = [b["prompt"] for b in batch]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(DEVICE)

    with torch.no_grad():
        outputs = policy.generate(
            **inputs,
            max_new_tokens=GEN_LEN,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    gen = []
    for i in range(outputs.shape[0]):
        prompt_len = inputs["attention_mask"][i].sum()
        gen.append(outputs[i, prompt_len:])

    gen = torch.nn.utils.rnn.pad_sequence(gen, batch_first=True)

    full_sequences = torch.cat([inputs["input_ids"], gen], dim=1)

    attention_mask = (full_sequences != tokenizer.pad_token_id).long()

    reward = reward_model(full_sequences, attention_mask).squeeze(-1)

    reward = (reward - reward.mean()) / (reward.std() + 1e-6)

    outputs_policy = policy(
        input_ids=full_sequences,
        attention_mask=attention_mask
    )

    outputs_ref = ref_model(
        input_ids=full_sequences,
        attention_mask=attention_mask
    )

    logits = torch.nan_to_num(outputs_policy.logits)
    ref_logits = torch.nan_to_num(outputs_ref.logits)

    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    ref_log_probs = torch.log_softmax(ref_logits[:, :-1], dim=-1)

    targets = full_sequences[:, 1:]

    token_logprob = log_probs.gather(
        -1,
        targets.unsqueeze(-1)
    ).squeeze(-1)

    ref_token_logprob = ref_log_probs.gather(
        -1,
        targets.unsqueeze(-1)
    ).squeeze(-1)

    mask = attention_mask[:, 1:]

    prompt_lens = inputs["attention_mask"].sum(dim=1)

    response_mask = mask.clone()

    for i, l in enumerate(prompt_lens):
        response_mask[i, :l] = 0

    kl = (token_logprob - ref_token_logprob)
    kl = (kl * response_mask).sum(-1) / (response_mask.sum(-1) + 1e-8)
    kl = kl.mean()

    seq_logprob = (token_logprob * response_mask).sum(-1) / (response_mask.sum(-1) + 1e-8)

    pg_loss = -(reward * seq_logprob).mean()

    loss = pg_loss + KL_BETA * kl

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

    optimizer.step()

free_gpu()

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

FINAL_DIR = f"{OUTPUT_DIR}/final_model"

policy.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print("Pipeline complete")
# ==========================================================
# BATCHED EVALUATION ON PRALEKHA TEST SET
# ==========================================================

import json

print("\nStarting evaluation on Pralekha TEST set")

TEST_BATCH_SIZE = 8

RESULT_DIR = f"{OUTPUT_DIR}/evaluation"
os.makedirs(RESULT_DIR, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    FINAL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(FINAL_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


def build_eval_prompt(example):

    lang_map = {
        "ben": "Bengali", "guj": "Gujarati", "hin": "Hindi", "kan": "Kannada",
        "mal": "Malayalam", "mar": "Marathi", "ori": "Odiya", "pan": "Punjabi",
        "tam": "Tamil", "tel": "Telugu", "urd": "Urdu"
    }

    target_lang = lang_map[example["tgt_lang"]]

    return (
        f"Translate the following sentence from English to {target_lang}.\n\n"
        f"English: {example['src_txt']}"
    )


all_scores = []

for split in language_splits:

    print(f"\nEvaluating {split}")

    test_dataset = load_dataset(
        "ai4bharat/Pralekha",
        "test",
        split=split
    )

    preds = []
    refs = []

    out_file = f"{RESULT_DIR}/{split}_predictions.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:

        for i in tqdm(range(0, len(test_dataset), TEST_BATCH_SIZE)):

            batch = test_dataset.select(
                range(i, min(i + TEST_BATCH_SIZE, len(test_dataset)))
            )

            prompts = [build_eval_prompt(x) for x in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            ).to(DEVICE)

            with torch.no_grad():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=GEN_LEN,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            for j in range(outputs.shape[0]):

                prompt_len = inputs["attention_mask"][j].sum()

                gen_tokens = outputs[j][prompt_len:]

                pred = tokenizer.decode(
                    gen_tokens,
                    skip_special_tokens=True
                )

                ref = batch[j]["tgt_txt"]

                preds.append(pred)
                refs.append(ref)

                record = {
                    "split": split,
                    "src": batch[j]["src_txt"],
                    "prediction": pred,
                    "reference": ref
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ======================================================
    # METRICS
    # ======================================================

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    chrfpp = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score

    score_record = {
        "split": split,
        "BLEU": round(bleu, 2),
        "CHRF": round(chrf, 2),
        "CHRF++": round(chrfpp, 2)
    }

    all_scores.append(score_record)

    print(score_record)

# ==========================================================
# SAVE METRIC SUMMARY
# ==========================================================

score_file = f"{RESULT_DIR}/scores.jsonl"

with open(score_file, "w") as f:
    for s in all_scores:
        f.write(json.dumps(s) + "\n")

print("\nEvaluation finished")
print("Scores saved to:", score_file)
print("Predictions saved to:", RESULT_DIR)
