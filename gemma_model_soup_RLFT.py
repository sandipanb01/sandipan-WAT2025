# ==========================================================
# INSTALL
# ==========================================================
# pip install transformers datasets peft sacrebleu accelerate tqdm

# ==========================================================
# IMPORTS
# ==========================================================

import os
import json
import random
import torch
import sacrebleu

from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)

# ==========================================================
# CONFIG
# ==========================================================

MODEL_NAME = "google/gemma-3-270m-it"

OUTPUT_DIR = "./outputs"

MAX_LEN = 4800
GEN_LEN = 4096

SEEDS = [1,2,3]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RL_STEPS = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LANGUAGE SPLITS
# ==========================================================

language_splits = [
    "eng_ben",
    "eng_guj",
    "eng_hin",
    "eng_kan",
    "eng_mal",
    "eng_mar",
    "eng_ori",
    "eng_pan",
    "eng_tam",
    "eng_tel",
    "eng_urd"
]

# ==========================================================
# LOAD PRALEKHA DATA
# ==========================================================

print("Loading dataset")

train_sets = []
val_sets = []

for split in language_splits:

    train_sets.append(
        load_dataset("ai4bharat/Pralekha","train",split=split)
    )

    val_sets.append(
        load_dataset("ai4bharat/Pralekha","validation",split=split)
    )

train_dataset = concatenate_datasets(train_sets)
val_dataset = concatenate_datasets(val_sets)

# ==========================================================
# TOKENIZER
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================================
# LANGUAGE MAP
# ==========================================================

lang_map = {
    "ben":"Bengali",
    "guj":"Gujarati",
    "hin":"Hindi",
    "kan":"Kannada",
    "mal":"Malayalam",
    "mar":"Marathi",
    "ori":"Odiya",
    "pan":"Punjabi",
    "tam":"Tamil",
    "tel":"Telugu",
    "urd":"Urdu"
}

# ==========================================================
# PROMPT BUILDER (STRICT VERBATIM)
# ==========================================================

def build_prompt(example):

    target_lang = lang_map[example["tgt_lang"]]

    messages = {
        "prompt":[
            {
                "role":"user",
                "content":f"Translate the following sentence from English to {target_lang}.\n\n"
                          f"English: {example['src_txt']}"
            }
        ],
        "completion":[
            {"role":"assistant","content":example["tgt_txt"]}
        ]
    }

    prompt = tokenizer.apply_chat_template(
        messages["prompt"],
        tokenize=False,
        add_generation_prompt=True
    )

    full = prompt + example["tgt_txt"]

    return {"text":full}

train_dataset = train_dataset.map(build_prompt)
val_dataset = val_dataset.map(build_prompt)

# ==========================================================
# TOKENIZATION
# ==========================================================

def tokenize(example):

    tok = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

    tok["labels"] = tok["input_ids"].copy()

    return tok

train_dataset = train_dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# ==========================================================
# LORA
# ==========================================================

def build_lora(model):

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[  "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    return get_peft_model(model,config)

# ==========================================================
# TRAIN MULTIPLE SEEDS
# ==========================================================

ckpts = []

for seed in SEEDS:

    print("Training seed",seed)

    torch.manual_seed(seed)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model = build_lora(base)

    out_dir = f"{OUTPUT_DIR}/seed_{seed}"

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        num_train_epochs=2,
        bf16=True,
        logging_steps=100,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False)
    )

    trainer.train()

    trainer.save_model(out_dir)

    ckpts.append(out_dir)

# ==========================================================
# MODEL SOUP
# ==========================================================

print("Building model soup")

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)

base = build_lora(base)

lora_weights = []

for p in ckpts:

    m = PeftModel.from_pretrained(base,p)

    sd = m.state_dict()

    lora = {k:v for k,v in sd.items() if "lora" in k}

    lora_weights.append(lora)

avg = {}

for k in lora_weights[0]:

    avg[k] = torch.stack([w[k] for w in lora_weights]).mean(0)

state = base.state_dict()

for k in avg:
    state[k] = avg[k]

base.load_state_dict(state)

SOUP_DIR = f"{OUTPUT_DIR}/soup"

base.save_pretrained(SOUP_DIR)

# ==========================================================
# BUILD PREFERENCE DATASET
# ==========================================================

prefs = []

for ex in train_dataset:

    src = ex["src_txt"]
    tgt = ex["tgt_txt"]
    lang = lang_map[ex["tgt_lang"]]

    prompt = f"Translate the following sentence from English to {lang}.\n\nEnglish: {src}"

    bad = tgt[::-1]

    prefs.append({
        "prompt":prompt,
        "chosen":tgt,
        "rejected":bad
    })

# ==========================================================
# SIMPLE REWARD MODEL
# ==========================================================

class RewardModel(torch.nn.Module):

    def __init__(self,base):

        super().__init__()

        self.base = base

        self.head = torch.nn.Linear(base.config.hidden_size,1)

    def forward(self,input_ids,attention_mask):

        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        h = out.hidden_states[-1][:,-1,:]

        return self.head(h)

base_rm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)

reward_model = RewardModel(base_rm).to(DEVICE)

opt = torch.optim.AdamW(reward_model.parameters(),lr=1e-5)

print("Training reward model")

for epoch in range(1):

    for p in tqdm(prefs):

        c = tokenizer(p["prompt"]+p["chosen"],return_tensors="pt").to(DEVICE)
        r = tokenizer(p["prompt"]+p["rejected"],return_tensors="pt").to(DEVICE)

        rc = reward_model(c["input_ids"],c["attention_mask"])
        rr = reward_model(r["input_ids"],r["attention_mask"])

        loss = -torch.nn.functional.logsigmoid(rc-rr).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

# ==========================================================
# RLHF
# ==========================================================

policy = AutoModelForCausalLM.from_pretrained(
    SOUP_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

policy = build_lora(policy)

optimizer = torch.optim.AdamW(policy.parameters(),lr=1e-6)

print("Running RLHF")

for step in tqdm(range(RL_STEPS)):

    sample = random.choice(prefs)

    prompt = sample["prompt"]

    inputs = tokenizer(prompt,return_tensors="pt").to(DEVICE)

    out = policy.generate(
        **inputs,
        max_new_tokens=GEN_LEN
    )

    gen = out[0][inputs["input_ids"].shape[1]:]

    txt = tokenizer.decode(gen,skip_special_tokens=True)

    tok = tokenizer(txt,return_tensors="pt").to(DEVICE)

    reward = reward_model(tok["input_ids"],tok["attention_mask"])

    loss = -reward.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

FINAL_DIR = f"{OUTPUT_DIR}/final_model"

policy.save_pretrained(FINAL_DIR)

# ==========================================================
# EVALUATION
# ==========================================================

print("Evaluating final model")

model = AutoModelForCausalLM.from_pretrained(
    FINAL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

for split in language_splits:

    print("Testing",split)

    dataset = load_dataset(
        "ai4bharat/Pralekha",
        "test",
        split=split
    )

    preds = []
    refs = []

    for ex in tqdm(dataset):

        tgt = lang_map[ex["tgt_lang"]]

        messages = {
            "prompt":[
                {
                    "role":"user",
                    "content":f"Translate the following sentence from English to {tgt}.\n\n"
                              f"English: {ex['src_txt']}"
                }
            ]
        }

        prompt = tokenizer.apply_chat_template(
            messages["prompt"],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt,return_tensors="pt").to(DEVICE)

        out = model.generate(
            **inputs,
            max_new_tokens=GEN_LEN
        )

        gen = out[0][inputs["input_ids"].shape[1]:]

        text = tokenizer.decode(gen,skip_special_tokens=True)

        preds.append(text)
        refs.append(ex["tgt_txt"])

    bleu = sacrebleu.corpus_bleu(preds,[refs]).score
    chrf = sacrebleu.corpus_chrf(preds,[refs]).score

    print(split,"BLEU:",round(bleu,2),"chrF:",round(chrf,2))

print("Pipeline complete")
