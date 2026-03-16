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

BATCH_SIZE = 4
KL_BETA = 0.02

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LANGUAGE SPLITS
# ==========================================================

language_splits = [
    "eng_ben","eng_guj","eng_hin","eng_kan","eng_mal",
    "eng_mar","eng_ori","eng_pan","eng_tam","eng_tel","eng_urd"
]

# ==========================================================
# LOAD PRALEKHA DATA
# ==========================================================

print("Loading dataset")

train_sets = []
val_sets = []

for split in language_splits:

    train_sets.append(load_dataset("ai4bharat/Pralekha","train",split=split))
    val_sets.append(load_dataset("ai4bharat/Pralekha","validation",split=split))

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
    "ben":"Bengali","guj":"Gujarati","hin":"Hindi","kan":"Kannada",
    "mal":"Malayalam","mar":"Marathi","ori":"Odiya","pan":"Punjabi",
    "tam":"Tamil","tel":"Telugu","urd":"Urdu"
}

# ==========================================================
# PROMPT BUILDER (UNCHANGED)
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
# TOKENIZATION (PACKED TRAINING)
# ==========================================================

print("Tokenizing datasets")

def tokenize(example):

    tok = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LEN
    )

    return {"input_ids": tok["input_ids"]}

train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

# ==========================================================
# PACKING FUNCTION
# ==========================================================

def pack_dataset(dataset):

    packed_input_ids = []

    current = []

    current_len = 0

    for ex in dataset:

        ids = ex["input_ids"]

        if current_len + len(ids) > MAX_LEN:

            packed_input_ids.append(current)

            current = []
            current_len = 0

        current.extend(ids)
        current_len += len(ids)

    if len(current) > 0:
        packed_input_ids.append(current)

    packed = {
        "input_ids": packed_input_ids,
        "labels": packed_input_ids
    }

    return packed

print("Packing sequences")

train_packed = pack_dataset(train_dataset)
val_packed = pack_dataset(val_dataset)

from datasets import Dataset

train_dataset = Dataset.from_dict(train_packed)
val_dataset = Dataset.from_dict(val_packed)

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# ==========================================================
# LORA
# ==========================================================

def build_lora(model):

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
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
# GREEDY MODEL SOUP
# ==========================================================

print("Building greedy soup")

def eval_model(path):

    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(MODEL_NAME),
        path
    ).to(DEVICE)

    sample = val_dataset.select(range(200))

    preds=[]
    refs=[]

    for ex in sample:

        ids = ex["input_ids"].unsqueeze(0).to(DEVICE)

        out = model.generate(ids,max_new_tokens=4096)

        text = tokenizer.decode(out[0],skip_special_tokens=True)

        preds.append(text)
        refs.append(tokenizer.decode(ex["labels"],skip_special_tokens=True))

    bleu = sacrebleu.corpus_bleu(preds,[refs]).score

    return bleu

scores = [(p,eval_model(p)) for p in ckpts]

scores.sort(key=lambda x:x[1],reverse=True)

soup = scores[0][0]

print("Best checkpoint:",soup)

SOUP_DIR = f"{OUTPUT_DIR}/soup"

base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
base = build_lora(base)

best = PeftModel.from_pretrained(base,soup)

best.save_pretrained(SOUP_DIR)

# ==========================================================
# BUILD REWARD DATASET (FIXED)
# ==========================================================

print("Building preference dataset")

policy = AutoModelForCausalLM.from_pretrained(
    SOUP_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prefs=[]

sample = train_dataset.select(range(2000))

for ex in tqdm(sample):

    src = ex["input_ids"].unsqueeze(0).to(DEVICE)

    out = policy.generate(src,max_new_tokens=4096)

    pred = tokenizer.decode(out[0],skip_special_tokens=True)

    ref = tokenizer.decode(ex["labels"],skip_special_tokens=True)

    if pred!=ref:

        prefs.append({
            "chosen":ref,
            "rejected":pred
        })

# ==========================================================
# REWARD MODEL
# ==========================================================

class RewardModel(torch.nn.Module):

    def __init__(self,base):

        super().__init__()

        self.base=base

        self.head=torch.nn.Linear(base.config.hidden_size,1)

    def forward(self,input_ids,attention_mask):

        out=self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        h=out.hidden_states[-1][:,-1,:]

        return self.head(h)

base_rm = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

reward_model = RewardModel(base_rm).to(DEVICE)

opt = torch.optim.AdamW(reward_model.parameters(),lr=1e-5)

print("Training reward model")

for i in tqdm(range(0,len(prefs),BATCH_SIZE)):

    batch=prefs[i:i+BATCH_SIZE]

    chosen = tokenizer(
        [b["chosen"] for b in batch],
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    rejected = tokenizer(
        [b["rejected"] for b in batch],
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    rc = reward_model(**chosen)
    rr = reward_model(**rejected)

    loss = -torch.nn.functional.logsigmoid(rc-rr).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

# ==========================================================
# RLHF (KL REGULARIZED)
# ==========================================================

policy = AutoModelForCausalLM.from_pretrained(
    SOUP_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

ref_model = AutoModelForCausalLM.from_pretrained(
    SOUP_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

optimizer = torch.optim.AdamW(policy.parameters(),lr=1e-6)

print("Running RLHF")

for step in tqdm(range(RL_STEPS)):

    batch=random.sample(prefs,BATCH_SIZE)

    texts=[b["chosen"] for b in batch]

    tok=tokenizer(texts,return_tensors="pt",padding=True).to(DEVICE)

    logits=policy(**tok).logits
    ref_logits=ref_model(**tok).logits

    reward=reward_model(**tok)

    kl=torch.nn.functional.kl_div(
        logits.log_softmax(-1),
        ref_logits.softmax(-1),
        reduction="batchmean"
    )

    loss=-(reward.mean()-KL_BETA*kl)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

FINAL_DIR=f"{OUTPUT_DIR}/final_model"

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

    dataset = load_dataset("ai4bharat/Pralekha","test",split=split)

    preds=[]
    refs=[]

    for ex in tqdm(dataset):

        tgt = lang_map[ex["tgt_lang"]]

        messages={
            "prompt":[
                {
                    "role":"user",
                    "content":f"Translate the following sentence from English to {tgt}.\n\n"
                              f"English: {ex['src_txt']}"
                }
            ]
        }

        prompt=tokenizer.apply_chat_template(
            messages["prompt"],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs=tokenizer(prompt,return_tensors="pt").to(DEVICE)

        out=model.generate(
            **inputs,
            max_new_tokens=GEN_LEN
        )

        gen=out[0][inputs["input_ids"].shape[1]:]

        text=tokenizer.decode(gen,skip_special_tokens=True)

        preds.append(text)
        refs.append(ex["tgt_txt"])

    bleu=sacrebleu.corpus_bleu(preds,[refs]).score
    chrf=sacrebleu.corpus_chrf(preds,[refs]).score

    print(split,"BLEU:",round(bleu,2),"chrF:",round(chrf,2))

print("Pipeline complete")
