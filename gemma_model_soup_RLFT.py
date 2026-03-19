# ==========================================================
# IMPORTS
# ==========================================================

import os
import gc
import random
import torch
import sacrebleu

from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import DataCollatorForSeq2Seq

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    Trainer,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)

# ==========================================================
# SPEED SETTINGS
# ==========================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
    return round(bleu,2), round(chrf,2)

# ==========================================================
# LANGUAGE SPLITS
# ==========================================================

language_splits = [
"eng_ben","eng_guj","eng_hin","eng_kan","eng_mal",
"eng_mar","eng_ori","eng_pan","eng_tam","eng_tel","eng_urd"
]

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading Pralekha dataset")

train_sets=[]
val_sets=[]

for split in language_splits:

    train_sets.append(
        load_dataset("ai4bharat/Pralekha","train",split=split)
    )

    val_sets.append(
        load_dataset("ai4bharat/Pralekha","dev",split=split)
    )

train_dataset = concatenate_datasets(train_sets)
val_dataset = concatenate_datasets(val_sets)

# ==========================================================
# TOKENIZER
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True
)

# ==========================================================
# PROMPT BUILDER
# ==========================================================

def build_prompt_wat(example, tokenizer):

    ref = example["tgt_txt"]
    tgt_lang = example["tgt_lang"]

    lang_map = {
        "ben":"Bengali","guj":"Gujarati","hin":"Hindi","kan":"Kannada",
        "mal":"Malayalam","mar":"Marathi","ori":"Odiya","pan":"Punjabi",
        "tam":"Tamil","tel":"Telugu","urd":"Urdu"
    }

    target_lang = lang_map[tgt_lang]

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

    return {
        "text":full,
        "prompt":prompt,
        "reference":ref
    }

train_dataset=train_dataset.map(
    build_prompt_wat,
    fn_kwargs={"tokenizer":tokenizer}
)

val_dataset=val_dataset.map(
    build_prompt_wat,
    fn_kwargs={"tokenizer":tokenizer}
)

reward_dataset = train_dataset.select(range(len(train_dataset)))

# ==========================================================
# TOKENIZATION
# ==========================================================

def tokenize(example):

    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=False
    )["input_ids"]

    full = example["prompt"] + example["reference"]

    tok = tokenizer(
        full,
        truncation=True,
        max_length=MAX_LEN
    )

    labels = tok["input_ids"].copy()

    prompt_len = len(prompt_ids)

    labels[:prompt_len] = [-100]*prompt_len

    return {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "labels": labels
    }

train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# ==========================================================
# LORA
# ==========================================================

def build_lora(model):

    model.config.use_cache = False

    config=LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    return get_peft_model(model,config)

# ==========================================================
# TRAIN MULTIPLE SEEDS
# ==========================================================

ckpts=[]

for seed in SEEDS:

    print("Training seed",seed)

    torch.manual_seed(seed)

    base=AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model=build_lora(base)

    out=f"{OUTPUT_DIR}/seed_{seed}"

    args=TrainingArguments(
        output_dir=out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        num_train_epochs=2,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=100,
        save_strategy="epoch"
    )

    trainer=Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(out)

    ckpts.append(out)

    free_gpu()

# ==========================================================
# MODEL SOUP
# ==========================================================

print("Building model soup")

lora_weights=[]

for p in ckpts:

    base=AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base=build_lora(base)

    m=PeftModel.from_pretrained(base,p)

    sd=m.state_dict()

    lora={k:v for k,v in sd.items() if "lora" in k}
    lora_weights.append(lora)

avg={}

for k in lora_weights[0]:

    avg[k]=torch.stack(
        [w[k] for w in lora_weights]
    ).mean(0)

base=AutoModelForCausalLM.from_pretrained(MODEL_NAME)
base=build_lora(base)

state=base.state_dict()

for k in avg:
    state[k]=avg[k]

base.load_state_dict(state)

SOUP_DIR=f"{OUTPUT_DIR}/soup"
base.save_pretrained(SOUP_DIR)

free_gpu()

# ==========================================================
# BUILD PREFERENCE DATASET
# ==========================================================

print("Building preference dataset")

policy=AutoModelForCausalLM.from_pretrained(
SOUP_DIR,
torch_dtype=torch.bfloat16,
device_map="auto"
)

prefs=[]

sample=reward_dataset.select(range(5000))

for i in tqdm(range(0,len(sample),BATCH_SIZE)):

    batch = sample[i:i+BATCH_SIZE]

    prompts=[ex["prompt"] for ex in batch]

    inputs=tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        out=policy.generate(
            **inputs,
            max_new_tokens=GEN_LEN,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )

    for j,ex in enumerate(batch):

        gen = out[j][inputs["input_ids"].shape[1]:]

        pred=tokenizer.decode(
            gen,
            skip_special_tokens=True
        )

        ref=ex["reference"]

        if pred!=ref:

            prefs.append({
                "prompt":ex["prompt"],
                "chosen":ref,
                "rejected":pred
            })

free_gpu()

# ==========================================================
# REWARD MODEL
# ==========================================================

class RewardModel(torch.nn.Module):

    def __init__(self,base):

        super().__init__()

        self.base=base
        self.head=torch.nn.Linear(
        base.config.hidden_size,1)

    def forward(self,input_ids,attention_mask):

        out=self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        lengths = attention_mask.sum(dim=1)-1

        h = out.hidden_states[-1][
            torch.arange(len(lengths)), lengths
        ]

        return self.head(h)

reward_base=AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

reward_model=RewardModel(reward_base).to(DEVICE)
reward_model.train()

opt=torch.optim.AdamW(
reward_model.parameters(),
lr=1e-5
)

print("Training reward model")

for i in tqdm(range(0,len(prefs),BATCH_SIZE)):

    batch=prefs[i:i+BATCH_SIZE]

    chosen=[b["prompt"]+b["chosen"] for b in batch]
    rejected=[b["prompt"]+b["rejected"] for b in batch]

    chosen=tokenizer(chosen,return_tensors="pt",padding=True,truncation=True).to(DEVICE)
    rejected=tokenizer(rejected,return_tensors="pt",padding=True,truncation=True).to(DEVICE)

    rc=reward_model(chosen["input_ids"],chosen["attention_mask"])
    rr=reward_model(rejected["input_ids"],rejected["attention_mask"])

    loss=-torch.nn.functional.logsigmoid(rc-rr).mean()

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

ref_model = AutoModelForCausalLM.from_pretrained(
SOUP_DIR,
torch_dtype=torch.bfloat16,
device_map="auto"
)

ref_model.eval()

for p in ref_model.parameters():
    p.requires_grad=False

optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)

for step in tqdm(range(RL_STEPS)):

    batch=random.sample(prefs,BATCH_SIZE)

    prompts=[b["prompt"] for b in batch]

    inputs=tokenizer(prompts,return_tensors="pt",padding=True,truncation=True).to(DEVICE)

    with torch.no_grad():
        outputs=policy.generate(
            **inputs,
            max_new_tokens=GEN_LEN,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )

    gen=outputs[:,inputs["input_ids"].shape[1]:]

    full_sequences = torch.cat([inputs["input_ids"], gen], dim=1)

    attention_mask = torch.ones_like(full_sequences).to(DEVICE)

    reward = reward_model(
        full_sequences,
        attention_mask
    )

    reward=(reward-reward.mean())/(reward.std()+1e-8)

    outputs_policy = policy(
        input_ids=full_sequences,
        attention_mask=attention_mask
    )

    outputs_ref = ref_model(
        input_ids=full_sequences,
        attention_mask=attention_mask
    )

    logits = outputs_policy.logits
    ref_logits = outputs_ref.logits

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

    prompt_len = inputs["input_ids"].shape[1]

    response_mask = mask.clone()
    response_mask[:, :prompt_len] = 0

    kl = ((token_logprob - ref_token_logprob) * response_mask).sum(-1).mean()

    seq_logprob = (token_logprob * response_mask).sum(-1) / response_mask.sum(-1)

    pg_loss = -(reward.squeeze() * seq_logprob).mean()

    loss = pg_loss + KL_BETA * kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

free_gpu()

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

FINAL_DIR=f"{OUTPUT_DIR}/final_model"

policy.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print("Pipeline complete")
