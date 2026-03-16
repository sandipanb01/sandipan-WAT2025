# ==========================================================
# INSTALL
# ==========================================================
# pip install transformers datasets peft sacrebleu accelerate tqdm

# ==========================================================
# IMPORTS
# ==========================================================

import os
import gc
import json
import random
import torch
import sacrebleu

from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets, Dataset

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
# PROMPT BUILDER (ADVISOR VERSION)
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

# ==========================================================
# SAVE COPY FOR REWARD DATASET
# ==========================================================

reward_dataset = train_dataset

# ==========================================================
# TOKENIZATION
# ==========================================================

def tokenize(example):

    tok=tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LEN
    )

    return {"input_ids":tok["input_ids"]}

train_dataset=train_dataset.map(tokenize)
val_dataset=val_dataset.map(tokenize)

# ==========================================================
# PACKING
# ==========================================================

def pack_dataset(dataset):

    packed=[]
    current=[]
    length=0

    for ex in dataset:

        ids=ex["input_ids"]

        if length+len(ids)>MAX_LEN:
            packed.append(current)
            current=[]
            length=0

        current+=ids
        length+=len(ids)

    if current:
        packed.append(current)

    return Dataset.from_dict({
        "input_ids":packed,
        "labels":packed
    })

print("Packing dataset")

train_dataset=pack_dataset(train_dataset)
val_dataset=pack_dataset(val_dataset)

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
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        num_train_epochs=2,
        bf16=True,
        logging_steps=100,
        save_strategy="epoch"
    )

    trainer=Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False)
    )

    trainer.train()

    trainer.save_model(out)

    ckpts.append(out)

# ==========================================================
# MODEL SOUP
# ==========================================================

print("Building model soup")

base=AutoModelForCausalLM.from_pretrained(
MODEL_NAME,
torch_dtype=torch.bfloat16
)

base=build_lora(base)

lora_weights=[]

for p in ckpts:

    m=PeftModel.from_pretrained(base,p)

    sd=m.state_dict()

    lora={k:v for k,v in sd.items() if "lora" in k}

    lora_weights.append(lora)

avg={}

for k in lora_weights[0]:

    avg[k]=torch.stack(
        [w[k] for w in lora_weights]
    ).mean(0)

state=base.state_dict()

for k in avg:
    state[k]=avg[k]

base.load_state_dict(state)

SOUP_DIR=f"{OUTPUT_DIR}/soup"

base.save_pretrained(SOUP_DIR)

# ==========================================================
# BUILD REWARD DATASET
# ==========================================================

print("Building preference dataset")

policy=AutoModelForCausalLM.from_pretrained(
SOUP_DIR,
torch_dtype=torch.bfloat16,
device_map="auto"
)

prefs=[]

sample=reward_dataset.select(range(2000))

for ex in tqdm(sample):

    prompt=ex["prompt"]

    inputs=tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    out=policy.generate(
        **inputs,
        max_new_tokens=GEN_LEN
    )

    gen = out[0][inputs["input_ids"].shape[1]:]

    pred=tokenizer.decode(
        gen,
        skip_special_tokens=True
    )

    ref=ex["reference"]

    if pred!=ref:

        prefs.append({
            "prompt":prompt,
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
        self.head=torch.nn.Linear(
        base.config.hidden_size,1)

    def forward(self,input_ids,attention_mask):

        out=self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        h=out.hidden_states[-1][:,-1,:]

        return self.head(h)

reward_base=AutoModelForCausalLM.from_pretrained(MODEL_NAME)

reward_model=RewardModel(reward_base).to(DEVICE)

opt=torch.optim.AdamW(
reward_model.parameters(),
lr=1e-5
)

print("Training reward model")

for i in tqdm(range(0,len(prefs),BATCH_SIZE)):

    batch=prefs[i:i+BATCH_SIZE]

    chosen=[b["prompt"]+b["chosen"] for b in batch]
    rejected=[b["prompt"]+b["rejected"] for b in batch]

    chosen=tokenizer(
        chosen,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    rejected=tokenizer(
        rejected,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    rc=reward_model(
        input_ids=chosen["input_ids"],
        attention_mask=chosen["attention_mask"]
    )

    rr=reward_model(
        input_ids=rejected["input_ids"],
        attention_mask=rejected["attention_mask"]
    )

    loss=-torch.nn.functional.logsigmoid(rc-rr).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

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

optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)

for step in tqdm(range(RL_STEPS)):

    batch=random.sample(prefs,BATCH_SIZE)

    prompts=[b["prompt"] for b in batch]

    inputs=tokenizer(
        prompts,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    outputs=policy.generate(
        **inputs,
        max_new_tokens=GEN_LEN
    )

    gen=outputs[:,inputs["input_ids"].shape[1]:]

    decoded=tokenizer.batch_decode(
        gen,
        skip_special_tokens=True
    )

    combined=[p+d for p,d in zip(prompts,decoded)]

    tok=tokenizer(
        combined,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    reward=reward_model(
        input_ids=tok["input_ids"],
        attention_mask=tok["attention_mask"]
    )

    logits=policy(**tok).logits
    ref_logits=ref_model(**tok).logits

    log_probs=torch.nn.functional.log_softmax(logits,-1)
    ref_probs=torch.nn.functional.softmax(ref_logits,-1)

    kl=torch.nn.functional.kl_div(
        log_probs,
        ref_probs,
        reduction="batchmean"
    )

    chosen=tok["input_ids"]

    token_logprob=log_probs.gather(
        -1,
        chosen.unsqueeze(-1)
    ).squeeze(-1)

    seq_logprob=token_logprob.mean(-1)

    pg_loss=-(reward.squeeze()*seq_logprob).mean()

    loss=pg_loss+KL_BETA*kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

FINAL_DIR=f"{OUTPUT_DIR}/final_model"

policy.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

# ==========================================================
# EVALUATION
# ==========================================================

print("Evaluating final model")

model=AutoModelForCausalLM.from_pretrained(
FINAL_DIR,
torch_dtype=torch.bfloat16,
device_map="auto"
)

tokenizer=AutoTokenizer.from_pretrained(FINAL_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token=tokenizer.eos_token

for split in language_splits:

    print("\nTesting",split)

    dataset=load_dataset(
    "ai4bharat/Pralekha",
    "test",
    split=split
    )

    dataset=dataset.map(
        build_prompt_wat,
        fn_kwargs={"tokenizer":tokenizer}
    )

    preds=[]
    refs=[]

    for ex in tqdm(dataset):

        inputs=tokenizer(
        ex["prompt"],
        return_tensors="pt"
        ).to(DEVICE)

        out=model.generate(
        **inputs,
        max_new_tokens=GEN_LEN
        )

        gen=out[0][inputs["input_ids"].shape[1]:]

        text=tokenizer.decode(
        gen,
        skip_special_tokens=True
        )

        preds.append(text)
        refs.append(ex["reference"])

    bleu,chrf=calc_metrics(preds,refs)

    print(split,"BLEU:",bleu,"chrF:",chrf)

print("Pipeline complete")
