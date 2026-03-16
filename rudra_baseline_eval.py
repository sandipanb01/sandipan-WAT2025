# ===============================
# IMPORTS
# ===============================

import json
import os
from tqdm import tqdm
import torch
import sacrebleu

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ===============================
# SETTINGS
# ===============================

MODEL_ID = "google/gemma-3-1b-it"

OUTPUT_DIR = "./base_model_eval"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# GPU CLEAN
# ===============================

def free_gpu():
    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ===============================
# METRICS
# ===============================

def calc_metrics(preds, refs):

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    return round(bleu, 2), round(chrf, 2)


# ===============================
# PROMPT BUILDER (Advisor Template)
# ===============================

def build_prompt_wat(example, tokenizer):

    ref = example["tgt_txt"]
    tgt_lang = example["tgt_lang"]

    lang_map = {
        "ben": "Bengali",
        "guj": "Gujarati",
        "hin": "Hindi",
        "kan": "Kannada",
        "mal": "Malayalam",
        "mar": "Marathi",
        "ori": "Odiya",
        "pan": "Punjabi",
        "tam": "Tamil",
        "tel": "Telugu",
        "urd": "Urdu"
    }

    target_lang = lang_map[tgt_lang]

    # ---- PROMPT TEMPLATE ----
    messages = {
        "prompt": [
            {
                "role": "user",
                "content": f"Translate the following sentence from English to {target_lang}.\n\n"
                f"English: {example['src_txt']}",
            }
        ],
        "completion": [
            {"role": "assistant", "content": example["tgt_txt"]}
        ],
    }

    # Inference uses prompt only
    prompt = tokenizer.apply_chat_template(
        messages["prompt"],
        tokenize=False,
        add_generation_prompt=True,
    )

    tokens = tokenizer(
        prompt,
        truncation=True,
        padding=False,
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "reference": ref,
    }


# ===============================
# INFERENCE
# ===============================

def evaluate_wat(model, tokenizer, dataset, batch_size=4):

    predictions = []
    references = []

    print("Running inference...")

    for i in tqdm(range(0, len(dataset), batch_size)):

        batch = dataset[i:i+batch_size]

        padded = tokenizer.pad(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
            padding=True,
            return_tensors="pt",
        )

        input_ids = padded["input_ids"].to(model.device)
        attention_mask = padded["attention_mask"].to(model.device)

        with torch.no_grad():

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4096,
                do_sample=False,
                use_cache=True,
            )

        new_tokens = outputs[:, input_ids.shape[1]:]

        decoded = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )

        predictions.extend(decoded)

        refs = dataset[i:i+batch_size]["reference"]
        references.extend(refs)

    return references, predictions


# ===============================
# LANGUAGE SPLITS
# ===============================

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


# ===============================
# LOAD MODEL
# ===============================

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ===============================
# EVALUATE
# ===============================

for split in language_splits:

    free_gpu()

    print("\n==============================")
    print(f"Evaluating {split}")
    print("==============================")

    dataset = load_dataset(
        "ai4bharat/Pralekha",
        "test",
        split=split
    )

    dataset = dataset.map(
        build_prompt_wat,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=16
    )

    references, predictions = evaluate_wat(
        model,
        tokenizer,
        dataset
    )

    bleu, chrf = calc_metrics(predictions, references)

    results = {
        "model": MODEL_ID,
        "split": split,
        "BLEU": bleu,
        "CHRF": chrf
    }

    results_path = f"{OUTPUT_DIR}/{split}_base.json"

    with open(results_path, "w") as f:
        json.dump(results, f)

    print("Saved:", results_path)


print("\n✅ Base model evaluation complete.")
print("Results stored in:", OUTPUT_DIR)
