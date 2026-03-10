import os
import json
from tqdm import tqdm
import torch
import sacrebleu

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
from peft import PeftModel
from huggingface_hub import login

# ---------------- Hugging Face Authentication ----------------
HF_TOKEN = "HF_TOKEN"  # <-- Replace with your HF token
login(HF_TOKEN)

# ---------------- GPU Utilities ----------------
def free_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ---------------- Metrics ----------------
def calc_metrics(preds, refs):
    if len(preds) == 0 or len(refs) == 0:
        return 0.0, 0.0
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

# ---------------- Language Map ----------------
LANG_MAP = {
    "ben": "Bengali", "guj": "Gujarati", "hin": "Hindi",
    "kan": "Kannada", "mal": "Malayalam", "mar": "Marathi",
    "ori": "Odiya", "pan": "Punjabi", "tam": "Tamil",
    "tel": "Telugu", "urd": "Urdu",
}

# ---------------- Prompt Builder ----------------
def build_prompt_wat(example, tokenizer):
    src = example["src_txt"]
    tgt_lang = example["tgt_lang"]
    target_lang = LANG_MAP.get(tgt_lang, tgt_lang)
    prompt = f"Translate the following text from English to {target_lang}:\nEnglish: {src}\n{target_lang}: "
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(prompt, truncation=True, padding=False)
    return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"], "reference": example["tgt_txt"]}

# ---------------- Evaluation ----------------
def evaluate_wat(model, tokenizer, dataset, batch_size=4):
    predictions, references = [], []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        padded = tokenizer.pad({"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}, padding=True, return_tensors="pt")
        input_ids = padded["input_ids"].to(model.device)
        attention_mask = padded["attention_mask"].to(model.device)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4096, do_sample=False, use_cache=True)
        new_tokens = outputs[:, input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        predictions.extend(decoded)
        references.extend(batch["reference"])
    return references, predictions

# ---------------- Model Evaluation ----------------
def evaluate_model(base_model, model_path, output_dir, dtype, lora_subfolder=None):
    print(f"\nEvaluating: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=dtype, token=HF_TOKEN)
    if lora_subfolder:
        try:
            model = PeftModel.from_pretrained(model, model_path, subfolder=lora_subfolder, device_map="auto", torch_dtype=dtype, token=HF_TOKEN)
        except Exception as e:
            print(f"Warning: LoRA merge failed, continuing with base weights.\n{e}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    DATASET_NAME = "ai4bharat/Pralekha"
    configs = [c for c in get_dataset_config_names(DATASET_NAME) if c.startswith("eng_")]
    results, bleu_all, chrf_all = {}, [], []
    for config in configs:
        print(f"\n--- Evaluating split: {config} ---")
        dataset = load_dataset(DATASET_NAME, config, split="test")
        dataset = dataset.map(build_prompt_wat, fn_kwargs={"tokenizer": tokenizer})
        refs, preds = evaluate_wat(model, tokenizer, dataset)
        bleu, chrf = calc_metrics(preds, refs)
        results[config] = {"BLEU": bleu, "CHRF": chrf}
        bleu_all.append(bleu)
        chrf_all.append(chrf)
    results["avg"] = {"BLEU": round(sum(bleu_all)/len(bleu_all), 2) if bleu_all else 0.0,
                      "CHRF": round(sum(chrf_all)/len(chrf_all), 2) if chrf_all else 0.0}
    output_path = os.path.join(output_dir, f"wat_{model_path.split('/')[-1]}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results → {output_path}")

# ---------------- Main ----------------
def main():
    base_model = "google/gemma-3-1b-it"  # public base
    adapted_model = "ibm-iitr-mt-research/gemma-3-1b-it_wat_sft"
    output_dir = "./output"
    dtype = torch.bfloat16
    os.makedirs(output_dir, exist_ok=True)
    # Only load main repo and subfolders as LoRA
    lora_checkpoints = ["checkpoint-24438", "checkpoint-48876", "checkpoint-73314", None]  # None means base
    for ckpt in lora_checkpoints:
        free_gpu()
        subfolder = ckpt
        model_path = adapted_model
        evaluate_model(base_model, model_path, output_dir, dtype, lora_subfolder=subfolder)

if __name__ == "__main__":
    main()
