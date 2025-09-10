#!/usr/bin/env python
# finetune_gemma3_1b_pralekha_doc_all.py

import argparse
import os
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="Base model (e.g., google/gemma-3-1b or local dir)")
    p.add_argument("--dataset", type=str, default="ai4bharat/Pralekha",
                   help="Dataset HF hub ID (default: ai4bharat/Pralekha)")
    p.add_argument("--output_dir", type=str, default="./gemma-3-1b-ft-doc")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)  # doc-level → long seqs
    p.add_argument("--lr", type=float, default=2e-5)
    return p.parse_args()

def preprocess_dataset(dataset, tokenizer, max_length=4096):
    """
    Convert dataset rows into chat-formatted training examples.
    Assumes fields: 'src' (source doc), 'tgt' (target doc).
    """
    def preprocess(example):
        messages = [
            {"role": "user", "content": example["src"]},
            {"role": "assistant", "content": example["tgt"]}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
        )
    return dataset.map(preprocess, batched=False, remove_columns=dataset.column_names)

def main():
    args = parse_args()

    print(f"🔹 Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="bfloat16",
    )

    print(f"Loading Pralekha dataset: {args.dataset}")
    # Get all configs (language pairs)
    all_configs = load_dataset(args.dataset).builder_configs.keys()
    print(f"Found configs: {list(all_configs)}")

    train_splits = []
    eval_splits = []

    for config in all_configs:
        print(f"Processing config: {config}")
        ds = load_dataset(args.dataset, config)

        train_split = preprocess_dataset(ds["train"], tokenizer, max_length=4096)
        train_splits.append(train_split)

        if "validation" in ds:
            eval_split = preprocess_dataset(ds["validation"], tokenizer, max_length=4096)
            eval_splits.append(eval_split)

    # Merge all language pairs into one dataset
    train_dataset = concatenate_datasets(train_splits)
    eval_dataset = concatenate_datasets(eval_splits) if eval_splits else None

    print(f"Combined train size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Combined eval size: {len(eval_dataset)}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_steps=200,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    print("Saving fine-tuned model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Final doc-level multilingual model saved at {args.output_dir}")

if __name__ == "__main__":
    main()
