import json
from pathlib import Path
import sacrebleu
import torch
import re 
from transformers import AutoModelForCausalLM, AutoTokenizer

eng_file = Path("/home/janvi/transformers/WAT25_IndicDoc/data/dev/eng_hin/doc.eng.fixed.jsonl")
tel_file = Path("/home/janvi/transformers/WAT25_IndicDoc/data/dev/eng_hin/doc.hin.fixed.jsonl")


def load_first_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
        return json.loads(line)["text"]

eng_doc = load_first_jsonl(eng_file)
tel_ref = load_first_jsonl(tel_file)

print("\n--- First English Doc ---\n")
print(eng_doc[:500] + ("..." if len(eng_doc) > 500 else ""))  # print first 500 chars


model_name = "sarvamai/sarvam-translate"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda:0")


messages = [
    {"role": "system", "content": f"Translate the text below to Hindi."},
    {"role": "user", "content": eng_doc}
]


text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096,       
        do_sample=False,
        repetition_penalty=1.1     
    )

# Remove input tokens from output
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
translation = tokenizer.decode(output_ids, skip_special_tokens=True)

def normalize(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)     # collapse multiple spaces
    return text

translation = normalize(translation)
tel_ref = normalize(tel_ref)

print("\n--- Model Translation ---\n") 
print(translation[:500] + ("..." if len(translation) > 500 else ""))

print("\n--- reference ---\n") 
print(tel_ref[:500] + ("..." if len(tel_ref) > 500 else ""))

bleu = sacrebleu.corpus_bleu([translation], [[tel_ref]])
chrf = sacrebleu.corpus_chrf([translation], [[tel_ref]], word_order=2)

print("\n--- Evaluation (first doc onlyy) ---")
print(f"BLEU score: {bleu.score:.2f}")
print(f"chrF2 score: {chrf.score:.2f}")

