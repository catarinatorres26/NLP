import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "NLP/project/lora_out/adapter"

# Perguntas finais SEM respostas (30)
TEST_QUESTIONS = "NLP/project/data/test_30.jsonl"

OUT_PATH = "NLP/project/results/test_finetuned.jsonl"

SYSTEM_PROMPT = (
    "You are a healthcare-focused assistant. Answer safely and concisely. "
    "Return ONLY a valid JSON object with keys: short_answer, confidence_level, clinical_notes. "
    "confidence_level must be one of: high, medium, low. "
    "clinical_notes can be an empty string if not needed. "
    "Do not include any extra keys or surrounding text."
)

def extract_json(text: str):
    text = text.strip()
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            obj = json.loads(text[s:e+1])
            return {
                "short_answer": str(obj.get("short_answer","")).strip(),
                "confidence_level": str(obj.get("confidence_level","")).strip().lower(),
                "clinical_notes": str(obj.get("clinical_notes","")).strip(),
            }
        except Exception:
            pass
    return {
        "short_answer": text[:500],
        "confidence_level": "low",
        "clinical_notes": "Output parsing failed; treat as low confidence.",
    }

def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(TEST_QUESTIONS, "r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            q = (ex.get("question") or ex.get("instruction") or ex.get("prompt") or "").strip()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {q}"},
            ]

            if hasattr(tok, "apply_chat_template"):
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"SYSTEM: {SYSTEM_PROMPT}\nUSER: Question: {q}\nASSISTANT:"

            inputs = tok(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tok.eos_token_id,
                )

            text = tok.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            parsed = extract_json(text)

            record = {"question": q, **parsed}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved finetuned outputs to: {OUT_PATH}")

if __name__ == "__main__":
    main()
