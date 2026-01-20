import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a healthcare-focused assistant. Answer safely and concisely. "
    "Return ONLY a valid JSON object with keys: short_answer, confidence_level, clinical_notes. "
    "confidence_level must be one of: high, medium, low. "
    "clinical_notes can be an empty string if not needed. "
    "Do not include any extra keys or surrounding text."
)

def pick_question(ex: dict) -> str:
    # suporta vários formatos
    for key in ["question", "prompt", "instruction", "input"]:
        if key in ex and isinstance(ex[key], str) and ex[key].strip():
            return ex[key].strip()
    return ""

def pick_response(ex: dict) -> dict:
    # formato A: campos ao nível de topo
    if any(k in ex for k in ["short_answer", "confidence_level", "clinical_notes"]):
        return {
            "short_answer": ex.get("short_answer", ""),
            "confidence_level": ex.get("confidence_level", ""),
            "clinical_notes": ex.get("clinical_notes", ""),
        }
    # formato B: response é dict
    if "response" in ex and isinstance(ex["response"], dict):
        r = ex["response"]
        return {
            "short_answer": r.get("short_answer", ""),
            "confidence_level": r.get("confidence_level", ""),
            "clinical_notes": r.get("clinical_notes", ""),
        }
    return {"short_answer": "", "confidence_level": "", "clinical_notes": ""}

def to_output_json(resp: dict) -> str:
    obj = {
        "short_answer": (resp.get("short_answer") or "").strip(),
        "confidence_level": (resp.get("confidence_level") or "").strip().lower(),
        "clinical_notes": (resp.get("clinical_notes") or "").strip(),
    }
    return json.dumps(obj, ensure_ascii=False)

def convert_split(in_path: str, out_path: str):
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            q = pick_question(ex)
            if not q:
                raise ValueError(f"Missing question/instruction in example: {ex}")

            resp = pick_response(ex)
            cl = (resp.get("confidence_level") or "").strip().lower()
            if cl not in {"high", "medium", "low"}:
                raise ValueError(f"Invalid confidence_level='{cl}' in example with instruction: {q}")

            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {q}"},
                    {"role": "assistant", "content": to_output_json(resp)},
                ]
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"Saved {n} records -> {out_path}")

def main():
    # AJUSTA ESTES CAMINHOS AOS TEUS FICHEIROS REAIS:
    train_in = "NLP/project/data/train.jsonl"  # muda se necessário
    dev_in   = "NLP/project/data/dev_questions.jsonl"    # muda se necessário

    train_out = "NLP/project/data/sft_train.jsonl"
    dev_out   = "NLP/project/data/sft_val.jsonl"

    convert_split(train_in, train_out)
    convert_split(dev_in, dev_out)

    print("Done. (O test de 30 sem respostas fica apenas para inferência/avaliação.)")

if __name__ == "__main__":
    main()
