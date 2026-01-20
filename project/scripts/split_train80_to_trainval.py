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
    for key in ["question", "prompt", "instruction", "input"]:
        if key in ex and isinstance(ex[key], str) and ex[key].strip():
            return ex[key].strip()
    return ""

def pick_response(ex: dict) -> dict:
    if "response" in ex and isinstance(ex["response"], dict):
        r = ex["response"]
        return {
            "short_answer": r.get("short_answer", ""),
            "confidence_level": r.get("confidence_level", ""),
            "clinical_notes": r.get("clinical_notes", ""),
        }
    return {
        "short_answer": ex.get("short_answer", ""),
        "confidence_level": ex.get("confidence_level", ""),
        "clinical_notes": ex.get("clinical_notes", ""),
    }

def to_output_json(resp: dict) -> str:
    obj = {
        "short_answer": (resp.get("short_answer") or "").strip(),
        "confidence_level": (resp.get("confidence_level") or "").strip().lower(),
        "clinical_notes": (resp.get("clinical_notes") or "").strip(),
    }
    return json.dumps(obj, ensure_ascii=False)

def to_sft_record(ex: dict) -> dict:
    q = pick_question(ex)
    if not q:
        raise ValueError(f"Missing question/instruction in: {ex}")

    resp = pick_response(ex)
    cl = (resp.get("confidence_level") or "").strip().lower()
    if cl not in {"high", "medium", "low"}:
        raise ValueError(f"Invalid confidence_level='{cl}' for instruction: {q}")

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {q}"},
            {"role": "assistant", "content": to_output_json(resp)},
        ]
    }

def read_jsonl(path: Path) -> list[dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(items: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in items:
            rec = to_sft_record(ex)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main(
    train_path="NLP/project/data/train.jsonl",
    dev_path="NLP/project/data/dev.jsonl",
    out_train="NLP/project/data/sft_train.jsonl",
    out_val="NLP/project/data/sft_val.jsonl",
):
    train_path = Path(train_path)
    dev_path = Path(dev_path)
    out_train = Path(out_train)
    out_val = Path(out_val)

    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)

    write_jsonl(train, out_train)
    write_jsonl(dev, out_val)

    print(f"Saved train: {out_train} ({len(train)})")
    print(f"Saved val:   {out_val} ({len(dev)})")

if __name__ == "__main__":
    main()

