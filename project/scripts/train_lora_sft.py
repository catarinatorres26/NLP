import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct" 
TRAIN_PATH = "NLP/project/data/sft_train.jsonl"
VAL_PATH   = "NLP/project/data/sft_val.jsonl"
OUT_DIR    = "NLP/project/lora_out/adapter"

# LoRA (config "completa" standard)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Treino
MAX_LEN = 1024
EPOCHS = 3
LR = 2e-4
GRAD_ACCUM = 16

# 4-bit (recomendado na A10G)
USE_4BIT = True

# Target modules típicos para Llama/Mistral (se o teu modelo for Qwen, ajustamos depois)
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

def format_messages(messages, tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # fallback
    out = []
    for m in messages:
        out.append(f"{m['role'].upper()}: {m['content']}")
    return "\n".join(out) + "\n"

def tokenize_record(example, tokenizer):
    text = format_messages(example["messages"], tokenizer)
    toks = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False)
    toks["labels"] = toks["input_ids"].copy()
    return toks

def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_config = None
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        quantization_config=quant_config,
    )

    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})
    ds = ds.map(lambda ex: tokenize_record(ex, tok), remove_columns=ds["train"].column_names)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.03,
        logging_steps=20,
        save_steps=50,
        save_total_limit=3,
        fp16=not torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        report_to="none",
        optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
    )

    trainer.train()

    trainer.model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"Saved LoRA adapter to: {OUT_DIR}")

if __name__ == "__main__":
    main()
