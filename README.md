# NLP
# NLP Project – Prompting and Fine-Tuning for Text Classification

## Overview
This project investigates different Natural Language Processing (NLP) approaches for text classification, focusing on **zero-shot**, **one-shot**, and **few-shot prompting**, as well as **supervised fine-tuning using LoRA adapters**.  
The goal is to analyse how prompt design and fine-tuning influence model performance and robustness.

All experiments were developed and executed using **AWS SageMaker**.

---

## Project Structure

NLP/
├── 01_inspecao.ipynb
├── 02_zero_shot_baseline.ipynb
├── 03_one_shot.ipynb
├── 04_few_shots.ipynb
├── 05_evaluation.ipynb
├── FicheiroTreino.ipynb
│
├── project/
│ ├── data/ # Datasets (JSONL format)
│ ├── prompts/ # Prompt templates
│ ├── scripts/ # Training and inference scripts
│ ├── results/ # Evaluation outputs and metrics
│ └── requirements.txt
│
└── README.md

---

## Methodology

### Prompting Strategies
- **Zero-shot prompting**  
  The model performs the task without seeing any task-specific examples.
- **One-shot prompting**  
  A single labeled example is provided within the prompt.
- **Few-shot prompting**  
  Multiple labeled examples are included to guide the model’s predictions.

### Fine-Tuning
- **Supervised Fine-Tuning (SFT)** using **LoRA adapters**
- Training performed on task-specific data splits
- Multiple checkpoints evaluated during training

### Evaluation
- Quantitative evaluation using classification metrics
- Qualitative analysis of selected predictions
- Comparative analysis across:
  - Zero-shot
  - One-shot
  - Few-shot
  - Fine-tuned models

---

## Notebooks Description

| Notebook | Description |
|--------|-------------|
| `01_inspecao.ipynb` | Dataset inspection and exploratory analysis |
| `02_zero_shot_baseline.ipynb` | Zero-shot baseline experiments |
| `03_one_shot.ipynb` | One-shot prompting experiments |
| `04_few_shots.ipynb` | Few-shot prompting experiments |
| `05_evaluation.ipynb` | Final evaluation and comparison |

---

## Requirements

All dependencies required to run the project are listed in:

project/requirements.txt

Install with:
```bash
pip install -r project/requirements.txt
Execution Environment
Python 3
AWS SageMaker
Hugging Face Transformers
PyTorch
PEFT (LoRA)
Results
All predictions, evaluation metrics, and analysis outputs are available in:
project/results/
This includes:
CSV summary tables
JSONL prediction files
Confidence and calibration analyses
Notes
Large model weights and training artifacts are not included in the repository due to GitHub file size limitations.
The repository is structured for academic evaluation and reproducibility.
Training outputs can be regenerated using the provided scripts.
Author
Academic project developed in the context of Natural Language Processing coursework.
