# TigerLM-HF-Deploy: LoRA Fine-Tuning of TigerLM-1B-it for Bangla MCQ Answering

## Abstract

This project fine-tunes the **TigerLM-1B-it** instruction-tuned language model on a **Bangla multiple-choice question (MCQ) answering** task using **Low-Rank Adaptation (LoRA)**, a parameter-efficient fine-tuning (PEFT) technique. The adapted model achieves **83.33% accuracy** on a held-out benchmark of 12 domain-specific Bangla MCQs spanning biology and chemistry. A standalone evaluation pipeline reports per-question predictions alongside aggregate metrics, serving as a reproducible benchmark for Bangla LLM evaluation.

## Problem Statement

Large language models (LLMs) exhibit strong performance in high-resource languages such as English, but their effectiveness in **low-resource languages like Bengali (Bangla)** remains underexplored. Existing Bangla LLM evaluations often lack structured, reproducible benchmarks with quantitative metrics. This work addresses the gap by:

1. Constructing a **curated MCQ benchmark** in Bangla covering secondary-level science topics.
2. Applying **LoRA fine-tuning** to adapt a 1B-parameter instruction-tuned model for the MCQ-answering task.
3. Implementing a **deterministic evaluation framework** that reports accuracy and per-instance diagnostics.

## Methodology

### Base Model

We use [**TigerLM-1B-it**](https://huggingface.co/md-nishat-008/TigerLLM-1B-it) (md-nishat-008/TigerLLM-1B-it), a 1-billion-parameter instruction-tuned causal language model. The model was chosen for its demonstrated capability in Bengali text generation and its suitability for deployment in resource-constrained environments.

### Parameter-Efficient Fine-Tuning with LoRA

**Low-Rank Adaptation (LoRA)** (Hu et al., 2021) reduces the number of trainable parameters by factorizing weight updates into low-rank matrices. For a pre-trained weight matrix `W ∈ ℝᵈˣᵏ`, the update `ΔW` is represented as `ΔW = BA`, where `B ∈ ℝᵈˣʳ`, `A ∈ ℝʳˣᵏ`, and the rank `r ≪ min(d, k)`. During fine-tuning, only `A` and `B` are updated while `W` remains frozen.

**Hyperparameters:**

| Parameter       | Value  |
|-----------------|--------|
| LoRA rank (`r`) | 16     |
| LoRA alpha      | 32     |
| LoRA dropout    | 0.05   |
| Target modules  | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj` |
| Task type       | Causal language model |
| Merge weights   | Yes (adapter merged into base for inference) |

All seven linear projection layers in the transformer architecture (attention and feed-forward blocks) are targeted, enabling comprehensive adaptation with minimal parameter overhead.

## Dataset

The benchmark dataset (`benchmark_dataset.json`) comprises **12 Bangla multiple-choice questions** on biology and chemistry topics (e.g., diabetes pathophysiology, human anatomy, chemical formulae). Each entry follows this schema:

```json
{
  "question": "প্রশ্ন টেক্সট",
  "options": {
    "A": "বিকল্প ক",
    "B": "বিকল্প খ",
    "C": "বিকল্প গ",
    "D": "বিকল্প ঘ"
  },
  "answer": "A"
}
```

The dataset covers the following question types:

| Category       | Count |
|----------------|-------|
| Human Biology  | 10    |
| Chemistry      | 2     |
| **Total**      | **12**|

The Bengali script (`ক`/`খ`/`গ`/`ঘ`) is used for option labels and answer keys.

## Evaluation Methodology

The evaluation pipeline (`evaluate.py`) proceeds as follows:

1. **Loading:** The base model and fine-tuned LoRA adapter are loaded and merged. The merged model is set to evaluation mode.
2. **Prompt Construction:** Each MCQ is formatted into a structured prompt in Bangla:
   ```
   প্রশ্ন: {question}
   ক. {option_A}
   খ. {option_B}
   গ. {option_C}
   ঘ. {option_D}
   সঠিক উত্তর:
   ```
3. **Inference:** The model generates a continuation using **greedy decoding** (`do_sample=False`, `max_new_tokens=10`) to ensure deterministic output.
4. **Answer Extraction:** The generated text is parsed with a regular expression to extract either a Bengali letter (`ক`/`খ`/`গ`/`ঘ`) or a Latin letter (`A`–`D`). The extracted letter is mapped to the canonical answer key.
5. **Scoring:** The predicted answer is compared against the ground truth. Accuracy is computed as `correct / total × 100`.

## Results

| Metric   | Value   |
|----------|---------|
| Accuracy | **83.33%** (10 / 12) |

### Per-Question Breakdown

| # | Question Topic         | Predicted | Ground Truth | Correct |
|---|------------------------|-----------|--------------|---------|
| 1 | Diabetes characteristic| A         | A            | ✓       |
| 2 | Blood pressure control | C         | C            | ✓       |
| 3 | Insulin secretion      | B         | B            | ✓       |
| 4 | Largest human organ    | C         | C            | ✓       |
| 5 | Type 2 diabetes cause  | B         | B            | ✓       |
| 6 | Chromosome pairs       | C         | C            | ✓       |
| 7 | Water pH               | C         | C            | ✓       |
| 8 | Cardiac cycle          | C         | C            | ✓       |
| 9 | Diabetes diet control  | B         | B            | ✓       |
|10| RBC lifespan           | C         | C            | ✓       |
|11| Sulfuric acid formula  | —         | C            | ✗       |
|12| Blood clotting vitamin | —         | C            | ✗       |

Two errors occurred on chemistry-related items (sulfuric acid formula and blood clotting vitamin), suggesting the model may be less robust on **chemistry domain knowledge** compared to biology.

## Error Analysis

The two misclassified instances share common characteristics:

- Both are **chemistry questions** (inorganic/formula and nutrition/biochemistry), while the training data likely contained predominantly biology content.
- In both cases, the model failed to emit any valid option label, producing an empty or non-conforming response that could not be parsed.
- This indicates a **domain coverage gap** rather than a simple answer confusion, implying that additional chemistry-domain training data would likely resolve these errors.

## Limitations

1. **Small benchmark size (12 questions):** The evaluation set is insufficient for statistical confidence in the reported accuracy.
2. **Imbalanced answer distribution:** 9 of 12 questions have the correct answer `C`, which may inflate accuracy if the model learns a positional bias.
3. **Single-domain coverage:** The dataset covers only science topics; general knowledge and humanities questions are absent.
4. **Single base model:** Only TigerLM-1B-it was evaluated; comparisons with other Bangla-capable models (e.g., BanglaBERT, mT5) are not performed.
5. **No training split documented:** The LoRA adapter weights are provided, but the full fine-tuning procedure (training hyperparameters, dataset splits, number of epochs) is not reproduced here.

## Future Improvements

1. **Expand the benchmark** to 100+ questions across diverse domains (literature, history, civics, mathematics).
2. **Balance answer key distribution** to eliminate positional bias in accuracy measurements.
3. **Evaluate multiple base models** (e.g., BanglaT5, Llama-3-Bangla, mGPT) under identical LoRA configurations.
4. **Report confidence intervals** and per-category accuracy breakdowns.
5. **Implement few-shot prompting** as a baseline comparison against fine-tuning.
6. **Release training script and hyperparameters** for full reproducibility.
7. **Explore alternative PEFT methods** such as QLoRA, AdaLoRA, or DoRA.

## Project Structure

```
TigerLM-HF-Deploy/
├── .gitignore                     # Git ignore rules
├── app.py                         # Interactive inference script for Bangla Q&A
├── benchmark_dataset.json         # Curated Bangla MCQ benchmark (12 questions)
├── evaluate.py                    # Full evaluation pipeline with metrics
├── model/                         # LoRA adapter weights and tokenizer files
│   ├── adapter_config.json        # LoRA hyperparameter configuration
│   ├── adapter_model.safetensors  # Trained LoRA adapter weights
│   ├── added_tokens.json          # Tokenizer additional tokens
│   ├── special_tokens_map.json    # Special token mappings
│   ├── tokenizer.model            # SentencePiece tokenizer model
│   └── tokenizer_config.json      # Tokenizer configuration
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation (this file)
```

## How to Run

### Prerequisites

- Python 3.10+
- PyTorch (≥2.0)
- `transformers`, `peft`, `sentencepiece`, `torch`

Install dependencies:

```bash
pip install transformers peft sentencepiece torch
```

### Evaluation

Run the full benchmark evaluation:

```bash
python evaluate.py
```

The script loads the base model from Hugging Face, applies the LoRA adapter from `./model`, runs inference on all 12 benchmark questions, and prints per-question results along with final accuracy.

### Interactive Inference

For free-form Bangla question answering:

```bash
python app.py
```

Modify the prompt in `app.py` to ask custom questions. The model will generate a response using sampling-based decoding (`temperature=0.7`).

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{tigerlm-bangla-mcq,
  author = {},
  title = {LoRA Fine-Tuning of TigerLM-1B-it for Bangla MCQ Answering},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/TigerLM-HF-Deploy}
}
```

## License

This project is released for academic and research purposes. The base model TigerLM-1B-it is subject to its own license terms.
