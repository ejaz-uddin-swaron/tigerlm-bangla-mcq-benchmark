# TigerLM-HF-Deploy: LoRA Fine-Tuning of TigerLM-1B-it for Bangla MCQ Answering

## Abstract

This project fine-tunes the **TigerLM-1B-it** instruction-tuned language model on a **Bangla multiple-choice question (MCQ) answering** task using **Low-Rank Adaptation (LoRA)**, a parameter-efficient fine-tuning (PEFT) technique. The adapted model achieves **73.33% accuracy** on an expanded held-out benchmark of **30 Bangla medical MCQs**. A standalone evaluation pipeline reports per-question predictions alongside aggregate metrics and generates visual performance artifacts (accuracy chart and confusion matrix), serving as a reproducible benchmark for Bangla LLM evaluation.

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

The benchmark dataset (`benchmark_dataset.json`) comprises **30 Bangla multiple-choice questions** focused on medical and health science topics (e.g., human anatomy, disease pathophysiology, pharmacology, biochemistry). Each entry follows this schema:

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

The dataset covers the following categories:

| Category                         | Count |
|----------------------------------|-------|
| Human Anatomy & Physiology       | 10    |
| Disease Pathology & Diagnosis    | 8     |
| Pharmacology & Treatment         | 4     |
| Biochemistry & Nutrition         | 5     |
| Microbiology & Infectious Disease| 3     |
| **Total**                        | **30**|

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
| Accuracy | **73.33%** (22 / 30) |

### Performance Visualizations

The evaluation pipeline automatically generates two diagnostic figures:

- **`accuracy_chart.png`** — A bar chart comparing correct vs. incorrect prediction counts.
- **`confusion_matrix.png`** — A 4×4 confusion matrix over answer labels (A–D), revealing specific answer confusion patterns.

### Per-Question Breakdown

| # | Question Topic                         | Predicted | Ground Truth | Correct |
|---|----------------------------------------|-----------|--------------|---------|
| 1 | Diabetes characteristic                | B         | A            | ✗       |
| 2 | Blood pressure control organ           | C         | C            | ✓       |
| 3 | Insulin secretion source               | B         | B            | ✓       |
| 4 | Largest human organ                    | B         | C            | ✗       |
| 5 | Type 2 diabetes cause                  | B         | B            | ✓       |
| 6 | Chromosome pairs in humans             | C         | C            | ✓       |
| 7 | pH of water                            | C         | C            | ✓       |
| 8 | Cardiac cycle definition               | C         | C            | ✓       |
| 9 | Diabetes diet control                  | B         | B            | ✓       |
|10| RBC lifespan                           | C         | C            | ✓       |
|11| Sulfuric acid formula                  | C         | C            | ✓       |
|12| Blood clotting vitamin                 | C         | C            | ✓       |
|13| Heart layer in contact with blood      | C         | C            | ✓       |
|14| Squint (strabismus) affected muscle    | C         | B            | ✗       |
|15| Brain region for balance               | B         | B            | ✓       |
|16| Vitamin D deficiency disease           | B         | B            | ✓       |
|17| AIDS causative virus                   | C         | C            | ✓       |
|18| White blood cell function              | B         | B            | ✓       |
|19| Tuberculosis primary affected organ    | C         | B            | ✗       |
|20| Elevated WBC count condition           | C         | B            | ✗       |
|21| Body temperature regulator             | B         | B            | ✓       |
|22| Dengue platelet count change           | B         | B            | ✓       |
|23| Largest gland in human body            | B         | B            | ✓       |
|24| Hypertension meaning                   | C         | C            | ✓       |
|25| Pre-surgery analgesic class            | B         | B            | ✓       |
|26| O− blood group designation             | C         | B            | ✗       |
|27| Coronary artery disease vessel         | B         | B            | ✓       |
|28| Penicillin target organism             | B         | B            | ✓       |
|29| Hemoglobin oxygen-binding component    | A         | C            | ✗       |
|30| UTI affected anatomical sites          | C         | D            | ✗       |

The model correctly answered **22 of 30** questions. Eight errors occurred, primarily on items requiring specialized anatomical or pathophysiological knowledge.

## Error Analysis

The eight misclassified questions reveal several systematic patterns:

1. **Anatomical & physiological knowledge gaps:** Three errors (#4 — largest organ, #14 — squint muscle, #19 — tuberculosis organ) involve specific anatomical facts. The model defaulted to option `C` across these cases, suggesting a positional bias or uncertainty rather than genuine knowledge.

2. **Terminology confusion:** For #1 (diabetes characteristic), the model chose `B` (blood lipids) instead of `A` (blood sugar), indicating possible confusion between diabetes and hyperlipidemia — two commonly co-occurring metabolic conditions.

3. **Hematology & biochemistry nuance:** Questions `#26` (O− blood group) and `#29` (hemoglobin oxygen binding) require domain-specific knowledge. The model picked a related but imprecise option in both cases — `C` ("donor" without "universal") instead of `B` ("universal donor") for O−, and `A` (globin protein) instead of `C` (iron) for oxygen binding.

4. **Multi-site disease knowledge:** The UTI question (`#30`, correct answer `D`: all listed sites) was answered as `C` (bladder only). While the bladder is the most common UTI site, the question required understanding that the entire urinary tract can be involved.

5. **Answer distribution bias:** Notably, 6 of the 8 incorrect predictions favored option `C`, consistent with the training data's answer distribution imbalance noted in the original 12-question benchmark.

These patterns suggest that while the model has acquired broad medical knowledge, it struggles with **fine-grained anatomical specificity** and exhibits a **residual positional bias** toward option `C` when uncertain.

## Limitations

1. **Moderate benchmark size (30 questions):** While tripled from the original 12, the evaluation set remains modest for robust statistical conclusions.
2. **Imbalanced answer distribution:** The dataset exhibits some answer bias, which may inflate accuracy if the model exploits positional shortcuts.
3. **Single-domain coverage:** The benchmark focuses exclusively on medical topics; general knowledge, humanities, and other domains are not represented.
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
├── benchmark_dataset.json         # Curated Bangla MCQ benchmark (30 questions)
├── evaluate.py                    # Full evaluation pipeline with metrics
├── accuracy_chart.png             # Bar chart: correct vs. incorrect counts
├── confusion_matrix.png           # 4×4 confusion matrix over answer labels
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
- `transformers`, `peft`, `sentencepiece`, `torch`, `scikit-learn`, `matplotlib`

Install dependencies:

```bash
pip install transformers peft sentencepiece torch scikit-learn matplotlib
```

### Evaluation

Run the full benchmark evaluation:

```bash
python evaluate.py
```

The script loads the base model from Hugging Face, applies the LoRA adapter from `./model`, runs inference on all 30 benchmark questions, prints per-question results along with final accuracy, and generates two diagnostic figures: `accuracy_chart.png` and `confusion_matrix.png`.

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
  url = {https://github.com/ejaz-uddin-swaron/TigerLM-HF-Deploy}
}
```

## License

This project is released for academic and research purposes. The base model TigerLM-1B-it is subject to its own license terms.
