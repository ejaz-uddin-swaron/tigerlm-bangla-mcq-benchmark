import json
import re
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model_name: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(item: dict) -> str:
    options = item["options"]
    prompt = (
        f"প্রশ্ন: {item['question']}\n"
        f"ক. {options['A']}\n"
        f"খ. {options['B']}\n"
        f"গ. {options['C']}\n"
        f"ঘ. {options['D']}\n"
        f"সঠিক উত্তর:"
    )
    return prompt


def extract_answer(text: str) -> str | None:
    match = re.search(r"[কখগঘ]", text)
    if match:
        mapping = {"ক": "A", "খ": "B", "গ": "C", "ঘ": "D"}
        return mapping[match.group()]
    match = re.search(r"\b[A-D]\b", text)
    if match:
        return match.group()
    return None


def run_inference(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=10,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def main():
    base_model_name = "md-nishat-008/TigerLLM-1B-it"
    adapter_path = "./model"
    dataset_path = "benchmark_dataset.json"

    model, tokenizer = load_model(base_model_name, adapter_path)
    dataset = load_dataset(dataset_path)

    correct = 0
    total = len(dataset)
    y_true = []
    y_pred = []

    for i, item in enumerate(dataset, 1):
        prompt = build_prompt(item)
        response = run_inference(model, tokenizer, prompt)
        predicted = extract_answer(response)
        ground_truth = item["answer"]

        y_true.append(ground_truth)
        y_pred.append(predicted)

        if predicted == ground_truth:
            correct += 1

        print(f"[{i}] প্রশ্ন: {item['question']}")
        print(f"   পূর্বাভাস: {predicted}")
        print(f"   সঠিক উত্তর: {ground_truth}")
        print("-" * 60)

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"FINAL RESULTS")
    print(f"Accuracy: {accuracy:.2f}%")

    incorrect = total - correct
    fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
    categories = ["Correct", "Incorrect"]
    counts = [correct, incorrect]
    ax_bar.bar(categories, counts)
    ax_bar.set_title("Model Performance Summary")
    ax_bar.set_xlabel("Correct / Incorrect")
    ax_bar.set_ylabel("Count")
    for idx, val in enumerate(counts):
        ax_bar.text(idx, val + 0.1, str(val), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("accuracy_chart.png", dpi=150)
    plt.close(fig_bar)
    print(f"Accuracy chart saved to accuracy_chart.png")

    valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if valid:
        y_true_val, y_pred_val = zip(*valid)
        labels = ["A", "B", "C", "D"]
        cm = confusion_matrix(y_true_val, y_pred_val, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("TigerLM MCQ Confusion Matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.colorbar(im)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to confusion_matrix.png")
    else:
        print("No valid predictions to compute confusion matrix.")


if __name__ == "__main__":
    main()
