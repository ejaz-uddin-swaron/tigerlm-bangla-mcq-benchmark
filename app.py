import re
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "md-nishat-008/TigerLLM-1B-it"
ADAPTER_PATH = "./model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()
model.eval()

BANGLA_TO_LATIN = {"ক": "A", "খ": "B", "গ": "C", "ঘ": "D"}


def build_prompt(question: str, opt_a: str, opt_b: str, opt_c: str, opt_d: str) -> str:
    return (
        f"প্রশ্ন: {question}\n"
        f"ক. {opt_a}\n"
        f"খ. {opt_b}\n"
        f"গ. {opt_c}\n"
        f"ঘ. {opt_d}\n"
        f"সঠিক উত্তর:"
    )


def extract_answer(text: str) -> str | None:
    match = re.search(r"[কখগঘ]", text)
    if match:
        return BANGLA_TO_LATIN[match.group()]
    match = re.search(r"\b[A-D]\b", text)
    if match:
        return match.group()
    return None


def predict(question: str, opt_a: str, opt_b: str, opt_c: str, opt_d: str) -> str:
    if not all([question, opt_a, opt_b, opt_c, opt_d]):
        return "Please fill in all fields."

    prompt = build_prompt(question, opt_a, opt_b, opt_c, opt_d)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=10,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    latin = extract_answer(generated)

    if latin is None:
        return f"Could not parse answer. Model output: {generated}"

    bangla_map = {"A": "ক", "B": "খ", "C": "গ", "D": "ঘ"}
    bangla = bangla_map[latin]
    return f"উত্তর: {bangla} ({latin})"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Question (প্রশ্ন)", placeholder="e.g. ডায়াবেটিস রোগের প্রধান বৈশিষ্ট্য কোনটি?"),
        gr.Textbox(label="Option A (ক)", placeholder="e.g. রক্তে শর্করার মাত্রা বেড়ে যাওয়া"),
        gr.Textbox(label="Option B (খ)", placeholder="e.g. রক্তে চর্বির মাত্রা বেড়ে যাওয়া"),
        gr.Textbox(label="Option C (গ)", placeholder="e.g. রক্তে প্রোটিনের মাত্রা বেড়ে যাওয়া"),
        gr.Textbox(label="Option D (ঘ)", placeholder="e.g. রক্তে লবণের মাত্রা বেড়ে যাওয়া"),
    ],
    outputs=gr.Textbox(label="Predicted Answer"),
    title="TigerLM Bangla MCQ Solver",
    description="Enter a Bangla multiple-choice question and four options. The model predicts the correct answer.",
)

if __name__ == "__main__":
    demo.launch()
