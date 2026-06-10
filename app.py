from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "md-nishat-008/TigerLLM-1B-it"
adapter_path = "./model"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Attach LoRA
model = PeftModel.from_pretrained(base_model, adapter_path)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate("ডায়াবেটিসের লক্ষণ কী?"))