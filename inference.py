from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel, PeftConfig
import torch
import time
import evaluate
import pandas as pd
import numpy as np

huggingface_dataset_name = "mbpp"

dataset = load_dataset(huggingface_dataset_name)


model_name_hf = "google/flan-t5-base"
original_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_hf, torch_dtype=torch.float32
)  # TODO: don't work bfloat16
tokenizer_original = AutoTokenizer.from_pretrained(model_name_hf)

model_name = "./t5-summary-checkpoint-base"
peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_hf, torch_dtype=torch.float32
)  # TODO: don't work bfloat16
tokenizer_instruct = AutoTokenizer.from_pretrained(
    model_name_hf
)  # TODO: fix to work with own tokenizer.
instruct_model = PeftModel.from_pretrained(
    peft_model_base, model_name, is_trainable=False
)

def format_instruction(text, code):
	return f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the following Task:

### Task:
{text}

### Response:
{code}
"""

index = 1
instruct = dataset['test'][index]['text']
code = dataset['test'][index]['code']
prompt = format_instruction(dataset['test']['text'][index], dataset['test']['code'][index])

input_original_ids = tokenizer_original(prompt, return_tensors="pt").input_ids
input_instruct_ids = tokenizer_instruct(prompt, return_tensors="pt").input_ids

original_model_outputs = original_model.generate(
    input_ids=input_original_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1),
)
original_model_text_output = tokenizer_original.decode(
    original_model_outputs[0], skip_special_tokens=True
)

instruct_model_outputs = instruct_model.generate(
    input_ids=input_instruct_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1),
)
instruct_model_text_output = tokenizer_instruct.decode(
    instruct_model_outputs[0], skip_special_tokens=True
)

dash_line = "-".join("" for _ in range(100))
print(dash_line)
print(f"SOURCE INSTRUCT:\n{instruct}")
print(dash_line)
print(f"BASELINE CODE:\n{code}")
print(dash_line)
print(f"ORIGINAL MODEL:\n{original_model_text_output}")
print(dash_line)
print(f"INSTRUCT MODEL:\n{instruct_model_text_output}")
