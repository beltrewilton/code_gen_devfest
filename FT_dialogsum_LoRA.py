import time


import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments)

device = "mps"
weights_path: str = "./weights"
tokeniz_path: str = "./tokenizer"
datasets_path: str = "./dataset"
huggingface_dataset_name: str = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name, cache_dir=datasets_path)

print(dataset)


model_name = "google/flan-t5-base"
original_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, torch_dtype=torch.float32, cache_dir=weights_path
)  # bfloat16
original_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=tokeniz_path)


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


print(print_number_of_trainable_model_parameters(original_model))


index = 200

dialogue = dataset["test"][index]["dialogue"]
summary = dataset["test"][index]["summary"]

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = tokenizer.decode(
    original_model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True,
)

# dash_line = '-'.join('' for x in range(100))
# print(dash_line)
# print(f'INPUT PROMPT:\n{prompt}')
# print(dash_line)
# print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
# print(dash_line)
# print(f'MODEL GENERATION - ZERO SHOT:\n{output}')


def tokenize_function(example):
    start_prompt = "Summarize the following conversation.\n\n"
    end_prompt = "\n\nSummary: "
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example["input_ids"] = tokenizer(
        prompt, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids
    example["labels"] = tokenizer(
        example["summary"], padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids

    return example


# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    [
        "id",
        "topic",
        "dialogue",
        "summary",
    ]
)


print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)


import os

os.environ["WANDB_DISABLED"] = "true"

output_dir = f"./dialogue-summary-training-{str(int(time.time()))}"

original_model.to(device)

lora_config = LoraConfig(
    r=64,  # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,  # FLAN-T5
)

peft_model = get_peft_model(original_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,  # Wilton
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=1,
    # max_steps=1,
    report_to=None,
    fp16=False,
    bf16=False,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()


# save model in local
peft_model_path = "./peft-dialog-summary-checkpoint-base"
trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)


# Empty VRAM
del original_model
del peft_model
del trainer
import gc

gc.collect()
gc.collect()


torch.cuda.empty_cache()  # PyTorch thing

gc.collect()

print("Terminé.....")
