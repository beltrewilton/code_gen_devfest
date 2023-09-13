import time
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from pprint import pprint

device = "mps"
model_name = "google/flan-t5-base"
dataset_name = "mbpp"

def load_from_hf():
    dataset = load_dataset(dataset_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, dataset, tokenizer

t5_model, dataset, tokenizer = load_from_hf()

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters:\t{trainable_model_params:,}\nall model parameters:\t\t{all_model_params:,}\n% of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


print(print_number_of_trainable_model_parameters(t5_model))


def format_task(text):
	return f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the following Task:

### Task:
{text}

### Response:
"""

def format_response(code):
     return f"""{code}"""


index = 217
instruct = dataset['train'][index]['text']
code = dataset['train'][index]['code']
prompt = format_task(instruct)
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
generated = tokenizer.decode(
    t5_model.generate(
        inputs,
        max_new_tokens=300,
    )[0],
    skip_special_tokens=True,
)

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT INSTRUCTION:\n{instruct}')
print(dash_line)
print(f'BASELINE CODE:\n{code}\n')
print(dash_line)
print(f'MODEL CODE GENERATION - ZERO SHOT:\n{generated}')


def tokenize_function(example):
    tasks = [format_task(t) for t in example['text']]
    example["input_ids"] = tokenizer(tasks, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)

    responses = [format_response(c) for c in example['code']]
    example["labels"] = tokenizer(text_target=responses, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    
    return example



tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    [
        "task_id",
        "text",
        "code",
        "test_list",
        "test_setup_code",
        "challenge_test_list"
    ]
)

lora_config = LoraConfig(
    r=16,  # Rank
    # lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.01,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,  # FLAN-T5
)

output_dir = f"./t5-summary-training-{str(int(time.time()))}"

peft_model = get_peft_model(t5_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))


training_args = TrainingArguments(
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # save_total_limit=3,
    # load_best_model_at_end=True,
    # metric_for_best_model="loss",
    # optim="adamw_torch",
    # gradient_checkpointing=True,
    # torch_compile=True, # Test it

    output_dir=output_dir,
    # auto_find_batch_size=True,
    learning_rate=1e-3,  # Wilton
    # num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=1,
    max_steps=1,
    report_to="tensorboard",
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
peft_model_path = "./t5-summary-checkpoint-base"
trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

