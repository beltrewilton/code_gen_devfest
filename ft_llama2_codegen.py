import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from random import randrange
import pprint
import time

from trl import SFTTrainer

device = "mps"
dataset_name = "mbpp"
model_id = "NousResearch/Llama-2-7b-hf"
new_model = "devfest_{0}"

dataset = load_dataset(dataset_name, split="train")



# Show a random example
# pprint.pprint(dataset[randrange(len(dataset))], width=90)


def format_instruction(sample):
	return f"""### Instruction:
Use the Task below given to write the Response, which is a programming code that can solve the following Task:

### Task:
{sample['text']}

### Response:
{sample['code']}
"""

# Show a formatted instruction
print(format_instruction(dataset[randrange(len(dataset))]))


# Load the pretrained model
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.config.pretraining_tp = 1


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).to(device)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
)

output_dir = f"./llama-summary-training-{str(int(time.time()))}"

# Define the training arguments
args = TrainingArguments(
	evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    # metric_for_best_model="loss",
    # optim="adamw_torch",
    # gradient_checkpointing=True,
    # torch_compile=True, # Test it
    output_dir=output_dir,
    num_train_epochs=1,
    logging_steps=200,
    learning_rate=1e-3,
    fp16=False,
    bf16=False,
    report_to="tensorboard",
    seed=42,
	device=device
)


# Create the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
	device=device
)


# train
trainer.train() # there will not be a progress bar since tqdm is disabled

# save model in local
trainer.save_model()


# Empty VRAM
del model
del trainer
import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache() # PyTorch thing???
gc.collect()


# Reload the trained and saved model and merge it then we can save the whole model


new_model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
)

# Merge LoRA and base model
merged_model = new_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model",safe_serialization=True)
tokenizer.save_pretrained("merged_model")