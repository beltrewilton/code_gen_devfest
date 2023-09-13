from datasets import load_dataset
from random import randrange

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM


from trl import SFTTrainer



# The model that you want to train from the Hugging Face hub
model_id = "NousResearch/Llama-2-7b-hf"
# The instruction dataset to use
# dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
dataset_name = "mbpp"
#dataset_name = "HuggingFaceH4/CodeAlpaca_20K"
# Dataset split
dataset_split= "train"
# Fine-tuned model name
new_model = "llama-2-7b-int4-python-code-20k"
# Huggingface repository
hf_model_repo="edumunozsala/"+new_model
# Load the entire model on the GPU 0
device_map = {"": 0}
device_map = "auto"

################################################################################
# bitsandbytes parameters
################################################################################
# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_double_nested_quant = False

################################################################################
# QLoRA parameters
################################################################################
# LoRA attention dimension
lora_r = 64
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# TrainingArguments parameters
################################################################################
# Output directory where the model predictions and checkpoints will be stored
output_dir = new_model
# Number of training epochs
num_train_epochs = 1
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False
# Batch size per GPU for training
per_device_train_batch_size = 2
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1 # 2
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4 #1e-5
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = "paged_adamw_32bit"
# Learning rate schedule
lr_scheduler_type = "cosine" #"constant"
# Number of training steps (overrides num_train_epochs)
max_steps = -1
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = False
# Save checkpoint every X updates steps
save_steps = 0
# Log every X updates steps
logging_steps = 25
# Disable tqdm
disable_tqdm= False

################################################################################
# SFTTrainer parameters
################################################################################
# Maximum sequence length to use
max_seq_length = 2048 #None
# Pack multiple short examples in the same input sequence to increase efficiency
packing = True #False



# Load dataset from the hub
dataset = load_dataset(dataset_name, split=dataset_split)
# Show dataset size
print(f"dataset size: {len(dataset)}")
# Show an example
print(dataset[randrange(len(dataset))])



import pprint

# Show a random example
pprint.pprint(dataset[randrange(len(dataset))], width=90)


# Set the instruction format for Mostly Basic Python Problems (mbpp)
def format_instruction(sample):
	return f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the following Task:

### Task:
{sample['text']}

### Input:
{sample['test_list'][randrange(len(sample['test_list']))]}

### Response:
{sample['code']}
"""


def format_instruction(sample):
	return f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the following Task:

### Task:
{sample['text']}

### Response:
{sample['code']}
"""




# Show a formatted instruction
print(format_instruction(dataset[randrange(len(dataset))]))



# Get the type
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_use_double_quant=use_double_nested_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype
)



# Load the pretrained model
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache = False, device_map=device_map)
model.config.pretraining_tp = 1

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


