import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from peft import LoraConfig, TaskType, get_peft_model


class MBPPDataset(Dataset):
    def __init__(self, ds, split: str = "train") -> None:
        super().__init__()
        self.ds = ds[0 if split == "train" else 1]
        print()

    # TODO: Example programs synthesized (few-shot) by our largest model.
    def format_task(self, task, test_list):
        PROMPT_DICT = {
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{task}\n\n### Response:"
            ),
            "prompt_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request."
                f"Your code should satisfy these tests:\n{test_list}\n\n"
                f"### Instruction:\n{task}\n\n### Response:"
            ),
        }
        instruct = PROMPT_DICT["prompt_input"] if test_list != "" else PROMPT_DICT["prompt_no_input"]
        return instruct

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        text = self.format_task(self.ds[index]["text"], self.ds[index]["test_list"])
        code = self.ds[index]["code"]

        return text, code


def print_trainable_parameters(desc: str, model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    dash_line = "-".join("" for x in range(100))
    return f"{dash_line}\n{desc}\ntrainable model parameters:\t{trainable_model_params:,}\nall model parameters:\t\t{all_model_params:,}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%\n{dash_line}\n"


def load_resources(
    dataset_name: str,
    split: list,
    model_name: str,
    device: str,
    inference: bool = False,
):
    dataset = (
        load_dataset(dataset_name, split=split) if dataset_name is not None else None
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)

    model.enable_input_require_grads()
    print(print_trainable_parameters(f"Base Model: {model_name}", model))

    # modules_to_save : also trained but not LoRA applied.
    if not inference:
        lora_config = LoraConfig(
            r=16,  # Rank
            target_modules=["q", "v"],
            task_type=TaskType.SEQ_2_SEQ_LM,  # FLAN-T5
        )
        model = get_peft_model(model, lora_config)
        print(print_trainable_parameters(f"LoRA Model: {model_name}", model))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, dataset


def get_dls(dataset, batch_size: int):
    train = MBPPDataset(dataset)
    val = MBPPDataset(dataset, split="validation")

    train = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    val = DataLoader(dataset=val, batch_size=1, shuffle=True)

    return train, val


def get_test_dl(dataset, batch_size: int = 1):
    test = MBPPDataset(dataset, split="test")
    test = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
    return test
