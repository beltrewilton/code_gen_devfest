import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from peft import LoraConfig, TaskType, get_peft_model


class MBPPDataset(Dataset):
    def __init__(self, model_name: str, dataset_name: str, split: str = "train") -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ds = load_dataset(dataset_name, split=split)

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
        text = self.format_task(self.ds[index]["text"], "")
        code = text + self.ds[index]["code"] + self.tokenizer.eos_token

        # TODO: ver este max_length=512, maybe 1024 can work.
        model_inputs = self.tokenizer(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        labels = self.tokenizer(code, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        model_inputs["decoder_input_ids"] = copy.deepcopy(labels["input_ids"])

        # changing labels: convert all tokens in the duplicate prefix prompt and the padding part to -100
        eos_token_id = self.tokenizer.eos_token_id
        for x, y in zip(model_inputs["input_ids"], labels["input_ids"]):
            label_prefix_len = torch.where(x == eos_token_id)[0].item() if eos_token_id in x else len(x)
            y[:label_prefix_len] = torch.tensor([-100] * label_prefix_len)

            if eos_token_id in y:
                pad_len = len(y) - torch.where(y == eos_token_id)[0][0].item() - 1
                if pad_len > 0:
                    y[torch.where(y == eos_token_id)[0][0].item() + 1:] = torch.tensor([-100] * pad_len)

        # shift labels to the right as the decoder input and add decoder start token id
        decoder_start_id = self.tokenizer.eos_token_id
        for z in model_inputs["decoder_input_ids"]:
            # z[1:] = z[:-1] # memory error alocation
            z[0] = decoder_start_id

        model_inputs["labels"] = copy.deepcopy(labels["input_ids"])
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]

        return model_inputs


def print_trainable_parameters(desc: str, model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    dash_line = "-".join("" for x in range(100))
    return f"{dash_line}\n{desc}\ntrainable model parameters:\t{trainable_model_params:,}\nall model parameters:\t\t{all_model_params:,}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%\n{dash_line}\n"


def load_model(
    model_name: str,
    device: str,
    inference: bool = False,
):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)

    # Fix fom pytorch forum
    model.enable_input_require_grads()
    print(print_trainable_parameters(f"Base Model: {model_name}", model))

    # modules_to_save : also trained but not LoRA applied.
    # if not inference:
    #     lora_config = LoraConfig(
    #         r=16,  # Rank
    #         target_modules=["q", "v"],
    #         task_type=TaskType.SEQ_2_SEQ_LM,  # FLAN-T5
    #     )
    #     model = get_peft_model(model, lora_config)
    #     print(print_trainable_parameters(f"LoRA Model: {model_name}", model))

    return model


def get_dls(model_name: str, dataset_name: str, batch_size: int):
    train = MBPPDataset(model_name, dataset_name)
    val = MBPPDataset(model_name, dataset_name, split="validation")

    train = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    val = DataLoader(dataset=val, batch_size=1, shuffle=True)

    return train, val


def get_test_dl(dataset, model_name: str, dataset_name: str, batch_size: int):
    test = MBPPDataset(dataset, model_name, dataset_name, split="test")
    test = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
    return test
