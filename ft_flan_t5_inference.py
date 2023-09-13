import torch
from torch.utils.data import DataLoader, Dataset
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")

from code_gen_util import get_test_dl, load_model


def compare_models(task: str, base_resp: str, orig_resp: str, ft_resp: str):
    dash_line = "-".join("" for x in range(100))
    pprint(dash_line)
    pprint(f"TASK (Prompt):\n{task}")
    pprint(f"BASELINE CODE:\n{base_resp}\n")
    pprint(f"(ORIGINAL) MODEL CODE GENERATION - ZERO SHOT:\n{orig_resp}")
    pprint(f"(FINETUNED) MODEL CODE GENERATION - ZERO SHOT:\n{ft_resp}")
    pprint(dash_line)
    print("\n")


dataset_name = "mbpp"
device = "mps"

####
source_model = "google/flan-t5-base"
# load_from_checkpoint
ft_model_name = "/Users/beltre.wilton/apps/code_gen_devfest/outputs/flan-t5-base-7-47.92197974522909-1694608836"
####

orig_model, orig_tokenizer, dataset = load_model(
    dataset_name, ["train", "test"], source_model, device, inference=True
)
ft_model, ft_tokenizer, _ = load_model(
    None, ["test"], ft_model_name, device, inference=True
)
test_dl = get_test_dl(dataset)


def test(orig_model, orig_tokenizer, ft_model, ft_tokenizer, test_dl, device):
    for i, x in enumerate(test_dl):
        task, resp = x
        ft_model.eval()
        orig_model.eval()
        with torch.no_grad():
            input_ids = ft_tokenizer(
                task, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            # labels = ft_tokenizer(resp, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
            # outputs = model(input_ids=input_ids, labels=labels)
            outputs = ft_model.generate(input_ids)  # max_length
            # logits = outputs.logits
            ft_resp = ft_tokenizer.decode(outputs[0], skip_special_tokens=True)

            input_ids = orig_tokenizer(
                task, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            # labels = orig_tokenizer(resp, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
            outputs = orig_model.generate(input_ids)
            orig_resp = orig_tokenizer.decode(outputs[0], skip_special_tokens=True)

            compare_models(task, resp, orig_resp, ft_resp)

            if i == 1:
                print("End!")
                break


if __name__ == "__main__":
    test(orig_model, orig_tokenizer, ft_model, ft_tokenizer, test_dl, device)
