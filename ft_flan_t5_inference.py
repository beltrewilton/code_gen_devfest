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
  
  # 407

dataset_name = "mbpp"
device = "mps"

####
source_model = "google/flan-t5-base"
# load_from_checkpoint
ft_model_name = "/Users/beltre.wilton/Documents/devfest_docs/flan-t5-base-9-1.3386965940395992-MBPP"
####

orig_model = load_model(source_model, device, inference=True)
ft_model = load_model(ft_model_name, device, inference=True)
test_dl, _ = get_test_dl(ft_model_name, dataset_name, batch_size=3)
orig_tokenizer = AutoTokenizer.from_pretrained(source_model)
ft_tokenizer = AutoTokenizer.from_pretrained(ft_model_name)


def test(orig_model, orig_tokenizer, ft_model, ft_tokenizer, test_dl, device):
    for model_inputs in test_dl:
        for input_ids, task, resp in zip(model_inputs["input_ids"], model_inputs["instruct_text"], model_inputs["code_text"]):
            input_ids = input_ids.to(device)
            # # labels = model_inputs["labels"]
            # task = model_inputs["instruct_text"]
            # resp = model_inputs["code_text"]

            ft_model.eval()
            orig_model.eval()
            with torch.no_grad():
                outputs = ft_model.generate(input_ids)
                ft_resp = ft_tokenizer.decode(outputs[0], skip_special_tokens=True)

                outputs = orig_model.generate(input_ids)
                orig_resp = orig_tokenizer.decode(outputs[0], skip_special_tokens=True)

                compare_models(task, resp, orig_resp, ft_resp) 


if __name__ == "__main__":
    test(orig_model, orig_tokenizer, ft_model, ft_tokenizer, test_dl, device)
