import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from tqdm import tqdm

from code_gen_util import get_dls, load_resources


device = "cuda" if torch.cuda.is_available() else "mps"
model_name = "google/flan-t5-base"
dataset_name = "mbpp"


## save best model utility
class KeepBestModel:
    def __init__(self):
        self.best_valid_loss = float("inf")
        self.out = "./outputs/flan-t5-base-{0}"

    def __call__(
        self, current_valid_loss, epoch, model, optimizer, loss_criterion, tokenizer
    ):
        try:
            print(
                f"Current Valid Loss: {current_valid_loss}, Best Valid Loss: {self.best_valid_loss}"
            )
            # if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")

            # LoRA weigths + base weights
            model = model.merge_and_unload()
            suffix = f"{epoch+1}-{str(self.best_valid_loss)}-{str(int(time.time()))}"
            model.save_pretrained(save_directory=self.out.format(suffix))
            tokenizer.save_pretrained(save_directory=self.out.format(suffix))
        except Exception as ex:
            print(f"**************************************")
            print(ex)
            print(f"**************************************")



def train(
    model: T5ForConditionalGeneration,
    train_dataloader,
    eval_dataloader,
    epochs,
    tokenizer,
    device,
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, eps=1e-9)
    keep = KeepBestModel()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch+1:02d}")
        for task, resp in batch_iterator:
            encoding = tokenizer(task, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = encoding.input_ids.to(device)
            attention_mask = encoding.attention_mask.to(device)
            labels = tokenizer(resp, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            
            # altertative version: 
            loss_computed = outputs.loss # (yes! the same result) 
            # loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            train_loss += loss_computed.item()
            loss_computed.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # jug ????
            optimizer.step()
            optimizer.zero_grad()
            #

        # each epoch chech for the best model
        keep(
            train_loss / len(batch_iterator),
            epoch,
            model,
            optimizer,
            loss_fn,
            tokenizer,
        )


if __name__ == "__main__":
    t5_model, tokenizer, dataset = load_resources(dataset_name, ["train", "validation"], model_name, device, )
    train_dataloader, eval_dataloader = get_dls(dataset=dataset, batch_size=32)
    train(t5_model, train_dataloader, eval_dataloader, 10, tokenizer, device)
