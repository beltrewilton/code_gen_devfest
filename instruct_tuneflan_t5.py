import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5LayerCrossAttention

from tqdm import tqdm

from code_gen_util import get_dls, load_model, print_trainable_parameters


device = "cuda" if torch.cuda.is_available() else "mps"
torch.cuda.empty_cache()
model_name = "google/flan-t5-base"
dataset_name = "code_x_glue_ct_code_to_text"

### Balance: $15.682
### Processing epoch 01:   5%|▊  | 838/15739 [13:07<3:52:58,  1.07it/s]
## 6:34 PM

# Balance:  $14.085
# Processing epoch 01:  25%|███▏ | 3880/15739 [1:00:43<3:05:30,  1.07it/s]
# 7:22 PM


## save best model utility
class KeepBestModel:
    def __init__(self):
        self.best_valid_loss = float("inf")
        self.out = "./outputs/flan-t5-base-{0}"

    def __call__(
        self, current_valid_loss, epoch, model, optimizer, tokenizer
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
            # model = model.merge_and_unload()
            suffix = f"{epoch+1}-{str(self.best_valid_loss)}-{str(int(time.time()))}"
            model.save_pretrained(save_directory=self.out.format(suffix))
            tokenizer.save_pretrained(save_directory=self.out.format(suffix))
        except Exception as ex:
            print(f"**************************************")
            print(ex)
            print(f"**************************************")


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def freeze_decoder_except_xattn_codegen(model):
    print(f'Para before freezing: {model.num_parameters():,}, trainable para: {get_model_size(model)}')
    for param in model.decoder.parameters():
        param.requires_grad = False

    # num_decoder_layers = model.decoder.config.n_layer 
    num_decoder_layers = model.decoder.config.num_layers
    # for i in range(num_decoder_layers):
    for j in range(len(model.decoder.block)):
        block = model.decoder.block[j]
        # each_decoder_layer = model.decoder.transformer.h[i]
        # each_decoder_layer = model.decoder.block[1]
        # if hasattr(model.decoder.block[1].layer[1], 'crossattention'):
        for i in range(len(block.layer)):
            layer = block.layer[i]
            if isinstance(layer, T5LayerCrossAttention):
                for param in layer.parameters():
                    param.requires_grad = True
                layer.to(torch.float32)

        # if hasattr(each_decoder_layer, 'alpha_xattn'):
        #     each_decoder_layer.alpha_xattn.requires_grad = True
    print(f'Para after freezing: {model.num_parameters():,}, trainable para: {get_model_size(model)}')



def train(
    model: T5ForConditionalGeneration,
    train_dataloader,
    eval_dataloader,
    epochs,
    tokenizer,
    device,
):
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, eps=1e-9)
    keep = KeepBestModel()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch+1:02d}")
        for i, model_inputs in enumerate(batch_iterator):
            input_ids = model_inputs["input_ids"].to(device)
            input_ids = input_ids.view(-1, input_ids.size(-1))
            attention_mask = model_inputs["decoder_attention_mask"].to(device).view(-1, input_ids.size(-1))
            labels = model_inputs["labels"].to(device).view(-1, input_ids.size(-1))
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            # logits = outputs.logits
            
            # altertative version: 
            loss_computed = outputs.loss # (yes! the same result) 
            # loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            train_loss += loss_computed.item()
            loss_computed.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # jug ????
            optimizer.step()
            optimizer.zero_grad()
            #

            if i == 5:
                print('Finish!')
                break

        # each epoch chech for the best model
        keep(
            train_loss / len(batch_iterator),
            epoch,
            model,
            optimizer,
            tokenizer,
        )


if __name__ == "__main__":
    t5_model = load_model(model_name, device)
    train_dataloader, eval_dataloader, tokenizer = get_dls(model_name=model_name, dataset_name=dataset_name, batch_size=16)
    print(f"  ==> Loaded model from {model_name}, model size {t5_model.num_parameters():,}")
    freeze_decoder_except_xattn_codegen(t5_model)
    print(print_trainable_parameters(f"Base Model (freeze_decoder): {model_name}", t5_model))
    train(t5_model, train_dataloader, eval_dataloader, 2, tokenizer, device)
