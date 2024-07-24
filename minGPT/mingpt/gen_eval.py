import torch
from torch.utils.data.dataloader import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer

import pickle

import sys
sys.path.append("/home/jupyter/YD/ZeoPrecLLM/ZeoPrec/minGPT")
from mingpt.data import ZeoEvalDataset




def generate(model, prompt, external_rep=None, num_samples=10, steps=20, do_sample=True):
        
    # tokenize the input prompt into integer input sequence
    tokenizer = BPETokenizer()
    x = tokenizer(prompt).to(model.device)
    
    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, external_rep=external_rep, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    
    for i in range(y.shape[0]):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-'*80)
        print(out)



def generate_batch(model, data_loader, steps=20, do_sample=True, device='auto'):
    """
    If use_mingpt == False, then model_type must not be None (we need to load a specified pretrained tokenizer)

    data_loader: the torch dataloader that contains prompts and external reps used for generation
    """
    for batch in data_loader:
        batch = [t.to(model.device) for t in batch]
        x, zeo_rep, syn_rep = batch
    
        # forward the model `steps` times to get samples, in a batch
        y = model.generate(x, external_rep=(zeo_rep, syn_rep), max_new_tokens=steps, do_sample=do_sample, top_k=40)
        
        for i in range(x.shape[0]):
            out = tokenizer.decode(y[i].cpu().squeeze())
            print('-'*80)
            print(out)





if __name__ == "__main__":

    ########## 1. Read the preprocessed data ##########
    data_dir = f"/home/jupyter/YD/ZeoPrecLLM/ZeoPrec/prec_dataset/"

    with open(data_dir + "train_dataset.pkl", 'rb') as file:
        train_dataset = pickle.load(file)
    with open(data_dir + "val_dataset.pkl", 'rb') as file:
        val_dataset = pickle.load(file)
    with open(data_dir + "test_dataset.pkl", 'rb') as file:
        test_dataset = pickle.load(file)
    with open(data_dir + "info.pkl", 'rb') as file:
        info = pickle.load(file)

    train_dataset = ZeoEvalDataset("train", train_dataset.data_records, info)
    val_dataset = ZeoEvalDataset("val", val_dataset.data_records, info)
    test_dataset = ZeoEvalDataset("test", test_dataset.data_records, info)
    # setup the dataloader
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=64
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=64
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=64
    )


    ########## 2. Generate precursors with batched prompts and external reps ##########
    generate_batch(model, train_loader, steps=info["max_total_length"], do_sample=True, device='auto')