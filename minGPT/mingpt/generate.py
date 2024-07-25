import torch
from torch.utils.data.dataloader import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer

import pickle
import copy
import argparse, os

import sys
sys.path.append("/home/jupyter/YD/ZeoPrecLLM/ZeoPrec/minGPT")
from mingpt.data import ZeoEvalDataset
from mingpt.bpe import *
from mingpt.model import VectraGPT




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



def batched_generate(model, data_loader, steps=20, do_sample=True, device='cpu'):
    """
    If use_mingpt == False, then model_type must not be None (we need to load a specified pretrained tokenizer)

    data_loader: the torch dataloader that contains prompts and external reps used for generation
    """
    tokenizer = get_encoder()
    data_records = copy.deepcopy(data_loader.dataset.data_records)

    for batch in data_loader:
        batch = [t.to(device) for t in batch]
        ids, x, zeo_rep, syn_rep = batch
    
        # forward the model `steps` times to get samples, in a batch
        y = model.generate(x, external_rep=(zeo_rep, syn_rep), max_new_tokens=steps, do_sample=do_sample, top_k=40)
        
        for i in range(x.shape[0]):
            vocabID_seq = y[i].cpu().squeeze().numpy()
            vocabID_to_tokenID = data_loader.dataset.vocabID_to_tokenID
            tokenID_seq = [vocabID_to_tokenID[v_id] for v_id in vocabID_seq]
            tokenID_seq = [t_id for t_id in tokenID_seq if t_id!=-1]
            out = tokenizer.decode(tokenID_seq)     # input list into encoder.decode function for text output
            
            # slice the generated precursors and record together with true precursors for reference
            dp_id = ids[i].item()
            zeo_text = data_records[dp_id]["zeo"]
            prompt = f"{zeo_text} # Precursors:"
            pred_prec = out.replace(prompt, "").strip()
            if "<|endoftext|>" in pred_prec:
                pred_prec = pred_prec.split("<|endoftext|>")[0]

            data_records[dp_id]["precs_gen"] = pred_prec

    return data_records




if __name__ == "__main__":
    
    ########## 0. Configurate the session ##########
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="gpt-mini")  
    parser.add_argument('--device', default="auto")   
    parser.add_argument('--ckpt_start', type=int, default=10)  
    parser.add_argument('--ckpt_end', type=int, default=100) 
    parser.add_argument('--ckpt_step', type=int, default=10) 

    args = parser.parse_args()
    model_type = args.model_type
    device = args.device
    ckpt_start = args.ckpt_start
    ckpt_end = args.ckpt_end
    ckpt_step = args.ckpt_step

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    ########## 2. Load trained model ##########
    save_dir = f"/home/jupyter/YD/ZeoPrecLLM/saved_models/{model_type}/"
    for epoch in range(ckpt_start, ckpt_end+ckpt_step, ckpt_step):
        print(f"Generating for model saved at epoch {epoch}")
        trained_stuff = torch.load(save_dir + f"epoch{epoch}_model.pth")

        model_config = trained_stuff["model_config"]
        model_config.model_type = None
        model = VectraGPT(model_config)
        model.load_state_dict(trained_stuff["model_state_dict"])
        model = model.to(torch.double)
        model = model.to(device)
        
    ########## 3. Generate precursors with batched prompts and external reps ##########
    
        gen_dir = f"/home/jupyter/YD/ZeoPrecLLM/generation_analysis/{model_type}/epoch_{epoch}/"
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)

        train_gen_records = batched_generate(model, train_loader, steps=info["max_total_length"], do_sample=True, device=device)
        
        with open(gen_dir + "train_set_records.pkl", 'wb') as file:
            pickle.dump(train_gen_records, file)