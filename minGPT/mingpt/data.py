import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import pandas as pd
import numpy as np
import pickle

import sys
sys.path.append("/home/jupyter/YD/ZeoPrecLLM/ZeoPrec/minGPT")
from mingpt.bpe import *

data_dir = "/home/jupyter/YD/ZeoPrecLLM/ZeoPrec/prec_dataset/"


def collate_zeo_datasets():
    ### Get the BPE encoder ###
    enc = get_encoder()

    ### Process zeolite precursor datasets ###
    processed_data = {
        "train": [],
        "val": [],
        "test": []
    }
    vocab = set()   # keep record of the overall vocab to determine the vocab size of the problem
    max_total_length = 0  # keep record of the maximum sequence length
    max_prompt_length = 0
    for split in ["train", "val", "test"]:
        ##### Read in the raw data #####
        df = pd.read_csv(data_dir + f"prec_dataset_{split}.csv")
        zeo_cols = [x for x in df.columns if 'zeo_' in x]   # zeolite features
        syn_cols = [x for x in df.columns if 'syn_' in x]   # synthesis gel features

        ##### Tokenize and convert the dataset #####
        for idx in range(len(df)):
            ####### Read the raw data feats #######
            zeo = df.iloc[idx]
            zeo_text = zeo["zeo"]   # the zeolite structure code, e.g. GME, CHA, BEA
            zeo_rep = zeo[zeo_cols] # the external zeolite structural representation
            syn_rep = zeo[syn_cols] # the external synthesis gel representation
            prec_text = zeo["precs"]    # the precursors as text
            ####### Tokenize and process as input, output tokens #######
            text = f"{zeo_text} # Precursors: {prec_text}"
            tokens = enc.encode(text)    # tokenize the entire text sequence
            prompt = f"{zeo_text} # Precursors: "
            prompt_tokens = enc.encode(prompt)
            input_tokens = tokens[:-1]  # shift
            output_tokens = tokens[1:]  # shift
            mask_len = len(enc.encode(f"{zeo_text} # Precursors: ")) - 1
            output_tokens[:mask_len] = [-1]*mask_len   # mask the read-in input tokens in the output sequence
            ####### Store the data point into processed_data #######
            processed_data[split].append(
                {
                    "zeo": zeo_text,
                    "precs": prec_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "zeo_rep": zeo_rep,
                    "syn_rep": syn_rep,
                    "prompt_tokens": prompt_tokens
                }
            )
            ####### Update vocab and max_length info #######
            vocab = vocab.union(set(list(tokens)))
            max_total_length = max(max_total_length, len(input_tokens))
            max_prompt_length = max(max_prompt_length, len(prompt_tokens))


    vocab = set(sorted(list(vocab)))    # sort the token ids within the corpus vocab

    info = {
        "vocab_size": len(vocab) + 1,   # an additional one used for padding/mask token
        "max_total_length": max_total_length,
        "max_prompt_length": max_prompt_length,
        "tokenID_to_vocabID": {**{tok_id:i+1 for i, tok_id in enumerate(vocab)}, **{-1:0}},  # mask token id -1 is mapped to vocab id 0
        "vocabID_to_tokenID": {**{i+1:tok_id for i, tok_id in enumerate(vocab)}, **{0:-1}}
    }

    ### Collate the datasets ###
    train_dataset = ZeoDataset("train", processed_data["train"], info)
    val_dataset = ZeoDataset("val", processed_data["val"], info)
    test_dataset = ZeoDataset("test", processed_data["test"], info)

    with open(data_dir + "info.pkl", 'wb') as file:
        pickle.dump(info, file)

    with open(data_dir + "train_dataset.pkl", 'wb') as file:
        pickle.dump(train_dataset, file)
    with open(data_dir + "val_dataset.pkl", 'wb') as file:
        pickle.dump(val_dataset, file)
    with open(data_dir + "test_dataset.pkl", 'wb') as file:
        pickle.dump(test_dataset, file)

    return train_dataset, val_dataset, test_dataset, info       







class ZeoDataset(Dataset):
    """ 
    Dataset for the Zeolite precursor generation problem. E.g.
    Input: GME # Precursors: -> Output: sodium silicate, sodium aluminate
    Which will feed into the transformer shifted and concatenated as:
    input:  GME # Precursors: sodium silicate, sodium
    output: I I I sodium silicate, sodium aluminate
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, data_records, info):
        ##### Read in the data of the specified split as pandas df #####
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.data_records = data_records
        self.max_total_length = info["max_total_length"]
        self.vocab_size = info["vocab_size"]
        self.tokenID_to_vocabID = info["tokenID_to_vocabID"]
        self.vocabID_to_tokenID = info["vocabID_to_tokenID"]
    
    def __len__(self):
        return len(self.data_records)
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.max_total_length

    def __getitem__(self, idx):
        ### Get the record for the data point at the specified idx ###
        zeo = self.data_records[idx]

        ### Get the input and output tokens, pad to max_total_length and then convert to torch tensors ###
        x = zeo["input_tokens"]
        y = zeo["output_tokens"]
        ##### Padding #####
        x = [-1]*(self.max_total_length-len(x)) + x        
        y = [-1]*(self.max_total_length-len(y)) + y
        ##### Map token ids to vocab ids #####
        x = [self.tokenID_to_vocabID[t_id] for t_id in x]
        y = [self.tokenID_to_vocabID[t_id] for t_id in y]
        ##### Convert to batchable torch tensors #####
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        ### Get the external representations and concert to torch tensors ###
        zeo_rep = torch.tensor(zeo["zeo_rep"])
        syn_rep = torch.tensor(zeo["syn_rep"])
        

        return x, zeo_rep, syn_rep, y
    




class ZeoEvalDataset(Dataset):
    """ 
    Evaluation dataset for the Zeolite precursor generation problem. E.g.
    Prompt: GME # Precursors: -> Generation: sodium silicate, sodium aluminate
    """

    def __init__(self, split, data_records, info):
        ##### Read in the data of the specified split as pandas df #####
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.data_records = data_records
        self.max_prompt_length = info["max_prompt_length"]
        self.vocab_size = info["vocab_size"]
        self.tokenID_to_vocabID = info["tokenID_to_vocabID"]
        self.vocabID_to_tokenID = info["vocabID_to_tokenID"]
    
    def __len__(self):
        return len(self.data_records)
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.max_prompt_length

    def __getitem__(self, idx):
        ### Get the record for the data point at the specified idx ###
        zeo = self.data_records[idx]

        ### Get the input and output tokens, pad to max_total_length and then convert to torch tensors ###
        x = zeo["prompt_tokens"]
        ##### Padding #####
        x = [-1]*(self.max_prompt_length-len(x)) + x
        ##### Map token ids to vocab ids #####
        x = [self.tokenID_to_vocabID[t_id] for t_id in x]
        ##### Convert to batchable torch tensors #####
        x = torch.tensor(x, dtype=torch.long)

        ### Get the external representations and concert to torch tensors ###
        zeo_rep = torch.tensor(zeo["zeo_rep"])
        syn_rep = torch.tensor(zeo["syn_rep"])
        

        return x, zeo_rep, syn_rep