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
    max_length = 0  # keep record of the maximum sequence length
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
                    "syn_rep": syn_rep
                }
            )
            ####### Update vocab and max_length info #######
            vocab = vocab.union(set(list(tokens)))
            max_length = max(max_length, len(input_tokens))
    
    ### Save processed datasets ###
    pd.DataFrame.from_records(processed_data["train"]).to_csv(data_dir + "processed_train.csv", index=False)
    pd.DataFrame.from_records(processed_data["val"]).to_csv(data_dir + "processed_val.csv", index=False)
    pd.DataFrame.from_records(processed_data["test"]).to_csv(data_dir + "processed_test.csv", index=False)

    ### Collate the datasets ###
    train_dataset = ZeoDataset(processed_data["train"])
    val_dataset = ZeoDataset(processed_data["val"])
    test_dataset = ZeoDataset(processed_data["test"])

    info = {
        "vocab_size": len(vocab),
        "max_length": max_length
    }

    with open(data_dir + "info.pkl", 'wb') as file:
        pickle.dump(info, file)

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
        self.max_length = info["max_length"]
        self.vocab_size = info["vocab_size"]
    
    def __len__(self):
        return len(self.df)
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.max_length

    def __getitem__(self, idx):
        ### Get the record for the data point at the specified idx ###
        zeo = self.data_records[idx]

        ### Get the input and output tokens, pad to max_length and then convert to torch tensors ###
        x = zeo["input_tokens"]
        y = zeo["output_tokens"]
        ##### Padding #####
        x = x + [-1]*(self.max_length-len(x))        
        y = y + [-1]*(self.max_length-len(y))
        ##### Convert to batchable torch tensors #####
        x = torch.tensor([x], dtype=torch.long) # [x] to create a "batch_dimension" of 1
        y = torch.tensor([y], dtype=torch.long)

        ### Get the external representations and concert to torch tensors ###
        zeo_rep = torch.tensor([zeo["zeo_rep"]])    # [zeo["zeo_rep"]] to create a "batch_dimension" of 1
        syn_rep = torch.tensor([zeo["syn_rep"]])
        

        return (x, zeo_rep, syn_rep), y