import evaluate

import pandas as pd
import numpy as np

import argparse, yaml

rouge = evaluate.load("rouge")

if __name__ == '__main__':

    ########## 0. Configurate the session ##########
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="gpt-mini") 
    parser.add_argument('--ckpt_start', type=int, default=10)  
    parser.add_argument('--ckpt_end', type=int, default=100) 
    parser.add_argument('--ckpt_step', type=int, default=10) 

    args = parser.parse_args()
    model_type = args.model_type
    ckpt_start = args.ckpt_start
    ckpt_end = args.ckpt_end
    ckpt_step = args.ckpt_step

    ########## 1. Load the data ##########
    for epoch in range(ckpt_start, ckpt_end+ckpt_step, ckpt_step):
        print(f"Evaluating checkpoint at epoch {epoch}")
        gen_dir = f"/home/jupyter/YD/ZeoPrecLLM/generation_analysis/{model_type}/epoch_{epoch}/"

        for split in ["train", "val", "test"]:
            

        pd.read_csv(gen_dir + f"train_df.csv")

