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
            df = pd.read_csv(gen_dir + f"{split}_df.csv")
            print(f"{split[0].upper()+split[1:]}: ")

            df["precision"] = None
            df["recall"] = None

            for idx in range(len(df)):
                precs_true = list(df["precs"])[idx].split(",")
                precs_true = [mat.strip() for mat in precs_true]
                precs_pred = list(df["precs_gen"])[idx].split(",")
                precs_pred = [mat.strip() for mat in precs_pred]
                
                count_shared = len(set(precs_true).intersection(precs_pred))
                df["precision"][idx] = count_shared/len(precs_true)
                df["recall"][idx]    = count_shared/len(precs_pred)
            
            df.to_csv(gen_dir + f"{split}_df.csv", index=False)
            
            # rouge
            rouge_results = rouge.compute(predictions=list(df["precs_gen"]), 
                                        references=list(df["precs"]))
            
            print(rouge_results)

            # accuracy
            accuracy = {"precision": df["precision"].mean(),
                        "recall": df["recall"].mean()}
            print(accuracy)
            print("\n")

