import evaluate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse, yaml
import os

if __name__ == '__main__':

    ########## 0. Configurate the session ##########
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="gpt-mini") 

    args = parser.parse_args()
    model_type = args.model_type
    external_rep_modes = args.external_rep_modes

    ########## 1. Load eval metrics ##########
    dfs = {}
    for external_rep_mode in list(external_rep_modes):
        df = pd.read_csv(f"eval_metrics_summary.csv")
        dfs[external_rep_mode] = df


    ########## 2. Plots: Compare splits in one mode ##########
        for metric in ["precision", "recall", "F1", "rouge1", "rouge2", "rougeL", "rougeLsum"]:
            df_train = df[(df["split"]=="train")]
            df_val = df[(df["split"]=="val")]
            df_test = df[(df["split"]=="test")]
            
            plt.figure()
            plt.plot(df_train["epoch"], df_train[metric], label="train")
            plt.plot(df_val["epoch"], df_val[metric], label="val")
            plt.plot(df_test["epoch"], df_test[metric], label="test")
            plt.legend()
            plt.title(f"{model_type}_mode{external_rep_mode}_{metric}")
            plt.savefig(f"eval_metrics_{metric}.jpg")
            plt.show()
    

    ########## 3. Plots: Compare modes ##########
    save_path = f"{model_type}_comparison/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    for metric in ["precision", "recall", "F1", "rouge1", "rouge2", "rougeL", "rougeLsum"]:
        for split in ["train", "val", "test"]:
            plt.figure()

            for external_rep_mode, df in dfs.items():
                df_split = df[(df["split"]==split)]
                plt.plot(df_split["epoch"], df_split[metric], label=f"mode{external_rep_mode}")

            plt.legend()
            plt.title(f"{model_type}_{split}_{metric}")
            plt.savefig(save_path + f"{split}_{metric}.jpg")
            plt.show()
