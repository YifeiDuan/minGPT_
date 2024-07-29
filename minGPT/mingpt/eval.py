import evaluate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    metrics_records = []
    ########## 1. Load the data ##########
    for epoch in range(ckpt_start, ckpt_end+ckpt_step, ckpt_step):
        print(f"Evaluating checkpoint at epoch {epoch}")
        gen_dir = f"/home/jupyter/YD/ZeoPrecLLM/generation_analysis/{model_type}/epoch_{epoch}/"

        for split in ["train", "val", "test"]:
            df = pd.read_csv(gen_dir + f"{split}_df.csv")
            print(f"{split[0].upper()+split[1:]}: ")

    ########## 2. Calculate metrics ##########
            df["precision"] = None
            df["recall"] = None

            for idx in range(len(df)):
                precs_true = str(list(df["precs"])[idx]).split(",")
                precs_true = [mat.strip() for mat in precs_true]
                precs_pred = str(list(df["precs_gen"])[idx]).split(",")
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
            precision = df["precision"].mean()
            recall = df["recall"].mean()
            f1 = 2/(1/precision + 1/recall)
            accuracy = {"precision": precision,
                        "recall": recall,
                        "F1": f1}
            print(accuracy)
            print("\n")

    ########## 3. Save record ##########
            record = {
                "model": model_type,
                "epoch": epoch,
                "split": split
            }
            record = {**record, **accuracy, **rouge_results}

            metrics_records.append(record)

    df_metrics = pd.DataFrame.from_records(metrics_records)
    df_metrics.to_csv(f"/home/jupyter/YD/ZeoPrecLLM/generation_analysis/{model_type}/eval_metrics_summary.csv", index=False)

    ########## 4. Plots ##########
    for split in ["train", "val", "test"]:
        df = df_metrics[(df_metrics["split"]==split)]
        plt.figure()
        plt.plot(df["epoch"], df["precision"], label="precision")
        plt.plot(df["epoch"], df["recall"], label="recall")
        plt.plot(df["epoch"], df["F1"], label="F1")
        plt.savefig(f"/home/jupyter/YD/ZeoPrecLLM/generation_analysis/{model_type}/eval_metrics_accuracy.jpg")
        plt.show()

        plt.figure()
        plt.plot(df["epoch"], df["rouge1"], label="rouge1")
        plt.plot(df["epoch"], df["rouge2"], label="rouge2")
        plt.plot(df["epoch"], df["rougeL"], label="rougeL")
        plt.plot(df["epoch"], df["rougeLsum"], label="rougeLsum")
        plt.savefig(f"/home/jupyter/YD/ZeoPrecLLM/generation_analysis/{model_type}/eval_metrics_rouge.jpg")
        plt.show()


