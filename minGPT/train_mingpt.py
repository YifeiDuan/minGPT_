import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import matplotlib.pyplot as plt

import argparse, yaml
import os
import sys
sys.path.append("/home/jupyter/YD/ZeoPrecLLM/ZeoPrec/minGPT")

from mingpt.data import *
from mingpt.model import VectraGPT
from mingpt.trainer import Trainer




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="gpt-mini")   
    parser.add_argument('--learning_rate', type=float, default=5e-4) # the model we're using is so small that we can go a bit faster
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--device',type=str, default="auto")

    args = parser.parse_args()

    model_type = args.model_type
    learning_rate = args.learning_rate
    max_iters = args.max_iters
    num_workers = args.num_workers
    device = args.device


    ########## 1. Read and collate data ##########
    train_dataset, val_dataset, test_dataset, info = collate_zeo_datasets()


    ########## 2. Initialize the minGPT model ##########
    model_config = VectraGPT.get_default_config()
    model_config.model_type = model_type
    model_config.vocab_size = train_dataset.get_vocab_size()
    print(f"vocab size: {model_config.vocab_size}")
    model_config.block_size = train_dataset.get_block_size()
    model_config.external_dim = train_dataset[0][1].shape[0] + train_dataset[0][2].shape[0]   # lens of zeo_rep and syn_rep
    print(f"External rep dimension: {model_config.external_dim}")
    model = VectraGPT(model_config)
    model = model.to(torch.float)


    ########## 3. Config the trainer ##########
    train_config = Trainer.get_default_config()
    train_config.learning_rate = learning_rate
    train_config.max_iters = max_iters
    train_config.num_workers = num_workers
    train_config.device = device
    trainer = Trainer(train_config, model, train_dataset)

    
    ########## 4. Start training ##########
    train_losses = []
    save_dir = "/home/jupyter/YD/ZeoPrecLLM/saved_models"
    def batch_end_callback(trainer):
        train_losses.append(trainer.loss.item())

        if trainer.iter_num % 100 == 0:
            ### print loss ###
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
            
            ### save checkpoint ###
            save_dict = {
                    'model_state_dict': model.state_dict(),
                    "model_config": model_config,
                    "train_config": train_config,
                    'model_name': f"{model_type}_ckpt{trainer.iter_num}",
                    "train_loss": trainer.loss.item()
                }

            torch.save(save_dict, save_dir + f"{model_type}_ckpt{trainer.iter_num}_model.pth")

    trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()

    plt.figure()
    plt.plot(train_losses)
    plt.show()