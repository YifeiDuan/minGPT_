import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import argparse, yaml
import os

from mingpt.data import *
from mingpt.model import VectraGPT
from mingpt.trainer import Trainer




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="gpt-mini")   
    parser.add_argument('--learning_rate', default=5e-4) # the model we're using is so small that we can go a bit faster
    parser.add_argument('--max_iters', default=1000)
    parser.add_argument('--num_workers', default=0)

    args = parser.parse_args()

    model_type = args.model_type
    learning_rate = args.learning_rate
    max_iters = args.max_iters
    num_workers = args.num_workers


    ########## 1. Read and collate data ##########
    train_dataset, val_dataset, test_dataset, info = collate_zeo_datasets()


    ########## 2. Initialize the minGPT model ##########
    model_config = VectraGPT.get_default_config()
    model_config.model_type = model_type
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model = VectraGPT(model_config)


    ########## 3. Config the trainer ##########
    train_config = Trainer.get_default_config()
    train_config.learning_rate = learning_rate
    train_config.max_iters = max_iters
    train_config.num_workers = num_workers
    trainer = Trainer(train_config, model, train_dataset)

    
    ########## 4. Start training ##########
    save_dir = "/home/jupyter/YD/ZeoPrecLLM/saved_models"
    def batch_end_callback(trainer):
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