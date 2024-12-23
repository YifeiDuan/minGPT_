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
    # parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--device',type=str, default="auto")
    parser.add_argument('--external_rep_mode',type=int, default=1)

    args = parser.parse_args()

    model_type = args.model_type
    learning_rate = args.learning_rate
    # max_iters = args.max_iters
    max_epochs = args.max_epochs
    save_every = args.save_every
    num_workers = args.num_workers
    device = args.device
    external_rep_mode = args.external_rep_mode


    ########## 1. Read the preprocessed data ##########
    data_dir = f"/home/jupyter/YD/ZeoPrecLLM/ZeoPrec/prec_dataset/"

    with open(data_dir + "train_dataset.pkl", 'rb') as file:
        train_dataset = pickle.load(file)


    ########## 2. Initialize the minGPT model ##########
    model_config = VectraGPT.get_default_config()
    model_config.model_type = model_type
    model_config.vocab_size = train_dataset.get_vocab_size()
    print(f"vocab size: {model_config.vocab_size}")
    model_config.block_size = train_dataset.get_block_size()
    model_config.max_text_len = train_dataset.get_block_size()
    model_config.external_rep_mode = external_rep_mode
    if external_rep_mode == 2:
        model_config.block_size += 1    # add the "structure token" into block size
    if external_rep_mode != 0:
        model_config.external_dim = train_dataset[0][1].shape[0] + train_dataset[0][2].shape[0]   # lens of zeo_rep and syn_rep
    else: # mode = 0 (no external rep as conditions)
        model_config.external_dim = 0
    print(f"External rep dimension: {model_config.external_dim}")
    model = VectraGPT(model_config)
    model = model.to(torch.double)


    ########## 3. Config the trainer ##########
    train_config = Trainer.get_default_config()
    train_config.learning_rate = learning_rate
    # train_config.max_iters = max_iters
    train_config.max_epochs = max_epochs
    train_config.num_workers = num_workers
    train_config.device = device
    trainer = Trainer(train_config, model, train_dataset, external_rep_mode=external_rep_mode)

    
    ########## 4. Start training ##########
    train_losses = []
    save_dir = f"/home/jupyter/YD/ZeoPrecLLM/saved_models/{model_type}_mode{external_rep_mode}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    def batch_end_callback(trainer):
        train_losses.append(trainer.loss.item())
    
    def epoch_end_callback(trainer):
        ### print epoch record ###
        print(f"epoch_dt {trainer.epoch_dt * 1000:.2f}ms; epoch {trainer.epoch}: train loss {trainer.epoch_loss.item():.5f}")

        ### save checkpoint ###
        if trainer.epoch % save_every == 0 or trainer.epoch == max_epochs:            
            save_dict = {
                    'model_state_dict': model.state_dict(),
                    "model_config": model_config,
                    "train_config": train_config,
                    'model_name': f"{model_type}_epoch{trainer.epoch}",
                    "train_loss": trainer.loss.item()
                }

            torch.save(save_dict, save_dir + f"epoch{trainer.epoch}_model.pth")

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.set_callback('on_epoch_end', epoch_end_callback)

    trainer.run()

    plt.figure()
    plt.plot(train_losses)
    plt.show()