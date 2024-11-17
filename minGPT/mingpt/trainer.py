"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

import sys
sys.path.append("/home/jupyter/YD/ZeoPrecLLM/ZeoPrec/minGPT")
from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.external_dim = None
        return C

    def __init__(self, config, model, train_dataset, external_rep_mode=1):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        self.external_rep_mode = external_rep_mode

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.epoch = 0
        self.batch_num = 0
        self.epoch_time = time.time()
        epoch_batch_losses = []
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                # Previous epoch ends, stats a new one
                self.epoch += 1
                self.batch_num = 0

                tnow = time.time()
                self.epoch_dt = tnow - self.epoch_time
                self.epoch_time = tnow

                self.epoch_loss = torch.mean(torch.tensor(epoch_batch_losses, dtype=float))
                epoch_batch_losses = []

                self.trigger_callbacks('on_epoch_end')

                # termination conditions
                if config.max_epochs is not None and self.epoch >= config.max_epochs:
                    break
                
                data_iter = iter(train_loader)
                batch = next(data_iter)
                
            batch = [t.to(self.device) for t in batch]
            x, y = batch


            # forward the model
            logits, self.loss = model(x, 
                                    targets=y) 
            # self.loss is the mean CE loss of data points in the batch
            # pass in not only explicit input, but also a tuple of all external latent rep

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            epoch_batch_losses.append(self.loss.item())
            self.batch_num += 1
            self.trigger_callbacks('on_batch_end')
            