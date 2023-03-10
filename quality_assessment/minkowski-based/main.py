import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import argparse

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc

from data import DInterface
from model import MInterface

def main(args):
    pl.seed_everything(args.seed)
    data_module = DInterface(**vars(args))

    model = None

    criterion = torch.nn.MSELoss()
    model_module = MInterface(model, criterion, **vars(args))

    trainer = Trainer.from_argparse_args(args, accelerator='gpu', devices=1)

    trainer.fit(model_module, data_module)


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--train", type=bool, default=True)

    # DATA
    parser.add_argument("--voxel_size", type=int, default=5)
    parser.add_argument("--train_info", type=str, default="./data_cfgs/LS-PCQA/train.csv")
    parser.add_argument("--train_dir", type=str, default="/public/DATA/lhh/LS-PCQA/samples_with_MOS")
    parser.add_argument("--val_info", type=str, default="./data_cfgs/LS-PCQA/test.csv")
    parser.add_argument("--val_dir", type=str, default="/public/DATA/lhh/LS-PCQA/samples_with_MOS")
    parser.add_argument("--test_info", type=str, default="./data_cfgs/LS-PCQA/test.csv")
    parser.add_argument("--test_dir", type=str, default="/public/DATA/lhh/LS-PCQA/samples_with_MOS")
    # Model

    parser = Trainer.add_argparse_args(parser)

    # Hyper Parameters
        # Optimizer
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.95)

        # Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="exp")
    parser.add_argument("--lr_decay_steps", type=float, default=10)
    parser.add_argument("--lr_decay_min_lr", type=float, default=1e-8)
    parser.add_argument("--lr_decay_rate", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.95)

        # Training
    parser.add_argument("--batch_size", type=int, default=5)

    parser.set_defaults(max_epochs=100)

    args = parser.parse_args()


    main(args)