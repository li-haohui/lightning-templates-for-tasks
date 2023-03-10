import torch

from easydict import EasyDict as edict
import pytorch_lightning as pl

try:
    from .dataloader import get_loader
except:
    raise NotImplementedError("Implement Dataset and get_loader in dataloader.py firstly !")

class DataInterface(pl.LightningDataModule):
    def __init__(
        self,
        **kargs
    ):
        super().__init__()
        self.args = edict(kargs)

    def train_dataloader(self):
        return get_loader(phase="train", args=self.args)

    def val_dataloader(self):
        return get_loader(phase="val", args=self.args)

    def test_dataloader(self):
        return get_loader(phase="test", args=self.args)

