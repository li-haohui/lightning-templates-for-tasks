import torch
import numpy as np
import MinkowskiEngine as ME

from scipy.stats import pearsonr, spearmanr, kendalltau

import pytorch_lightning as pl
import torch.optim.lr_scheduler as lrs

class ModelInterface(pl.LightningModule):
    def __init__(self, model, criterion, **kargs):
        super().__init__()
        self.save_hyperparameters(kargs)
        self.model = model
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        stensor = ME.SparseTensor(
            coordinates=batch["coordinates"], features=batch["features"]
        )
        mos = torch.tensor(batch["labels"], device=self.device).double()

        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()

        loss = self.criterion(self(stensor).squeeze(-1).double(), mos)

        self.log("train_loss", loss, batch_size=mos.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        stensor = ME.SparseTensor(
            coordinates=batch["coordinates"], features=batch["features"]
        )
        mos = torch.tensor(batch["labels"], device=self.device).double()
        pred = self(stensor).squeeze(-1).double()
        loss = self.criterion(pred, mos)

        self.log('dev_loss', loss, batch_size=mos.size(0))

        return {
            "pred": pred.detach().cpu().numpy(),
            "labels": mos.detach().cpu().numpy()
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def validation_epoch_end(self, outputs):
        pred = np.array([], dtype=np.float64)
        mos = np.array([], dtype=np.float64)

        for o in outputs:
            pred = np.append(pred, o["pred"])
            mos = np.append(mos, o["labels"])

        plcc, srocc, krocc = pearsonr(pred, mos)[0], spearmanr(pred, mos)[0], kendalltau(pred, mos)[0]

        self.log("val_plcc", plcc, on_epoch=True)
        self.log("val_srocc", srocc, on_epoch=True)
        self.log("val_krocc", krocc, on_epoch=True)


        printstr = "Epoch[{}] -> PLCC: {:.4f}, SROCC: {:.4f}, KROCC: {:.4f}"
        self.print(printstr.format(self.current_epoch, plcc, srocc, krocc))


    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            elif self.hparams.lr_scheduler == "exp":
                scheduler = lrs.ExponentialLR(optimizer, gamma=self.hparams.gamma)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]



