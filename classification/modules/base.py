import torch
import torch.nn as nn
import numpy as np
from typing import List,Tuple, Dict, Any, Optional
from omegaconf import OmegaConf
from classification.utils.general import initialize_from_config
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from imblearn.over_sampling import SMOTE

from torchmetrics import JaccardIndex,Dice
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score,confusion_matrix
from sklearn.metrics import precision_score,accuracy_score,roc_auc_score, recall_score,f1_score

class MLPclassifica(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 data_key,
                 in_channels,
                 weight_decay,
                 dropout=0.2,
                 over_sampling: str = None,
                 loss_weight: list=[1,1]
                 ):
        super(MLPclassifica, self).__init__()
        self.over_sampling = over_sampling
        self.num_classes = num_classes
        self.data_key = data_key
        self.weight_decay = weight_decay
        # if loss is not None:
        #     self.loss = initialize_from_config(loss)
        weight = torch.tensor(loss_weight, dtype=torch.float)
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.input = nn.Sequential(
            nn.Linear(in_channels,64),

        )
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=64,out_features=128,bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.classifica = nn.Sequential(
            nn.Linear(512, 2),
        )


    def forward(self,x):
        x = x.view(x.shape[0],-1)

        x = self.input(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        logits = self.classifica(x)

        return logits

    def init_from_ckpt(self,path: str,ignore_keys: List[str] = list()):
        sd = torch.load(path,map_location='cpu')['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def get_input(self, batch: Tuple[Any, Any], key: str = 'dna_data') -> Any:
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.dtype == torch.double:
            x = x.float()
        return x.contiguous()

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        x = self.get_input(batch, self.data_key)
        y = batch['label']

        if self.over_sampling == 'SOMTE':
            smote = SMOTE(random_state=42,k_neighbors=1)
            x, y = smote.fit_resample(x.cpu(), y.cpu())
            x = torch.from_numpy(x).to(self.device)
            y = torch.from_numpy(y).to(self.device)

        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = self.get_input(batch, self.data_key)
        y = batch['label']
        logits = self(x)
        preds = nn.functional.sigmoid(logits).argmax(1)
        loss = self.loss(logits, y)
        output = {'y_true':y,'y_pred':preds,'loss_step':loss}
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return output

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        y_true = []
        y_pred = []

        for output in outputs:
            y_true.extend(output['y_true'].cpu().numpy().flatten())
            y_pred.extend(output['y_pred'].cpu().numpy().flatten())
        if len(np.unique(y_true)) < 2:
            # 处理只有一个类别的情况
            self.log("val/auc", 0.0, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        else:
            preci_score = precision_score(y_true, y_pred)
            acc_score = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            self.log("val/preci_score", preci_score, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log("val/acc_score", acc_score, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log("val/auc", auc, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log("val/recall", recall, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log("val/f1", f1, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        optimizers = [torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=self.weight_decay)]

        total_epochs = self.trainer.max_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=total_epochs)

        schedulers = [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        ]

        return optimizers, schedulers



    # def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
    #     log = dict()
    #     x = self.get_input(batch, self.data_key).to(self.device)
    #     y = batch['label']
    #     # log["originals"] = x
    #     out = self(x)
    #     out = torch.nn.functional.softmax(out,dim=1)
    #     predict = out.argmax(1)
    #
    #     log["image"] = x
    #     log["label"] = y
    #     log["predict"] = predict
    #     return log

class DoubleHeadMLPclassifica(MLPclassifica):
    def __init__(self,
                 num_classes,
                 data_key,
                 dna_in_channels,
                 dna_out_channels,
                 ppi_in_channels,
                 ppi_out_channels,
                 weight_decay,
                 dropout=0.2,
                 over_sampling: str = None,
                 loss_weight: list=[1,1]
                 ):
        super(DoubleHeadMLPclassifica, self).__init__(
            num_classes,
            data_key,
            dna_out_channels+ppi_out_channels,
            weight_decay,
            dropout,
            over_sampling,
            loss_weight,
        )
        self.over_sampling = over_sampling
        self.num_classes = num_classes
        self.data_key = data_key
        self.weight_decay = weight_decay
        # if loss is not None:
        #     self.loss = initialize_from_config(loss)
        weight = torch.tensor(loss_weight, dtype=torch.float)
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)

        self.dna_input = nn.Sequential(
            nn.Linear(dna_in_channels,dna_out_channels),
        )

        self.ppi_input = nn.Sequential(
            nn.Linear(ppi_in_channels,ppi_out_channels),
        )

    def forward(self,dna_x,ppi_x):
        dna_x = dna_x.view(dna_x.shape[0],-1)
        ppi_x = ppi_x.view(ppi_x.shape[0],-1)

        dna_x = self.dna_input(dna_x)
        ppi_x = self.ppi_input(ppi_x)

        x = torch.cat([dna_x,ppi_x],dim=1)

        x = self.input(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        logits = self.classifica(x)

        return logits

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        dna_x = self.get_input(batch, 'dna_data')
        ppi_x = self.get_input(batch, 'ppi_data')
        y = batch['label']

        # if self.over_sampling == 'SOMTE':
        #     smote = SMOTE(random_state=42,k_neighbors=1)
        #     x, y = smote.fit_resample(x.cpu(), y.cpu())
        #     x = torch.from_numpy(x).to(self.device)
        #     y = torch.from_numpy(y).to(self.device)

        logits = self(dna_x,ppi_x)
        loss = self.loss(logits, y)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        dna_x = self.get_input(batch, 'dna_data')
        ppi_x = self.get_input(batch, 'ppi_data')
        y = batch['label']
        logits = self(dna_x,ppi_x)

        preds = nn.functional.sigmoid(logits).argmax(1)
        loss = self.loss(logits, y)
        output = {'y_true':y,'y_pred':preds,'loss_step':loss}
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return output

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        y_true = []
        y_pred = []

        for output in outputs:
            y_true.extend(output['y_true'].cpu().numpy().flatten())
            y_pred.extend(output['y_pred'].cpu().numpy().flatten())
        if len(np.unique(y_true)) < 2:
            # 处理只有一个类别的情况
            self.log("val/auc", 0.0, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        else:
            preci_score = precision_score(y_true, y_pred)
            acc_score = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            self.log("val/preci_score", preci_score, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log("val/acc_score", acc_score, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log("val/auc", auc, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log("val/recall", recall, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log("val/f1", f1, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
