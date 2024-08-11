import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm
import pytorch_lightning as pl
from seld_tcn_pytorch import SELDTCNModel
from torch.utils.data import Dataset, DataLoader
import numpy as np

class tcn_lightning(pl.LightningModule):
    
    def __init__(self,
                 tcn_seld,
                 metrics:dict,
                 loss_fn):
        super().__init__()
        self.tcn_seld = tcn_seld
        self.loss_fn = loss_fn
        self.metrics = metrics

    def forward(self,x):
        out = self.tcn_seld(out)
    
    def training_step(self,batch,batch_idx):
        loss,out,y = self.common_step(batch=batch,batch_idx=batch_idx)

        metrics_val = {'train_loss':loss}
        
        for key in self.metrics:
            metrics_val[key] = self.metrics[key](out,y)

        self.log_dict(metrics_val,on_step=True,on_epoch=False,prog_bar=True)

        return {'loss':loss,'scores':out,'y': y}
    
    def validation_step(self,batch,batch_idx):

        loss,out,y = self.common_step(batch=batch,batch_idx=batch_idx)

        metrics_val = {'val_loss':loss}
        
        for key in self.metrics:
            metrics_val[key] = self.metrics[key](out,y)

        self.log_dict(metrics_val,on_step=True,on_epoch=False,prog_bar=True)

        return {'loss':loss,'scores':out,'y': y}

    def common_step(self,batch,batch_idx):
        x,y = batch
        out = self.forward(x)
        loss = self.loss_fn(out,y)
        return loss, out, y
    
    def configure_optimizers(self):
        return optim.Adam(self.tcn_seld.parameters(),lr = 1e-3)

class CustomDataset(Dataset):
    def __init__(self, n_samples=1000, n_features=48000, n_classes=2, std=1.0):
        self.data = []
        self.labels = []
        
        # Generating clusters
        for i in range(n_classes):
            mean = torch.randn(n_features) * 10  # Different mean for each class
            std = np.full(mean.shape,std)
            std = torch.Tensor(std)

            for j in range(n_samples//n_classes):
                cluster_data = torch.normal(mean=mean, std=std)
                self.data.append(cluster_data)
                
            self.labels.append(torch.full((n_samples // n_classes,), i, dtype=torch.long))
        
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        
        self.train_dataset = CustomDataset(n_samples=10000)
        self.val_dataset = CustomDataset(n_samples=2000)
        self.test_dataset = CustomDataset(n_samples=2000)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# Usage
if __name__ == "__main__":
    data_module = CustomDataModule(batch_size=32)
    data_module.setup()
    
    for batch in data_module.train_dataloader():
        x, y = batch
        print(f"Batch shape: {x.shape}, Labels: {y}")
        break
