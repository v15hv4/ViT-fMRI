import glob
import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# dataset class for the GenericObjectDecoding dataset
class GODData(Dataset):
    DATA_PATH = "data/processed"

    def __init__(self, subject="01", session_id="01", task="perception", train=True, transform=None):
        self.subject = subject
        self.session = f"{task}{'Training' if train else 'Test'}{session_id}"
        self.length = len(glob.glob(f"{self.DATA_PATH}/sub-{subject}/ses-{self.session}/fmris/*"))
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # load data
        fmri = np.load(f"{self.DATA_PATH}/sub-{self.subject}/ses-{self.session}/fmris/{idx}.npy")
        category = np.load(f"{self.DATA_PATH}/sub-{self.subject}/ses-{self.session}/categories/{idx}.npy")

        # convert to tensor
        fmri = torch.from_numpy(fmri).float()
        category = torch.from_numpy(category).long() - 1 # 0-indexed categories

        # apply transforms if necessary
        if self.transform:
            fmri = self.transform(fmri)

        return fmri, category

# lightning data module for the GenericObjectDecoding dataset
class GODDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, batch_size=8):
        super().__init__()

        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)