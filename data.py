import glob
import numpy as np
import nibabel as nib

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

# dataset class for the GenericObjectDecoding dataset
class GODData(Dataset):
    DATA_PATH = "data/10classes"

    def __init__(self, subject="01", transform=None, rois=[]):
        self.subject = subject
        # self.session = f"{task}{'Training' if train else 'Test'}{session_id}"
        self.length = len(glob.glob(f"{self.DATA_PATH}/sub-{subject}/dataset/fmris/*"))
        self.transform = transform

        # construct ROI mask
        roi_mask = np.zeros((3, 50, 64, 64))
        for roi in rois:
            roi_mask_LH = nib.load(f"{self.DATA_PATH}/sub-{subject}/masks/sub-{subject}_mask_LH_{roi}.nii.gz").get_fdata()
            roi_mask_LH = roi_mask_LH.transpose(2, 1, 0)
            roi_mask_LH = np.vstack([roi_mask_LH[np.newaxis, ...]] * 3)

            roi_mask_RH = nib.load(f"{self.DATA_PATH}/sub-{subject}/masks/sub-{subject}_mask_RH_{roi}.nii.gz").get_fdata()
            roi_mask_RH = roi_mask_RH.transpose(2, 1, 0)
            roi_mask_RH = np.vstack([roi_mask_RH[np.newaxis, ...]] * 3)

            roi_mask += roi_mask_LH
            roi_mask += roi_mask_RH

        # if ROI list is empty, use all voxels
        if len(rois) == 0:
            roi_mask = np.ones((3, 50, 64, 64))

        self.roi_mask = roi_mask.astype(bool)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # load data
        fmri = np.load(f"{self.DATA_PATH}/sub-{self.subject}/dataset/fmris/{idx}.npy")
        category = np.load(f"{self.DATA_PATH}/sub-{self.subject}/dataset/categories/{idx}.npy")

        # apply ROI mask to fMRI data
        fmri = np.where(self.roi_mask, fmri, 0)

        # convert to tensor
        fmri = torch.from_numpy(fmri).float()
        category = torch.from_numpy(category).long() - 1 # 0-indexed categories

        # apply transforms if necessary
        if self.transform:
            fmri = self.transform(fmri)

        return fmri, category

# lightning data module for the GenericObjectDecoding dataset
class GODDataModule(pl.LightningDataModule):
    def __init__(self, full_data, val_frac=0.2, batch_size=16):
        super().__init__()

        self.full_data = full_data
        self.val_frac = val_frac
        self.batch_size = batch_size

    def setup(self, stage=""):
        full_size = len(self.full_data)
        val_size = int(self.val_frac * full_size)
        train_size = full_size - val_size

        self.train_data, self.val_data = random_split(self.full_data, [train_size, val_size])

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)