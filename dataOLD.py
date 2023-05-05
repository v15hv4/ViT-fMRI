import glob
import numpy as np
import pandas as pd
import nibabel as nib

import torch
from torch.utils.data import Dataset

# dataset class for the GenericObjectDecoding dataset
class GODData(Dataset):
    FEATURES_PATH = "data/ds001246/derivatives/preproc-spm/output"
    TARGETS_PATH = "data/ds001246"
    TRAIN_CATEGORIES_PATH = "data/ds001246/stimulus_ImageNetTraining.csv"
    TEST_CATEGORIES_PATH = "data/ds001246/stimulus_ImageNetTest.csv"

    def __init__(
            self, 
            subject="01", 
            session_id="01", 
            task="perception", 
            train=True,
            limit_size=None,
        ):
        session = f"{task}{'Training' if train else 'Test'}{session_id}"

        # load data
        feature_runs = sorted(glob.glob(f"{self.FEATURES_PATH}/sub-{subject}/ses-{session}/func/*"))
        target_runs = sorted(glob.glob(f"{self.TARGETS_PATH}/sub-{subject}/ses-{session}/func/*events*"))
        categories = pd.read_csv(self.TRAIN_CATEGORIES_PATH if train else self.TEST_CATEGORIES_PATH, sep="\t", header=None)

        # process features and targets
        features = []
        targets = []

        for f_run, t_run in zip(feature_runs, target_runs):
            features_run = nib.load(f_run).get_fdata()
            targets_run = pd.read_csv(t_run, sep="\t")

            # remove resting states
            features_run_pp = features_run[:, :, :, 8:-2]
            targets_run_pp = targets_run[targets_run["event_type"] != "rest"]

            # reshape features into (N, C, D, W, H)
            features_run_pp = features_run_pp.transpose(3, 2, 1, 0).reshape(-1, 3, 50, 64, 64)

            # extract category labels
            targets_run_pp = targets_run_pp.merge(categories, left_on="stim_id", right_on=1)[2]
            targets_run_pp = targets_run_pp.to_numpy().reshape(-1, 1)

            features.append(features_run_pp)
            targets.append(targets_run_pp)

        features = np.vstack(features)
        targets = np.vstack(targets)

        # convert and store as tensors
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).long() - 1

        # flatten targets
        self.targets = self.targets.squeeze()

        # limit dataset size
        if limit_size is not None:
            self.features = self.features[:limit_size]
            self.targets = self.targets[:limit_size]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        target = self.targets[index]
        return feature, target