import glob
import numpy as np

from torch.utils.data import Dataset

# dataset class for the GenericObjectDecoding dataset
class GODData(Dataset):
    DATA_PATH = "data/processed"

    def __init__( self, subject="01", session_id="01", task="perception", train=True, transform=None):
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

        # apply transforms if necessary
        if self.transform:
            fmri = self.transform(fmri)
            category = self.transform(category)

        return fmri, category