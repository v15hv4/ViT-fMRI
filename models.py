import torch
import torch.optim as optim
import pytorch_lightning as pl

import numpy as np

from sklearn.metrics import accuracy_score, f1_score

from transformers import VideoMAEConfig, VideoMAEForVideoClassification

# VideoMAE model for fMRI classification
class fMRIClassifier(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-3, batch_size=8):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size

        # instantiate model
        config = VideoMAEConfig(
            image_size=64,
            num_channels=3,
            num_frames=50,
            num_labels=num_classes,
            problem_type="single_label_classification",
        )
        self.model = VideoMAEForVideoClassification(config)

    def forward(self, x):
        return self.model(**x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        fmris, labels = batch

        batch_fmris = torch.stack([f.permute(1, 0, 2, 3) for f in fmris])
        batch_labels = torch.stack(labels)

        outputs = self({
            "pixel_values": batch_fmris,
            "labels": batch_labels,
        })
        loss = outputs.loss

        self.log("train_loss", loss.item(), sync_dist=True, batch_size=self.batch_size)

        return { "loss": loss }

    def validation_step(self, batch, batch_idx):
        fmris, labels = batch

        batch_fmris = torch.stack([f.permute(1, 0, 2, 3) for f in fmris])
        batch_labels = torch.stack(labels)

        outputs = self({
            "pixel_values": batch_fmris,
            "labels": batch_labels,
        })
        preds = np.argmax(outputs.logits.detach().cpu(), axis=-1)

        # TODO: add more metrics; 2v2 cosine dist etc.
        acc = accuracy_score(batch_labels.cpu(), preds)
        f1 = f1_score(batch_labels.cpu(), preds, average="macro")

        self.log("val_acc", acc, sync_dist=True, batch_size=self.batch_size)
        self.log("val_f1", f1, sync_dist=True, batch_size=self.batch_size)

        return { "val_acc": acc, "val_f1": f1 }

    def predict_step(self, batch, batch_idx):
        fmris, _ = batch

        batch_fmris = torch.stack([f.permute(1, 0, 2, 3) for f in fmris])

        outputs = self({
            "pixel_values": batch_fmris,
        })
        preds = np.argmax(outputs.logits.detach().cpu(), axis=-1)

        return preds
