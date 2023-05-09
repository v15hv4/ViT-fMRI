import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data import GODData, GODDataModule
from models import fMRIClassifier

from argparse import ArgumentParser

# config
parser = ArgumentParser()
parser.add_argument("--subject", type=str)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--accelerator", type=str, default="auto")
parser.add_argument("--devices", type=str, default="auto")
parser.add_argument("--strategy", type=str, default="ddp_find_unused_parameters_false")
parser.add_argument("--wandb-project", type=str, default="fMRI-Classification-10")
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--ckpt-root", type=str, default="/scratch/vishva.saravanan/ViT-fMRI")
parser.add_argument("--val-freq", type=int, default=10)
args = parser.parse_args()

# instantiate data
rois = ["V1d", "V1v", "V2d", "V2v", "V3d", "V3v", "hV4", "LOC", "FFA", "PPA", "HVC"]
normalize = lambda x: x / x.max()
data = GODData(
    subject=args.subject, 
    transform=normalize,
    rois=rois,
)
data = GODDataModule(data, val_frac=0.2, batch_size=args.batch_size)

# instantiate model
if args.ckpt is not None:
    print(f"Loading checkpoint: {args.ckpt}")
    model = fMRIClassifier.load_from_checkpoint(args.ckpt)
else:
    model = fMRIClassifier(num_classes=10, lr=args.lr)

# use wandb logger
wandb_logger = WandbLogger(project=args.wandb_project, config={"seed": args.seed})

# intantiate trainer
trainer = pl.Trainer(
    max_epochs=args.epochs,
    devices=args.devices,
    accelerator=args.accelerator,
    strategy=args.strategy,
    default_root_dir=args.ckpt_root,
    check_val_every_n_epoch=args.val_freq,
    logger=wandb_logger,
)

# train model
trainer.fit(model, data)
