"""
This script trains a baseline model on the Sen12MS dataset and tests on DFC2020
"""

from common.dfc2020_datamodule import DFC2020DataModule
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from model import ResNet18
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='A simple script with argparse')

    # Add optional arguments
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--dataset-root', type=str, default="datasets")
    parser.add_argument('--accelerator', type=str, default="auto")
    parser.add_argument('--fast-dev-run', action='store_true')

    # Parse command-line arguments
    args = parser.parse_args()
    return args
def main(args):
    datamodule = DFC2020DataModule(root=args.dataset_root, batch_size=args.batchsize, workers=args.num_workers)

    model = ResNet18(in_channels=10, num_classes=8)

    wandb_logger = WandbLogger(project="datacentric-paper")

    callbacks = [
        ModelCheckpoint(
            dirpath='weights',
            monitor='val_loss',
            filename='RN18-epoch{epoch:02d}-val_loss{val_loss:.2f}',
            save_last=True
        )
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        log_every_n_steps=5,
        fast_dev_run=args.fast_dev_run,
        accelerator=args.accelerator,
        callbacks=callbacks,
        logger=wandb_logger)

    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    args = parse_args()
    main(args)