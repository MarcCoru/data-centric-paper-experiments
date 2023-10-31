"""
This script trains a baseline model on the Sen12MS dataset and tests on DFC2020
"""

from common.dfc2020_datamodule import DFC2020DataModule
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from model import ResNet18

datamodule = DFC2020DataModule(root="datasets", batch_size=512, workers=2)

model = ResNet18(in_channels=10, num_classes=8)

wandb_logger = WandbLogger(project="datacentric-paper")

callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=10),
    ModelCheckpoint(
        dirpath='weights',
        monitor='val_loss',
        filename='RN18-epoch{epoch:02d}-val_loss{val_loss:.2f}',
        save_last=True
    )
]

trainer = pl.Trainer(
    max_epochs=100,
    log_every_n_steps=5,
    fast_dev_run=False,
    callbacks=callbacks,
    logger=wandb_logger)

trainer.fit(model=model, datamodule=datamodule)
