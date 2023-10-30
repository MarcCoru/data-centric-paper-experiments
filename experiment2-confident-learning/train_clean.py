"""
This script trains a model on the clean Sen12MS dataset
"""
from common.sen12ms import AllSen12MSDataset
from common.sen12ms import Sen12MSDataModule
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from common.model import ResNet18
from label_noise import clean_dataset

clean_dataset(root="/../../experiment/sen12ms", path_to_ckpt='weights/RN18-epochepoch=17-val_lossval_loss=0.37.ckpt',
              model=ResNet18, dataset=AllSen12MSDataset, train=True, val=False, test=False)  # creates a cleaned csv file

datamodule = Sen12MSDataModule(root="/../../experiment/sen12ms_clean", batch_size=256, workers=10)

model = ResNet18(in_channels=15, num_classes=10)

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
