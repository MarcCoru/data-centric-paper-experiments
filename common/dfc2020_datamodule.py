import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from common.transforms import get_classification_transform
from common.dfc2020 import DFCDataset

class DFC2020DataModule(pl.LightningDataModule):
    def __init__(self, root="datasets", batch_size=32, workers=0):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.workers = workers

    def setup(self, stage=None):
        self.train_ds = DFCDataset(self.root, "train", augment=True)
        self.valid_ds = DFCDataset(self.root, "val", augment=False)
        self.test_ds = DFCDataset(self.root, "test", augment=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.workers,
                          shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.workers,
                          shuffle=False, drop_last=False)
