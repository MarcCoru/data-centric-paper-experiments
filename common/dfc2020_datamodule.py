import os
import rasterio
from glob import glob
from tqdm import tqdm

import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from common.transforms import get_classification_transform


class DFC2020Dataset(Dataset):
    def __init__(self, root, prefix, transform):
        super(DFC2020Dataset, self).__init__()

        self.transform = transform

        tifs = glob(os.path.join(root, "lc_" + prefix, "*.tif"))

        self.paths = []
        for t in tqdm(tifs):
            self.paths.append((t, t.replace('lc', 's1'), t.replace('lc', 's2')))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        lc_path, s1_path, s2_path = self.paths[index][0], self.paths[index][1], self.paths[index][2]

        with rasterio.open(s1_path, "r") as src:
            s1 = src.read()
        with rasterio.open(s2_path, "r") as src:
            s2 = src.read()
        with rasterio.open(lc_path, "r") as src:
            lc = src.read(1)

        input, target = self.transform(s1, s2, lc)

        return input, target, lc_path


class DFC2020DataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=32, workers=8):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.workers = workers

    def setup(self):
        self.train_ds = DFC2020Dataset(self.root, "validation", get_classification_transform(augment=True))
        self.valid_ds = DFC2020Dataset(self.root, "validation", get_classification_transform(augment=False))
        self.test_ds = DFC2020Dataset(self.root, "0", get_classification_transform(augment=False))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.workers,
                          shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.workers,
                          shuffle=False, drop_last=False)
