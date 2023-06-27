import torch
import os
import pandas as pd
from common.data import trainregions, valregions, holdout_regions
import h5py
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from common.transforms import get_classification_transform

class AllSen12MSDataset(Dataset):
    def __init__(self, root, fold, transform, classes=None, seasons=None):
        super(AllSen12MSDataset, self).__init__()

        self.transform = transform

        self.h5file_path = os.path.join(root, "sen12ms.h5")
        index_file = os.path.join(root, "sen12ms.csv")
        self.paths = pd.read_csv(index_file, index_col=0)

        if fold == "train":
            regions = trainregions
        if fold == "trainval":
            regions = trainregions + valregions
        elif fold == "val":
            regions = valregions
        elif fold == "test":
            regions = holdout_regions
        elif fold == "all":
            regions = holdout_regions + valregions + trainregions
        else:
            raise AttributeError("one of meta_train, meta_val, meta_test must be true or "
                                 "fold must be in 'train','val','test'")

        mask = self.paths.region.isin(regions)
        print(f"fold {fold} specified. Keeping {mask.sum()} of {len(mask)} tiles")
        self.paths = self.paths.loc[mask]
        if classes is not None:
            mask = self.paths.maxclass.isin(classes)
            print(f"classes {classes} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]
        if seasons is not None:
            mask = self.paths.season.isin(seasons)
            print(f"seasons {seasons} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]

        # shuffle the tiles once
        self.paths = self.paths.sample(frac=1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths.iloc[index]

        with h5py.File(self.h5file_path, 'r') as data:
            s2 = data[path.h5path + "/s2"][()]
            s1 = data[path.h5path + "/s1"][()]
            label = data[path.h5path + "/lc"][()]

        image, target = self.transform(s1, s2, label)

        return image, target, path.h5path

class RegionSen12MSDataset(torch.utils.data.Dataset):
    def __init__(self, root, region, fold, transform, classes=None, seasons=None, train_test_ratio=0.75, random_seed=0):
        super(RegionSen12MSDataset, self).__init__()
        assert fold in ["train", "test"], "splitting tiles o region randomly. only train or tet folds are allowed"
        assert type(region) == int, "region must be specified as int according to the regions in data.py"

        self.transform = transform

        self.h5file_path = os.path.join(root, "sen12ms.h5")
        index_file = os.path.join(root, "sen12ms.csv")
        self.paths = pd.read_csv(index_file, index_col=0)

        if region in trainregions:
            group = "training regions"
        if region in valregions:
            group = "validation regions"
        if region in holdout_regions:
            group = "hold-out regions"

        mask = self.paths.region == region
        print(f"region {region} specified ({group}). Keeping {mask.sum()} of {len(mask)} tiles")
        self.paths = self.paths.loc[mask]
        if classes is not None:
            mask = self.paths.maxclass.isin(classes)
            print(f"classes {classes} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]
        if seasons is not None:
            mask = self.paths.season.isin(seasons)
            print(f"seasons {seasons} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]

        # fix random state with seed to ensure same mask if invoked with fold==train or fold==test
        mask = np.random.RandomState(random_seed).rand(len(self.paths)) < train_test_ratio
        if fold == "test":
            # invert mask
            mask = ~mask

        self.paths = self.paths[mask]
        print(f"fold {fold} specified. Keeping {mask.sum()} of {len(mask)} tiles")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths.iloc[index]

        with h5py.File(self.h5file_path, 'r') as data:
            s2 = data[path.h5path + "/s2"][()]
            s1 = data[path.h5path + "/s1"][()]
            label = data[path.h5path + "/lc"][()]

        image, target = self.transform(s1, s2, label)

        return image, target, path.h5path

class Sen12MSDataModule(pl.LightningDataModule):
    def __init__(self, root="/data/sen12ms", batch_size=32, workers=8, split="random"):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.workers = workers

        self.split = split

    def setup(self, stage: str):
        if self.split == "regionwise":
            self.train_ds = AllSen12MSDataset(self.root, "train", get_classification_transform(augment=True))
            self.valid_ds = AllSen12MSDataset(self.root, "val", get_classification_transform(augment=False))
        elif self.split == "random":
            dataset = AllSen12MSDataset(self.root, "trainval", get_classification_transform(augment=True))
            self.train_ds, self.valid_ds = torch.utils.data.random_split(dataset, [0.8,0.2])

        self.test_ds = AllSen12MSDataset(self.root, "test", get_classification_transform(augment=False))
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.evalu_ds, batch_size=self.batch_size, shuffle=False)
