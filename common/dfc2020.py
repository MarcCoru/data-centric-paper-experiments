import os
import glob
import numpy as np
import pandas as pd
import json

import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import torch
from torch.utils.data import Dataset

classes = ["Forest", "Shrubland", "Grassland", "Wetlands", "Croplands",
           "Urban/Built-up", "Barren", "Water"]

def augmentation(input):
    if np.random.rand() < 0.5:
        input = torch.fliplr(input)

    # horizontal flip
    if np.random.rand() < 0.5:
        input = torch.flipud(input)

    # rotate
    n_rotations = np.random.choice([0, 1, 2, 3])
    input = np.rot90(input, k=n_rotations, axes=(1, 2)).copy()
    return input

def load_s2_10Bands(loc):
    """ loads all the 10m and 20m GSD
    bands of the s2 image.
    https://docs.google.com/spreadsheets/d/1DASiXTg0gC4EGDa-FLIOVXpuVMcWSoesC4Vu0-C9wkk/edit?usp=sharing
    """
    with rasterio.open(loc,"r") as src:
        data = src.read((2,3,4,5,6,7,8,9,12,13))
    return data

def load_lc_data(loc):
    with rasterio.open(loc,"r") as src:
        data = src.read(1)
    return  data 

def preprocess_s2(x):
    return np.clip(x/10000,0,1)

class sen12msDFC_labelTransform:

    def __init__(self):
        # only classes that do exist in sen12ms DFC dataset are
        # 1, 2, 4, 5, 6, 7, 9, 10
        # so map them from zero to seven
        #
        #                       0     1  2     3    4  5  6  7    8     9  10
        self.lut = np.array( [np.nan, 0, 1, np.nan, 2, 3, 4, 5, np.nan, 6, 7] )

    def __call__(self, x):
        return self.lut[x]

class DFCDataset(Dataset):

    def __init__(self,
                 dataset_directory,
                 train_val_test,
                 augment=False):

        self.dataset_directory = dataset_directory
        self.train_val_test = train_val_test

        if augment:
            self.transforms = augmentation
        else:
            self.transforms = None
        
        # load the file with the realive locations of all data
        with open("datasets/locations_dfc.json", "r") as f:
            self.relative_locations_all = json.load(f)

            # some checks
            assert len(set(self.relative_locations_all["train"]).intersection(set(self.relative_locations_all["test"]))) == 0
            assert len(set(self.relative_locations_all["train"]).intersection(set(self.relative_locations_all["val"]))) == 0
            assert len(set(self.relative_locations_all["val"]).intersection(set(self.relative_locations_all["test"]))) == 0

        # choose train val or test images
        self.relative_locations = self.relative_locations_all[self.train_val_test]

        # commented out, to shuffle once in the dataloader (if necessary)
        # np.random.shuffle(self.relative_locations)

        # look up table for label transform
        self.label_trans = sen12msDFC_labelTransform()

    def __getitem__(self, i):
    
        s2_loc = os.path.join(self.dataset_directory,
                              self.relative_locations[i])

        label_loc = os.path.join(self.dataset_directory,
                                 self.relative_locations[i].replace("s2_","dfc_"))

        assert os.path.isfile(s2_loc),f"image {s2_loc} not found"
        assert os.path.isfile(label_loc),f"image {label_loc} not found"

        # read s2 image from disk
        # only take the 10 bands with 10 or 20m GSD
        data_s2 = load_s2_10Bands(s2_loc).astype("float32")
        data_s2 = preprocess_s2(data_s2)
        data_s2 = torch.Tensor(data_s2)

        # label is the maximal occuring
        # landcover class
        # we tranform to 0-8 classifcation sceme
        # via self.label_trans
        landcover_data = load_lc_data(label_loc)
        label_non_transformed = np.argmax( np.bincount(landcover_data.flatten()) )
        label = self.label_trans(label_non_transformed)
        label = torch.Tensor([label]).long()

        if self.transforms is not None:
            data_s2 = self.transforms(data_s2)

        return data_s2, label, self.relative_locations[i]

    def __len__(self):
        return len(self.relative_locations)


class DFCDatasetBatchLoader:
    def __init__(self, root, train_scores):
        super().__init__()
        self.dataset_directory = root
        self.train_scores = train_scores

        # look up table for label transform
        self.label_trans = sen12msDFC_labelTransform()

        self.transforms = augmentation

    def get_batch(self, batch_indexes):
        images = torch.empty((len(batch_indexes), 10, 256, 256))
        targets = []

        for i, index in enumerate(batch_indexes):
            s2_loc = os.path.join(self.dataset_directory, self.train_scores[i][0])

            label_loc = os.path.join(self.dataset_directory,
                                     self.train_scores[i][0].replace("s2_", "dfc_"))

            assert os.path.isfile(s2_loc), f"image {s2_loc} not found"
            assert os.path.isfile(label_loc), f"image {label_loc} not found"

            # read s2 image from disk
            # only take the 10 bands with 10 or 20m GSD
            data_s2 = load_s2_10Bands(s2_loc).astype("float32")
            data_s2 = preprocess_s2(data_s2)
            data_s2 = torch.Tensor(data_s2)

            # label is the maximal occuring landcover class
            # we tranform to 0-8 classifcation sceme via self.label_trans
            landcover_data = load_lc_data(label_loc)
            label_non_transformed = np.argmax(np.bincount(landcover_data.flatten()))
            label = self.label_trans(label_non_transformed)
            # label = torch.Tensor([label]).long()

            if self.transforms is not None:
                data_s2 = self.transforms(data_s2)

            images[i] = torch.from_numpy(data_s2)
            targets.append(label)

        return images, torch.from_numpy(np.asarray(targets)).long()


if __name__ == "__main__":

    # make the location json file
    # to have a fix split.
    # dont run code if not necessary!!
    if False:

        dataset_directory = "/home/user/data/sen12msDFC/"
        random_state = np.random.RandomState(42)
        all_train_val_imgs = glob.glob(os.path.join(dataset_directory,"s2_0","**","*_s2_*.tif"),recursive=True)
        print(len(all_train_val_imgs))
        all_test_imgs = glob.glob(os.path.join(dataset_directory,"s2_validation","**","*_s2_*.tif"),recursive=True)
        print(len(all_test_imgs))

        train_locs = [x.replace(dataset_directory,"") for x in all_train_val_imgs[:4000]]
        val_locs = [x.replace(dataset_directory,"") for x in all_train_val_imgs[4000:]]
        test_locs = [x.replace(dataset_directory,"") for x in all_test_imgs]
    
        assert len(set(train_locs).intersection(set(val_locs))) == 0
        assert len(set(train_locs).intersection(set(test_locs))) == 0
        assert len(set(val_locs).intersection(set(test_locs))) == 0

        with open("datasets/locations_dfc.json", "w+") as dst:
            json.dump({"train":train_locs,"val":val_locs,"test":test_locs},dst)


    ds = Dataset_Sen12msDFC("/home/user/data/sen12msDFC",
                            "train")

    ds[0]

    pass