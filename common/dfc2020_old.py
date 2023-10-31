import os

import numpy as np
import rasterio
from glob import glob
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset

class DFCDataset(Dataset):
    def __init__(self, dfcpath, split="train"):
        super(DFCDataset, self).__init__()
        self.dfcpath = dfcpath

        if split == "train":
            dfcpath = os.path.join(dfcpath, "dfc_0")
        elif split == "test":
            dfcpath = os.path.join(dfcpath, "dfc_validation")

        indexfile = os.path.join(dfcpath, "index.csv")

        if os.path.exists(indexfile) and False:
            print(f"loading {indexfile}")
            index = pd.read_csv(indexfile)
        else:
            tifs = glob(os.path.join(dfcpath, "*.tif"))

            index_dict = []
            for t in tqdm(tifs):
                basename = os.path.basename(t)
                path = t.replace(dfcpath, "")

                # remove leading slash if exists
                path = path[1:] if path.startswith("/") else path

                seed, season, type, region, tile = basename.split("_")

                with rasterio.open(os.path.join(dfcpath, path), "r") as src:
                    arr = src.read()

                classes, counts = np.unique(arr, return_counts=True)

                maxclass = classes[counts.argmax()]

                N_pix = len(arr.reshape(-1))
                counts_ratio = counts / N_pix

                # multiclass labelled with at least 10% of occurance following Schmitt and Wu. 2021
                multi_class = classes[counts_ratio > 0.1]
                multi_class_fractions = counts_ratio[counts_ratio > 0.1]

                s2path = os.path.join(f"{seed}_{season}", f"s2_{region}", basename.replace("dfc", "s2"))
                assert os.path.exists(os.path.join(dfcpath, s2path))

                lcpath = os.path.join(f"{seed}_{season}", f"lc_{region}", basename.replace("dfc", "lc"))
                assert os.path.exists(os.path.join(dfcpath, lcpath))

                index_dict.append(
                    dict(
                        basename=basename,
                        dfcpath=path,
                        seed=seed,
                        season=season,
                        region=region,
                        tile=tile,
                        maxclass=maxclass,
                        multi_class=multi_class,
                        multi_class_fractions=multi_class_fractions,
                        s1path=s1path,
                        s2path=s2path,
                        lcpath=lcpath
                    )
                )
            index = pd.DataFrame(index_dict)
            print(f"saving {indexfile}")
            index.to_csv(indexfile)

        index = index.reset_index()
        self.index = index.set_index(["region", "season"])
        self.region_seasons = self.index.index.unique().tolist()
        print(self.region_seasons)
        self.index.sort_index()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        row = self.index.loc[self.index["index"] == item].iloc[0]

        with rasterio.open(os.path.join(self.dfcpath, row.s1path), "r") as src:
            s1 = src.read()

        with rasterio.open(os.path.join(self.dfcpath, row.s2path), "r") as src:
            s2 = src.read()

        with rasterio.open(os.path.join(self.dfcpath, row.lcpath), "r") as src:
            lc = src.read(1)

        input, target = self.transform(s1, s2, lc)

        return input, target

if __name__ == '__main__':
    print("test")
    dfcpath = "/Users/marc/projects/data-centric-paper-experiments/datasets"
    ds = DFCDataset(dfcpath=dfcpath, split="train")