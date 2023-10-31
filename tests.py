import unittest

from common.dfc2020 import DFCDataset
from common.dfc2020_datamodule import DFC2020DataModule
import torch

class TestDFCDataset(unittest.TestCase):
    def test_dataset(self):
        train_ds = DFCDataset("datasets",
                                "train")

        val_ds = DFCDataset("datasets",
                                      "val")

        test_ds = DFCDataset("datasets",
                                      "test")

        self.assertEqual(len(train_ds), 4000)  # add assertion here
        self.assertEqual(len(val_ds), 1128)  # add assertion here
        self.assertEqual(len(test_ds), 986)  # add assertion here

        X, y = train_ds[0]
        self.assertEqual(X.shape, torch.Size([10, 256, 256]))



class TestDFCDataModule(unittest.TestCase):

    def test_dataloader(self):
        dm = DFC2020DataModule(batch_size = 32)
        dm.setup()
        train_dl = dm.train_dataloader()
        for X,y in train_dl:
            self.assertEqual(X.shape, torch.Size([32, 10, 256, 256]))
            break


if __name__ == '__main__':
    unittest.main()
