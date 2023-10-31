import os
import sys
import argparse
import datetime

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, jaccard_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from common.dfc2020_old import DFCDataset
from common.model import ResNet18
from common.transforms import get_classification_transform


@torch.no_grad()
def test(dfc_root_folder, weights):
    # Load datamodule and dataloader
    datamodule = DFCDataset(dfc_root_folder, None, get_classification_transform(augment=False))
    dataloader = torch.utils.data.DataLoader(datamodule, batch_size=128, shuffle=False, num_workers=4, drop_last=False)

    # Load model
    model = ResNet18(in_channels=15, num_classes=10)
    model.load_state_dict(torch.load(weights)['state_dict'])
    model.cuda()
    model.eval()

    all_targets = []
    all_preds = []
    first = True
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        images, targets = batch[0], batch[1]
        images = images.float().cuda()
        preds = model(images)
        print(targets.shape, preds.shape)

        if first:
            all_targets = targets.cpu().numpy()
            all_preds = preds.cpu().numpy()
            first = False
        else:
            all_targets = np.concatenate((all_targets, targets.cpu().numpy()))
            all_preds = np.concatenate((all_preds, preds.cpu().numpy()))

    print('before', all_targets.shape, all_preds.shape, all_preds[0:2, :])
    all_preds = np.argmax(all_preds, axis=1)
    print('after', all_targets.shape,  all_preds.shape, all_preds[0:2])

    acc = accuracy_score(all_targets, all_preds)
    conf_m = confusion_matrix(all_targets, all_preds)
    f1_s_w = f1_score(all_targets, all_preds, average='micro')
    kappa = cohen_kappa_score(all_targets, all_preds)
    jaccard = jaccard_score(all_targets, all_preds, average='micro')

    print("---- Test -- Time " + str(datetime.datetime.now().time()) +
          " Overall Accuracy= " + "{:.4f}".format(acc) +
          " F1 score weighted= " + "{:.4f}".format(f1_s_w) +
          " Kappa= " + "{:.4f}".format(kappa) +
          " Jaccard= " + "{:.4f}".format(jaccard) +
          " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
          )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfc_root_folder", type=str,
                        default="/home/kno/datasets/df2020/DFC_Public_Dataset/DFC_Public_Dataset/",
                        required=False, help="Path to the DFC2020 dataset")
    parser.add_argument("--weights", type=str, required=False,
                        default="../weights/RN18-epochepoch=15-val_lossval_loss=0.43.ckpt",
                        help="Path to the pre-trained weights")

    args = parser.parse_args()
    test(args.dfc_root_folder, args.weights)
