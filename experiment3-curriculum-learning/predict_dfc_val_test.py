import os
import sys
import argparse
import datetime

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             cohen_kappa_score, jaccard_score, balanced_accuracy_score)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from common.model import ResNet18
from common.dfc2020_datamodule import DFC2020DataModule


@torch.no_grad()
def test(dfc_root_folder, test_or_val, weights):
    # get test dataloader
    datamodule = DFC2020DataModule(dfc_root_folder, batch_size=256, workers=8)
    datamodule.setup('train')
    if test_or_val == 'val' or test_or_val == 'validation':
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()

    # Load model
    model = ResNet18(in_channels=10, num_classes=8)
    try:
        model.load_state_dict(torch.load(weights)['state_dict'])
    except KeyError:
        model.load_state_dict(torch.load(weights))
    model.cuda()
    model.eval()

    all_targets = []
    all_preds = []
    first = True
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        images, targets = batch[0], batch[1]
        images = images.float().cuda()
        preds = model(images)

        if first:
            all_targets = np.squeeze(targets.cpu().numpy())
            all_preds = preds.cpu().numpy()
            first = False
        else:
            all_targets = np.concatenate((all_targets, np.squeeze(targets.cpu().numpy())))
            all_preds = np.concatenate((all_preds, preds.cpu().numpy()))

    print('before', all_targets.shape, all_preds.shape, all_preds[0:2, :])
    all_preds = np.argmax(all_preds, axis=1)
    print('after', all_targets.shape, all_preds.shape, all_preds[0:2])
    print('bc', np.bincount(all_targets), np.bincount(all_preds))

    acc = accuracy_score(all_targets, all_preds)
    b_acc = balanced_accuracy_score(all_targets, all_preds)
    conf_m = confusion_matrix(all_targets, all_preds)
    f1_s_w = f1_score(all_targets, all_preds, average='macro')
    kappa = cohen_kappa_score(all_targets, all_preds)
    jaccard = jaccard_score(all_targets, all_preds, average='macro')

    print("---- Test -- Time " + str(datetime.datetime.now().time()) +
          " Overall Accuracy= " + "{:.4f}".format(acc) +
          " Balanced Accuracy= " + "{:.4f}".format(b_acc) +
          " F1 score weighted= " + "{:.4f}".format(f1_s_w) +
          " Kappa= " + "{:.4f}".format(kappa) +
          " Jaccard= " + "{:.4f}".format(jaccard) +
          " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
          )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfc_root_folder", type=str, required=False, default='/home/kno/datasets/df2020/',
                        help="Path to the DFC2020 dataset")
    parser.add_argument("--val_or_test", type=str, required=False, default='test', help="test or val")
    parser.add_argument("--weights", type=str, required=False,
                        default="weights/RN18-epochepoch=123-val_lossval_loss=0.23.ckpt",
                        help="Path to the pre-trained weights")

    args = parser.parse_args()
    test(args.dfc_root_folder, args.val_or_test, args.weights)
