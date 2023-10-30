from torch.utils.data import DataLoader
import lightning as pl
import torch
from skorch import NeuralNetClassifier
from cleanlab.classification import CleanLearning
from common.transforms import get_classification_transform
import pandas as pd
import os


def identify_noisy_labels(path_to_ckpt, model, dataset, batch_size=64, num_workers=10):

    # load trained model
    model = model.load_from_checkpoint(path_to_ckpt)

    # get all predictions, labels, and corresponding paths
    trainer = pl.Trainer()
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    output = trainer.predict(model, data_loader)

    preds = torch.cat([row['logits'] for row in output])
    preds = torch.nn.functional.softmax(preds, dim=-1).numpy()
    labels = torch.cat([row['labels'] for row in output]).numpy()
    paths = [item for row in output for item in row['paths']]

    # identify label issues
    model_skorch = NeuralNetClassifier(model)
    issues = CleanLearning(model_skorch).find_label_issues(labels=labels, pred_probs=preds)

    # return paths for possibly noisy samples
    issues['path'] = paths
    noisy_samples = issues.loc[issues['is_label_issue'] == True]

    return noisy_samples['path'].tolist()


def clean_dataset(root, path_to_ckpt, model, dataset, train, val, test):
    noisy_samples = []
    if train:
        train_dataset = dataset(root=root, fold='train', transform=get_classification_transform(augment=False))
        noisy_train_samples = identify_noisy_labels(path_to_ckpt=path_to_ckpt, model=model, dataset=train_dataset)
        noisy_samples += noisy_train_samples
        print(f"identified {len(noisy_train_samples)} potentially noisy samples in Train set.")
    if val:
        val_dataset = dataset(root=root, fold='val', transform=get_classification_transform(augment=False))
        noisy_val_samples = identify_noisy_labels(path_to_ckpt=path_to_ckpt, model=model, dataset=val_dataset)
        noisy_samples += noisy_val_samples
        print(f"identified {len(noisy_val_samples)} potentially noisy samples in Val set.")
    if test:
        test_dataset = dataset(root=root, fold='test', transform=get_classification_transform(augment=False))
        noisy_test_samples = identify_noisy_labels(path_to_ckpt=path_to_ckpt, model=model, dataset=test_dataset)
        noisy_samples += noisy_test_samples
        print(f"identified {len(noisy_test_samples)} potentially noisy samples in Test set.")

    paths = pd.read_csv(os.path.join(root, "sen12ms.csv"), index_col=0)
    paths_clean = paths.drop(paths[paths.h5path.isin(noisy_samples)].index.tolist())
    assert len(paths_clean) == len(paths) - len(noisy_samples), "noisy samples were not removed correctly"

    new_root = root + '_clean'
    os.makedirs(new_root, exist_ok=True)
    os.symlink(os.path.join(root, "sen12ms.h5"), os.path.join(new_root, "sen12ms.h5"))
    paths_clean.to_csv(os.path.join(new_root, "sen12ms.csv"))
