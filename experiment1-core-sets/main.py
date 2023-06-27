from common.model import ResNet18
from common.sen12ms import Sen12MSDataModule
from common.data import regionlonlat
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import geopandas as gpd
from skimage.exposure import equalize_hist

import math
from cvxopt import matrix, solvers

features_file = "features.npz"
checkpoint = "weights/RN18-epochepoch=15-val_lossval_loss=0.43.ckpt"
dataroot = "/data/sen12ms"

def main():
    features, ids, rgb = extract_features()

    # subsample
    msk = np.random.rand(features.shape[0]) > 0.9
    features, ids = features[msk], ids[msk]

    season, region, classname, tile = list(zip(*[id.split("/") for id in ids]))
    lon, lat = list(zip(*[regionlonlat[int(r)] for r in region]))
    lon, lat = np.array(lon), np.array(lat)
    # jitter
    lon, lat = lon + np.random.rand(lon.shape[0]) * 15, lat + np.random.rand(lon.shape[0]) * 15

    print(np.unique(region))

    #mask = np.stack([c == "Croplands" for c in classname])
    #features = features[mask]
    #season, region, classname = np.array(season)[mask], np.array(region)[mask], np.array(classname)[mask]
    #lon, lat = lon[mask], lat[mask]

    mask = np.stack([r == "4" for r in region])
    features_pca = TSNE(n_components=2).fit_transform(features)

    def to_idx(arr):
        uniques = list(np.unique(arr))
        return [uniques.index(s) for s in arr]


    plt.figure()
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=mask, cmap="Spectral")
    plt.title("region")

    plt.show()

    Z = features[~mask]
    X = features[mask]

    coeffs = kernel_mean_matching(X, Z, kern='rbf', B=10, sigma=10)

    plt.figure()
    plt.scatter(features_pca[~mask, 0], features_pca[~mask, 1], s=coeffs*10 + 1)
    plt.scatter(features_pca[mask, 0], features_pca[mask, 1], s=50)
    plt.title("coeffs")

    plt.figure()
    plt.hist(coeffs)

    fig, ax = plt.subplots()
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, color="gray")
    ax.scatter(lon[~mask], lat[~mask], c=coeffs*5, s=coeffs*2 + 1, cmap="Reds") # , edgecolor="white")
    ax.scatter(lon[mask], lat[mask], s=10)

    plt.show()

    print()



# an implementation of Kernel Mean Matching
# referenres:
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
def kernel_mean_matching(X, Z, kern='lin', B=1.0, eps=None, sigma=1.0):
    nx = X.shape[0]
    nz = Z.shape[0]
    if eps == None:
        eps = B / math.sqrt(nz)
    if kern == 'lin':
        K = np.dot(Z, Z.T)
        kappa = np.sum(np.dot(Z, X.T) * float(nz) / float(nx), axis=1)
    elif kern == 'rbf':
        K = compute_rbf(Z, Z, sigma)
        kappa = np.sum(compute_rbf(Z, X, sigma), axis=1) * float(nz) / float(nx)
    else:
        raise ValueError('unknown kernel')

    K = matrix(K.astype("float"))
    kappa = matrix(kappa.astype("float"))
    G = matrix(np.r_[np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz * (1 + eps), nz * (eps - 1), B * np.ones((nz,)), np.zeros((nz,))])

    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef

def compute_rbf(X, Z, sigma=1.0):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i, :] = np.exp(-np.sum((vx - Z) ** 2, axis=1) / (2.0 * sigma))
    return K

def extract_features():
    """
    either recomputes features from a model and the dataset or loads cached features
    :return: features and ids
    """
    if os.path.exists(features_file):
        features, ids, rgb = np.load("features.npz").values()
    else:

        datamodule = Sen12MSDataModule(root=dataroot, batch_size=512, workers=32)
        datamodule.setup("regionwise")

        model = ResNet18.load_from_checkpoint(checkpoint)
        model.eval()
        # get features instead of class logits
        model.model.fc = nn.Identity()
        dl = datamodule.val_dataloader()

        features, ids, rgb = [], [], []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dl), total=len(dl)):
                X,Y,id = batch
                y_features = model(X.cuda().float())

                rgb.append(equalize_hist(X[:, np.array([4, 3, 2])].numpy()))

                features.append(y_features.cpu().detach().numpy())
                ids.append(id)

        features = np.vstack(features)
        ids = np.hstack(ids)
        rgb = np.vstack(rgb)

        np.savez("features.npz", features=features, ids=ids, rgb=rgb)

    return features, ids, rgb


if __name__ == '__main__':
    main()