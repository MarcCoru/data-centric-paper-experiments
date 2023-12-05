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

    # much simpler debugging features
    #features = rgb.mean(axis=(-1,-2))

    # subsample
    msk = np.random.rand(features.shape[0]) > 0.75
    features, ids, rgb = features[msk], ids[msk], rgb[msk]
    def plot_sample(rgb, coeffs, idx):
        plt.figure()
        plt.imshow(rgb[idx].transpose(1,2,0))
        plt.title(f"{ids[idx]}-{coeffs[idx]}")
        plt.show()

    season, region, classname, tile = list(zip(*[id.split("/") for id in ids]))
    lon, lat = list(zip(*[regionlonlat[int(r)] for r in region]))
    lon, lat = np.array(lon), np.array(lat)
    # jitter
    #lon, lat = lon + np.random.rand(lon.shape[0]) * 15, lat + np.random.rand(lon.shape[0]) * 15

    print(np.unique(region))

    if True:
        ## Filter by class
        mask = np.stack([c == "Urban_Build-up" for c in classname])
        features = features[mask]
        ids = ids[mask]
        season, region, classname = np.array(season)[mask], np.array(region)[mask], np.array(classname)[mask]
        lon, lat = lon[mask], lat[mask]
        rgb = rgb[mask]

    region_mask = np.stack([r == "109" for r in region])

    Z = features[~region_mask]
    X = features[region_mask]

    coeffs = kernel_mean_matching(X, Z, kern='rbf', B=5, sigma=2)
    #coeffs = kernel_mean_matching(X, Z, kern='lin', B=3)


    plot_tsne(features, coeffs, region_mask)
    plt.show()

    # Histogram
    plt.figure()
    plt.hist(coeffs)

    fig, ax = plt.subplots()
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, color="gray")
    ax.scatter(lon[~region_mask], lat[~region_mask], c=coeffs*5, s=coeffs*2 + 1, cmap="Reds") # , edgecolor="white")
    ax.scatter(lon[region_mask], lat[region_mask], s=10)

    plt.show()

    idxs = np.argsort(coeffs[:,0])

    fig, axs = plt.subplots(3, 8, figsize=(8 * 3, 3 * 3))

    ## Base classes
    for ax, img, id in zip(axs[0], rgb[region_mask], ids[region_mask]):
        ax.imshow(img.transpose(1, 2, 0))
        ax.set_title(str(id))
        ax.axis("off")

    ## Most similar
    idxs_ = idxs[-8:]
    for ax, img, c, id in zip(axs[1], rgb[idxs_], coeffs[idxs_], ids[idxs_]):
        ax.imshow(img.transpose(1, 2, 0))
        ax.set_title(str(id) + f" ({float(c):.2f})")
        ax.axis("off")

    ## Least similar
    idxs_ = idxs[:8]
    for ax, img, c, id in zip(axs[2], rgb[idxs_], coeffs[idxs_], ids[idxs_]):
        ax.imshow(img.transpose(1, 2, 0))
        ax.set_title(str(id) + f" ({float(c):.2f})")
        ax.axis("off")

    plt.show()



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

def plot_tsne(features, coeffs, mask):
    features_pca = PCA(n_components=2).fit_transform(features)

    plt.figure()
    plt.scatter(features_pca[~mask, 0], features_pca[~mask, 1], s=coeffs*10 + 1, cmap="Reds")
    plt.scatter(features_pca[mask, 0], features_pca[mask, 1], s=50)
    plt.title("coeffs")



if __name__ == '__main__':
    main()