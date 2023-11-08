from common.model import ResNet18
from common.dfc2020_datamodule import DFC2020DataModule
from common.dfc_region_dataset import DFCRegionDataset, regions, bands
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
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import adapt

papercolors = {
    'prussianblue': '#04233A',
    'pennred': '#990000',
    'lapislazuli': '#3F6493',
    'sage': '#A4B494',
    'earthyellow': '#FFA552',
    'hookersgreen': '#59827A',
    'pearl': '#E0DCC2',
    'slategray': '#9AB5C4',
    'bittersweetshimmer': '#CC444B',
    'coral': '#FC7A57',
    'lightcoral': '#F28482',
    'burntsienna': '#EE6C4D',
    'skymagenta': '#B37BA4',
}

colors = {
    'KippaRing': '#E0DCC2',       # Pearl
    'MexicoCity': '#59827A',      # Hookers Green
    'CapeTown': '#CC444B',        # Bittersweet Shimmer
    'BandarAnzali': '#9AB5C4',    # Slate Gray
    'Mumbai': '#F28482',          # Light Coral
    'BlackForest': '#04233A',     # prussianblue
    'Chabarovsk': '#B37BA4'       # Sky Magenta
}

source_domain_color = papercolors["prussianblue"]
target_domain_color = papercolors["burntsienna"]

coordinates = {
    "BandarAnzali": (37.4686457254962, 49.474444963107395),
    "Mumbai": (19.088602296879547, 72.87157484146893),
    "MexicoCity": (19.43202018506947, -99.1195237076458),
    "CapeTown": (-33.926504094383525, 18.482963100299195),
    "BlackForest": (48.28424998396387, 7.994630448217265),
    "Chabarovsk": (48.48742588965333, 135.07296238176582),
    "KippaRing": (-27.224253825057243, 153.08326748125538)
}

import math
from cvxopt import matrix, solvers

checkpoint = "weights/RN18-epochepoch=123-val_lossval_loss=0.23.ckpt"

DFCPATH = "/Users/marc/projects/data-centric-paper-experiments/datasets/DFC_Public_Dataset"
DEVICE = "cpu"

target_region_name = "KippaRing"


def main():
    #(train_features, train_ids, train_rgb), (test_features, test_ids, test_rgb) = extract_features()

    def get_avg_bands(s2):
        s2 = s2 * 1e-4
        return s2.mean(1).mean(1)

    if not os.path.exists("exp1_data.npz"):
        X, y, reg = [], [], []
        for regionname, regionseason in tqdm(regions):
            ds = DFCRegionDataset(dfcpath=DFCPATH, region=(regionname, regionseason), transform=get_avg_bands)

            for i in range(len(ds)):
                s2, id = ds[i]
                X.append(s2)
                y.append(id)
                reg.append(regionname)

        X = np.stack(X)
        y = np.stack(y)
        reg = np.stack(reg)
        c = np.array([colors[r] for r in reg])
        np.savez("exp1_data.npz", X=X, y=y, reg=reg, c=c)
    else:
        f = np.load("exp1_data.npz")
        X, y, reg, c = f["X"], f["y"], f["reg"], f["c"]

    idxs = np.arange(X.shape[0], dtype=np.int32)
    np.random.shuffle(idxs)

    fig, ax = plt.subplots()
    ax.scatter(X[idxs,bands.index("S2B4")], X[idxs,bands.index("S2B8")], c=c[idxs], alpha=0.5)
    ax.set_xlabel("red surface reflectance in %")
    ax.set_ylabel("near-infrared surface reflectance in %")
    plt.show()

    mask = reg == target_region_name
    target_features = X[mask]
    target_c = c[mask]
    target_y = y[mask]
    target_reg = reg[mask]

    source_features = X[~mask]
    source_c = c[~mask]
    source_reg = reg[~mask]
    source_y = y[~mask]

    model = adapt.instance_based.KMM(estimator=RandomForestClassifier(random_state=0), Xt=None, kernel='linear', B=10., eps=None, max_size=1000, tol=None,
                             max_iter=100, copy=True, verbose=1, random_state=0)
    model.fit(source_features, source_y, Xt=target_features)
    target_y_pred = model.predict(target_features)

    print("with adapt")
    print(classification_report(y_true=target_y, y_pred=target_y_pred))
    print(accuracy_score(y_true=target_y, y_pred=target_y_pred))

    print("without coeffs")
    clf = RandomForestClassifier(random_state=0)

    clf.fit(source_features, source_y)
    target_y_pred = clf.predict(target_features)

    print(classification_report(y_true=target_y, y_pred=target_y_pred))
    print(accuracy_score(y_true=target_y, y_pred=target_y_pred))

    coeffs = kernel_mean_matching(target_features, source_features, kern='lin', B=10.)


    print("with coeffs")
    # with coefficients
    clf = RandomForestClassifier(random_state=0)
    clf.fit(source_features, source_y, sample_weight=coeffs[:,0])
    target_y_pred = clf.predict(target_features)

    print(classification_report(y_true=target_y, y_pred=target_y_pred))
    print(accuracy_score(y_true=target_y, y_pred=target_y_pred))

    fig, ax = plt.subplots()
    ax.hist(coeffs)
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(source_features[:, bands.index("S2B4")], source_features[:, bands.index("S2B8")], s=coeffs*5+1, c=source_domain_color, alpha=0.5)
    ax.scatter(target_features[:, bands.index("S2B4")], target_features[:, bands.index("S2B8")], marker="*", c=target_domain_color, s=30, alpha=0.5)
    ax.set_xlabel("red surface reflectance in %")
    ax.set_ylabel("near-infrared surface reflectance in %")
    ax.set_xlim(0,0.25)
    ax.set_ylim(0, 0.4)
    sns.despine(ax=None, top=True, right=True, left=False, bottom=False, offset=10, trim=True)
    plt.show()

    df = pd.DataFrame([source_reg, coeffs[:, 0]], index=["region", "coeff"]).T
    coeff_by_region = df.groupby("region").mean()

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    coords = np.array(list(coordinates.values()))
    coords_names = np.array(list(coordinates.keys()))
    is_target_region = coords_names == target_region_name
    #coords = coords[~is_target_region]

    fig, ax = plt.subplots()
    world.plot(ax=ax, color='lightgray')
    ax.scatter(coords[~is_target_region,1], coords[~is_target_region,0], c=source_domain_color, s = 1+coeff_by_region["coeff"].values.astype(float)*20)
    ax.scatter(coords[is_target_region, 1], coords[is_target_region, 0], c=target_domain_color, s=200, marker="*")
    sns.despine(ax=None, top=True, right=True, left=True, bottom=True, offset=0, trim=False)
    ax.set_xlabel("longitude in degree")
    ax.set_ylabel("latitude in degree")
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

def extract_features(overwrite=True):
    """
    either recomputes features from a model and the dataset or loads cached features
    :return: features and ids
    """

    def extract(ds):
        features, rgb, ids = [], [], []
        with torch.no_grad():
            for idx, (X,Y) in tqdm(enumerate(ds), total=len(ds)):
                X = torch.tensor(X, device=DEVICE, dtype=torch.float)
                y_features = model(X[None])

                rgb.append(equalize_hist(X[np.array([4, 3, 2])].numpy()))

                features.append(y_features.cpu().detach().numpy())
                ids.append(idx)

        features = np.vstack(features)
        rgb = np.vstack(rgb)
        return features, ids, rgb


    datamodule = DFC2020DataModule()
    datamodule.setup()

    model = ResNet18.load_from_checkpoint(checkpoint)
    model.eval()
    # get features instead of class logits
    model.model.fc = nn.Identity()

    model = model.to(DEVICE)

    if os.path.exists("train_features.npz") and not overwrite:
        train_tuple = np.load("train_features.npz").values()
    else:
        features, ids, rgb = extract(datamodule.train_ds)
        np.savez("train_features.npz", features=features, ids=ids, rgb=rgb)
        train_tuple = (features, ids, rgb)

    if os.path.exists("test_features.npz") and not overwrite:
        test_tuple = np.load("test_features.npz").values()
    else:
        features, ids, rgb = extract(datamodule.test_ds)
        np.savez("test_features.npz", features=features, ids=ids, rgb=rgb)
        test_tuple = (features, ids, rgb)

    return train_tuple, test_tuple

def plot_tsne(features, coeffs, mask):
    features_pca = PCA(n_components=2).fit_transform(features)

    plt.figure()
    plt.scatter(features_pca[~mask, 0], features_pca[~mask, 1], s=coeffs*10 + 1, cmap="Reds")
    plt.scatter(features_pca[mask, 0], features_pca[mask, 1], s=50)
    plt.title("coeffs")



if __name__ == '__main__':
    main()