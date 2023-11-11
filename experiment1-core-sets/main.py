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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from itertools import product

classids = np.array([1, 2, 4, 5, 6, 7, 9, 10])
classes = ["Forest", "Shrubland", "Grassland", "Wetlands", "Croplands",
           "Urban/Built-up", "Barren", "Water"]
class_lut = np.array( [np.nan, 0, 1, np.nan, 2, 3, 4, 5, np.nan, 6, 7] )

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
    "BandarAnzali": (37.4686457254962    , 49.474444963107395, "left" , "center"),
    "Mumbai":       (19.088602296879547  , 72.87157484146893 , "left" , "center"),
    "MexicoCity":   (19.43202018506947   , -99.1195237076458 , "left" , "center"),
    "CapeTown":     (-33.926504094383525 , 18.482963100299195, "left" , "center"),
    "BlackForest":  (48.28424998396387   , 7.994630448217265 , "left" ,  "bottom"),
    "Chabarovsk":   (48.48742588965333   , 135.07296238176582, "left" , "bottom"),
    "KippaRing":    (-27.224253825057243, 153.08326748125538 , "left" , "center")
}

DFCPATH = "/Users/marc/projects/data-centric-paper-experiments/datasets/DFC_Public_Dataset"
DEVICE = "cpu"

SAVEPATH = "/Users/marc/projects/data-centric-paper-experiments/experiment1-core-sets/results/"

from skimage.feature import graycomatrix, graycoprops

use_bands = ["S2B2", "S2B3", "S2B4", "S2B5",
                     "S2B6", "S2B7", "S2B8", "S2B8A",
                     "S2B11", "S2B12"]

def main(overwrite = False,
        normalize = True,
        keep_fraction = 0.33,
        seed = 0,
        figsize = (6,2.5),
        no_glcm=True):

    savepath = os.path.join(SAVEPATH, f"{seed}")
    os.makedirs(savepath, exist_ok=True)

    np.random.seed(seed)

    if not os.path.exists("exp1_data.npz") or overwrite:
        X, y, reg, c = extract_featuers()
        np.savez("exp1_data.npz", X=X, y=y, reg=reg, c=c)
    else:
        f = np.load("exp1_data.npz")
        X, y, reg, c = f["X"], f["y"], f["reg"], f["c"]
    y = class_lut[y].astype(int)

    if no_glcm:
        X = X[:,:20]

    # take only means
    #X = X[:,:13]
    if normalize:
        means = X.mean(0)
        stds = X.std(0)
        X -= means
        X /= stds
    else:
        means, stds = np.zeros(X.shape[1]), np.ones(X.shape[1])

    idxs = np.arange(X.shape[0], dtype=np.int32)
    np.random.RandomState(seed).shuffle(idxs)

    orig_X = X * stds + means

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(orig_X[idxs,use_bands.index("S2B4")] * 100, orig_X[idxs,use_bands.index("S2B8")] * 100, c=c[idxs], s=7, alpha=0.5)
    ax.set_xlabel("red reflectance in %")
    ax.set_ylabel("near-infrared reflectance in %")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 40)
    sns.despine(ax=None, top=True, right=True, left=False, bottom=False)
    plt.show()
    fig.savefig(os.path.join(savepath, "feature_space.pdf"), transparent=True, bbox_inches="tight", pad_inches=0)

    stats = []
    region_similarity_dfs = []
    for target_region_name in coordinates.keys():
        print(target_region_name)

        mask = reg == target_region_name
        target_features = X[mask]
        target_y = y[mask]

        source_features = X[~mask]
        source_reg = reg[~mask]
        source_y = y[~mask]

        model = get_model("KLIEP", seed=seed)
        model.fit(source_features, source_y, Xt=target_features)
        acc_with = model.score(target_features, target_y)
        print(model.best_params_)

        clf = RandomForestClassifier(random_state=seed)
        clf.fit(source_features, source_y)
        acc_without = clf.score(target_features, target_y)

        coeffs = model.predict_weights()
        sel_idxs = np.argsort(coeffs)[int(len(coeffs) * keep_fraction):]
        clf = RandomForestClassifier(random_state=seed)
        clf.fit(source_features[sel_idxs], source_y[sel_idxs])
        acc_selected = clf.score(target_features, target_y)

        stats.append(dict(
            region=target_region_name,
            acc_with=acc_with,
            acc_selected=acc_selected,
            acc_without=acc_without
        ))

        print("without coeffs")
        print(acc_without)
        print("with adapt")
        print(acc_with)

        coeffs = model.predict_weights() # kernel_mean_matching(target_features, source_features, kern='lin', B=10.)

        fig, ax = plt.subplots(figsize=figsize)
        orig_source_features = source_features * stds + means
        orig_target_features = target_features * stds + means

        ax.scatter(orig_source_features[:, use_bands.index("S2B4")] * 100,
                   orig_source_features[:, use_bands.index("S2B8")] * 100,
                   s=coeffs*5,
                   c=source_domain_color,
                   alpha=0.5)
        ax.scatter(orig_target_features[:, use_bands.index("S2B4")] * 100,
                   orig_target_features[:, use_bands.index("S2B8")] * 100,
                   marker="*",
                   c=target_domain_color,
                   s=30,
                   alpha=0.5)

        ax.set_xlabel("red reflectance in %")
        ax.set_ylabel("near-infrared reflectance in %")
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 40)
        sns.despine(ax=None, top=True, right=True, left=False, bottom=False)

        fig.savefig(os.path.join(savepath, f"{target_region_name}_scatter.pdf"), bbox_inches="tight", pad_inches=0)
        plt.show()

        df = pd.DataFrame([source_reg, coeffs], index=["region", "coeff"]).T
        coeff_by_region = df.groupby("region").mean()

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        coords = np.array(list(coordinates.values()))[:, :2].astype(float)
        coords_names = np.array(list(coordinates.keys()))
        is_target_region = coords_names == target_region_name
        #coords = coords[~is_target_region]

        fig, ax = plt.subplots()
        world.plot(ax=ax, color='lightgray')
        coords_df = pd.DataFrame(coordinates, index=["lat", "lon", "ha", "va"]).T.reset_index().rename(columns={"index":"region"}).set_index("region")
        joined_df = pd.concat([coords_df, coeff_by_region],axis=1)
        joined_df["is_target"] = False
        joined_df.loc[target_region_name, "is_target"] = True

        # ax.scatter(coords[~is_target_region,1], coords[~is_target_region,0], c=source_domain_color, s = 1+coeff_by_region["coeff"].values.astype(float)*50)
        # ax.scatter(coords[is_target_region, 1], coords[is_target_region, 0], c=target_domain_color, s=200, marker="*")
        for name, row in joined_df.iterrows():

            if row.is_target:
                ax.scatter(row.lon, row.lat, marker="*", s=300, c=target_domain_color)
            else:
                s = 1 + (row.coeff * 50)
                ax.scatter(row.lon, row.lat, marker="o", s=s, c=source_domain_color)
            ax.text(row.lon+10, row.lat, name, ha=row.ha, va=row.va, fontsize=12)

        #for name, (lat, lon, ha, va) in coordinates.items():
        #    ax.text(lon, lat, name, ha=ha, va=va)
        sns.despine(ax=None, top=True, right=True, left=True, bottom=True, offset=0, trim=False)
        ax.set_xlabel("longitude in degree")
        ax.set_ylabel("latitude in degree")

        fig.savefig(os.path.join(savepath, f"{target_region_name}_map.pdf"), bbox_inches="tight", pad_inches=0)

        region_similarity_df = coeff_by_region.reset_index().rename(columns={"region": "source_region"})
        region_similarity_df["target_region"] = target_region_name
        region_similarity_dfs.append(region_similarity_df)

    df = pd.DataFrame(stats)
    df["diff_weighted"] = df.acc_with - df.acc_without
    df["diff_selected"] = df.acc_selected - df.acc_without
    print(df)
    df.to_csv(os.path.join(savepath, "results.csv"))

    latex_df = df.set_index("region").T * 100
    print(latex_df.to_latex(float_format="%0.2f"), file=open(os.path.join(savepath, "results.tex"), "w"))
    print(latex_df.to_latex(float_format="%0.2f"))

    plot_graph(pd.concat(region_similarity_dfs), savepath=savepath)

    print()

def get_features(s2):
    s2 = s2[[use_bands.index(i) for i in use_bands]]
    s2 = s2 * 1e-4
    means = s2.mean(1).mean(1)
    stds = s2.std(1).std(1)

    glcm_features = []
    for b in range(s2.shape[0]):
        band_values = s2[0]
        band_values -= band_values.min()
        band_values /= band_values.max()
        band_values *= 255
        glcm = graycomatrix(band_values.astype("uint8"), distances=[5], angles=[0], levels=256,
                     symmetric=True, normed=True)

        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        ASM = graycoprops(glcm, 'ASM')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        glcm_features.append([
            dissimilarity,
            correlation,
            contrast,
            energy,
            ASM,
            homogeneity
        ])
    glcm = np.array(glcm_features).reshape(-1)
    return np.hstack([means,stds,glcm])

def extract_featuers():
    X, y, reg = [], [], []
    for regionname, regionseason in tqdm(regions):
        ds = DFCRegionDataset(dfcpath=DFCPATH, region=(regionname, regionseason), transform=get_features)

        for i in range(len(ds)):
            s2, id = ds[i]
            X.append(s2)
            y.append(id)
            reg.append(regionname)

    X = np.stack(X)
    y = np.stack(y)
    reg = np.stack(reg)
    c = np.array([colors[r] for r in reg])
    return X, y, reg, c

def plot_graph(region_similarity_dfs, savepath=SAVEPATH):
    region_similarity_dfs = region_similarity_dfs.rename(columns={"coeff": "weight"})

    G = nx.from_pandas_edgelist(region_similarity_dfs,
                            source="source_region",
                            target="target_region",
                            edge_attr="weight",
                            create_using=nx.DiGraph())

    fig, ax = plt.subplots()
    pos = nx.circular_layout(G)
    edge_widths = [G.edges[u, v]['weight'] for (u, v) in G.edges()]

    edge_colormap = cm.Reds
    edge_norm = plt.Normalize(min(edge_widths), max(edge_widths))
    sm = plt.cm.ScalarMappable(cmap=edge_colormap, norm=edge_norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('average sample weight')

    node_color = [colors[node] for node in G.nodes()]

    nx.draw(G, pos, with_labels=False, ax=ax,
            node_size=300,
            font_size=9,
            node_color=node_color,
            edge_color=edge_widths,
            width=[w*2 for w in edge_widths],
            font_color="black",
            edge_cmap=edge_colormap,
            arrows=True, connectionstyle='arc3,rad=0.2')

    radius = 1.2  # Increase the radius for placing labels outside the circle
    label_pos = {}
    for node, (x, y) in pos.items():
        angle = np.arctan2(y, x)  # Calculate the angle of the node
        new_x = radius * np.cos(angle)  # Shift the node x-coordinate
        new_y = radius * np.sin(angle)  # Shift the node y-coordinate
        label_pos[node] = (new_x, new_y)
    #nx.draw_networkx_labels(G, pos=label_pos, font_size=9)
    # Rotate labels according to angles (theta)
    for node, (x, y) in label_pos.items():
        angle = np.arctan2(y, x)  # Calculate the angle of the node
        rotation_degrees = np.degrees(angle)
        if rotation_degrees > 0:
            rotation_degrees -= 180# Convert angle from radians to degrees
        rotation_degrees += 180
        plt.text(x, y, str(node), fontsize=12, rotation=rotation_degrees - 90,
                 horizontalalignment='center', verticalalignment='center')

    #nx.draw_networkx_edge_labels(G, pos=pos,
    #                             edge_labels=edge_labels, font_size=8)
    plt.margins(0.1)
    plt.axis('equal')
    plt.show()
    fig.savefig(os.path.join(savepath, "graph.pdf"),
                transparent=True,
                bbox_inches="tight",
                pad_inches=0)

def get_model(method = "KLIEP", seed=0):
    if method=="KMM":
        model = adapt.instance_based.KMM(estimator=RandomForestClassifier(random_state=seed),
                                         Xt=None,
                                         kernel='rbf',
                                         B=10, eps=None,
                                         max_size=1000,
                                         tol=None,
                                         max_iter=100,
                                         copy=True,
                                         verbose=False,
                                         random_state=seed,
                                         gamma=0.01)
    elif method == "LDM":
        model = adapt.instance_based.LDM(estimator=RandomForestClassifier(random_state=seed),
                                 Xt=None,
                                 copy=True,
                                 verbose=1,
                                 random_state=None)
    elif method == "KLIEP":
        model = adapt.instance_based.KLIEP(estimator=RandomForestClassifier(random_state=seed),
                                   Xt=None,
                                   kernel='rbf',
                                   sigmas=None,
                                   max_centers=100,
                                   cv=5,
                                   algo='FW',
                                   tol=1e-06,
                                   max_iter=2000,
                                   copy=True,
                                   verbose=False,
                                   gamma=[2,1.75,1.5,1.25,1.,0.75,0.5,0.25,0.1], # [2,1.75,1.5,1.25,1.,0.75,0.5,0.25,0.1]
                                   random_state=seed)
    return model

if __name__ == '__main__':
    for seed in [0]:
        main(seed=seed)