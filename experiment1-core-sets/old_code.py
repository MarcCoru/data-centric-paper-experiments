
def tune(source_features, source_y, target_features, target_y):
    """

        if not os.path.exists(f"{target_region_name}_tuneresults.csv"):
            df = tune(source_features, source_y, target_features, target_y)
            df.to_csv(f"{target_region_name}_tuneresults.csv")
        else:
            df = pd.read_csv(f"{target_region_name}_tuneresults.csv", index_col=0)
        hparams = df.sort_values("unsup_score").iloc[0]

    :param source_features:
    :param source_y:
    :param target_features:
    :param target_y:
    :return:
    """
    tune_stats = []
    for gamma, B in product([0.01, 0.1, 1, 10], [1, 10, 100]):
        model = adapt.instance_based.KMM(estimator=RandomForestClassifier(random_state=0),
                                         Xt=None,
                                         kernel='rbf',
                                         B=B, eps=None,
                                         max_size=1000,
                                         tol=None,
                                         max_iter=100,
                                         copy=True,
                                         verbose=1,
                                         random_state=0,
                                         gamma=gamma)
        model.fit(source_features, source_y, Xt=target_features)
        unsup_score = model.unsupervised_score(source_features, target_features)
        sup_score = model.score(target_features, target_y)
        tune_stats.append(
            dict(
                gamma=gamma,
                B=B,
                unsup_score=unsup_score,
                sup_score=sup_score
            )
        )
    df = pd.DataFrame(tune_stats)
    return df

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
