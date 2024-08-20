import umap
import numpy as np
import matplotlib.pyplot as plt

LOGD74_DIR = 'data/logd74'
ABALONE_DIR = 'data/abalone_age'
INHIBITION_DIR = 'data/inhibition_data'

def plot_umap(X, Y, title, colorbar_label, fig_path):
    plt.clf()

    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(X)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=Y,cmap='viridis', s=5)
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.savefig(fig_path)

def main():
    dataset_configs = [
        (
            f"{LOGD74_DIR}/fingerprints.npy", 
            f"{LOGD74_DIR}/logd74s.npy",
            'UMAP Embedding of Lipophilicity Dataset',
            'logd74',
            'explore/umap/lipo_umap.png'
        ),
        (
            f"{ABALONE_DIR}/X.npy",
            f"{ABALONE_DIR}/Y.npy",
            'UMAP Embedding of Abalone Dataset',
            'Age',
            'explore/umap/abalone_umap.png'
        ),
        (
            f"{INHIBITION_DIR}/X.npy",
            f"{INHIBITION_DIR}/y.npy",
            'UMAP Embedding of Inhibition Dataset',
            'Inhibition Percentage',
            'explore/umap/inhibition_umap.png'
        )
    ]

    for dataset_config in dataset_configs:
        feature_path, target_path, title, cb_label, fig_path = dataset_config
        X = np.load(feature_path)
        Y = np.load(target_path)
        plot_umap(X, Y, title, cb_label, fig_path)

if __name__ == "__main__":
    main()