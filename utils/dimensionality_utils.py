import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def apply_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)
    return pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])


def apply_tsne(df, n_components=2, perplexity=30, random_state=42):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    components = tsne.fit_transform(df)
    return pd.DataFrame(components, columns=[f"Dim{i+1}" for i in range(n_components)])
