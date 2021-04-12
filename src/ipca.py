from sklearn.decomposition import IncrementalPCA
import numpy as np

ipca = IncrementalPCA()


def init(n_comp):
    global ipca

    ipca = IncrementalPCA(n_components=n_comp)


def partial_fit_ipca(tile):

    global ipca
    ipca.partial_fit(tile)


def apply_ipca(tile):

    global ipca

    transformed = ipca.transform(tile)
    inv_transformed = ipca.inverse_transform(transformed).astype(np.uint8)

    return inv_transformed
