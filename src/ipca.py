from sklearn.decomposition import IncrementalPCA
import numpy as np

ipca = IncrementalPCA(n_components=(500))
started = False
i = 0


def run_pca(n, data):
    global ipca
    global started

    if (started == False):
        print("abc")

    ipca.partial_fit(data)
    tmp = ipca.transform(data)
    temp = ipca.inverse_transform(tmp)

    return temp


def fit_tiles(data):
    print("python running")
    global ipca

    ipca.partial_fit(data)


def add(j):
    global i
    i = i+j

    # ! hier muss der wert zurückgegeben werden, auch wenn in rust nichts damit gemacht wird!
    return i


def get():
    global i
    return i


def consume_tiles(tile):

    print("\n\nprinting geoengine tile from within python:")
    print(tile)


def partial_fit_ipca(tile):

    print("fitting")

    global ipca
    ipca.partial_fit(tile)


def apply_ipca(tile):

    global ipca

    print("transforming")

    tmp = ipca.transform(tile)
    temp = ipca.inverse_transform(tmp).astype(np.uint8)

    return temp
