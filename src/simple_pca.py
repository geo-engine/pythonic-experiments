from sklearn.decomposition import PCA
import numpy as np


def run_pca(n, data):
    print("python running")
    ppp = PCA(n)
    print(data.shape)

    red_transformed = ppp.fit_transform(data)
    red_inverted = ppp.inverse_transform(red_transformed).astype(np.uint8)

    print(red_inverted.shape)
    print("python finished")
    return np.zeros((4, 4))


# data = np.arange(16).reshape(4, 4) + 1
# pca = PCA(3)

# transformed = pca.fit_transform(data)
# inve = pca.inverse_transform(transformed)
# print(data)
# print(inve)
