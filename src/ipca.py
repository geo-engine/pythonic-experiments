from ast import In
from distutils.ccompiler import new_compiler
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=(50))
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
