"""
Master Thesis
Academic year 2021-2022

Authors:
    - GILSON Maxence
"""

###########
# Imports #
###########

from scipy.sparse import csgraph
from numpy import linalg
import numpy as np
from database import DB, DB_np, DB_Cat
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score, SPEC  # uni-variate
from skfeature.function.sparse_learning_based import MCFS
from sklearn.feature_selection import VarianceThreshold


def get_gap(S, k):
    value = 0
    Norm = 0
    for i in range(2, k):
        Norm += S[i]
    for i in range(2, k):
        for j in range(i + 1, k + 1):
            value += abs((S[i] - S[j]) / Norm)
    return value


def get_eigen_values(database):
    X = get_kernel_matrix(database)
    Y = csgraph.laplacian(X, normed=True)
    S = linalg.eigvals(Y)
    return np.sort(S)


def get_features_name_USFSM(arr):
    name_features = []
    for i in range(0, len(arr)):
        if arr[i] > 0:
            name_features.append(DB.columns.values[i])
    return name_features


def get_kernel_matrix(database):
    kernel = np.zeros((database.shape[0], database.shape[0]), dtype=float)
    for i, xi in enumerate(database):
        for j, xj in enumerate(database):
            kernel[i][j] = apply_kernel(xi, xj)
    return kernel


def apply_kernel(xi, xj):
    value = 0
    size = xi.size
    for i in range(0, size):
        if isinstance(xi[i], str):
            if xi[i] == xj[i]:
                value += 1 - 1
            else:
                value += 1 - 0
        else:
            value += 1 - abs(xj[i] - xi[i])
    return value / len(xi)


def get_affinity_matrix():
    return construct_W.construct_W(DB_np)


def get_feature_names(arr, n_features):
    sorted_idx = np.argsort(arr)
    return DB.columns[sorted_idx[-n_features:]]


def get_USFSM(Cat):
    if Cat:
        DB_USFSM = DB_Cat.to_numpy()
    else:
        DB_USFSM = DB.to_numpy()
    S = get_eigen_values(DB_USFSM)
    gap = get_gap(S, 6)
    n_features = 0
    phi = []
    for i in range(0, DB_USFSM.shape[1]):
        mod_DB = np.delete(DB_USFSM, i, 1)
        inter_S = get_eigen_values(mod_DB)
        inter_gap = get_gap(inter_S, 6)
        act_phi = gap - inter_gap
        phi.append(gap - inter_gap)
        if act_phi > 0:
            n_features += 1
    return get_features_name_USFSM(phi)


# Uni variate UFS
def get_LapScor(n_features):
    idx = lap_score.lap_score(DB_np, W=get_affinity_matrix())
    return get_feature_names(idx, n_features)


def get_SPEC(n_features):
    idx = SPEC.spec(DB_np, style=0, W=get_affinity_matrix())
    return get_feature_names(idx, n_features)


# Sparse learning based UFS
def get_MCFS(n_features, n_clusters):
    idx = MCFS.mcfs(DB_np, n_features=n_features, W=get_affinity_matrix(), n_clusters=n_clusters)
    return get_feature_names(idx, n_features)


# stat based UFS - uni-variate
def get_LowVar(threshold):
    select = VarianceThreshold(threshold)
    test = select.fit(DB)
    return test.get_feature_names_out()


def feature_selection(name_FS, n_features):
    names = ''
    if name_FS == 'Lap Score':
        names = get_LapScor(n_features).to_numpy()
    elif name_FS == 'SPEC':
        names = get_SPEC(n_features).to_numpy()
    elif name_FS == 'MCFS':
        names = get_MCFS(n_features=n_features, n_clusters=6).to_numpy()
    elif name_FS == 'USFSM':
        names = get_USFSM(Cat=True)
    elif name_FS == 'Low Variance':
        names = get_LowVar(0.215)
    return names
