import numpy as np


def Euclid_d(a, b):
    dist = np.linalg.norm(a - b, axis=1)
    return dist


def Manhattan_d(a, b):
    dist = np.linalg.norm(a - b, ord=1, axis=1)
    return dist


def Maharanovis_d(a, b, A):
    A_ = np.linalg.pinv(A)
    dist = []
    for a_d, b_d in zip(a, b):
        dist.append(np.sqrt(np.dot(np.dot((a_d - b_d).T, A_), (a_d - b_d))))
    return dist
