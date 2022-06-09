import numpy as np
from Req import LeakyReLUSoftmaxCCE


def majority_voting(X, Y, N, in_shape, out_shape, layer_sizes):
    mlps = [LeakyReLUSoftmaxCCE((28,28), 10, [16,16,16]) for _ in range(N)]
