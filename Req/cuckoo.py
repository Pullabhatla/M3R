import numpy as np
import random
from Req import LeakyReLUSoftmaxCCE


def cuckoo_search(X, Y, N, generations, p, l, in_shape, out_shape, layer_sizes):
    agents = [LeakyReLUSoftmaxCCE(in_shape, out_shape, layer_sizes) for _ in range(N)]
    generation_best = []

    generation_best.append(max(agents, key=lambda agent: agent.loss(X, Y)))
    for _ in range(generations):

        
        agents = sorted(agents, key=lambda agent: agent.loss(X, Y))[:int(p*len(agents))]
        generation_best.append(agents[0])
        

    return generation_best
