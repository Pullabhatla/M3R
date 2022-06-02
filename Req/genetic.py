import numpy as np
import random
from Req import LeakyReLUSoftmaxCCE


def genetic_algorithm(X, Y, pop_size, generations, selection_rate, mutation_rate, in_shape, out_shape, layer_sizes):
    n = len(layer_sizes) + 1
    agents = [LeakyReLUSoftmaxCCE(in_shape, out_shape, layer_sizes) for _ in range(pop_size)]
    generation_best = []
    for _ in range(generations):
        agents = sorted(agents, key=lambda agent: agent.loss(X, Y))[:int(selection_rate * len(agents))]
        generation_best.append(agents[0])
        children = []
        for _ in range((pop_size-int(selection_rate * len(agents)))//2):
            parent_1, parent_2 = random.sample(agents, 2)            
            child_1, child_2 = LeakyReLUSoftmaxCCE(in_shape, out_shape, layer_sizes), LeakyReLUSoftmaxCCE(in_shape, out_shape, layer_sizes)
            k = random.randrange(n)
            child_1.weights, child_1.biases = parent_1.weights[:k] + parent_2.weights[k:], parent_1.biases[:k] + parent_2.biases[k:]
            child_2.weights, child_2.biases = parent_2.weights[:k] + parent_1.weights[k:], parent_2.biases[:k] + parent_1.biases[k:]

            children += [child_1, child_2]
        agents = agents + children
        for agent in agents:
            for i in range(n):
                if np.random.uniform()<mutation_rate:
                    W = agent.weights[i]
                    pos = random.randrange(np.prod(W.shape))
                    W[np.unravel_index(pos, W.shape)] = np.random.randn()

                if np.random.uniform() < mutation_rate:
                    b = agent.biases[i]
                    pos = random.randrange(b.shape[0])
                    b[pos] = np.random.randn()
    generation_best.append(max(agents, key=lambda agent: agent.loss(X, Y)))

    return generation_best
