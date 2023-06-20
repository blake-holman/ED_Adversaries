import matplotlib.pyplot as plt
from Adversary import Adversary, Problem, hamming_dist, visualize
import itertools
import numpy as np

def exact_k(n, k):
    no_instances = [tuple([0] * n)]
    yes_instances = list(set(itertools.permutations(tuple([0] * (n-k) + [1] * k))))
    return Problem(no_instances, yes_instances)

def threshold_k(n, k):
    no_instances = set()
    yes_instances = set()

    for i in range(k):
        no_instances = no_instances.union(itertools.permutations(tuple([0] * (n-i) + [1] * i)))

    no_instances = list(no_instances)
    yes_instances = set(itertools.permutations(tuple([0]* (n-k) + [1]*k)))
    return Problem(no_instances, list(yes_instances))
