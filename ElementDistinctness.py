import numpy as np
import itertools 
from math import comb
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import matplotlib as mpl
from  Adversary import Adversary, Problem


def sort_funcs(func):
    return ''.join(str(i) for i in func)
def to_str(func):
    return ''.join(str(i) for i in func)

def ED(n):   
    permutations = list(itertools.permutations(list(range(n))))
    permutations.sort()
    no_family = {permutations[i]: i for i in range(len(permutations))}
    yes_family = set()

    for permutation in permutations:
        for i in range(len(permutation)):
            permutation = list(permutation)
            for v in range(n):
                if permutation[i] != v:
                    # print(permutation)
                    yes_family.add(tuple(permutation[:i] + [v] + permutation[i+1:]))

    yes_family = list(yes_family)
    yes_family.sort(key=sort_funcs)
    no_family = list(no_family)
    no_family.sort(key=sort_funcs)
    
    print(no_family)
    yes_family = {yes_family[i]: i for i in range(len(yes_family))}
    return Problem(no_family, list(yes_family))