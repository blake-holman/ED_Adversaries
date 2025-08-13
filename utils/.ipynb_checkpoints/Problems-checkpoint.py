import matplotlib.pyplot as plt
import itertools
import numpy as np
from copy import copy
def sort_lexico(iter):
    if not isinstance(iter, list):
        iter = list(iter)
    iter.sort(key=lambda L: ''.join([str(x) for x in L]))
    
class Problem():
    def __init__(self, no_instances, yes_instances, sort=True):
        self.n = len(no_instances[0])
        for instance in no_instances + yes_instances:
            assert len(instance) == self.n
        self.no_instances = no_instances
        self.yes_instances = yes_instances
        if sort:
            sort_lexico(self.no_instances)
            sort_lexico(self.yes_instances)
        self.instances = no_instances + yes_instances
        self.no_len = len(self.no_instances)
        self.yes_len = len(self.yes_instances)
        self.len = self.no_len + self.yes_len
        self.no_instance_to_index = {self.no_instances[i] : i for i in range(len(self.no_instances))}
        self.yes_instance_to_index = {self.yes_instances[i] : i for i in range(len(self.yes_instances))}
        self.instance_to_index = copy(self.no_instance_to_index)
        self.instance_to_index.update({instance: self.yes_instance_to_index[instance] + self.no_len for instance in self.yes_instance_to_index})
        self.no_labels = [''.join([str(x) for x in no]) for no in self.no_instances]
        self.yes_labels = [''.join([str(x) for x in yes]) for yes in self.yes_instances]

        self.alphabet = set()
        for instance in no_instances + yes_instances:
            for v in instance:
                self.alphabet.add(v)
        self.alphabet = list(self.alphabet)
        self.alphabet.sort()
        self.alphabet = tuple(self.alphabet)
        
    def __str__(self):
        to_print = 'No:' + str(self.no_instances) + '\n' + 'Yes:' + str(self.yes_instances)
        return to_print
def sort_lexico(iter):
    if not isinstance(iter, list):
        iter = list(iter)
    iter.sort(key=lambda L: ''.join([str(x) for x in L]))

def exact_k(n, k):
    no_instances = [tuple([0] * n)]
    yes_instances = list(set(itertools.permutations(tuple([0] * (n-k) + [1] * k))))
    return Problem(no_instances, yes_instances)

def threshold_k(n, k):
    no_instances = set()
    yes_instances = set()

    no_instances = no_instances.union(itertools.permutations(tuple([0] * (n-k+1) + [1] * (k-1))))
    no_instances = list(no_instances)
    yes_instances = set(itertools.permutations(tuple([0]* (n-k) + [1]*k)))
    return Problem(no_instances, list(yes_instances))
    
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