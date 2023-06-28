from Adversary import Adversary, Problem, to_str
import numpy as np
import itertools
from Examples import exact_k
from ElementDistinctness import ED
from Solvers import adv_solver
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] =1000
from copy import deepcopy as copy

def special_or_ed(n, k):
    n_list = list(range(n))
    permutations = list(itertools.permutations(n_list))
    no_fam = list(itertools.product(*([permutations]*k)))
    yes_fam = set()
    print('fam', no_fam)
    for no in no_fam:
        no = [list(n_) for n_ in no]
        for perm1_index in range(k):
            for perm2_index in range(perm1_index):
                for swap1_index in range(n):
                    for swap2_index in range(n):
                        perm1 = list(no[perm1_index])
                        perm2 = list(no[perm2_index])
                        curr1 = perm1[swap1_index]
                        curr2 = perm2[swap2_index]
                        if curr1 != curr2:
                            perm1[swap1_index] = curr2
                            perm2[swap2_index] = curr1
                            yes = copy(no)
                            yes[perm1_index] = perm1
                            yes[perm2_index] = perm2
                            yes_fam.add(tuple(tuple(y) for y in yes))
        
    yes_fam = list(yes_fam)
    
    no_fam = [to_str([to_str(n) for n in no]) for no in no_fam]
    yes_fam = [to_str([to_str(y) for y in yes]) for yes in yes_fam]
    no_fam.sort()
    yes_fam.sort()
    return Problem(no_fam, yes_fam)
    
prob_special = special_or_ed(3, 3)
special_ed_adv = adv_solver(prob_special, {'solver': 'MOSEK', 'verbose': True})
special_ed_adv.visualize_matrix(save='special.png')