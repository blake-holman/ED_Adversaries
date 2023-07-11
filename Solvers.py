import cvxpy as cp
import numpy as np
from Adversary import Adversary, Problem
from Examples import exact_k, threshold_k
import matplotlib.pyplot as plt
import itertools

def partial(problem, i):
    lang_size = problem.yes_len + problem.no_len
    D = np.zeros((lang_size, lang_size))
    for j in range(problem.yes_len):
        for k in range(problem.no_len):
            yes = problem.yes_instances[j]
            no = problem.no_instances[k]
            if yes[i] != no[i]:
                D[j + problem.no_len][k] = 1
                D[k][j + problem.no_len] = 1 
    return D
def big_mask_index_disagree(problem, i):
    lang_size = problem.yes_len + problem.no_len
    n = problem.n
    target = [np.zeros((lang_size, lang_size))]*i + [partial(problem, i)] + [np.zeros((lang_size, lang_size))]*(n-i-1)
    return np.block([target if j==i else [np.zeros((lang_size, lang_size))] * n for j in range(n)])


    
def type_mask(problem):
    lang_size = problem.yes_len + problem.no_len
    mask = np.zeros((lang_size, lang_size))
    for i in range(problem.yes_len):
        for j in range(problem.no_len):
            mask[i + problem.no_len][j] = 1
            mask[j][i + problem.no_len] = 1 
    return mask

def instance_mask(problem, instance):
    lang_size = problem.yes_len + problem.no_len
    mask = np.zeros((lang_size, lang_size))
    mask[problem.instance_to_index[instance]][problem.instance_to_index[instance]] = 1
    return mask 

def big_mask_instance(problem, instance):
    lang_size = problem.yes_len + problem.no_len
    n = problem.n
    
    # target = [np.zeros((lang_size, lang_size))]*i + [partial(problem, i)] + [np.zeros((lang_size, lang_size))]*(n-i-1)
    return np.block([[np.zeros((lang_size, lang_size))]*i + [instance_mask(problem, instance)] + [np.zeros((lang_size, lang_size))]*(n-i-1) for i in range(n)])
    
def big_mask_type(problem):
    lang_size = problem.yes_len + problem.no_len
    n = problem.n

def big_mask_index_disagree_type(problem, yes_instance, no_instance):
    yes = problem.instance_to_index[yes_instance]
    no = problem.instance_to_index[no_instance]
    lang_size = problem.yes_len + problem.no_len
    n = problem.n
    mask = np.zeros((lang_size, lang_size))
    mask[yes, no] = 1
    big = []
    for i in range(n):
        if yes_instance[i] != no_instance[i]:
            big.append([np.zeros((lang_size, lang_size))]*i + [mask] + [np.zeros((lang_size, lang_size))]*(n-i-1))
        else:
            big.append([np.zeros((lang_size, lang_size))]*n)
    return np.block(big)
    # target = [np.zeros((lang_size, lang_size))]*i + [partial(problem, i)] + [np.zeros((lang_size, lang_size))]*(n-i-1)
    return np.block([[np.zeros((lang_size, lang_size))]*i + [type_mask(problem)] + [np.zeros((lang_size, lang_size))]*(n-i-1) for i in range(n)])

def span_solver2(problem, solver_params=None):
    if solver_params is None:
        solver_params = {'solver':'MOSEK', 'verbose': True}
    lang_size = problem.yes_len + problem.no_len
    n = problem.n
    mat_size = lang_size * n
    X = cp.Variable((mat_size, mat_size), PSD=True)
    t = cp.Variable()
    # s = cp.Variable()
    I = np.identity(mat_size)
    constraints = [X >= np.zeros(X.shape)]
    super_mask = np.kron(np.identity(n), np.ones((lang_size, lang_size)))
    plt.imshow(np.ones(X.shape)-super_mask)
    plt.colorbar()
    plt.show()
    constraints += [cp.multiply(np.ones(X.shape)-super_mask, X)==np.zeros(X.shape)]
    # constraints = [np.linalg.matrix_rank(X) <= s]
    # constraints = [cp.trace(X) <= s]
    constraints += [cp.sum(cp.multiply(big_mask_index_disagree_type(problem, yes_i, no_i), X)) == 1 for yes_i, no_i in itertools.product(problem.yes_instances, problem.no_instances)]
    constraints += [cp.trace(cp.multiply(big_mask_instance(problem, instance), X)) <= t for instance in problem.no_instances + problem.yes_instances]
    prob = cp.Problem(cp.Minimize(t), constraints)
    print(solver_params)
    prob.solve(**solver_params)
    return prob.value, X.value

def span_solver(problem, solver_params=None):
    if solver_params is None:
        solver_params = {'solver':'MOSEK', 'verbose': True}
    lang_size = problem.yes_len + problem.no_len
    n = problem.n
    mat_size = lang_size * n
    X = cp.Variable((mat_size, mat_size), PSD=True)
    t = cp.Variable()
    I = np.identity(mat_size)
    constraints = []
    constraints += [cp.sum(cp.multiply(big_mask_index_disagree_type(problem, yes_i, no_i), X)) == 1 for yes_i, no_i in itertools.product(problem.yes_instances, problem.no_instances)]
    constraints += [cp.trace(cp.multiply(big_mask_instance(problem, instance), X)) <= t for instance in problem.no_instances + problem.yes_instances]
    prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve(**solver_params)
    return prob.value, X.value


def adv_solver(problem, solver_params=None):
    if solver_params is None:
        solver_params = {'solver':'MOSEK', 'verbose': True}
    lang_size = problem.yes_len + problem.no_len
    n = problem.n
    mat_size = lang_size 
    L = cp.Variable((lang_size, lang_size), symmetric=True)
    M = cp.Variable((lang_size))
    constraints = [cp.diag(M) >> cp.multiply(L, partial(problem, j)) for j in range(n)]
    constraints += [cp.sum(M) == 1]
    constraints += [M >= 0]
    # L is an adversary matrix
    constraints += [
        cp.multiply(L, np.ones((mat_size, mat_size)) - type_mask(problem)) == 0 
    ]
    opt_func = cp.sum(
        cp.multiply(
            type_mask(problem), L
        ))
    prob = cp.Problem(cp.Maximize(opt_func), constraints)
    prob.solve(**solver_params)
    return Adversary(problem, matrix=L.value)
    
