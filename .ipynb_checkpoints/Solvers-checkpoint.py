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
    mask = np.block([target if j==i else [np.zeros((lang_size, lang_size))] * n for j in range(n)])
    return mask + mask.T

    
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
    mask = np.block(big)
    return mask + mask.T
    # target = [np.zeros((lang_size, lang_size))]*i + [partial(problem, i)] + [np.zeros((lang_size, lang_size))]*(n-i-1)
    return np.block([[np.zeros((lang_size, lang_size))]*i + [type_mask(problem)] + [np.zeros((lang_size, lang_size))]*(n-i-1) for i in range(n)])

def span_solver2(problem, r, solver_params=None):
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
    # constraints += [cp.multiply(np.ones(X.shape)-super_mask, X)==np.zeros(X.shape)]
    # constraints = [np.linalg.matrix_rank(X) <= r]
    constraints = [cp.trace(X) <= r]
    constraints += [cp.scalar_product(big_mask_index_disagree_type(problem, yes_i, no_i), X) == 1 for yes_i, no_i in itertools.product(problem.yes_instances, problem.no_instances)]
    constraints += [cp.trace(cp.multiply(big_mask_instance(problem, instance), X)) <= t for instance in problem.no_instances + problem.yes_instances]
    prob = cp.Problem(cp.Minimize(t), constraints)
    print(solver_params)
    prob.solve(**solver_params)
    return prob.value, X.value

def ket(i, dim):
    v = np.zeros(dim)
    v[i] = 1
    return v.T

def span_dual_relax(problem, d, Q=0, solver_params=None):
    if solver_params is None:
        solver_params = {'solver':'MOSEK', 'verbose': True}
    lang_size = problem.yes_len + problem.no_len
    n = problem.n
    mat_size = lang_size * n + 1
    I = np.identity(mat_size)
    T = cp.Variable((d, d), symmetric=True)
    ts = {index: cp.Variable() for index in 
         list(itertools.product(problem.yes_instances, problem.no_instances)) + problem.instances}
    A_mats = {(yes_i, no_i): 
              np.block([[big_mask_index_disagree_type(problem, yes_i, no_i), np.zeros((mat_size-1, 1))],
                        [np.zeros((1, mat_size-1)),                          np.array([[0]])]])
              
              for yes_i, no_i in itertools.product(problem.yes_instances, problem.no_instances)}
    B_mats = {instance: 
              np.block([[big_mask_instance(problem, instance), np.zeros((mat_size-1, 1))],
                        [np.zeros((1, mat_size-1)),            np.array([[-1]])]]) 
                for instance in problem.no_instances + problem.yes_instances}
    A0 = np.zeros((mat_size, mat_size))
    A0[-1,-1] = 1
    big_block = cp.bmat([
        [A0,                             np.zeros((mat_size, d))],
        [np.zeros((d, mat_size)), -T]
    ])
    
    for (yes, no), Ai in A_mats.items():
        # print(ts[(yes, no)])
        big_block = big_block + ts[(yes, no)] * cp.bmat([
            [Ai,                             np.zeros((mat_size, d))],
            [np.zeros((d, mat_size)),        -1/d * np.eye(d)]
                                                            ])
    print('after')
    for instance, Bi in B_mats.items():
        # print(ts[instance]), 
        big_block = big_block + ts[instance] * cp.bmat([
            [Bi,                             np.zeros((mat_size, d))],
            [np.zeros((d, mat_size)),        np.zeros((d, d))]
        ])
    print('after2')
    
    constraints = [big_block >> 0]
    
    # Diagonally dominant constraint
    small_block = cp.Variable()
    for (yes, no), Ai in A_mats.items():
        small_block = small_block + ts[(yes, no)] * Ai
    for instance, Bi in B_mats.items():
        small_block = small_block + ts[(yes, no)] * Bi
    
    for i in range(mat_size):
        constraints += [
            cp.norm(small_block[i,:], 1) <= 2*small_block[i,i]
        ]
    
    prob = cp.Problem(cp.Maximize(cp.trace(T)), constraints)
    prob.solve(**solver_params)
    return small_block.value, big_block.value, T.value
    
def span_solver(problem, solver_params=None, mode=None, target=None):
    if target is None:
        target = np.ones(problem.yes_len)
    if solver_params is None:
        solver_params = {'solver':'MOSEK', 'verbose': True}
    lang_size = problem.yes_len + problem.no_len
    n = problem.n
    mat_size = lang_size * n
    X = cp.Variable((mat_size, mat_size), PSD=True)
    Y = None
    t = cp.Variable()
    I = np.identity(mat_size)
    constraints = []
    # if mode == '<=':
    #     print('here')
    #     constraints.append(X >= 0)
    if mode is not None:
        if mode == "pos":
            constraints += [X >= 0]
        if mode[0] == 'trace':
            constraints += [cp.trace(X) <= mode[1]]
        if mode[0] == 'logdet':
            print('det')
            eps = mode[1]
            r = mode[2]
            constraints += [cp.log_det(X+eps * np.eye(mat_size)) <= r]
            
        if mode[0] == 'block':
            Y = cp.Variable((mode[1], mat_size))
            # constraints += [Y == Yp - Yn]
            # print(list(itertools.product(list(range(Y.shape[0])), list(range(Y.shape[1])))))
            # print([Yp[i,j] - Yn[i,j] == cp.max(Yp[i,j], - Yn[i,j]) for i,j in itertools.product(list(range(Y.shape[0])), list(range(Y.shape[1])))])
            # constraints += [Yp + Yn <= cp.maximum(Yp, Yn)]
            
            constraints +=[
                cp.bmat([
                    [X, Y.T],
                    [Y, np.eye(mode[1])]
                        ]) >> 0,
                # cp.sum(X) <= cp.sum(Y) * np.sqrt(mode[2] * problem.len) + problem.yes_len * problem.no_len,
                # cp.trace(X) <= cp.sum(Y) * problem.len * n,
                cp.multiply(np.ones((mat_size, mat_size)) - np.kron(np.identity(n), np.ones((lang_size, lang_size))), X) == 0
            ]
            constraints += [
                cp.trace(X) >= cp.norm(Y, "fro")
            ]
            if len(mode) >= 3:
                constraints += [
                    cp.sum(Y) >= np.sqrt(mode[2] * problem.len)
                ]
                constraints += [
                    cp.sum(Y @ np.kron(np.ones(n), ket(i, problem.len))) >= mode[2] for i in range(problem.len)
                ]
            for index1, index2 in itertools.product(list(range(problem.yes_len)), list(range(problem.no_len))):
                instance1 = problem.yes_instances[index1]
                instance2 = problem.no_instances[index2]
                for j in range(n):
                    ket_instance1 = ket(index1, problem.len)
                    ket_instance2 = ket(index2, problem.len)
                    ket_index = ket(j, n)
                    keti = np.kron(ket_index, ket_instance1)
                    ketj = np.kron(ket_index, ket_instance2)
                    print('s')
                    constraints += [
                        (keti+ketj).T @ X @ (keti+ketj)>= cp.power(cp.norm(Y@(keti+ketj) , 2), 2),
                        (keti-ketj).T @ X @ (keti-ketj) >= cp.power(cp.norm(Y@(keti-ketj), 2), 2)
                    ]
                    # constraints += [
                    #     (keti+ketj).T @ X @ (keti+ketj) <= cp.sum(Y@(keti+ketj)) + 1,
                    #     (keti-ketj).T @ X @ (keti-ketj) <= cp.sum(Y@(keti+ketj))*problem.len - 1/n 
                    # ]
                constraints += [cp.sum_squares(Y[:, problem.instance_to_index[instance]]) <= t for instance in problem.no_instances + problem.yes_instances]

            
            
    constraints += [cp.sum(cp.multiply(big_mask_index_disagree_type(problem, yes_i, no_i), X)) == 1 for yes_i, no_i in itertools.product(problem.yes_instances, problem.no_instances)]
    if mode == '<=':
        print('doing <=')
        constraints += [cp.sum(cp.multiply(big_mask_index_disagree_type(problem, yes_i, no_i), X)) >= 1 for yes_i, no_i in itertools.product(problem.yes_instances, problem.no_instances)]
    else: 
        constraints += [cp.sum(cp.multiply(big_mask_index_disagree_type(problem, yes_i, no_i), X)) == target[problem.yes_instance_to_index[yes_i]] for yes_i, no_i in itertools.product(problem.yes_instances, problem.no_instances)]
        
    constraints += [cp.trace(cp.multiply(big_mask_instance(problem, instance), X)) <= t for instance in problem.no_instances + problem.yes_instances]
    if mode is not None and mode == 'min_trace':
        prob = cp.Problem(cp.Minimize(cp.trace(X)), constraints)
    else:
        prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve(**solver_params)
    if Y is not None:
        return prob.value, X.value, Y.value
    
    return prob.value, X.value

def real_adv_solver(problem):
    lang_size = problem.yes_len + problem.no_len
    n = problem.n
    mat_size = lang_size 
    L = cp.Variable((lang_size, lang_size), symmetric=True)
    constraints = [cp.norm(cp.multiply(L, partial(problem, j)), 2) <= 1 for j in range(n)]
    opt_prob = cp.Problem(cp.Maximize(cp.norm(L, 2)), constraints)
    opt_prob.solve()
    return Adversary(problem, matrix=L.value)
    
    
    
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
    return Adversary(problem, matrix=L.value), M.value

def outer(a, b):
    a = cp.Expression.cast_to_const(a)  # if a is an Expression, return it unchanged.
    assert a.ndim == 1
    b = cp.Expression.cast_to_const(b)
    assert b.ndim == 1
    a = cp.reshape(a, (a.size, 1))
    b = cp.reshape(b, (1, b.size))
    expr = a @ b
    return expr

def space_adv_prob(problem, solver_params=None):
    if solver_params is None:
        solver_params = {'solver':'MOSEK', 'verbose': False}
    lang_size = problem.yes_len + problem.no_len
    n = problem.n
    mat_size = lang_size 
    A = cp.Variable((lang_size, lang_size), symmetric=True, name='A')
    M = cp.Variable((lang_size), name='M')
    r = cp.Parameter(nonneg=True, name='r')
    space_var = cp.Variable(name='space_var')
    constraints = [cp.diag(M) + space_var * np.eye(lang_size) >> cp.multiply(A, partial(problem, j)) for j in range(n)]
    constraints += [cp.sum(M) == 1]
    constraints += [M >= 0]
    constraints += [space_var >= 0]
    # L is an adversary matrix
    constraints += [
        cp.multiply(A, np.ones((mat_size, mat_size)) - type_mask(problem)) == np.zeros((mat_size, mat_size))
    ]

    prob = cp.Problem(cp.Maximize(cp.sum(A) - r * space_var), constraints)
    # prob.solve(**solver_params)
    return prob

def space_adv_solver(opt_prob, solver_params=None):
    opt_prob.solve(opt_prob)
#     return opt_prob.
    
