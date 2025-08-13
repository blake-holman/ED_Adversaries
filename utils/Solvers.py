import cvxpy as cp
import numpy as np
from .Adversary import Adversary
from .Problems import Problem
# from Examples import exact_k, threshold_k
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

def instance_mask(problem, instance, instance2=None):
    if instance2 is None:
        instance2 = instance
    lang_size = problem.yes_len + problem.no_len
    mask = np.zeros((lang_size, lang_size))
    mask[problem.instance_to_index[instance]][problem.instance_to_index[instance2]] = 1
    mask[problem.instance_to_index[instance2]][problem.instance_to_index[instance]] = 1
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
    
def span_solver(problem, solver_params=None, target=None, return_mats=False):
    print(return_mats)
    if target is None:
        target = np.ones(problem.yes_len)
    if solver_params is None:
        solver_params = {'solver':'MOSEK', 'verbose': True}
    lang_size = problem.yes_len + problem.no_len
    n = problem.n
    mat_size = lang_size * n
    X = cp.Variable((mat_size, mat_size), PSD=True)
    t = cp.Variable()
    I = np.identity(mat_size)
    constraints = []
            
    if return_mats:
        obj_mat = np.zeros(X.shape)
        constraint_mats = [big_mask_index_disagree_type(problem, yes_i, no_i) for yes_i, no_i in itertools.product(problem.yes_instances, problem.no_instances)]
        constraint_mats += [big_mask_instance(problem, instance) for instance in problem.no_instances + problem.yes_instances]
        return constraint_mats
        
    constraints += [cp.sum(cp.multiply(big_mask_index_disagree_type(problem, yes_i, no_i), X)) == 1 for yes_i, no_i in itertools.product(problem.yes_instances, problem.no_instances)]
    
    # constraints += [cp.sum(cp.multiply(big_mask_index_disagree_type(problem, yes_i, no_i), X)) == target[problem.yes_instance_to_index[yes_i]] for yes_i, no_i in itertools.product(problem.yes_instances, problem.no_instances)]
        
    constraints += [cp.trace(cp.multiply(big_mask_instance(problem, instance), X)) <= t for instance in problem.no_instances + problem.yes_instances]

    prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve(**solver_params)

    
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
    
def relative_hadamard_norm_max(mat, relative_mat, D_len, solver="MOSEK", uni=True, threads=8, extra_constraints=None):
    print('shapes', )
    relative_shape = [0,0]
    relative_shape[0] = relative_mat.shape[0] // mat.shape[0]
    relative_shape[1] = relative_mat.shape[1] // mat.shape[1]
    relative_shape = tuple(relative_shape)
    
    # print(D1_len, len(D2), relative.shape)
    # relative_mat = np.block(relative)

    print(relative_mat.shape)
    Y = cp.Variable((D_len, D_len), hermitian=True)
    Lambda = cp.Variable((D_len, D_len), hermitian=True)
    print(relative_shape)
    Lambda_relative = cp.kron(Lambda, np.ones(relative_shape))
    Lambda_relative = cp.multiply(Lambda_relative, relative_mat)
    diag = cp.kron(cp.diag(cp.diag(Y)), np.eye(relative_shape[0]))
    if not uni:
        Lambda_relative = cp.bmat([
            [np.zeros(Lambda_relative.shape), Lambda_relative],
            [Lambda_relative.H, np.zeros(Lambda_relative.shape)] 
        ])
        diag = cp.kron(diag, np.eye(2))
    # print(W.shape, diag.shape)
    # W = Lambda_relative - diag
    # print(W)
    constraints = [cp.trace(Y) == 1, Lambda_relative << diag]
    if extra_constraints is not None:
        constraints = constraints + [eval(constraint) for constraint in extra_constraints]
        
    opt_prob = cp.Problem(cp.Maximize(2 * cp.sum(cp.real(cp.multiply(Lambda.T, mat)))), constraints)
    opt_prob.solve(verbose=True, solver=solver, mosek_params= {"MSK_IPAR_NUM_THREADS": threads})
    # opt_prob.solve(verbose=True, solver="SCS")
                # eps_rel=1e-4,
                # eps_infeas=1e-7)
    return Lambda.value, Y.value, Lambda_relative.value

def min_fill(adv, diffs, mask=None):
    n = diffs.shape[0] // adv.shape[0] 
    if mask is None:
        mask = np.zeros(adv.shape)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if adv[i,j] == 0:
                    mask[i,j] = 1
    print('mask?')
    visualize(mask)
    extra = cp.Variable(adv.shape, symmetric=True)
    constraints = [cp.multiply(mask, extra) == extra]
    adv_diff = cp.multiply(cp.kron((adv + extra), np.ones((n,n))), diffs)
    val = np.linalg.eigvalsh(np.kron(adv, np.ones((n,n)))* diffs)[-1]
    opt_prob = cp.Problem(cp.Minimize(cp.lambda_max(adv_diff)), constraints)
    opt_prob.solve(verbose=True)
    return extra.value, adv_diff.value