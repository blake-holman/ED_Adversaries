import numpy as np
from itertools import product, permutations, chain
import matplotlib.pyplot as plt
"""
    Problem/ Problem Matrices
"""

def adversary_k(cycles, k):
    print('getting shifted', 'k=', k)
    print(cycles[0])
    cycle_set = {tuple(cycle) for cycle in cycles}
    shifted = {cycle: get_all_k_shift(cycle, k) for cycle in cycles}
    shift_mat = np.zeros((len(cycles), len(cycles))) 
    print('filling matrix')
    curr_cycle = 0
    for cycle, shifts in shifted.items():
        if len(shifted)>= 10 and not curr_cycle % (len(shifted) // 10):
            print( np.round(curr_cycle /len(shifted), 3), end=' ')
        curr_cycle += 1
        for shift in shifts:
            # print(cycle, shift)
            shift_mat[cycles.index(cycle), cycles.index(shift)] = 1
    return shift_mat
    
def unitary_implementation(problem_mats):
    n = problem_mats[0][0].shape[0]
    problem_index_to_label = list(product(range(len(problem_mats)), range(n)))
    label_to_index = {problem_index_to_label[i]: i for i in range(len(problem_mats))}
    size = len(problem_index_to_label)
    problem_mask = np.zeros((size, size), dtype=complex)
    print(problem_index_to_label)
    start_end_vectors = []
    for oracle, unitary in problem_mats:
        for i in range(n):
            start_end_vectors.append(((np.eye(1, n, i).T).T, (unitary@np.eye(1, n, i).T).T))
    
    for i in range(size):
        for j in range(size):
            start_i, end_i = start_end_vectors[i]
            start_j, end_j = start_end_vectors[j]    
            
            problem_mask[i,j] = np.inner(np.conj(start_i), start_j)[0,0] - np.inner(np.conj(end_i), start_j)[0,0]
    oracles_for_diff = [problem_mats[i][0] for _ in range(n) for i in range(len(problem_mats))]
    diffs = get_oracle_diffs(oracles_for_diff)
    return problem_mask, diffs

def search_mask(funcs, assignment=None, marker=0):
    size = len(funcs)
    if assignment is None:
        assignment = search_perm_sorter_via_inverse(funcs)
    if assignment == 'same':
        assignment = {funcs[i]: i for i in range(len(funcs))}
    mask = np.zeros((size, size))
    for f in funcs:
        f0 = f.index(marker)
        for g in funcs:
            g0 = g.index(marker)
            if f0 != g0:
                mask[assignment[f], assignment[g]] = 1
    return mask
    
def mask_maker(cases, assignment=None):
    perms = None
    if assignment is None:
        for case in cases:
            cases.sort()
        perms = list(chain(cases))
        assignment = {i: perm for perm in case for i, case in enumerate(perms)}
    mat = np.zeros((len(assignment), len(assignment)))
    for case1 in cases:
        for case2 in cases:
            if case1 != case2:
                for perm1 in cases[case1]:
                    for perm2 in cases[case2]:
                        mat[assignment[perm1], assignment[perm2]] = 1
    return mat

def function_erasure_mask(funcs, target=0):
    n = len(funcs[0])
    start_vects = []
    end_vects = []
    for func in funcs:
        inv = func.index(target)
        end_vect = np.zeros((n, n)) 
        end_vect[:, 0] = 1
        end_vect[inv,0] = 0
        start_vect = np.zeros((n,n))
        for i in range(n):
            if i != inv:
                start_vect[i, func[i]] = 1
        start_vects.append(start_vect)
        end_vects.append(end_vect)
    mask = np.zeros((len(funcs), len(funcs)))
    for i in range(len(funcs)):
        # print(start_vects[i])
        for j in range(len(funcs)):
            # print(end_vects[i])
            mask[i,j] = np.trace(start_vects[i].T@start_vects[j]) - np.trace(end_vects[i].T@end_vects[j])
    return mask

def search_perm_sorter_via_inverse(perms):
    n = len(perms[0])
    print('n', n)
    perm_types = {i:[] for i in range(n)}
    for perm in perms:
        perm_types[perm.index(0)].append( perm)
    for t in perm_types:
        perm_types[t].sort()
    perm_assignment = {}
    curr_perm = 0
    for i, permsi in perm_types.items():
        permsi = list(permsi)
        for perm in permsi:
            perm_assignment[perm] = curr_perm
            curr_perm += 1
    # print(perm_assignment)
    return perm_assignment, perm_types

def get_n_cycles(n):
    cycles = permutations(range(n))
    perms = set()
    for cycle in cycles:
        perm = list(range(n))
        for i in range(n):
            perm[cycle[i]] = cycle[(i+1)%n]
        perms.add(tuple(perm))
    return list(perms)

def lv_search_mat(n, shift=None):
    cycles = get_n_cycles(n)
    # print(cycles)
    search_perm_sorter_via_inverse(cycles)
    assignment, perm_types = search_perm_sorter_via_inverse(cycles)
    one_pairs = dict()
    for cycle in cycles:
        one_pairs[cycle] = []
        if shift is None:
            for k in range(1, n-1):
                one_pairs[cycle] = one_pairs[cycle] + get_all_k_shift(cycle, k)
        else: 
            one_pairs[cycle] = get_all_k_shift(cycle, shift)

    mat = np.zeros((len(cycles), len(cycles)))
    for cycle in cycles:
        for pair in one_pairs[cycle]:
            mat[assignment[cycle], assignment[pair]] = 1
    return mat
    
def get_special_cycles(n, m, decision=False):
    perm_items = list(product(range(n), range(m)))
    index_cycles = get_cycles(n*m)
    preimage_dict = {str(y):[] for y in range(1, n)}
    cycles = [tuple(to_str(perm_items[index_cycle[i]]) for i in range(n*m)) for index_cycle in index_cycles]
    special_cycles = []
    # print(cycles)
    bad_cycles = []
    for cycle in cycles:
        if cycle[-1][1]=='0':
            # print(cycle)
            special_cycles.append(cycle)
            preimage_dict[cycle[-1][0]].append(cycle)
        else:
            bad_cycles.append(cycle)
    if decision:
        assignment_special = cycle_sort(special_cycles)
        assignment_bad = cycle_sort(bad_cycles)
        all_cycles = special_cycles + bad_cycles
        assignment = {all_cycles[i]: i for i in range(len(all_cycles))}
        preimage_dict = {0: special_cycles, 1: bad_cycles}
        return all_cycles, assignment, preimage_dict
    
    assignment = cycle_sort(special_cycles)
    for v, i in assignment.items():
        special_cycles[i] = v
        return special_cycles, assignment, preimage_dict

"""
    Masks
"""


def partial(instances, i):
    lang_size = len(instances)
    D = np.zeros((lang_size, lang_size))
    for j in range(lang_size):
        for k in range(lang_size):
            yes = instances[j]
            no = instances[k]
            if yes[i] != no[i]:
                D[j ][k] = 1
                D[k][j] = 1 
    return D
"""
Oracle Conversions
"""

def cycle_to_standard(cycle, nonstandard=True):
    n = len(cycle)
    if nonstandard:
        elements = list(cycle)
        elements.sort()
        cycle = tuple(elements.index(cycle[i]) for i in range(n))
    perm = list(range(n))
    for i in range(n):
        perm[cycle[i]] = cycle[(i+1)%n]
    return tuple(perm)
"""
    Specific Adversary Matrices
"""   
    
def get_shift(perm, k, l):
    shifted_cycle = perm[:k+1] + perm[l+1:] + perm[k+1: l+1]
    assert len(perm) == len(shifted_cycle) 
    return shifted_cycle

def get_all_k_shift(perm, k):
    n = len(perm)
    return list({get_shift(perm, k, j) for j in range(k+1, n-1) if perm[j][1]=='0'})


"""
    String/Helper Functions
"""

def visualize(mat, labels=None, to_string=False, save=None, title=None, xlabel=None, ylabel=None):
    if np.linalg.norm(np.imag(mat))>0:
        mat = np.block([np.real(mat), np.zeros(mat.shape), np.imag(mat)])
        if labels is not None:
            labels = [labels[0] + [""]*len(labels[0]) + labels[0], labels[1]]
    else: 
        mat = np.real(mat)
        
    fig, ax = plt.subplots()
    heatmap = ax.imshow(mat)
    plt.tight_layout()
    figh, figw = fig.get_size_inches()
    font_sizey = figh * 72  / 3 
    font_sizey = font_sizey / np.max(mat.shape)
    font_sizex = figw * 72  / 3 / np.max(mat.shape)
    
    # fig.set_size_inches(mat., mat.shape[1]/5)
    plt.colorbar(heatmap)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if labels is not None:
        fig.subplots_adjust(bottom=0.25, left=0.25)
        xlabels, ylabels = labels
        if to_string:
            xlabels = to_str_list(xlabels)
            ylabels = to_str_list(ylabels)
        ax.set_xticks(np.arange(mat.shape[1]), minor=False)
        ax.set_yticks(np.arange(mat.shape[0]), minor=False)
        ax.set_xticklabels(xlabels, rotation=90, fontsize=font_sizey)
        ax.set_yticklabels(ylabels, fontsize=font_sizey)
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()

def to_str_list(L):
    return [to_str(x) for x in L]

def hamming_dist(a, b):
    assert len(a) == len(b)
    return len([i for i in range(len(a)) if a[i] != b[i]])

def get_cycles(n):
    cycles = []
    for perm in permutations(range(1, n)):
        cycles.append((0,) + perm)
    return cycles
    
def to_str(l):
    return ''.join(str(a) for a in l)

def cycle_sort(cycles):
    cycles.sort(key=itemgetter(-1))
    return {cycles[i]: i for i in range(len(cycles))}

def nthRootsOfUnity2(n): # constant space, serial
    from cmath import exp, pi
    c = 2j * pi / n
    return [exp(k * c) for k in range(n)]
    
def real_phase_oracle(func):
    n = len(func)
    roots = nthRootsOfUnity2(n)
    # print(roots)
    oracle = np.zeros((n,n), dtype=complex)
    for i in range(n):
        oracle[i,i] = roots[func[i]]
    return oracle

def to_adversary_mat(Lambda, mu):
    G = np.zeros(Lambda.shape)
    for x in range(Lambda.shape[0]):
        for y in range(Lambda.shape[1]):
            G[x,y] = Lambda[x,y]/np.sqrt(mu[x,x] * mu[y,y])
            
    return G