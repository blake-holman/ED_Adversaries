from Solvers import adv_solver, span_solver, span_solver2, span_dual_relax
from Adversary import Adversary, Problem, to_str, visualize
import numpy as np
import matplotlib.pyplot as plt
import itertools
from Examples import exact_k, threshold_k
from ElementDistinctness import ED
from copy import deepcopy as copy
import scipy
import cvxpy as cp
import itertools
import matplotlib as mpl
mpl.rcParams['figure.dpi'] =200
class SpanProgram:
    def __init__(self, problem, I_dict, witnesses=None):
        self.witnesses = witnesses
        self.target = np.ones(problem.no_len)
        self.num_vects = np.sum([len(v) for i, v in I_dict.items()])
        self.ordered_I = [
            pair
            for pair in itertools.product(tuple(range(problem.n)), problem.alphabet)
        ]
        self.I_to_mat = {}
        self.counter = 0
        self.vect_list = []
        self.ticks = []
        self.I_dict = I_dict
        self.problem = problem
        for pair, v_set in I_dict.items():
            self.ticks.append(self.counter)
            self.I_to_mat[pair] = list(range(self.counter, self.counter + len(v_set)))
            self.counter += len(v_set)
            for vect in v_set:
                self.vect_list.append(vect)
            # self.vect_list.append(*v_set)
        self.A = np.array(self.vect_list).T

    def get_activated_A(self, x):
        activated_A = np.zeros(self.A.shape)
        for i in range(len(x)):
            interval = self.I_to_mat[(i, x[i])]
            activated_A[:, interval] = self.A[:, interval]
        return activated_A

    def apply(x, witness):
        return get_activated_A(x) @ witness

    def visualize_A(self):
        x_ticks = []
        for i in self.ordered_I:
            x_ticks.append(i)
            x_ticks += [""] * (len(self.I_dict[i]) - 1)
        visualize(self.A, (x_ticks, [to_str(no) for no in self.problem.no_instances]))

    def visualize_witnesses(self):
        y_ticks = []
        for i in self.ordered_I:
            y_ticks.append(i)
            y_ticks += [""] * (len(self.I_dict[i]) - 1)
        print("y:", len(y_ticks), self.witnesses[0].shape)
        print("x:", len(self.problem.yes_instances), len(self.witnesses))

        visualize(
            np.array(self.witnesses).T,
            ([to_str(yes) for yes in self.problem.yes_instances], y_ticks),
        )
def get_cholesky_fact(A, eps_pow=15):
    for i in range(eps_pow, 1, -1):
        print(i)
        try:
            curr_A = A + 10**-i * np.eye(A.shape[0])
            return np.round(scipy.linalg.cholesky(curr_A), i - 1)
        except:
            pass


def decompose_cholesky(L, problem):
    # print(problem.no_len)?
    partials = []
    for i in range(problem.n):
        full_partial = L[
            i * problem.len : (i + 1) * problem.len,
            i * problem.len : (i + 1) * problem.len,
        ]
        nonzero_columns = []
        for j in range(full_partial.shape[1]):
            v = full_partial[:, j]
            if np.linalg.norm(v) > 10**-3:
                # print('j', j)
                nonzero_columns.append(v)
        partials.append(np.array(nonzero_columns).T)
    return partials

def span_from_decomp(partials, problem):
    I = {}
    for j, b in itertools.product(range(problem.n), problem.alphabet):
        I[(j, b)] = []
        for i in range(partials[j].shape[1]):
            v = np.zeros(problem.no_len)
            for no_index in range(problem.no_len):
                no = problem.no_instances[no_index]
                if no[j] != b:
                    # print(j, i, no_index)
                    v[no_index] = partials[j][no_index, i]
            I[(j, b)].append(v)
    witnesses = []
    for yes_index in range(problem.yes_len):
        yes = problem.yes_instances[yes_index]
        w = []
        for (j, b), vects in I.items():
            if yes[j] != b:
                num_zeros = np.sum(v.shape[0] for v in vects)
                w += [0] * partials[j].shape[1]
            else:
                w += list(partials[j][problem.no_len + yes_index, :])
        # print(w)
        witnesses.append(np.array(w))

    return SpanProgram(problem, I, witnesses)

def sdp_to_span(X, prob, num_round=7):
    X2 = np.round(X, num_round)
    L = get_cholesky_fact(X2)
    partials = decompose_cholesky(L.T, prob)
    return span_from_decomp(partials, prob)