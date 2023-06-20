import numpy as np
import matplotlib.pyplot as plt
import scipy
from copy import copy
def sort_lexico(iter):
    if not isinstance(iter, list):
        iter = list(iter)
    iter.sort(key=lambda L: ''.join([str(x) for x in L]))

def hamming_dist(a, b):
    assert len(a) == len(b)
    return len([i for i in range(len(a)) if a[i] != b[i]])

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
        self.no_len = len(self.no_instances)
        self.yes_len = len(self.yes_instances)
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


def visualize(mat, labels=None, save=None):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(mat)
    plt.colorbar(heatmap)
    if labels is not None:
        fig.subplots_adjust(bottom=0.25, left=0.25)
        xlabels, ylabels = labels
        ax.set_xticks(np.arange(mat.shape[1]), minor=False)
        ax.set_yticks(np.arange(mat.shape[0]), minor=False)

        ax.set_xticklabels(xlabels, rotation=90)
        ax.set_yticklabels(ylabels)
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


class Adversary():
    def __init__(self, problem, matrix_assignment_func):
        self.problem = problem
        self.matrix = np.zeros((problem.yes_len, problem.no_len))
        for i in range(problem.yes_len):
            for j in range(problem.no_len):
                self.matrix[i, j] = matrix_assignment_func(problem.yes_instances[i], problem.no_instances[j])

    def partial_matrix(self, str_i, reduced=False):
        if reduced:
            pass
        else:
            partial = np.zeros(self.matrix.shape)
            for i in range(self.problem.yes_len):
                for j in range(self.problem.no_len):
                    if self.problem.no_instances[j][str_i] != self.problem.yes_instances[i][str_i]:
                        partial[i, j] = self.matrix[i, j]

        return partial

    def partial_norm(self, str_i):
        return np.linalg.norm(self.partial_matrix(str_i), 2)

    def norm(self):
        return np.linalg.norm(self.matrix, 2)

    def adv(self):
        return self.norm() / np.max([self.partial_norm(i) for i in range(self.problem.n)])

    def visualize_matrix(self, save=None):
        visualize(self.matrix, (self.problem.no_labels, self.problem.yes_labels), save=save)
        
    def visualize_partial(self, i, reduced=False):
        if reduced:
            pass
        else:
            visualize(self.partial_matrix(i), (self.problem.no_labels, self.problem.yes_labels))
