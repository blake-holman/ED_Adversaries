import numpy as np
import matplotlib.pyplot as plt
import scipy
from copy import copy
def sort_lexico(iter):
    if not isinstance(iter, list):
        iter = list(iter)
    iter.sort(key=lambda L: ''.join([str(x) for x in L]))

def to_str(L):
    return ''.join([str(x) for x in L])
def to_str_list(L):
    return [to_str(x) for x in L]

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
        self.instances = no_instances + yes_instances
        if sort:
            sort_lexico(self.no_instances)
            sort_lexico(self.yes_instances)
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


def visualize(mat, labels=None, to_string=False, save=None, complex=False):
    if complex:
        mat = np.block([np.real(mat), np.zeros(mat.shape), np.imag(mat)])
        if labels is not None:
            labels = [labels[0] + [""]*len(labels[0]) + labels[0], labels[1]]
    
    fig, ax = plt.subplots()
    heatmap = ax.imshow(mat)
    plt.tight_layout()
    figh, figw = fig.get_size_inches()
    font_sizey = figh * 72  / 3 
    font_sizey = font_sizey / np.max(mat.shape)
    font_sizex = figw * 72  / 3 / np.mat(mat.shape)
    
    # fig.set_size_inches(mat., mat.shape[1]/5)
    plt.colorbar(heatmap)
    if labels is not None:
        fig.subplots_adjust(bottom=0.25, left=0.25)
        xlabels, ylabels = labels
        if to_string:
            xlabels = to_str_list(xlabels)
            ylabels = to_str_list(ylabels)
        # print(xlabels)
        # print(ylabels)
        # print(mat.shape)
        ax.set_xticks(np.arange(mat.shape[1]), minor=False)
        ax.set_yticks(np.arange(mat.shape[0]), minor=False)
        # print(len(xlabels))
        ax.set_xticklabels(xlabels, rotation=90, fontsize=font_sizey)
        ax.set_yticklabels(ylabels, fontsize=font_sizey)
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()



class Adversary():
    def __init__(self, problem, matrix_assignment_func=None, matrix=None):
        self.problem = problem
        if matrix_assignment_func is not None:
            self.matrix = np.zeros((problem.yes_len, problem.no_len))
            for i in range(problem.yes_len):
                for j in range(problem.no_len):
                    self.matrix[i, j] = matrix_assignment_func(problem.yes_instances[i], problem.no_instances[j])
        elif matrix is not None:
            if matrix.shape == (problem.yes_len, problem.no_len):
                self.matrix = matrix
            elif matrix.shape == (problem.yes_len + problem.no_len, problem.yes_len + problem.no_len):
                self.matrix = matrix[problem.no_len:, :problem.no_len]
        else:
            print('mat:' + str(matrix))

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
        visualize(self.matrix, (self.problem.s, self.problem.yes_labels), save=save)
        
    def visualize_partial(self, i, reduced=False):
        if reduced:
            pass
        else:
            visualize(self.partial_matrix(i), (self.problem.no_labels, self.problem.yes_labels))
