import numpy as np
import matplotlib.pyplot as plt
import scipy
from copy import copy
from .Conversions import visualize







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
        # self.full_matrix = np.block([
        #     [np.zeros((problem.no_len, problem.no_len)), self.matrix],
        #     [self.matrix.T, np.zeros((problem.yes_len, problem.yes_len))]
        # ])
        
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
