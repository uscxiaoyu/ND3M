#coding=gbk
from __future__ import division
from numpy import pi,sqrt,log,e
from scipy.optimize import root
import networkx as nx
import numpy as np


class gener_pk:
    c = 1.2
    def __init__(self, d=6, g='power', k_list=np.arange(1,100)):
        self.d = d
        self.g = g
        self.k_list = k_list

    def gauss(self, x):
        pk_list = self.c * e ** (-(self.k_list - x[0]) ** 2 / (2 * x[1] ** 2)) / (x[1] * sqrt(2 * pi))
        return np.sum(pk_list) - 1, np.dot(pk_list, self.k_list) - self.d

    def logno(self, x):
        pk_list = self.c * e ** (-(log(self.k_list) - x[0]) ** 2 / (2 * x[1] ** 2)) / (
        self.k_list * x[1] * sqrt(2 * pi))
        return np.sum(pk_list) - 1, np.dot(pk_list, self.k_list) - self.d

    def expon(self, x):
        pk_list = x[1] * e ** (-self.k_list * x[0])
        return np.sum(pk_list) - 1, np.dot(pk_list, self.k_list) - self.d

    def power(self, x):
        pk_list = x[1] * (1.0 / self.k_list) ** x[0]
        return np.sum(pk_list) - 1, np.dot(pk_list, self.k_list) - self.d

    def get_pk(self):
        if self.g == 'gauss':
            sol = root(self.gauss, [6,1], args=(), method='lm')
            p_k = self.c * e ** (-(self.k_list - sol.x[0]) ** 2 / (2 * sol.x[1] ** 2)) / (sol.x[1] * sqrt(2 * pi))

        elif self.g == 'logno':
            sol = root(self.logno, [5,2], args=(), method='lm')
            p_k = self.c * e ** (-(log(self.k_list) - sol.x[0]) ** 2 / (2 * sol.x[1] ** 2)) / (
                     self.k_list * sol.x[1] * sqrt(2 * pi))
        elif self.g == 'expon':
            sol = root(self.expon, [0.5,1.1], args=(), method='lm')
            p_k = sol.x[1] * e ** (-self.k_list * sol.x[0])

        elif self.g == 'power':
            sol = root(self.power, [1, 1], args=(), method='lm')
            p_k = sol.x[1] * (1.0 / self.k_list) ** sol.x[0]

        else:
            p_k = np.zeros_like(self.k_list)
            p_k[int(self.d - 1)] = 1 + int(self.d) - self.d
            p_k[int(self.d)] = self.d - int(self.d)

        p_k[0] = 1 - sum(p_k[1:])  # in case of the sum is not equal to 1

        return p_k


class gener_random_graph:
    def __init__(self, n, k_list, pk_list):
        self.num_nodes = n
        self.k_list = k_list
        self.pk_list = pk_list
        self.d = np.dot(self.k_list, self.pk_list)
        self.num_edges = int(self.num_nodes*self.d/2)  # 一条边产生2度，需为整数
        while True:
            self.degre_sequance = np.random.choice(self.k_list, self.num_nodes, p=self.pk_list)
            if np.sum(self.degre_sequance) % 2 == 0:
                break

    def generate(self):
        G = nx.configuration_model(self.degre_sequance, create_using=None, seed=None)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))  # exclude self-loop edges
        num_of_edges = nx.number_of_edges(G)
        edges_list = G.edges()
        if num_of_edges > self.num_edges:
            edges_to_drop = num_of_edges - self.num_edges
            x = np.random.choice(num_of_edges, edges_to_drop, replace=False)
            for i in x:
                a, b = edges_list[i]
                G.remove_edge(a, b)

        elif num_of_edges < self.num_edges:
            edges_to_add = self.num_edges - num_of_edges
            x = np.random.choice(self.num_nodes, edges_to_add*2, replace=False)
            to_add_list = zip(x[:edges_to_add], x[edges_to_add:])
            G.add_edges_from(to_add_list)

        else:
            pass

        return G


if __name__ == '__main__':
    n = 10000
    g = 'power'
    d = 5
    k_list = np.arange(1, 100)
    g_pk = gener_pk(d, g, k_list)
    pk_list = g_pk.get_pk()
    g_graph = gener_random_graph(n, k_list, pk_list)
    random_network = g_graph.generate()
    print nx.number_of_nodes(random_network), nx.number_of_edges(random_network)
    print g_graph.pk_list, sum(g_graph.pk_list)
