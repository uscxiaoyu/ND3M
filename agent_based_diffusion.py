#coding=gbk
from __future__ import division
import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class agent_diffuse:
    def __init__(self, G, p_list, p, q, steps=25, seeds=[]):
        self.G = G
        self.p = p
        self.q = q
        self.steps = steps #steps for runing the agent-based model
        self.num_of_adopt = [] #non-cummulative adopters for each step
        self.p_list = p_list

        non_adopt_set = []
        for i in self.G.nodes_iter():
            if i in seeds:
                self.G.node[i]['state'] = True
            else:
                self.G.node[i]['state'] = False
                non_adopt_set.append(i)

        self.non_adopt_set = np.array(non_adopt_set)

    def shuffle_network(self):
        self.edge_list = nx.edges(self.G)
        self.num_of_edges = nx.number_of_edges(self.G)
        idx_set = np.random.random(self.num_of_edges) <= self.p_list  # select edges according to the probability p
        stublist = []
        for i,d in enumerate(idx_set): # deal with the edges
            if d==True:
                n1, n2 = self.edge_list[i]
                stublist.extend([n1, n2])  # add the nodes into the stublist
                self.G.remove_edge(n1, n2)  # remove the selected edge

        random.shuffle(stublist)  # shuffle the order of the stublist
        while stublist: # self-loop or repeated edges may exist
            n1 = stublist.pop()
            n2 = stublist.pop()
            self.G.add_edge(n1, n2)

    def single_diffuse(self):
        for i in self.G.nodes_iter():
            self.G.node[i]['neigh'] = self.G.neighbors(i)

        len_non = len(self.non_adopt_set)
        influ = np.zeros(len_non)
        for i in xrange(len_non):
            influ[i] = len([k for k in self.G.node[self.non_adopt_set[i]].get('neigh', []) if self.G.node[k]['state']])

        rand = np.random.random(len_non)
        prob = 1 - (1 - self.p) * (1 - self.q) ** influ
        upda = rand <= prob #generate sequences to be updated
        self.num_of_adopt.append(np.sum(upda)) #the number of adopter in this step

        for i in self.non_adopt_set[upda]: #update states
            self.G.node[i]['state'] = True

        self.non_adopt_set = self.non_adopt_set[rand>prob]  #update the set of nodes having state 'False'

    def diffuse(self):
        self.single_diffuse() #inital diffuse
        for i in xrange(self.steps - 1):
            self.shuffle_network() #shuffle the network before updating states
            self.single_diffuse()

        return self.num_of_adopt

if __name__ == '__main__':
    G = nx.gnm_random_graph(50000, 150000)
    p_list0,p_list1,p_list2 = 0,0.5,1
    p, q = 0.001, 0.1
    steps = 25
    t1 = time.clock()
    ab_diffuse1 = agent_diffuse(G, p_list0, p, q, steps)
    diff1 = ab_diffuse1.diffuse()

    ab_diffuse2 = agent_diffuse(G, p_list1, p, q, steps)
    diff2 = ab_diffuse2.diffuse()

    ab_diffuse3 = agent_diffuse(G, p_list2, p, q, steps)
    diff3 = ab_diffuse3.diffuse()

    print u'耗时:%.1f秒'%(time.clock()-t1)
    plt.plot(diff1, 'b-', lw=2)
    plt.plot(diff2, 'r-', lw=2)
    plt.plot(diff3, 'g-', lw=2)
    plt.xlabel('T')
    plt.ylabel('Number of adopters')
    plt.show()