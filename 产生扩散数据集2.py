#coding=utf-8
from gen_network import *
import networkx as nx
import time
import matplotlib.pyplot as plt


def _diffuse_(g, p, q, num_of_run=25):
    if not g.is_directed():
        g = g.to_directed()
        
    for i in g:
        g.node[i]['state'] = False
        
    non_set = np.array(g.nodes())
    num_of_adopt = []
    for u in range(num_of_run):
        #获取各节点已采纳邻居数量
        len_non = len(non_set)
        dose = np.zeros(len_non)
        for i in range(len_non):
            dose[i] = np.sum([1 for k in g.predecessors(non_set[i]) if g.node[k]['state']])
        #获取本时间步采纳者数量
        prob = 1 - (1 - p) * (1 - q) ** dose
        rand = np.random.random(len_non)
        upda = rand<=prob
        num_of_adopt.append(np.sum(upda))
        #更新本时间步已采纳节点的状态
        for i in non_set[upda]:
            g.node[i]['state'] = True
        
        non_set = non_set[rand > prob] #更新未采纳节点集合
    return num_of_adopt


# ### 产生随机网络
# #### 2. 幂律分布
# #### p:[0.001,0.021 ], q:[0.04,0.15]   21\*22
'''
n = 10000
g = 'logno'
d = 6
k_list = np.arange(1,100)
g_graph = gener_random_graph(n,d,k_list,g)
G = g_graph.generate()
print(nx.number_of_nodes(G), nx.number_of_edges(G))
#plt.plot(g_graph.k_list,g_graph.pk_list,'o')


to_save = []
for p in np.arange(0.001,0.02,0.001):
    t1 = time.clock()
    for q in np.arange(0.015,0.105,0.005):
        diff_cont = []
        for i in range(20):
            diff = _diffuse_(G,p,q,num_of_run=40)
            diff_cont.append(diff)

        mean_diff = np.mean(diff_cont,axis=0)
        to_save.append(np.concatenate(([p,q],mean_diff)))
        
    print('p:%s,time elapsed:%.2f s'%(p,time.clock()-t1))
    print('================================')

np.save('logno_diff', to_save)
'''
g = nx.barabasi_albert_graph(10000, 3)
p, q = 0.001, 0.05
diff_cont = []
t1 = time.clock()
for i in range(10):
    diff = _diffuse_(g, p, q, num_of_run=25)
    diff_cont.append(diff)

mean_diff = np.mean(diff_cont, axis=0)
print('p:%s,q:%s  time elapsed:%.2f s' % (p, q, time.clock() - t1))
print(u'最大值位置：%s' % np.argmax(mean_diff))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
for x in diff_cont:
    ax.scatter(range(25), x, s=5, alpha=0.5)

ax.plot(mean_diff, 'b-', label='Asyn', alpha=0.8, lw=3)
ax.set_xlabel('Time step')
ax.set_ylabel('Number of adopters')
ax.set_xlim([0, 25])
ax.set_ylim([0, np.max(diff_cont)*1.2])
ax.legend(loc='best')
plt.show()
