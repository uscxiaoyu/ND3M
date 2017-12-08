#coding=gbk
import sys
sys.path.append('/Users/xiaoyu/PycharmProjects/ND3M')
from gen_network import *
import time
import matplotlib.pyplot as plt

def _diffuse_(G,p,q,num_of_run=25):
    if not G.is_directed():
        G = G.to_directed()
        
    for i in G.nodes_iter():
        G.node[i]['state'] = False
        G.node[i]['prede'] = G.predecessors(i)
        
    non_set = np.array(G.nodes())
    num_of_adopt = []
    for u in xrange(num_of_run):
        #获取各节点已采纳邻居数量
        len_non = len(non_set)
        influ = np.zeros(len_non)
        for i in xrange(len_non):
            influ[i] = len([k for k in G.node[non_set[i]].get('prede',[]) if G.node[k]['state']])
      
        #获取本时间步采纳者数量
        prob = 1-(1-p)*(1-q)**influ
        rand = np.random.random(len_non)
        upda = rand<=prob
        num_of_adopt.append(np.sum(upda))
        #更新本时间步已采纳节点的状态
        for i in non_set[upda]:
            G.node[i]['state'] = True
        
        non_set = non_set[rand>prob] #更新未采纳节点集合         
    return num_of_adopt


# ### 产生随机网络
# #### 2. 幂律分布
# #### p:[0.001,0.021 ], q:[0.04,0.15]   21\*22
n = 10000
g = 'logno'
d = 6
k_list = np.arange(1,100)
g_graph = gener_random_graph(n,d,k_list,g)
G = g_graph.generate()
print nx.number_of_nodes(G),nx.number_of_edges(G)
#plt.plot(g_graph.k_list,g_graph.pk_list,'o')


to_save = []
for p in np.arange(0.001,0.02,0.001):
    t1 = time.clock()
    for q in np.arange(0.015,0.105,0.005):
        diff_cont = []
        for i in xrange(20):
            diff = _diffuse_(G,p,q,num_of_run=40)
            diff_cont.append(diff)

        mean_diff = np.mean(diff_cont,axis=0)
        to_save.append(np.concatenate(([p,q],mean_diff)))
        
    print 'p:%s,time elapsed:%.2f s'%(p,time.clock()-t1)
    print '================================'

np.save('C:\Users\XIAOYU\PycharmProjects\ND3M\logno_diff',to_save)
'''
p,q = 0.001,0.015
diff_cont = []
t1 = time.clock()
for i in xrange(10):
    diff = _diffuse_(G,p,q,num_of_run=40)
    diff_cont.append(diff)
mean_diff = np.mean(diff_cont,axis=0)
print 'p:%s,q:%s  time elapsed:%.2f s'%(p,q,time.clock()-t1)
print u'最大值位置：%s'%np.argmax(mean_diff)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
for x in diff_cont:
    ax.scatter(np.arange(40),x,s=5,alpha=0.5)

ax.plot(mean_diff,'b-',label='Asyn',alpha=0.8,lw=3)
ax.set_xlabel('Time step')
ax.set_ylabel('Number of adopters')
ax.set_xlim([0,40])
ax.set_ylim([0,np.max(diff_cont)*1.2])
ax.legend(loc='best')
plt.show()
'''