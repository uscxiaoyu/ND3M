#coding=gbk
import sys
sys.path.append('/Users/xiaoyu/PycharmProjects/ND3M')
from gen_network import *
import time

def _diffuse_(G,p,q,num_of_run=25):
    if not G.is_directed():
        G = G.to_directed()

    for i in G.nodes_iter():
        G.node[i]['state'] = False
        G.node[i]['prede'] = G.predecessors(i)

    non_set = np.array(G.nodes())
    num_of_adopt = []
    for u in xrange(num_of_run):
        #��ȡ���ڵ��Ѳ����ھ�����
        len_non = len(non_set)
        influ = np.zeros(len_non)
        for i in xrange(len_non):
            influ[i] = len([k for k in G.node[non_set[i]].get('prede',[]) if G.node[k]['state']])

        #��ȡ��ʱ�䲽����������
        prob = 1-(1-p)*(1-q)**influ
        rand = np.random.random(len_non)
        upda = rand<=prob
        num_of_adopt.append(np.sum(upda))
        #���±�ʱ�䲽�Ѳ��ɽڵ���״̬
        for i in non_set[upda]:
            G.node[i]['state'] = True

        non_set = non_set[rand>prob] #����δ���ɽڵ㼯��
    return num_of_adopt

# #### 2. ���ɷֲ�
# #### p:[0.001,0.021 ], q:[0.04,0.15]   21*22
n = 10000
g = 'expon'
d = 6
k_list = np.arange(1,100)
g_graph = gener_random_graph(n,d,k_list,g)
G = g_graph.generate()
#plt.plot(g_graph.k_list,g_graph.pk_list,'o')

to_save = []
for p in np.arange(0.001,0.02,0.001):
    t1 = time.clock()
    for q in np.arange(0.01,0.1,0.005):
        diff_cont = []
        for i in xrange(20):
            diff = _diffuse_(G,p,q,num_of_run=40)
            diff_cont.append(diff)

        mean_diff = np.mean(diff_cont,axis=0)
        to_save.append(np.concatenate(([p,q],mean_diff)))

    print 'p:%s,time elapsed:%.2f s'%(p,time.clock()-t1)
    print '================================'

np.save('C:\Users\XIAOYU\PycharmProjects\ND3M\%s_diff'%g,to_save)
