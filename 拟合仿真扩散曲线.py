#coding=gbk
import sys
sys.path.append('/Users/xiaoyu/PycharmProjects/ND3M')
from discrete_forecasting import *
import time

data_set = np.load('power_diff.npy')
for s in data_set:
    S = s[2:]
    p,q = s[:2]
    d = 6
    t1 = time.clock()
    rgs = Random_Grid_Search(S, d, g='gauss')
    est = rgs.optima_search()
    params = est[1:]
    r2 = rgs.r2(params)
    print '============================================='
    print '    Time elasped:', time.clock() - t1, 's'
    print '    p:%s, q:%s'%(p,q)
    print '    P:%.4f, Q:%.4f, M:%d'%params
    print '    r2:%.4f' %r2

    #绘图
    x = rgs.f(params)
    pl.text(0.2, max(S)*0.9, '$R^2=%.4f$'%r2)
    pl.plot(x, 'k-', lw=2, alpha=0.7, label='estimation')
    pl.plot(S, 'ro', ms=5, alpha=0.5, label='sales')
    pl.xlabel('Year')
    pl.ylabel('Sales')
    pl.legend(loc=4)
    pl.show()