#coding=utf-8
from discrete_forecasting import *
import multiprocessing

def func(g, s, n, b_idx):
    fore = Forecast(s, b_idx, n, g)
    res = fore.run()
    print 'Model:%s finished' % g
    return res


if __name__ == "__main__":
    data_set = {'room air conditioners': (np.arange(1949, 1962), [96, 195, 238, 380, 1045, 1230, 1267, 1828, 1586, 1673, 1800, 1580, 1500]),
                'color televisions': (np.arange(1963, 1971), [747, 1480, 2646, 5118, 5777, 5982, 5962, 4631]),
                'clothers dryers': (np.arange(1949, 1962), [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425, 1260, 1236]),
                'ultrasound': (np.arange(1965, 1979), [5, 3, 2, 5, 7, 12, 6, 16, 16, 28, 28, 21, 13, 6]),
                'mammography': (np.arange(1965, 1979), [2, 2, 2, 3, 4, 9, 7, 16, 23, 24, 15, 6, 5, 1]),
                'foreign language': (np.arange(1952, 1964), [1.25, 0.77, 0.86, 0.48, 1.34, 3.56, 3.36, 6.24, 5.95, 6.24, 4.89, 0.25]),
                'accelerated program': (np.arange(1952, 1964), [0.67, 0.48, 2.11, 0.29, 2.59, 2.21, 16.80, 11.04, 14.40, 6.43, 6.15, 1.15])}

    pool = multiprocessing.Pool(processes=4)
    tx = 'clothers dryers'
    g_cont = ['expon', 'power', 'gauss', 'logno']
    s = data_set[tx][1]
    n, b_idx = 3, 8

    result = []
    t = time.clock()
    Random_Grid_Search.t_n, Random_Grid_Search.s_n = 800, 200
    for i, g in enumerate(g_cont):
        result.append(pool.apply_async(func, (g, s, n, b_idx)))

    pool.close()
    pool.join()

    print u'共耗时%ds'%(time.clock() - t)
    print "Sub-process(es) done."
    for i, res in enumerate(result):
        print "Model:%s"%g_cont[i], res.get()