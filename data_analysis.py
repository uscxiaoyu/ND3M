#coding=utf-8
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

f1 = open('forecast result.pkl')
result_1 = pickle.load(f1)

power_dict = {'one_step':[], 'mul_step':[]}
logno_dict = {'one_step':[], 'mul_step':[]}
expon_dict = {'one_step':[], 'mul_step':[]}
gauss_dict = {'one_step':[], 'mul_step':[]}

dict_cont = [gauss_dict, expon_dict, power_dict, logno_dict]
g_cont = ['gauss', 'expon', 'power', 'logno']

for x in result_1:
    i = g_cont.index(x[0][-1])
    dict_cont[i]['one_step'].append(list(x[1][0]))
    dict_cont[i]['mul_step'].append(list(x[1][1]))

for i, g in enumerate(g_cont):
    one_ = np.array(dict_cont[i]['one_step'])
    mul_ = np.array(dict_cont[i]['mul_step'])
    print '========%s========'%g
    print '1 step ahead prediction:'
    print '   MAD: %.2f, MPAD: %.2f, MSE: %.2f' % tuple(np.nanmean(one_, axis=0))
    print '5 steps ahead prediction:'
    print '   MAD: %.2f, MPAD: %.2f, MSE: %.2f' % tuple(np.nanmean(mul_, axis=0))
    print
