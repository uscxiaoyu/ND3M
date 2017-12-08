#coding=utf-8
from __future__ import division
from numpy import exp,pi,sqrt,log
from scipy.optimize import root
from copy import deepcopy as dc
from random import random
import numpy as np
import time

class Gener_PK:
    c = 1.2
    def __init__(self, d=6, g=0, k_list = np.arange(1, 50)):
        self.d = d
        self.g = g
        self.k_list = k_list
        self.samp_cont = []

    def gauss(self, x):
        pk_list = self.c * exp(-(self.k_list - x[0]) ** 2 / (2 * x[1] ** 2)) / (x[1] * sqrt(2 * pi))
        return np.sum(pk_list) - 1, np.dot(pk_list, self.k_list) - self.d

    def logno(self, x):
        pk_list = self.c * exp(-(log(self.k_list) - x[0]) ** 2 / (2 * x[1] ** 2)) / (self.k_list * x[1] * sqrt(2 * pi))
        return np.sum(pk_list) - 1, np.dot(pk_list, self.k_list) - self.d

    def expon(self, x):
        pk_list = x[1] * exp(-self.k_list * x[0])
        return np.sum(pk_list) - 1, np.dot(pk_list, self.k_list) - self.d

    def power(self, x):
        pk_list = x[1] * self.k_list ** (-x[0])
        return np.sum(pk_list) - 1, np.dot(pk_list, self.k_list) - self.d

    def get_pk(self):
        if self.g == 1:
            sol = root(self.gauss, [6, 1], args=(), method='lm')
            p_k = self.c * exp(-(self.k_list - sol.x[0]) ** 2 / (2 * sol.x[1] ** 2)) / (sol.x[1] * sqrt(2 * pi))
        elif self.g == 2:
            sol = root(self.logno, [5, 2], args=(), method='lm')
            p_k = self.c * exp(-(log(self.k_list) - sol.x[0]) ** 2 / (2 * sol.x[1] ** 2)) / (
            self.k_list * sol.x[1] * sqrt(2 * pi))
        elif self.g == 3:
            sol = root(self.expon, [0.5, 1.1], args=(), method='lm')
            p_k = sol.x[1] * exp(-self.k_list * sol.x[0])
        else:
            p_k = np.zeros_like(self.k_list)
            p_k[int(self.d) - 1] = 1 + int(self.d) - self.d
            p_k[int(self.d)] = self.d - int(self.d)
        return p_k

class Gibbs_samp:
    def __init__(self, s, m, ini_values, k_list=np.arange(1,50)):
        self.s = s
        self.s_len = len(s)
        self.m = m
        self.k_list = k_list
        self.ini_values = ini_values
        self.samp_cont = [] #保存样本

    def func(self, x):
        p, q, c, d, g = x
        gener_pk = Gener_PK(d=d, g=g, k_list=self.k_list)
        Pk = gener_pk.get_pk()

        insta_diff = np.zeros(self.s_len)  # 区间扩散率
        F = np.zeros_like(self.k_list)
        theta = 0
        for i in range(self.s_len):
            delta_F = c * (1 - F) * (1 - (1 - p) * (1 - q) ** (self.k_list * theta))  # 各k对应的采纳率增长
            insta_diff[i] = np.dot(delta_F, Pk)  # 添加i+1时间步下的总采纳率增长
            F = F + delta_F  # 各k对应的采纳率
            theta = np.sum(self.k_list * Pk * F) / np.dot(Pk, self.k_list)  # 计算平均影响率

        return insta_diff

    def loglike(self, x):  # S为现实扩散数据,m为群体数量
        ins = self.func(x)
        if sum(ins) <= 0 or sum(ins) >= 1:
            return -np.inf
        else:
            return (m - sum(self.s)) * log(1 - sum(ins)) + np.dot(self.s, log(ins))

    def sample(self, num_samp=10000):
        t1 = time.clock()
        t2 = time.clock()
        p, q, c, d, g = self.ini_values
        params = [p, q, c, d, g]
        likel = self.loglike(self.ini_values)
        for i in xrange(num_samp):
            u_p = 0.0001 + 0.08 * random()
            u_q = 0.001 + 0.4 * random()
            u_c = 0.2 + 0.8 * random()
            u_g = np.random.choice([0, 1, 2, 3])  # 更新网络种类
            u_d = 3 + 9 * random()

            u_cont = [u_p, u_q, u_c, u_g, u_d]  # 待更新内容
            _params = params[:]
            for i, u in enumerate(u_cont):
                _params[i] = u
                u_likel = self.loglike(_params)
                if exp(u_likel - likel) > random():  # loglikelihood转化为likelihood,likehood = e**(loglikelihood)
                    params[i] = u
                    likel = dc(u_likel)

            self.samp_cont.append(params)
            if time.clock() - t1 > 600:
                print u'[%-20s]耗时%d秒' % (int(i / num_samp * 20) * '>', time.clock()-t2)
                t1 = time.clock()

if __name__ == '__main__':
    data_set = {'room air conditioners': (np.arange(1949, 1962), [96, 195, 238, 380, 1045, 1230, 1267, 1828, 1586, 1673, 1800, 1580, 1500]),
                'color televisions': (np.arange(1963, 1971), [747, 1480, 2646, 5118, 5777, 5982, 5962, 4631]),
                'clothers dryers': (np.arange(1949, 1962), [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425, 1260, 1236]),
                'ultrasound': (np.arange(1965, 1979), [5, 3, 2, 5, 7, 12, 6, 16, 16, 28, 28, 21, 13, 6]),
                'mammography': (np.arange(1965, 1979), [2, 2, 2, 3, 4, 9, 7, 16, 23, 24, 15, 6, 5, 1]),
                'foreign language': (np.arange(1952, 1964), [1.25, 0.77, 0.86, 0.48, 1.34, 3.56, 3.36, 6.24, 5.95, 6.24, 4.89, 0.25]),
                'accelerated program': (np.arange(1952, 1964), [0.67, 0.48, 2.11, 0.29, 2.59, 2.21, 16.80, 11.04, 14.40, 6.43, 6.15, 1.15])}
    china_set = {'color tv': (np.arange(1997, 2013),[2.6, 1.2, 2.11, 3.79, 3.6, 7.33, 7.18, 5.29, 8.42, 5.68, 6.57, 5.49, 6.48, 5.42, 10.72, 5.15]),
                 'mobile phone': (np.arange(1997, 2013), [1.7, 1.6, 3.84, 12.36, 14.5, 28.89, 27.18, 21.33, 25.6, 15.88, 12.3, 6.84, 9.02, 7.82, 16.39, 7.39])}

    k_list = np.arange(1,50)
    ini_values = [0.001, 0.1, 0.5, 6, 0]
    s = data_set['room air conditioners'][1]
    m = 20000
    t1 = time.clock()
    g_samp = Gibbs_samp(s=s, m=m, ini_values=ini_values, k_list=k_list)
    g_samp.sample(num_samp = 5000000)
    np.save('gibbs(room air conditioners)', g_samp.samp_cont)
    print u'完成，一共用时%d秒'%(time.clock() - t1)

    '''
        k_list = np.arange(1, 50)
        ini_values = [0.001, 0.1, 0.5, 6, 0]
        key_cont = ['room air conditioners', 'color televisions', 'clothers dryers']
        m_cont = [20000, 40000, 30000]

        for i, txt in enumerate(key_cont):
            print txt
            s = data_set[txt][1]
            m = m_cont[i]
            t1 = time.clock()
            g_samp = Gibbs_samp(s=s, m=m, ini_values=ini_values, k_list=k_list)
            g_samp.sample(num_samp = 500000)
            np.save('gibbs_%s'%txt, g_samp.samp_cont)
            print u'完成，一共用时%d秒'%(time.clock() - t1)
    '''
