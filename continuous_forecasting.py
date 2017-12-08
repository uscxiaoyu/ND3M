#coding=gbk
from __future__ import division
from scipy.integrate import odeint
from scipy.optimize import minimize,root,fixed_point
from copy import deepcopy as dc
from math import e
from numpy import exp,pi,sqrt,log
import time
import numpy as np
import matplotlib.pyplot as pl
pl.rcParams.update({'font.size': 15, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})


class Gener_PK:
    k_list = np.arange(1, 100)
    c = 1.2

    def __init__(self, d=6, g='power'):
        self.d = d
        self.g = g

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
        pk_list = x[1] * self.k_list ** (-x[0])
        return np.sum(pk_list) - 1, np.dot(pk_list, self.k_list) - self.d

    def get_pk(self):
        if self.g == 'gauss':
            sol = root(self.gauss, [6, 1], args=(), method='lm')
            p_k = self.c * e ** (-(self.k_list - sol.x[0]) ** 2 / (2 * sol.x[1] ** 2)) / (sol.x[1] * sqrt(2 * pi))
        elif self.g == 'logno':
            sol = root(self.logno, [5, 2], args=(), method='lm')
            p_k = self.c * e ** (-(log(self.k_list) - sol.x[0]) ** 2 / (2 * sol.x[1] ** 2)) / (
            self.k_list * sol.x[1] * sqrt(2 * pi))
        elif self.g == 'expon':
            sol = root(self.expon, [0.5, 1.1], args=(), method='lm')
            p_k = sol.x[1] * e ** (-self.k_list * sol.x[0])
        else:
            sol = root(self.power, [1, 1], args=(), method='lm')
            p_k = sol.x[1] * self.k_list ** (-sol.x[0])

        return p_k


class Random_Grid_Search:
    t_n = 500  # 抽样量
    c_n = 50  # 保留参数量
    threshold = 1e-4  # 循环停止阈值
    orig_points = []  # 初始化边界点
    k_list = np.arange(1, 100)

    def __init__(self, s, d=6, g='power'):  # 初始化实例参数
        self.s, self.s_len = np.array(s), len(s)
        self.d, self.g = d, g
        self.T = np.arange(self.s_len + 1)  # 包括0时刻扩散量
        self.F = np.zeros_like(self.k_list)
        get_pk = Gener_PK(self.d)
        get_pk.g = self.g
        self.p_k = get_pk.get_pk()
        self.para_range = [[1e-6, 0.1], [1e-3 / d, 1 / d], [sum(s), 5 * sum(s)]]  # 参数范围
        self.p_range = [[1e-6, 0.1], [1e-3 / d, 1 / d], [sum(s), 5 * sum(s)]]  # 用于产生边界节点的参数范围

    def gener_orig(self):  # 递归产生边界点
        if len(self.p_range) == 0:
            return
        else:
            pa = self.p_range[-1]
            if self.orig_points == []:
                self.orig_points = [[pa[0]], [pa[1]]]  # 初始化,排除orig_points为空的情形
            else:
                self.orig_points = [[pa[0]] + x for x in self.orig_points] + [[pa[1]] + x for x in
                                                                              self.orig_points]  # 二分裂
            self.p_range.pop()
            return self.gener_orig()

    def sample(self, c_range):  # 抽样参数点
        p_list = []
        for pa in c_range:
            if isinstance(pa[0], float):
                x = (pa[1] - pa[0]) * np.random.random(self.t_n) + pa[0]
            else:
                x = np.random.randint(low=pa[0], high=pa[1] + 1, size=self.t_n)
            p_list.append(x)

        p_list = np.array(p_list).T
        return p_list.tolist()

    def evolve(self, F, t, p, q):
        theta = np.sum(F * self.p_k * self.k_list) / self.d
        return (1 - F) * (p + q * self.k_list * theta)

    def f(self, params):
        p, q, m = params
        t = self.T
        track = m * self.p_k * odeint(self.evolve, self.F, t, args=(p, q))
        accum_diff = np.sum(track, axis=1)
        insta_diff = np.zeros_like(self.s)
        for i in range(self.s_len):
            insta_diff[i] = accum_diff[i + 1] - accum_diff[i]
        return insta_diff

    def r2(self, params):
        f_act = self.f(params)
        tse = np.sum(np.square(self.s - f_act))
        mean_y = np.mean(self.s)
        ssl = np.sum(np.square(self.s - mean_y))
        R_2 = (ssl - tse) / ssl
        return R_2

    def mse(self, params):  # 定义适应度函数（mse）
        a = self.f(params)
        sse = np.sum(np.square(self.s - a))
        return np.sqrt(sse) / self.s_len  # 均方误

    def optima_search(self):
        self.gener_orig()
        c_range = dc(self.para_range)
        samp = self.sample(c_range)
        solution = sorted([self.mse(x)] + x for x in samp + self.orig_points)[:self.c_n]
        u = 1
        while 1:
            params_min = np.min(np.array(solution), 0)  # 最小值
            params_max = np.max(np.array(solution), 0)  # 最大值
            c_range = [[params_min[j + 1], params_max[j + 1]] for j in range(len(c_range))]  # 重新定界
            samp = self.sample(c_range)
            solution = sorted([[self.mse(x)] + x for x in samp] + solution)[:self.c_n]
            r = sorted([x[0] for x in solution])
            v = (r[-1] - r[0]) / r[0]
            if v < self.threshold:
                break
            if u > 100:
                print 'Searching ends in 100 runs'
                break
            u += 1
        return solution[0]  # sse,p,q,m

if __name__ == '__main__':
    data_set = {'room air conditioners': (np.arange(1949, 1962), [96, 195, 238, 380, 1045, 1230, 1267, 1828, 1586, 1673, 1800, 1580, 1500]),
                'color televisions': (np.arange(1963, 1971), [747, 1480, 2646, 5118, 5777, 5982, 5962, 4631]),
                'clothers dryers': (np.arange(1949, 1962), [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425, 1260, 1236]),
                'ultrasound': (np.arange(1965, 1979), [5, 3, 2, 5, 7, 12, 6, 16, 16, 28, 28, 21, 13, 6]),
                'mammography': (np.arange(1965, 1979), [2, 2, 2, 3, 4, 9, 7, 16, 23, 24, 15, 6, 5, 1]),
                'foreign language': (np.arange(1952, 1964), [1.25, 0.77, 0.86, 0.48, 1.34, 3.56, 3.36, 6.24, 5.95, 6.24, 4.89, 0.25]),
                'accelerated program': (np.arange(1952, 1964), [0.67, 0.48, 2.11, 0.29, 2.59, 2.21, 16.80, 11.04, 14.40, 6.43, 6.15, 1.15])}
    china_set = {'color tv': (np.arange(1997, 2013), [2.6, 1.2, 2.11, 3.79, 3.6, 7.33, 7.18, 5.29, 8.42, 5.68, 6.57, 5.49, 6.48, 5.42, 10.72,5.15]),
                 'mobile phone': (np.arange(1997, 2013),[1.7, 1.6, 3.84, 12.36, 14.5, 28.89, 27.18, 21.33, 25.6, 15.88, 12.3, 6.84, 9.02,7.82, 16.39, 7.39])}

    tx = 'clothers dryers'
    S = data_set[tx][1]
    d = 6
    t1 = time.clock()
    rgs = Random_Grid_Search(S, d, g='power')
    est = rgs.optima_search()
    params = est[1:]
    r2 = rgs.r2(params)
    print '    Time elasped:', time.clock() - t1, 's'
    print '    r2:%.4f,    p:%.4f,   q:%.4f,    m:%d' % tuple([r2] + params)

    # 绘图
    x = rgs.f(params)
    year = data_set[tx][0]
    pl.text(min(year) + 0.2, max(S) * 0.9, '$R^2=%.4f$' % r2)
    pl.plot(year, x, 'k-', lw=2, alpha=0.7, label='estimation')
    pl.plot(year, S, 'ro', ms=5, alpha=0.5, label='sales')
    pl.xlabel('Year')
    pl.ylabel('Sales')
    pl.legend(loc=4)
    pl.show()