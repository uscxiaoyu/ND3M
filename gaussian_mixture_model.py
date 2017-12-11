#coding=utf-8
from __future__ import division
from numpy import exp,pi,sqrt,log
from scipy.optimize import minimize, root
from copy import deepcopy
import numpy as np
import time

class Gener_PK:
    c = 1.2

    def __init__(self, d=6, g=0, k_list=np.arange(1,50)):
        self.d = d
        self.g = g
        self.k_list = k_list

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
        elif self.g == 0:
            p_k = np.zeros_like(self.k_list)
            p_k[int(self.d) - 1] = 1 + int(self.d) - self.d
            p_k[int(self.d)] = self.d - int(self.d)
        return p_k


class Random_Grid_Search:

    def __init__(self, s, m, p_list, k_list=np.arange(1,50), t_n=500):  # 初始化实例参数
        self.s, self.s_len = np.array(s), len(s)
        self.m = m
        self.k_list = k_list
        self.para_range = [[1e-6, 0.1], [1e-4, 0.9], [0.1, 0.9]]  # p, q, c 参数范围
        self.p_list = p_list  # 各网络的权重
        self.t_n = t_n
        self.pk_cont = []
        for i in [0, 1, 2, 3]:
            gener_pk = Gener_PK(d=6, g=i, k_list=self.k_list)
            pk = gener_pk.get_pk()
            self.pk_cont.append(pk)

    def gener_orig(self, p_range, or_points):  # 递归产生边界点
        if not p_range:  # 如果p_range中已没有参数范围
            return or_points
        else:
            pa = p_range.pop()
            if not or_points:  # 如果or_points中无参数
                or_points = [[pa[0]], [pa[1]]]  # 初始化,排除orig_points为空的情形
            else:
                or_points = [[pa[0]] + x for x in or_points] + [[pa[1]] + x for x in or_points]  # 二分裂
            return self.gener_orig(p_range, or_points)

    def sample(self, c_range):  # 抽样
        p_list = [(pa[1] - pa[0]) * np.random.random(self.t_n) + pa[0] for pa in c_range]
        p_list = np.array(p_list).T
        return p_list.tolist()

    def func(self, g, x):
        pk = self.pk_cont[g]
        p, q, c = x  # c为最终累积扩散率
        inst_diff = np.zeros(self.s_len, dtype=np.float64)  # 非累积扩散率
        f = np.zeros_like(self.k_list, dtype=np.float64)
        theta = 0
        for i in range(self.s_len):
            delta_f = c * (1 - f) * (1 - (1 - p) * (1 - q) ** (self.k_list * theta))  # 各k对应的采纳率增长
            inst_diff[i] = np.dot(delta_f, pk)  # 添加i+1时间步下的总采纳率增长
            f = f + delta_f  # 各k对应的采纳率
            theta = np.sum(self.k_list * pk * f) / np.dot(pk, self.k_list)  # 计算平均影响率
        return inst_diff

    def mix_func(self, x):
        diff_cont = np.array([self.func(g, x) for g in range(4)])
        return np.average(diff_cont, axis=0, weights=self.p_list)  # 期望扩散率

    def neg_loglike(self, x):  # S为现实扩散数据, m为群体数量
        ins = self.mix_func(x)
        if sum(ins) <= 0 or sum(ins) >= 1:
            return np.concatenate(([np.inf], x))
        else:
            y = (self.m - sum(self.s)) * log(1 - sum(ins)) + np.dot(self.s, log(ins))
            return np.concatenate(([-y], x))

    def optima_search(self, c_n=100, threshold=1e-8):  # 搜索过程
        orig_points = self.gener_orig(deepcopy(self.para_range), or_points=[])  # 生成初始搜索点
        samp = self.sample(self.para_range)
        temp = np.apply_along_axis(self.neg_loglike, axis=1, arr=samp+orig_points)
        solution = sorted(temp.tolist())[:c_n]
        for u in range(100):
            par_min, par_max = np.nanmin(solution, axis=0)[1:], np.nanmax(solution, axis=0)[1:]  # 最小、最小值
            c_range = [[par_min[j], par_max[j]] for j in range(par_min.size)]  # 重新定界
            samp = self.sample(c_range)
            temp = np.apply_along_axis(self.neg_loglike, axis=1, arr=samp)  # 生成
            solution = sorted(temp.tolist() + solution)[:c_n]
            r = solution[:][0]
            v = (r[-1] - r[0]) / r[0]
            if v < threshold:  # 终止条件1：达到阈值终止
                break
        else: # 终止条件2：搜索轮次大于100
            print 'Searching ends in 100 runs'

        return solution[0]  # neg_loglike, p, q, c


class GMM:  # gaussian mixture model
    def __init__(self, s, m, k_list=np.arange(1, 50)):
        self.s = s
        self.s_len = len(s)
        self.m = m
        self.k_list = k_list
        self.pk_cont = []
        for i in [0, 1, 2, 3]:
            gener_pk = Gener_PK(d=6, g=i, k_list=self.k_list)
            pk = gener_pk.get_pk()
            self.pk_cont.append(pk)

    def func(self, g, x):
        p, q, c = x  # c为最终累积扩散率
        pk = self.pk_cont[g]
        inst_diff = np.zeros(self.s_len, dtype=np.float64)  # 非累积扩散率
        f = np.zeros_like(self.k_list, dtype=np.float64)
        theta = 0
        for i in range(self.s_len):
            delta_f = c * (1 - f) * (1 - (1 - p) * (1 - q) ** (self.k_list * theta))  # 各k对应的采纳率增长
            inst_diff[i] = np.dot(delta_f, pk)  # 添加i+1时间步下的总采纳率增长
            f = f + delta_f  # 各k对应的采纳率
            theta = np.sum(self.k_list * pk * f) / np.dot(pk, self.k_list)  # 计算平均影响率
        return inst_diff

    def mix_func(self, p_list, x):
        diff_cont = np.array([self.func(g, x) for g in range(4)])
        return np.average(diff_cont, axis=0, weights=p_list)  # 期望扩散率

    def neg_loglike(self, p_list, params=[0.005, 0.05, 0.8]):  # S为现实扩散数据,m为群体数量
        ins = self.mix_func(p_list, x=params)
        if sum(ins) <= 0 or sum(ins) >= 1:
            return np.inf
        else:
            return - (self.m - sum(self.s)) * log(1 - sum(ins)) - np.dot(self.s, log(ins))

    def excep_max(self, threshold=1e-8):
        p_list = np.array([0.25, 0.25, 0.25, 0.25])  # 初始化
        # 定义约束条件
        cons = ({'type':'eq', 'fun':lambda x:np.sum(x)-1},
                {'type':'ineq', 'fun':lambda x:x[0]},
                {'type':'ineq', 'fun':lambda x:1 - x[0]},
                {'type':'ineq', 'fun':lambda x:x[1]},
                {'type':'ineq', 'fun':lambda x:1 - x[1]},
                {'type':'ineq', 'fun':lambda x:x[2]},
                {'type':'ineq', 'fun':lambda x:1 - x[2]},
                {'type':'ineq', 'fun':lambda x:x[3]},
                {'type':'ineq', 'fun':lambda x:1 - x[3]})

        rgs = Random_Grid_Search(self.s, self.m, p_list, k_list=self.k_list, t_n=500)
        flag = 1
        run = 1
        res_cont = []
        while flag > threshold:
            # Exceptation
            rgs.p_list = p_list
            res1 = rgs.optima_search(threshold=threshold)
            res_cont.append(res1 + [list(p_list)])
            params = res1[1:]
            # Maximization
            res2 = minimize(self.neg_loglike, p_list, args=(params,), constraints=cons, method='SLSQP')
            p_list = np.array(res2.x)
            res_cont.append([self.neg_loglike(p_list, params=params)] + params + [list(p_list)])
            flag = abs(res_cont[-1][0] - res_cont[-2][0]) / res_cont[-1][0]
            print '%d: {negtive loglikelihood:%.4f, flag:%.4e}' % (run, res_cont[-1][0], flag)
            run += 1
        return res_cont[-1]

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

    t1 = time.clock()
    s = data_set['clothers dryers'][1]
    m = np.sum(s) * 2
    k_list = np.arange(1, 50)
    ini_values = [0.001, 0.1, 0.5, 6, 0]
    gmm = GMM(s, m)
    res = gmm.excep_max(threshold=1e-7)
    print u'完成，一共用时%d秒'%(time.clock() - t1)
