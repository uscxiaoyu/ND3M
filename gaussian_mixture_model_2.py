#for python3
#coding=utf-8
from __future__ import division
from numpy import exp,pi,sqrt,log
from scipy.optimize import minimize, root, differential_evolution
import matplotlib.pyplot as plt
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
            sol = root(self.gauss, np.array([6, 1]), args=(), method='lm')
            p_k = self.c * exp(-(self.k_list - sol.x[0]) ** 2 / (2 * sol.x[1] ** 2)) / (sol.x[1] * sqrt(2 * pi))
        elif self.g == 2:
            sol = root(self.logno, np.array([5, 2]), args=(), method='lm')
            p_k = self.c * exp(-(log(self.k_list) - sol.x[0]) ** 2 / (2 * sol.x[1] ** 2)) / (
            self.k_list * sol.x[1] * sqrt(2 * pi))
        elif self.g == 3:
            sol = root(self.expon, np.array([0.5, 1.1]), args=(), method='lm')
            p_k = sol.x[1] * exp(-self.k_list * sol.x[0])
        elif self.g == 0:
            p_k = np.zeros_like(self.k_list)
            p_k[int(self.d) - 1] = 1 + int(self.d) - self.d
            p_k[int(self.d)] = self.d - int(self.d)
        return p_k


class GMM:  # gaussian mixture model

    def __init__(self, s, m, k_list=np.arange(1, 50)):
        self.s = s
        self.s_len = len(s)
        self.m = m
        self.k_list = k_list
        self.pk_cont = []
        self.p_list = 0.25 * np.ones(4)
        self.x = np.array([0.001, 0.1, 0.5])
        for i in [0, 1, 2, 3]:
            gener_pk = Gener_PK(d=6, g=i, k_list=self.k_list)
            pk = gener_pk.get_pk()
            self.pk_cont.append(pk)

    def func(self, x, g):
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

    def mix_func(self, x, p_list):
        diff_cont = np.array([self.func(x, g) for g in range(4)])
        return np.average(diff_cont, axis=0, weights=p_list)  # 期望扩散率

    def neg_loglike1(self, x):  # 负对数似然函数, 以(p, q, c)作为参数
        ins = self.mix_func(x, self.p_list)
        if sum(ins) <= 0 or sum(ins) >= 1:
            return np.inf
        else:
            return - (self.m - sum(self.s)) * log(1 - sum(ins)) - np.dot(self.s, log(ins))

    def neg_loglike2(self, p_list):  # 负对数似然函数, 以p_list作为参数
        ins = self.mix_func(self.x, p_list)
        if sum(ins) <= 0 or sum(ins) >= 1:
            return np.inf
        else:
            return - (self.m - sum(self.s)) * log(1 - sum(ins)) - np.dot(self.s, log(ins))

    def optima_search1(self):  # 针对(p, q, c)优化
        bounds = ((1e-4, 0.1), (0.001, 1), (0.01, 0.9))
        #sol = minimize(self.neg_loglike1, self.x, bounds=bounds, method='SLSQP')
        sol = differential_evolution(self.neg_loglike1, bounds=bounds)
        self.x = np.array(sol.x)
        return sol.fun  # [neg_loglike, p, q, c]

    def optima_search2(self):  # 针对p_list优化
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x[0]},
                {'type': 'ineq', 'fun': lambda x: 1 - x[0]},
                {'type': 'ineq', 'fun': lambda x: x[1]},
                {'type': 'ineq', 'fun': lambda x: 1 - x[1]},
                {'type': 'ineq', 'fun': lambda x: x[2]},
                {'type': 'ineq', 'fun': lambda x: 1 - x[2]},
                {'type': 'ineq', 'fun': lambda x: x[3]},
                {'type': 'ineq', 'fun': lambda x: 1 - x[3]})

        sol = minimize(self.neg_loglike2, self.p_list, constraints=cons, method='SLSQP')
        self.p_list = np.array(sol.x)
        return sol.fun  # [neg_loglike, p, q, c]

    def excep_max(self, iters=50, threshold=1e-6):
        self.optima_search1()
        res2 = self.optima_search2()
        res_cont = [(res2, self.x, self.p_list)]
        for i in range(iters):
            self.optima_search1()  # Exceptation
            res2 = self.optima_search2()  # Maximization
            res_cont.append((res2, self.x, self.p_list))
            flag = abs(res_cont[-2][0] - res_cont[-1][0]) / res_cont[-1][0]
            print('EM---%d:{%.2f, %.2f}' % (i+1, res_cont[-2][0], res_cont[-1][0]), 'Flag: %.2e' % flag)
            if flag < threshold:
                break
        else:
            print('Iteration exceeds 50!')

        return sorted(res_cont)[0]

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

    m_cont = {'clothers dryers': 15960, 'room air conditioners':17581, 'color televisions':38619}
    t1 = time.clock()
    txt = 'clothers dryers'
    s = data_set[txt][1]
    m = m_cont[txt]
    gmm = GMM(s, m)
    gmm.p_list = 0.25 * np.ones(4)  # 设定p_list的初值
    gmm.x = np.array([0.001, 0.1, 0.15])  # np.array([0.001, 0.05, 0.4]) for room air conditionrs #  设定(p, q, c)的初值
    # 优化
    res = gmm.excep_max(threshold=1e-6)
    print('-Loglikelihood: %.2f' % res[0], 'p:%.4f, q:%.4f, c:%.4f' % tuple(res[1]),
          'Ba: %.4f, Gauss: %.4f, Logno:%.4f, Expon:%.4f' % tuple(res[2]), sep='\n')
    print(u'完成，一共用时%d秒'%(time.clock() - t1))
    '''
    # 绘图
    diff_curve = gmm.mix_func(gmm.x, gmm.p_list) * m
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(diff_curve, 'k-', lw=1.5)
    ax.plot(s, 'ro', ms=8, alpha=0.5)
    plt.show()'''
