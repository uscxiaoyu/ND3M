# -*- coding: utf-8 -*-
from __future__ import division
from copy import deepcopy as dc
from numpy import exp
import numpy as np
import time


class Bass_Estimate:
    t_n = 500  # 抽样量
    c_n = 100  # 保留参数量
    threshold = 1e-4  # 循环停止阈值
    orig_points = []  # 初始化边界点
    def __init__(self, s, para_range):  # 初始化实例参数
        self.s, self.s_len = np.array(s),len(s)
        self.para_range = para_range  # 参数范围
        self.p_range = dc(self.para_range)  # 用于产生边界节点的参数范围

    def gener_orig(self):  # 递归产生边界点
        if len(self.p_range) == 0:
            return
        else:
            pa = self.p_range[-1]
            if self.orig_points == []:
                self.orig_points = [[pa[0]], [pa[1]]]  # 初始化,排除orig_points为空的情形
            else:
                self.orig_points = [[pa[0]] + x for x in self.orig_points] + [[pa[1]] + x for x in self.orig_points]  # 二分裂
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

    def f(self, params): # 如果要使用其它模型，可以重新定义
        p, q, m = params
        t_list = np.arange(1, self.s_len + 1)
        a = np.array([1 - exp(- (p + q) * t) for t in t_list])
        b = np.array([1 + q / p * exp(- (p + q) * t) for t in t_list])
        diffu_cont = m * a / b

        adopt_cont = np.zeros_like(diffu_cont)
        adopt_cont[0] = diffu_cont[0]
        for t in xrange(1, self.s_len):
            adopt_cont[t] = diffu_cont[t] - diffu_cont[t - 1]
        return adopt_cont

    def mse(self, params):  # 定义适应度函数（mse）
        a = self.f(params)
        sse = np.sum(np.square(self.s - a))
        return np.sqrt(sse) / self.s_len  # 均方误

    def r2(self, params):  # 求R2
        f_act = self.f(params)
        tse = np.sum(np.square(self.s - f_act))
        mean_y = np.mean(self.s)
        ssl = np.sum(np.square(self.s - mean_y))
        R_2 = (ssl-tse)/ssl
        return R_2

    def optima_search(self):
        self.gener_orig()   #产生边界节点
        c_range = dc(self.para_range)
        samp = self.sample(c_range)
        solution = sorted([self.mse(x)]+x for x in samp+self.orig_points)[:self.c_n]
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
        r2 = self.r2(solution[0][1:])
        result = solution[0][1:]+[r2]

        return result  # p,q,m,r2


class Bass_forecast:
    def __init__(self, s, n, b_idx):
        self.s = s
        self.n = n
        self.s_len = len(s)
        self.b_idx = b_idx

    def f(self, params, T): # 如果要使用其它模型，可以重新定义
        p, q, m = params
        t_list = np.arange(1, T + 1)
        a = np.array([1 - exp(- (p + q) * t) for t in t_list])
        b = np.array([1 + q / p * exp(- (p + q) * t) for t in t_list])
        diffu_cont = m * a / b

        adopt_cont = np.zeros_like(diffu_cont)
        adopt_cont[0] = diffu_cont[0]
        for t in xrange(1, T):
            adopt_cont[t] = diffu_cont[t] - diffu_cont[t - 1]

        return adopt_cont

    def predict(self):
        pred_cont = []
        for i in xrange(self.s_len - 1 - self.b_idx):  # 拟合次数
            idx = self.b_idx + 1 + i
            x = self.s[: idx]
            para_range = [[1e-5, 0.1], [1e-5, 0.8], [sum(x), 15 * sum(x)]]
            bass_est = Bass_Estimate(x, para_range)
            est = bass_est.optima_search()
            params = est[:3]  # est: p, q, m, r2
            pred_s = self.f(params, self.s_len)
            pred_cont.append(pred_s[idx:])

        self.pred_res = pred_cont

    def one_step_ahead(self):
        pred_cont = np.array([x[0] for x in self.pred_res])
        mad = np.mean(np.abs(pred_cont - self.s[self.b_idx + 1:]))
        mape = np.mean(np.abs(pred_cont - self.s[self.b_idx + 1:]) / self.s[self.b_idx + 1:])
        mse = np.mean(np.sqrt(np.sum(np.square(pred_cont - self.s[self.b_idx + 1:]))))

        return mad, mape, mse

    def n_step_ahead(self):
        pred_cont = np.array([x[: self.n] for x in self.pred_res if self.n <= len(x)])
        act_cont = np.array([self.s[self.b_idx + i : self.b_idx + i + self.n] for i in range(self.s_len - self.b_idx - self.n)])
        mad = np.mean(np.abs(pred_cont - act_cont))
        mape = np.mean(np.abs(pred_cont - act_cont) / act_cont)
        mse = np.mean(np.sqrt(np.sum(np.square(pred_cont - act_cont))))

        return mad, mape, mse

    def run(self):
        self.predict()
        one_cont = self.one_step_ahead()
        n_cont = self.n_step_ahead()
        return [one_cont, n_cont]


if __name__=='__main__':
    data_set = {'room air conditioners': (np.arange(1949, 1962), [96, 195, 238, 380, 1045, 1230, 1267, 1828, 1586, 1673,
                                                                  1800, 1580, 1500]),
                'color televisions': (np.arange(1963, 1971), [747, 1480, 2646, 5118, 5777, 5982, 5962, 4631]),
                'clothers dryers': (np.arange(1949, 1962), [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425,
                                                            1260, 1236]),
                'ultrasound': (np.arange(1965, 1979), [5, 3, 2, 5, 7, 12, 6, 16, 16, 28, 28, 21, 13, 6]),
                'mammography': (np.arange(1965, 1979), [2, 2, 2, 3, 4, 9, 7, 16, 23, 24, 15, 6, 5, 1]),
                'foreign language': (np.arange(1952, 1964), [1.25, 0.77, 0.86, 0.48, 1.34, 3.56, 3.36, 6.24, 5.95, 6.24,
                                                             4.89, 0.25]),
                'accelerated program': (np.arange(1952, 1964), [0.67, 0.48, 2.11, 0.29, 2.59, 2.21, 16.80, 11.04, 14.40,
                                                                6.43, 6.15, 1.15])}

    china_set = {'color tv': (np.arange(1997, 2013), [2.6, 1.2, 2.11, 3.79, 3.6, 7.33, 7.18, 5.29, 8.42, 5.68, 6.57,
                                                      5.49, 6.48, 5.42, 10.72, 5.15]),
                 'mobile phone': (np.arange(1997, 2013), [1.7, 1.6, 3.84, 12.36, 14.5, 28.89, 27.18, 21.33, 25.6, 15.88,
                                                          12.3, 6.84, 9.02, 7.82, 16.39, 7.39])}

    S = data_set['room air conditioners'][1]

    '''
    para_range = [[1e-5, 0.1], [1e-5, 0.8], [sum(S), 10 * sum(S)]]
    bassest = Bass_Estimate(S, para_range)

    bassest.c_n = 60
    bass_estimates = bassest.optima_search()
    print 'p:%.4f   q:%.4f   m:%d   r2:%.4f' % tuple(bass_estimates)
    '''

    bass_fore = Bass_forecast(S, n=3, b_idx=8)
    res = bass_fore.run()

    print u'1步向前预测:',
    print 'MAD:%.2f  MAPE:%.2f  MSE:%.2f' % res[0]
    print u'3步向前预测:',
    print 'MAD:%.2f  MAPE:%.2f  MSE:%.2f' % res[1]
