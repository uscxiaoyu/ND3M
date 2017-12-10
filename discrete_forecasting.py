#coding=utf-8
from __future__ import division
from scipy.optimize import root
from copy import deepcopy as dc
from numpy import pi,sqrt,log,exp
import time
import numpy as np
import matplotlib.pyplot as pl
pl.rcParams.update({'font.size': 15, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})


class Gener_PK:
    '''
    产生5种网络度分布: gaussian, lognormal, exponential, power-law, one or two point
    '''
    c = 1.2 #度分布函数的默认参数
    def __init__(self, d=6, g='power', k_list=np.arange(1, 50)):
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
        pk_list = x[1] * (1.0 / self.k_list) ** x[0]
        return np.sum(pk_list) - 1, np.dot(pk_list, self.k_list) - self.d

    def get_pk(self):
        if self.g == 'gauss':
            sol = root(self.gauss, [6, 1], args=(), method='lm')
            p_k = self.c * exp(-(self.k_list - sol.x[0]) ** 2 / (2 * sol.x[1] ** 2)) / (sol.x[1] * sqrt(2 * pi))
        elif self.g == 'logno':
            sol = root(self.logno, [5, 2], args=(), method='lm')
            p_k = self.c * exp(-(log(self.k_list) - sol.x[0]) ** 2 / (2 * sol.x[1] ** 2)) / (
            self.k_list * sol.x[1] * sqrt(2 * pi))
        elif self.g == 'expon':
            sol = root(self.expon, [0.5, 1.1], args=(), method='lm')
            p_k = sol.x[1] * exp(-self.k_list * sol.x[0])
        elif self.g == 'power':
            sol = root(self.power, [1, 1], args=(), method='lm')
            p_k = sol.x[1] * (1.0 / self.k_list) ** sol.x[0]
        else:
            p_k = np.zeros_like(self.k_list)
            p_k[int(self.d - 1)] = 1 + int(self.d) - self.d
            p_k[int(self.d)] = self.d - int(self.d)

        p_k[0] = 1 - sum(p_k[1:])
        return p_k


class Network_Diffuse:
    def __init__(self, s, d=6, k_list=np.arange(1, 50), g='power'):
        self.k_list = k_list
        self.d = d  # average degree
        self.g = g  # genre of the network degree distribution
        self.s = np.array(s)
        self.len_s = len(self.s)  # length of the empirical data set
        self.F = np.zeros_like(self.k_list)  # initialization of cummulative number of adopters

    def gener_simulation(self, params):  # params:[genre of the degree distribution,average degree, p, q, m]
        self.d = params[0]
        get_pk = Gener_PK(self.d)
        get_pk.g = self.g
        self.p_k = get_pk.get_pk()
        self.p_k[0] = 1 - sum(self.p_k[1:])
        self.k_list = get_pk.k_list

        p, q, m = params[1:]
        accum_diff = np.zeros(self.len_s + 1)  # 累积扩散率，初始扩散为0
        insta_diff = np.zeros(self.len_s)  # 非累积扩散率
        v_f = np.zeros_like(self.k_list)
        theta = 0
        for i in range(self.len_s):
            temp = p + q * self.k_list * theta  # 限制影响小于等于1
            influ = np.array([u if u <= 1 else 1 for u in temp])
            delta_f = (1 - v_f) * influ  # 各k对应的采纳率增长
            insta_diff[i] = np.dot(delta_f, self.p_k)  # 添加i+1时间步下的总采纳率增长
            v_f = v_f + delta_f  # 各k对应的采纳率
            theta = np.sum(self.k_list * self.p_k * v_f) / self.d  # 计算平均影响率
            accum_diff[i+1] = np.dot(v_f, self.p_k)  # 添加i+1时间步的总采纳率

        return m * insta_diff

    def residuals(self, params):  # 残差平方和
        return np.sum(np.square(self.s - self.gener_simulation(params)))

    def mse(self, params):  # 均方误差
        simul_s = self.gener_simulation(params)
        return np.mean(np.square(simul_s - self.s))

    def cal_r(self, params):
        a = self.gener_simulation(params)
        sse = np.sum(np.square(self.s - a))
        ave_y = np.mean(self.s)
        ssl = np.sum(np.square(self.s - ave_y))
        r_2 = (ssl - sse) / ssl
        return r_2


class Random_Grid_Search:
    t_n = 300  # 抽样量
    c_n = 30  # 保留参数量
    threshold = 1e-4  # 循环停止阈值
    orig_points = []  # 初始化边界点

    def __init__(self, s, k_list=np.arange(1,50), g='power'):  # 初始化实例参数
        self.s, self.s_len = np.array(s), len(s)
        self.k_list, self.g = k_list, g
        self.para_range = [[1e-6, 0.1], [1e-4, 1], [0.7 * sum(s), 5 * sum(s)], [3, 20]]  # 参数范围
        self.d_range = range(self.para_range[-1][0], self.para_range[-1][1] + 1)
        self.pk_cont = []  # 度密度分布容器
        for d in self.d_range:
            get_pk = Gener_PK(k_list=self.k_list, d=d, g=self.g)
            p_k = get_pk.get_pk()
            self.pk_cont.append(p_k)

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

    def sample(self, c_range):  # 抽样参数点
        p_list = []
        for i, pa in enumerate(c_range):
            if i <= 1:  # p, q 取自浮点数随机数
                x = (pa[1] - pa[0]) * np.random.random(self.t_n) + pa[0]
            else:  # m, d 取自整数随机数
                x = np.random.randint(low=pa[0], high=pa[1]+1, size=self.t_n)
            p_list.append(x)

        p_list = np.array(p_list).T
        return p_list.tolist()

    def f(self, params):  # 离散网络扩散模型 params:[p, q, m, d]
        p, q, m, d = params
        idx = self.d_range.index(d)
        p_k = self.pk_cont[idx]

        accum_diff = np.zeros(self.s_len+1)  # 累积扩散率，初始扩散为0
        insta_diff = np.zeros(self.s_len)  # 非累积扩散率
        v_f = np.zeros_like(self.k_list)  # 初始化各k对应的累积采纳率v_f
        theta = 0  # 初始化连接到已采纳者的概率
        for i in xrange(self.s_len):
            temp = p + q * self.k_list * theta
            influ = np.array([u if u <= 1 else 1 for u in temp])  # 限制隔间的最大采纳率为1
            delta_f = (1 - v_f) * influ  # 各k对应的采纳率增长
            insta_diff[i] = np.dot(delta_f, p_k)  # 添加i+1时间步下的总采纳率增长
            v_f = v_f + delta_f  # 各k对应的采纳率
            theta = np.sum(self.k_list * p_k * v_f) / np.dot(self.k_list, p_k)  # 计算平均影响率
            accum_diff[i+1] = np.dot(v_f, p_k)  # 添加i+1时间步的总采纳率

        return m * insta_diff

    def r2(self, params):  # 计算R^2
        f_act = self.f(params)
        tse = np.sum(np.square(self.s - f_act))
        mean_y = np.mean(self.s)
        ssl = np.sum(np.square(self.s - mean_y))
        r_2 = (ssl - tse)/ssl
        return r_2

    def mse(self, params):  # 定义适应度函数mse
        a = self.f(params)
        sse = np.sum(np.square(self.s - a))
        return np.mean(np.sqrt(sse))  # 均方误

    def optima_search(self):  # 搜索过程
        orig_points = self.gener_orig(self.para_range[:], or_points=[])  # 生成初始搜索点
        samp = self.sample(self.para_range[:])
        solution = sorted([self.mse(x)] + x for x in samp + orig_points)[:self.c_n]
        u = 1
        while True:
            par_min, par_max = np.min(np.array(solution), 0), np.max(np.array(solution), 0)  # 最小最小值
            c_range = [[par_min[j + 1], par_max[j + 1]] for j in range(len(self.para_range))]  # 重新定界
            samp = self.sample(c_range)
            solution = sorted([[self.mse(x)] + x for x in samp] + solution)[:self.c_n]
            r = sorted([x[0] for x in solution])
            v = (r[-1] - r[0]) / r[0]
            if v < self.threshold:  # 终止条件1：达到阈值终止
                break

            if u > 100: # 终止条件2：搜索轮次大于100
                print 'Searching ends in 100 runs'
                break

            u += 1
        return solution[0]  # sse,p,q,m,d


class Forecast:
    def __init__(self, s, b_idx, n, g='power', k_list=np.arange(1, 100)):
        self.s = s
        self.n = n
        self.s_len = len(s)
        self.b_idx = b_idx
        self.g = g
        self.k_list = k_list

    def f(self, params, T):  # 离散网络扩散模型 params:[p, q, m, d]
        p, q, m, d = params
        get_pk = Gener_PK(k_list=self.k_list, d=d, g=self.g)
        p_k = get_pk.get_pk()

        accum_diff = np.zeros(T + 1)  # 累积扩散率，初始扩散为0
        insta_diff = np.zeros(T)  # 非累积扩散率
        v_f = np.zeros_like(self.k_list)  # 初始化各k对应的累积采纳率v_f
        theta = 0  # 初始化连接到已采纳者的概率
        for i in xrange(T):
            temp = p + q * self.k_list * theta
            influ = np.array([u if u <= 1 else 1 for u in temp])  # 限制隔间的最大采纳率为1
            delta_f = (1 - v_f) * influ  # 各k对应的采纳率增长
            insta_diff[i] = np.dot(delta_f, p_k)  # 添加i+1时间步下的总采纳率增长
            v_f = v_f + delta_f  # 各k对应的采纳率
            theta = np.sum(self.k_list * p_k * v_f) / np.dot(self.k_list, p_k)  # 计算平均影响率
            accum_diff[i+1] = np.dot(v_f, p_k)  # 添加i+1时间步的总采纳率

        return m * insta_diff

    def predict(self):
        pred_cont = []
        print 'Model:%s  ' % self.g
        for i in range(self.s_len - 1 - self.b_idx):
            t1 = time.clock()
            idx = self.b_idx + 1 + i
            x = self.s[:idx]
            rgs = Random_Grid_Search(x, k_list=self.k_list, g=self.g)
            est = rgs.optima_search()
            params = est[1:]
            pred_s = self.f(params, self.s_len)
            pred_cont.append(pred_s[idx:])
            print u'模型%s, 当前为第%d/%d个, 用时%d秒'%(self.g, i + 1, self.s_len - 1 - self.b_idx,  time.clock() - t1)

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
    data_set = {'room air conditioners':(np.arange(1949, 1962), [96, 195, 238, 380, 1045, 1230, 1267, 1828, 1586, 1673, 1800, 1580, 1500]),
                'color televisions':(np.arange(1963, 1971), [747, 1480, 2646, 5118, 5777, 5982, 5962, 4631]),
                'clothers dryers':(np.arange(1949, 1962), [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425, 1260, 1236]),
                'ultrasound':(np.arange(1965, 1979), [5, 3, 2, 5, 7, 12, 6, 16, 16, 28, 28, 21, 13, 6]),
                'mammography':(np.arange(1965, 1979), [2, 2, 2, 3, 4, 9, 7, 16, 23, 24, 15, 6, 5, 1]),
                'foreign language':(np.arange(1952, 1964), [1.25, 0.77, 0.86, 0.48, 1.34, 3.56, 3.36, 6.24, 5.95, 6.24, 4.89, 0.25]),
                'accelerated program':(np.arange(1952, 1964),[0.67, 0.48, 2.11, 0.29, 2.59, 2.21, 16.80, 11.04, 14.40, 6.43, 6.15, 1.15])}

    china_set = {'color tv': (np.arange(1997, 2013),[2.6, 1.2, 2.11, 3.79, 3.6, 7.33, 7.18, 5.29, 8.42, 5.68, 6.57, 5.49, 6.48, 5.42, 10.72,
                               5.15]),
                 'mobile phone': (np.arange(1997, 2013),[1.7, 1.6, 3.84, 12.36, 14.5, 28.89, 27.18, 21.33, 25.6, 15.88, 12.3, 6.84, 9.02,
                                   7.82, 16.39, 7.39])}

    tx = 'room air conditioners'
    s = data_set[tx][1]
    Random_Grid_Search.t_n, Random_Grid_Search.s_n = 800, 200
    fore = Forecast(s, n=3, g='power', b_idx=8)
    res = fore.run()

    print u'1步向前预测:',
    print 'MAD:%.2f  MAPE:%.2f  MSE:%.2f' % res[0]
    print u'3步向前预测:',
    print 'MAD:%.2f  MAPE:%.2f  MSE:%.2f' % res[1]

    '''
    t1 = time.clock()
    rgs = Random_Grid_Search(S, k_list=np.arange(1, 50), g='power')
    est = rgs.optima_search()
    params = est[1:]
    r2 = rgs.r2(params)
    print 'Data set:%s, the genre of the degree distribution: %s'%(tx, rgs.g)
    print '    Time elapsed: %.2fs'%(time.clock() - t1)
    print '    r2:%.4f,    p:%.4f,   q:%.4f,    m:%d,    average degree:%d' % tuple([r2] + params)

    #绘图
    x = rgs.f(params)
    year = china_set[tx][0]
    fig = pl.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ax.text(min(year) + 0.2, max(S) * 0.9, '$R^2=%.4f$'%r2)
    ax.plot(year, x, 'k-', lw=2, alpha=0.7, label='estimation')
    ax.plot(year, S, 'ro', ms=5, alpha=0.5, label='sales')
    ax.set_xlabel('Year')
    ax.set_ylabel('Sales')
    ax.legend(loc='best')
    pl.show()
    '''