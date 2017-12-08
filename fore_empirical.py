#coding=utf-8
from discrete_forecasting import *
import multiprocessing
import pickle

def func(g, s, n, b_idx):
    fore = Forecast(s, b_idx, n, g)
    res = fore.run()
    raw_data = fore.pred_res
    print 'Model:%s finished' % g
    return res, raw_data

if __name__ == '__main__':
    f = open('empirical diffusion data set.txt', 'r')
    empi_data = pickle.load(f)
    g_cont = ['gauss', 'expon',  'power', 'logno']
    task_list = []
    for u in empi_data:
        d_set = empi_data[u]
        for v in d_set:
            for g in g_cont:
                task_list.append((u, v, g))

    pool = multiprocessing.Pool(processes=6)
    f_result = []
    Random_Grid_Search.t_n, Random_Grid_Search.s_n = 800, 200
    for i, task in enumerate(task_list):
        u, v, g = task
        print i, u, v, g
        s = empi_data[u][v][1:]
        b_idx = np.argmax(s) + 1
        n = 5
        f_result.append(pool.apply_async(func, (g, s, n, b_idx)))

    pool.close()
    pool.join()

    result_1 = []
    result_2 = []
    for i, res in enumerate(f_result):
        u, v, g = task_list[i]
        temp = res.get()
        result_1.append(temp[0])
        result_2.append(temp[1])

    f1 = open('forecast result.pkl', 'wb')
    f2 = open('forecast result(raw).pkl', 'wb')
    pickle.dump(result_1, f1)
    pickle.dump(result_2, f2)
