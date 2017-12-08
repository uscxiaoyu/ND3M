#coding=utf-8
from estimate_bass import *
import multiprocessing
import pickle

def func(s, n, b_idx):
    t1 = time.clock()
    fore = Bass_forecast(s, n, b_idx)
    res = fore.run()
    raw_data = fore.pred_res
    print u'任务完成，耗时%.1fs'%(time.clock() - t1)
    return res, raw_data


if __name__ == '__main__':
    f = open('empirical diffusion data set.txt', 'r')
    empi_data = pickle.load(f)
    task_list = []
    for u in empi_data:
        d_set = empi_data[u]
        for v in d_set:
            task_list.append((u, v))

    pool = multiprocessing.Pool(processes=6)
    f_result = []
    for i, task in enumerate(task_list):
        u, v = task
        print i, u, v
        s = empi_data[u][v][1:]
        b_idx, n = np.argmax(s) + 1, 5
        f_result.append(pool.apply_async(func, (s, n, b_idx)))

    pool.close()
    pool.join()

    result_1 = []
    result_2 = []
    for i, res in enumerate(f_result):
        u, v = task_list[i]
        temp = res.get()
        result_1.append(temp[0])
        result_2.append(temp[1])

    f1 = open('forecast result of Bass.pkl', 'wb')
    f2 = open('forecast result(raw) of Bass.pkl', 'wb')
    pickle.dump(result_1, f1)
    pickle.dump(result_2, f2)
    f1.close()
    f2.close()