data_path='/media/xuweijia/DATA/代码/python_test/data/Criteo/demo_data/'
file_name='train.csv'

data_path='/media/xuweijia/新加卷/Kaggle数据集/h-and-m-personalized-fashion-recommendations/'
file_name='transactions_train.csv'

csv_file=data_path+file_name

import time
from memory_profiler import profile
import psutil
import pandas as pd
import time
import sys
import tracemalloc
#@profile
def readall():
    t0=time.time()
    raw_df = pd.read_csv(csv_file)
    #raw_df.head(10)
    t1=time.time()
    print(sys._getframe().f_code.co_name, t1 - t0,"s")

#@profile
def readallmap():
    t0=time.time()
    raw_df = pd.read_csv(csv_file,memory_map=True)    # e
    t1=time.time()
    print(sys._getframe().f_code.co_name, t1 - t0,"s")

# gan
#@profile
def readchunk():
    n = 1000
    i = 0
    t0=time.time()
    with pd.read_csv(csv_file, chunksize=500) as reader:  # 返回的是一个迭代器，类型是TextFileReader
        for chunk in reader:                              # 顺序500行的小df。带header
            #print(i)
            #print(chunk)
            i += 1
            if i == n:
                break
            pass
    t1=time.time()
    print(sys._getframe().f_code.co_name, t1 - t0, "s")

#@profile
def readchunkmap(): # map the file object directly onto memory and access the data directly from there.
                    # Using this option can improve performance because there is no longer any I/O overhead
    n = 1000
    i = 0
    t0 = time.time()
    with pd.read_csv(csv_file, chunksize=500, memory_map=True) as reader:  # 返回的是一个迭代器，类型是TextFileReader
        for chunk in reader:
            #print(i)
            #print(chunk)
            i += 1
            if i == n:
               break
            pass
    t1 = time.time()
    print(sys._getframe().f_code.co_name, t1 - t0,"s")


def readchunkMem():
    n = 1000
    i = 0
    x = []
    y = []  # 记录内存曲线
    multiplier = {'B': 1, 'KiB': 1024, 'MiB': 1048576}  # 内存单位转化
    #t0 = time.perf_counter()
    tracemalloc.start()  # 另一个内存检测工具
    snapshot0 = tracemalloc.take_snapshot()  # 当前内存
    with pd.read_csv(csv_file, chunksize=500) as reader:  # 返回的是一个迭代器，类型是TextFileReader
        for chunk in reader:  # 每次顺序返回一个小df.  # 是一个500行,带header小df.  内存维持在7M左右，不会一直涨
            i += 1
            snapshot1 = tracemalloc.take_snapshot()                # 现在内存
            top_stats = snapshot1.compare_to(snapshot0, 'lineno')  # 差异 增长的内存
                                                                   # 是占用内存最大的所有子进程
            for stat in top_stats:
                if 'readdf.py' in str(stat): # 必须找出是本程序产生的。否则有很多启动之类的
                    x.append(i)
                    mem=str(stat).split('average=')[1].split(' ')
                    y.append(float(mem[0])*multiplier[mem[1]])
                    break
            if i == n:
                break
    #t1 = time.perf_counter()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y, 'D', color='black', label='Experiment')   # 单位是B  312B
    #plt.plot(x, np.dot(x, 4), color='red', label='Expect')  # float32的预期占用空间
    plt.title('Memery Difference vs Array Length')
    plt.xlabel('Number Array Length')
    plt.ylabel('Memory Difference')
    plt.legend()
    plt.savefig('comp_mem.png')  # 内存不增长

# python  -m memory_profiler readdf.py
if __name__ == '__main__':
    # readall()
    # readallmap()
    # readchunk()
    # readchunkmap()
    readchunkMem()