

import time
from memory_profiler import profile
import psutil
import pandas as pd
import time
import sys
import tracemalloc
import numpy as np
@profile
def npload():   # 被赋的行越多，内存占用越多。被赋的所有行都在内存里了
    # 载入
    filename = "my_np_mmap_Big2"
    loadmp = np.load(filename, mmap_mode='r+')
    # 赋值：
    for i in range(0, 10000, 1):
        loadmp[i:i + 1, :] = np.ones((1, 100000))  # 只赋值一部分，内存不会奔溃。但遍历的越多，内存占用越多
    del loadmp

    filename = "my_np_mmap_Big2"

# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
#     11    111.2 MiB    111.2 MiB           1   @profile
#     12                                         def npload1():
#     13                                             # 载入
#     14    111.2 MiB      0.0 MiB           1       filename = "my_np_mmap_Big2"
#     15    111.2 MiB      0.0 MiB           1       loadmp = np.load(filename, mmap_mode='r+') 一开始不占内存
#     16                                             # 赋值：
#     17   7741.6 MiB      0.0 MiB       10001       for i in range(0, 10000, 1):
#     18   7741.6 MiB   7630.4 MiB       10000           loadmp[i:i + 1, :] = np.ones((1, 100000))  # 只赋值一部分，内存不会奔溃。但遍历的越多，内存占用越多
#     19    112.3 MiB  -7629.3 MiB           1       del loadmp

# 1000行，内存减少
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
#     11    111.3 MiB    111.3 MiB           1   @profile
#     12                                         def npload1():
#     13                                             # 载入
#     14    111.3 MiB      0.0 MiB           1       filename = "my_np_mmap_Big2"
#     15    111.3 MiB      0.0 MiB           1       loadmp = np.load(filename, mmap_mode='r+')
#     16                                             # 赋值：
#     17    875.2 MiB      0.0 MiB        1001       for i in range(0, 1000, 1):
#     18    875.2 MiB    763.9 MiB        1000           loadmp[i:i + 1, :] = np.ones((1, 100000))  # 只赋值一部分，内存不会奔溃。但遍历的越多，内存占用越多
#     19    112.3 MiB   -762.9 MiB           1       del loadmp
#     20
#     21    112.3 MiB      0.0 MiB           1       filename = "my_np_mmap_Big2"



@profile(precision=4)
def npuse():
    filename = "my_np_mmap_Big2"
    # 载入
    loadmp = np.load(filename, mmap_mode='r+')
    a=np.zeros((5,100000))                                   # 占位变量。只占a大小的内存
    for i in range(0,100000,5):
        a=loadmp[i:i+5,:]                                    # 很稳定。如果只是给一个固定变量用的话。不随行数增加而变化

    b=np.zeros((1,100000))                                   # 占位变量。只占b大小的内存
    for i in range(0,100000,1):
        b=loadmp[i:i+1,:]

# 遍历10000行
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
#     48    111.2 MiB    111.2 MiB           1   @profile
#     49                                         def npuse():
#     50    111.2 MiB      0.0 MiB           1       filename = "my_np_mmap_Big2"
#     51                                             # 载入
#     52    111.2 MiB      0.0 MiB           1       loadmp = np.load(filename, mmap_mode='r+')
#     53    111.5 MiB      0.3 MiB           1       a=np.zeros((1,100000))
#     54    111.5 MiB      0.0 MiB       10001       for i in range(0,10000,1):
#     55    111.5 MiB      0.0 MiB       10000           a=loadmp[i:i+1,:]

# 遍历100000行 一样
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
#     48    110.6 MiB    110.6 MiB           1   @profile
#     49                                         def npuse():
#     50    110.6 MiB      0.0 MiB           1       filename = "my_np_mmap_Big2"
#     51                                             # 载入
#     52    110.6 MiB      0.0 MiB           1       loadmp = np.load(filename, mmap_mode='r+')
#     53    110.8 MiB      0.3 MiB           1       a=np.zeros((1,100000))
#     54    110.8 MiB      0.0 MiB      100001       for i in range(0,100000,1):
#     55    110.8 MiB      0.0 MiB      100000           a=loadmp[i:i+1,:]

# python  -m memory_profiler readdf.py
if __name__ == '__main__':
    filename = "my_np_mmap_Big2"
    m = 100000
    n = 100000
    mmap = np.lib.format.open_memmap(filename, dtype='float64', mode='w+', shape=(m, n))  # 保存了形状，数值类型。能写入 80G
    mem1 = psutil.virtual_memory().used / 1024 ** 2
    del mmap

    #npload()  # 赋值。内存会增加
    npuse()   # 用