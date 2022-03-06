fname='/media/xuweijia/DATA/代码/python_test/data/Criteo/kaggle-display-advertising-challenge-dataset/train.txt'
import functools
from functools import partial
import time
from memory_profiler import profile
import psutil

from functools import wraps  # 装饰器
def time_count(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start,"s")
        return result
    return wrapper

#from memory_profiler import profile
@profile(precision=4)
#@time_count
def read():
    mem=psutil.virtual_memory().used/1024**2   # M  已用内存
    with open(fname, 'rb') as f:          # 大文件。
        s=f.read()                        # 直接read. 默认size是-1。将全部内容放到一个字符串变量中
        mem = psutil.virtual_memory().used / 1024 ** 2   # 已用内存
        del s                             # 用于从文件读取指定的字节数，如果未给定或为负则读取所有

@profile(precision=4)
#@time_count
def readline():
    mem0 = psutil.virtual_memory().used / 1024 ** 2
    mem_used=[]
    n=1000
    i=0
    with open(fname, 'rb') as f:          # 按行迭代f。如果原始就是二进制写入的，可以直接读入。
        for line in f:                    # 内部逻辑等同于line=f.readline() 。内容是一行一行返回的，不会占用太多内存. 但可能慢很多
            line.strip()                  # 包括 "\n"。strip()一下
            mem = psutil.virtual_memory().used / 1024 ** 2
            mem_used.append(mem -mem0)
            i+=1
            if(i==n):
                break

#@profile
@profile(precision=4)
#@time_count
def readchunk():                         #  如果一个文件的内容全部在一行. 或者想流式处理，每次只读取一部分，可以采用f.read(size)。适合流式统计一些内容
    mem0 = psutil.virtual_memory().used / 1024 ** 2
    mem_used=[]
    n=1000
    i=0
    block_size = 1024 * 8                #  单位是字节。一个字母一个字节
    with open(fname) as f:
        while True:
            chunk = f.read(block_size)   # 每次读取8k字节:   8k个字母。如果每次读一个字母，可以设置成1
            if not chunk:                # 当文件没有更多内容时，read调用将会返回空字符串 ''
                break
            mem = psutil.virtual_memory().used / 1024 ** 2
            mem_used.append(mem -mem0)
            i+=1
            if(i==n):
                break
'''
# 可以把read变成一个迭代器。用到再输出生成的内容。每次迭代，从文件输出一部分.但速度不如上一个
def chunked_file_reader(f, block_size=1024 * 8):
    """生成器函数：分块读取文件内容
    """
    #method1:
     # while True:
     #    chunk = f.read(block_size)    # 每次输出8k字节，通过yield返回。控制每次读取所需的内存大小
     #    if not chunk:
     #        break
     #    yield chunk

    # method2:
    # iter(callable, sentinel):传入一个函数callable。__next__()方法无法带参数调用该函数对象迭代，直到结果是sentinel停止
    # 因为__next__()无法把参数传给该函数，只能调用无参函数。所以这里传入一个partial函数： A=partital(read,arg1,..)  之后调用无参函数A(),相当于调用partital(arg1,...)
    # iter每次__next__()，都调用该函数。直到返回结果是sentinel停止
    for chunk in iter(partial(f.read, block_size), ''):
        yield chunk                     # 每次输出8k字节，通过yield返回。控制每次读取所需的内存大小

# 使用该迭代器，处理具体业务逻辑
def readchunk2():
    with open(fname) as fp:
        for chunk in chunked_file_reader(fp):
            pass
            #chunk.count('9')
            #break
'''


def test_partial():
    # 取原函数和一部分参数，作为原函数的partial function
    # 原函数
    def add(a,b):
        print(a+b)
        return a+b

    # 原函数的部分函数。 第一个参数是原函数。之后的各个参数原函数的部分/全部参数。 固定住这部分参数
    partadd= partial(add,1)   # 固定了原函数的一部分参数

    # 之后调用该函数时，只需要传另一部分参数即可。输出3
    partadd(2)

    partall = partial(add,1,2)  # 固定原函数的全部参数

    partall()                   # 调用该函数时，作为无参函数调用。 相当于调用参数固定的原函数 输出3


# 查看内存。函数添加profile：python  -m memory_profiler readtext.py
# 查看时间。函数添加时间装饰器（不同时添加）
if __name__ == '__main__':
    read()
    readline()
    readchunk()

# 耗时：
# read 3.443248748779297 s
# readline1 6.094823837280273 s  也挺慢的 11G数据 6s。但占用内存小。适合按行处理
# readchunk 3.233468770980835 s   chunk的速度比line的快。看场合。速度上这个好点。要是不按行处理的话
# readchunk2 3.438969850540161 s

# read 3.4175891876220703 s
# readline 35.37549304962158 s      jupyter跑的结果 line也挺慢的
# readchunk 10.200038194656372 s
# readchunk2 12.153430223464966 s
