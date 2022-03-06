import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np

batch_size = 2
# 假设共4个样本。
# 包含3类特征
# 特征1： 映射好的输入序列。每个样本对应序列不等长。类似句子vector，点击vector
text = np.array([
    [1,0,3,3,4,5,6,0], # 样本1对应序列
    [1,0,3], 
    [2,4,0,2,3], 
    [1,2,0,1]])       # 样本4对应序列  各样本不等长
# 另外一个稠密类特征X
X=np.array([
    [0,0.1,2],      # 样本1的该特征
    [1,0.4,5],
    [2,0.6,5],
    [3,0.5,5]       # 样本4的该特征
])
# label y
label = np.array([1,0,0,1])

class MyDataset(Dataset):
    def __init__(self, seq, X,label):
        self.seq = seq
        self.X=X
        self.label = label

    def __len__(self):
        return len(self.label)

    # 输出样本index的序列特征，其他特征,label。其中序列特征是一个映射过id的nparray(1,T)
    def __getitem__(self, index):
        #return self.seq[index], self.X[index],self.label[index] 
        return torch.Tensor(self.seq[index]),self.X[index],self.label[index]

# 初始化数据集
data=MyDataset(text,X,label)  

# 重新组织该batch内的所有样本
# 输入batch:[sample1,sample2,...sampleB]。 B个样本.每个样本是一个含n个元素的tuple,对应__getitem__的n个输出
def collate_fn(batch):
    B=len(batch)         # 一共有batch个样本
    for sample in batch: # 每个样本是Dataset的一个输出。包含该样本的n个元素
        #print(sample)
        #print(sample[0]) # 输出样本的第一个元素  （对应getitem的第一个位置的元素）
        #print(sample[1]) # 输出样本的第二个元素
        #print(sample[2])
        break
    
    #常见写法1： padding. 
    # __getitem__某个元素是一个序列。 对B个样本的该特征进行padding
    
    #1 首先找到batch里该特征的最大长度，作为该batch的maxT
    #  find the max len in each batch
    alltextlens=[len(sample[0]) for sample in batch]   # 假设序列特征是__getitem__的第一个输出元素。
    max_T=max(alltextlens)                             # 该batch里该序列特征的最大长度
    
    #2 该序列特征对应的(B,T)填充
    x = torch.LongTensor(B, max_T).zero_()           # （B,maxT）  默认padding, 0  不padding处填真实index
    x_mask=torch.LongTensor(B, max_T).fill_(1)       # （B,maxT） 表示哪些地方是mask的，不需要取注意力/softmax. mask的地方置1
    
    for i,sample in enumerate(batch):  # 填充该batch的每个序列特征
        T=len(sample[0])               # 该样本对应序列的真实长度
        x[i,:T].copy_(sample[0])       # 比赋值快
        x_mask[i, :T].fill_(0)         # mask的地方是1. 真实位置处是0
        
    # 常见写法2： 类似默认写法，把B个样本的每个特征组织成一个元素
     
    element_tuples=list(zip(*batch)) # zip后得到n个tuple. 每个tuple是B个样本的对应位置元素
        
    batch1=element_tuples[1]         # tuple.含B个元素. 每个元素是是B个样本位置1的输出。这里每个元素等长
                                     # 同 [sample[1] for sample in batch]
    # (array([14, 46, 59, 68, 77, 91, 92, 104, 115, 117, 127, 139, 150,           # 样本1 位置1输出的元素
    #         168, 175, 197, 213, 240, 249, 257, 258, 0, 279, 284, 305, 309,
    #         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    #  array([15, 26, 59, 68, 77, 86, 92, 103, 115, 117, 127, 139, 150,           # 样本2 位置1输出的元素
    #         167, 175, 197, 212, 219, 0, 0, 258, 0, 276, 284, 0, 0,
    #         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    #  array([15, 37, 59, 68, 78, 86, 92, 104, 115, 117, 127, 139, 150,          # 样本3 位置1输出的元素。 一行特征
    #         172, 175, 197, 212, 219, 250, 257, 258, 0, 275, 284, 299, 309,
    #         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))

    X=torch.Tensor(batch1)           # 把这个list,转成tensor(类似np), 变成（B,n）  （torch.Size([3, 39])）


    batch2=element_tuples[2]
    y=torch.Tensor(batch2)          # (B,1)
    
    return x,x_mask,X,y

dl = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
for batch_id,(x,x_mask,X,y) in enumerate(dl):
    print("batch_id:",batch_id)
    print("x_text:",x)
    print("x_mask:",x_mask)
    print("x_dense:",X)
    print("y:",y)
