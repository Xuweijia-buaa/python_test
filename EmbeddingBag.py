# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import Embedding,EmbeddingBag
# 定义一个[5,3]的embeddingbag
# 类似于embedding，也保存着一个（5，3）的embedding
# 不同的是在取用的时候，是给出要聚合的embedding的index,按找offset对input分组后。输出每组聚合后的embedding，而非原始embedding本身 (比如sum后的)
m=5
d=3
embedding_sum = nn.EmbeddingBag(m, d, mode="sum")  # 其中bag的方式是sum(默认是mean)
weights=embedding_sum.weight.data                  # # the learnable weights of the module of shape (m, d)
# tensor([[ 0.5944,  0.9436, -0.4434],
#         [-1.0945,  0.6171,  0.0950],
#         [ 0.1976, -0.2007,  0.6636],
#         [-0.4998,  0.0247, -0.8053],
#         [-0.8881, -0.4650, -0.4865]])
# 用自定义的weight初始化来调试
weight = torch.FloatTensor([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9],
                            [10,11,12],
                            [13,14,15]])
embedding_sum = nn.EmbeddingBag.from_pretrained(weight,mode="sum")
#embedding_sum.weight.data = torch.tensor(weight, requires_grad=True) 相同。facebook用的这个

input=torch.tensor([1,2,4,4], dtype=torch.long)   # 要从embedding table中抽取的位置
offsets=torch.tensor([0,2], dtype=torch.long)     # 把input根据offsets分为多个bags.每个bag对应的embedding分别sum
                                                  # 其中这个offsets会把input分为两个bag：input[0:2]和input[2:]
                                                  # 第一个bag,是[1,2].第二个bag是[4,4]
                                                  # 每个bag输出一个结果（根据mode聚合）
result=embedding_sum(input,offsets)
# tensor([[11., 13., 15.],               # sum(input[0:2]: 1,2)
#         [26., 28., 30.]])              # sum(input[2:]:  4,4)


# 长度为n的offset,对应n-1个bag:
offsets=torch.tensor([0,2,4], dtype=torch.long)      #                                 input[0:2],input[2:4],input[4:]
input=torch.tensor([0,1,2,3,4], dtype=torch.long)    # 按照offset切分成了3个bag,分别聚合： [0,1] ,[2,3], [4]
result=embedding_sum(input,offsets)
# tensor([[ 5.,  7.,  9.],
#         [17., 19., 21.],
#         [13., 14., 15.]])
offsets=torch.tensor([0,2,4], dtype=torch.long)      #                                 input[0:2],input[2:4],input[4:]
input=torch.tensor([1,2,4,5,0,6], dtype=torch.long)  # 按照offset切分成了3个bag,分别聚合： [1,2] ,[4,5], [0，6]。 后2个有超出范围的，直接置0
result=embedding_sum(input,offsets)
# tensor([[11., 13., 15.],
#         [ 0.,  0.,  0.],
#         [ 0.,  0.,  0.]])
offsets=torch.tensor([0], dtype=torch.long)          #  只有一个bag,是所有input embedding的聚合:input[0:]
input=torch.tensor([1,2,0], dtype=torch.long)
result=embedding_sum(input,offsets)
# tensor([[12., 15., 18.]])
offsets=torch.tensor([0,0,2], dtype=torch.long)      #  3个bag: input[0:0]  input[0:2]  input[2:]
input=torch.tensor([1,2,0], dtype=torch.long)        #             空         [1，2]       [0]
result=embedding_sum(input,offsets)
# tensor([[ 0.,  0.,  0.],
#         [11., 13., 15.],
#         [ 1.,  2.,  3.]])
offsets=torch.tensor([0,0,3], dtype=torch.long)      #  3个bag: input[0:0]  input[0:3]  input[3:] 最后一个bag超过了长度，是空
input=torch.tensor([1,2,0], dtype=torch.long)        #             空         [1，2,0]        空
result=embedding_sum(input,offsets)
# tensor([[ 0.,  0.,  0.],
#         [12., 15., 18.],
#         [ 0.,  0.,  0.]])

# 等价于
embed=nn.Embedding(5,3)
embed.weight.data=weight
result2=torch.sum(embed(input),dim=0)
# tensor([12., 15., 18.], grad_fn=<SumBackward1>)
print("ok")
