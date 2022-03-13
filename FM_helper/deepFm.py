import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# deepFM： 标准的。 或者上述k个元素作为特征统一拼到deep的
# 主要先留着初始化方式，备用
class deepFM(nn.Module):
    def __init__(self,n_field,n_features,embed_size,hiddens,dropout=0,batchnorm=True):
        super(deepFM, self).__init__()
        """
        标准DeepFM
        n_field: 原始特征数目
        n_features: 离散特征one-hot之后,和dense的总特征个数. 这里包含了一个缺失值特征向量，paddding成0
        deep每层：linear + (bn) + relu + (dropout).  输出hidden层。  最后按需单独linear到1
        """
        
        input_size=n_field*embed_size       # 输入deep的特征是所有原始特征的embedding拼接
        self.dropout=dropout
        self.batchnorm=batchnorm
        self.mlp=self.build_mlp(input_size,hiddens)                           #多层mlp.输出hidden  是一个ModuleList
        self.finallinear=nn.Linear(hiddens[-1],1,bias=True)                   #最后一层linear
        
        self.__init_weight__()                                              # 初始化权重。(可以按情况调用。现在用kaiming)
        
    def __init_weight__(self):              # 初始化权重
        for layer in self.mlp:
            if (layer.__class__.__name__=='Linear'):  # 每层初始化
                self.init_linear(layer)           
        self.init_linear(self.finallinear)
    
    # nn.linear  默认是U[-1/sqrt(in) ,1/sqrt(in)],但drml、deepFM用的xvar_normal
    def init_linear(self,layer):
#    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  #sqrt(2/inn) 或者 sqrt(2/out)
#    nn.init.constant_(layer.bias, 0)
        #自定义xvar_normal_
        inn=layer.in_features
        out=layer.out_features
        std1=np.sqrt(2.0 / (inn + out))
        std2=np.sqrt(2.0 /out)   
        W=np.random.normal(0.0, std1, size=(out, inn)).astype(np.float32)
        b = np.random.normal(0.0, std2, size=out).astype(np.float32)
        layer.weight.data.copy_(torch.tensor(W))
        layer.bias.data.copy_(torch.tensor(b))

    def build_mlp(self,input_size,hiddens):
        """
        hiddens:[256,56,32]
        输出最后一层的hidden节点 （dropout+激活后的），可直接linear+sigmoid到deepscore. 也可以拼其他特征后再linear
        """        
        layers=nn.ModuleList()
        
        hiddens.insert(0,input_size)     #  [inputsize,256,56,32]
        num_layer=len(hiddens)-1         #  3层
        
        for i in range(0,len(hiddens)-1):
            
            # 线性层
            in_dim=hiddens[i]
            out_dim=hiddens[i+1]
            layer=nn.Linear(in_dim,out_dim,bias=True)
            layers.append(layer)
            
            # BN
            if self.batchnorm:
                layers.append(nn.BatchNorm1d(out_dim)) 
            
            # active + dropout
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
                
        return layers  # 拼成一个前后连接的网络。可以在forward里作为一个模块被整体调用
    
    
    # TODO:finish forward and train_loop
    def forward(self,f_p,f_x): # B,n.  n是每个样本的原始特征数目
        pass 
        
        
        
