import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from importlib import reload
from torch.utils.data import DataLoader,Dataset,RandomSampler
import sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
import argparse
import random
import logging
logger = logging.getLogger(__name__)
from transformers import (AdamW, get_linear_schedule_with_warmup)
from torchkeras import summary
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()                             # 超参数
parser.add_argument('--embed_size', type=int, default=8)
parser.add_argument('--hiddens', type=str, default='256,56,32',help='deepfm的mlp层数')
parser.add_argument('--dropout', type=float, default=0,help='默认没有dropout')
parser.add_argument('--batchnorm', type=bool, default=False,help='默认没有batchnorm')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=8e-5,help='优化器步长')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='adamaW')
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--no-cuda', type=bool, default=False,help='是否用GPU')
parser.add_argument('--gpu', type=int, default=0,help='GPU设备id')
parser.add_argument('--random_seed', type=int, default=2020)
parser.add_argument('--eval_steps', type=int, default=500)
parser.add_argument('--display_steps', type=int, default=100)
parser.add_argument('--eval_batch_size', type=int, default=4096)
args = parser.parse_args(args=[])                              # jupyter里需要加args=[]

data_path='/media/xuweijia/DATA/代码/python_test/data/Criteo/demo_data/'
file_name='train.csv'
# get raw data
raw_df=pd.read_csv(os.path.join(data_path+file_name))
raw_df=raw_df.drop(["Id"],axis=1)
raw_df
# 分别找出连续列/离散列
def col_type(df):
    dis_col=[]
    con_col=[]
    columns=df.columns.tolist()
    for c in columns:
        if df[c].dtype=='int64' or df[c].dtype=='float':
            con_col.append(c)
        else:
            dis_col.append(c)
    return dis_col,con_col
dis_col,con_col=col_type(raw_df)
con_col.remove("Label")
label="Label"

# 填充缺失值
null_token = '<NULL>'
raw_df[dis_col]=raw_df[dis_col].fillna(null_token)
raw_df[con_col]=raw_df[con_col].fillna(0)
    
#设置随机种子。每次训练固定  
def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)


ss=StandardScaler()
ss.fit(raw_df[con_col])                           #归一化
raw_df[con_col]=ss.transform(raw_df[con_col])     # 测试做相同处理

from  Fmdata import FeaturePosTrans
from FMmodel import FM
f_trans=FeaturePosTrans(dis_col,con_col,0)           # 映射到对应id. 出现10次以下的作为UNK
f_trans.fit(raw_df)
feature_pos,feature_values=f_trans.transform(raw_df,label)  # 测试集做相同处理。用相同的原始con_col,dis_col。 label只是用来删除掉该列
cols=dis_col+con_col

args.dis_col=dis_col
args.con_col=con_col
args.f_trans=f_trans
args.hiddens=args.hiddens.split(',')
# 设置设备为某固定gpu
args.cuda = (not args.no_cuda)  and  (torch.cuda.is_available())
if args.cuda:
    torch.cuda.set_device(args.gpu)

class WrapModel(object):
    def __init__(self,args=None,state_dict=None):
        # 初始化模型
        n_field=len(args.dis_col+args.con_col)                # 原始特征数目
        n_features=len(args.f_trans.feature_id_map)           # 连续+离散特征one-hot后的特征总数目 (算上一个NAN padding)
        model=FM(n_field,n_features,args.embed_size,args.dropout)
        #self.model=deepFM(n_field,n_features,args.embed_size,args.hiddens,args.dropout,args.batchnorm)
        
        if state_dict:                                        # 加载保存过的模型参数（如果有）
            model.load_state_dict(state_dict)
       
        device = torch.device("cuda:0" if args.cuda else "cpu") # 放gpu上 （相比.cuda(),优先使用这个。方便在不同设备上切换）
        model.to(device) 
        
        self.model=model
        args.device=device
        self.args=args
        set_seed(args)

        
    def train(self,train_dataset,dev_dataset=None):        # 参考jupyter的loss画图
        args=self.args
        model=self.model
        
        # 设置train-loader. 每次都是随机取（无放回）
        sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size,num_workers=4)
        
        # 设置优化器
        parameters = [p for p in self.model.parameters() if p.requires_grad]  # 可只优化模型的部分层/部分parameter
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(parameters, lr=args.lr,momentum=0.9,weight_decay=0.08)
        elif args.optimizer == 'adamax':
            optimizer = optim.Adamax(parameters,weight_decay=0.08)
        elif args.optimizer == 'adamaW':
            optimizer = AdamW(parameters, lr=args.lr, eps=1e-8,weight_decay=0.08)
        
        # 定义lr本身的scheduler. 
        # 在多个epoch过程中，调节优化器中lr本身的大小。每个step通过schcduler.step()改变lr本身的值。optimizer里的lr被同步修改。
        # 但只改变lr的值。其他算法仍同optimizer
        # 在限定step内，让lr从0线性增加到设定值。防止初始lr较大，模型不稳定。之后认为模型稳定.再线性减小，到最终训练完lr减小到0. 过程中优化器算法本身不变
        num_training_steps=int(len(train_loader)*args.epochs)  # 完整训练过程中总的steps:每个step对应一个batch.
        num_warmup_steps=int(num_training_steps*0.2)               # 预热期steps的数目：占总step的20%。在这些step内，让lr从0线性增加到设定值。此时认为模型稳定了
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps,num_training_steps) # 先预热一些step.再逐渐减小。
        print("最初的lr:{}".format(scheduler.get_lr()))  # 此时是0  增到lr后，最终训练完仍减小到0
                                                                                                                              
        # train-loop
        logger.info("***** Running training *****")
        logger.info("  总的样本数量 = %d", len(train_dataset))
        logger.info("  epoch数目 = %d", args.epochs) 
        logger.info("  train batch size = %d",args.batch_size)
        logger.info("  所有epoch总的steps = %d", num_training_steps) 
        
        global_step = 0                      # 记录总的step
        best_metric=0.0

        model.zero_grad()                    # 把模型所有参数的梯度都置为0 （optimizer.zero_grad是把优化器里参数的梯度置为0）
        
        for epoch in enumerate(range(args.epochs)):                 # 每个epoch
            
            loss_sum=0.0                                     # 用来记录每个epoch到此时的平均loss
          
            for step, batch in enumerate(train_loader):  # dataset按batch-size轮询一遍。每个batch一个step
                
                # 输入放到对应设备上。同模型
                f_p,f_v,y= (x.to(args.device) for x in batch)   # trian、dev时才用y. test时没有y,dataset传入id列或任意列。但不用
                del batch
                
                model.train()  # train mode
                
                logits=model(f_p,f_v)                                  # 输出logits,before sigmoid: (B,1)
                
                batch_loss=self.compute_loss(logits,y)                 # 计算该batch的loss (TODO:可以添加L2)
                
                optimizer.zero_grad()                                  # 清空之前的参数梯度
                batch_loss.backward()                                  # 根据loss重新计算模型参数梯度
                #print(batch_loss.item())
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)   # 对算好的梯度 grad_clip
                
                optimizer.step()                                      # 根据梯度更新参数
                scheduler.step()                                      # 更新lr值
                global_step += 1
                
                
                # 打印每段时间的平均loss.和此时的train_metric
                loss_sum+=batch_loss.item()                                 # n个step内的平均。 item():转为float
                if args.display_steps % (step+1)==0:
                    print('Epoch = {} | step = {}/{} | loss = {:.2f}'.format(epoch,step+1,len(train_loader),loss_sum/(step+1)))
                # loss_sum=0.0
                #    pass
                          
                # 评估。好的话保存
            #break
                          
    def compute_loss(self,logits,labels):
        '''
        计算一个batch的loss
        logits:(B,1) before sigmoid
        labels:(B,1) 每个样本的取值0/1
        '''
        loss=F.binary_cross_entropy_with_logits(logits,labels.float()) #  mean [yn⋅logσ(xn)+(1−yn)⋅log(1−σ(xn))]
        return loss
        
    def infer(self,test_dataset):
        pass
        
    
    def eval_metric(self,labels,preds):
        pass
    
    
    def save(self,filename):
        '保存超参数和内部模型的模型参数'
        params = {
            'state_dict': self.model.state_dict(),   # 只按参数名称保存模型所有参数。是一个dict.不保存模型结构
            'args': self.args,
        }
        torch.save(params, filename)           # 保存static等到指定文件中。 这里可以保存字典，之后通过load加载进来（底层pickle）
    
    
    @staticmethod
    def load(filename):   
        '直接根据文件名，返回加载好内部模型参数的WraPModel（init时通过state_dict）.可以在外边直接调用：WrapModel.load(f)'
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage  # load默认直接加载到GPU.  这样指定，先加载到cpu上。再load_state
        )
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        
        return WrapModel(args,state_dict)     
    
    def export_onnx(self,onnx_filename):
        '需要用到nn.Module等'
        pass

class Mydata(Dataset):
    def __init__(self,fv,fp,target,mode='train'):
        super(Mydata, self).__init__()
        self.fv=fv           # np: m,n.  每个样本的特征取值
        self.fp=fp           #           每个样本的特征位置.如果太大以后可以放np文件名.或切成多个文件,每次只打开一个(类似drml)
        self.target=target   #           如果mode==train/valid.对应y. mode==test。可以传入样本id。infer时不用
    def __len__(self):
        return len(self.fv)
    def __getitem__(self, index):
        return self.fp[index,:],self.fv[index],self.target[index]  # 提前做好了映射。当然映射也可以在这里做。

skf=StratifiedKFold(n_splits=5,random_state=2020,shuffle=True)    # 分割器。按y值对给定的样本划分。返回分好的样本id 每轮4:1
for i, (train_index, dev_index) in enumerate(skf.split(raw_df, raw_df[label])):  # 可用于df. 根据y，输出分割后的train/dev样本位置
    logger.info("训练第%s折对应的模型", i)  # 每个fold训一个模型

    train_fv = feature_values[cols].iloc[train_index].values  # 按样本位置，从转换好的df里取训练样本
    train_fp = feature_pos[cols].iloc[train_index].values
    train_label = raw_df[label].iloc[train_index].values
    trainDataset = Mydata(train_fv, train_fp, train_label)  # 对应的dataset

    dev_fv = feature_values[cols].iloc[dev_index].values  # dev场景下，主要是计算指标。model不输出loss
    dev_fp = feature_pos[cols].iloc[dev_index].values
    dev_label = raw_df[label].iloc[dev_index].values  # test用raw_df[label]
    dev_dataset = Mydata(dev_fv, dev_fp, dev_label)

    # train(train_dataset,dev_dataset)
    break

CTRmodel=WrapModel(args)
#CTRmodel.train(trainDataset)

summary(CTRmodel.model,input_shape=[(39,),(39,)]) # 不用自己加batch-size