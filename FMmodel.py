# 标准FM。但加了2边的权重。初始化方式待定
class FM(nn.Module):
    def __init__(self,n_field,n_features,embed_size,dropout=0.2):
        """
        标准FM
        n_field: 原始离散特征数目
        n_features: 离散特征one-hot之后,和dense的总特征个数. 这里包含了一个缺失值特征向量，paddding成0
        """
        super(FM, self).__init__()
        self.n_field=n_field             # 原始特征数目
        self.n_features=n_features       # 连续+离散特征的总数目 (离散特征one-hot+unique化后的。且算上最终的一个padding)
        self.k=embed_size
        
        self.W=nn.Embedding(n_features,1,padding_idx=0)                       # 每个特征对应的wi。位置0是0对梯度无贡献。embedding默认是N(0,1)
        self.w0=nn.Parameter(torch.zeros([1,]))                               # b初始化为0
        self.feature_embed=nn.Embedding(n_features,embed_size,padding_idx=0)  # 每个特征对应的embedding
        self.droplayer=nn.Dropout(dropout)
        
        self.weight=nn.Parameter(0.5*torch.ones([2,]))                        # 一阶，二阶score的权重 
                                                                              
        self.__init_weight__()                                                # 初始化权重(可以不调用，用默认的)
            
    def __init_weight__(self):              # 初始化权重.默认是N(0,1)
        inn=n_features-1
        # 自定义uniform_，参考drml （deepFM用的tf.random_normal）
        init_embed = np.random.uniform(low=-np.sqrt(1 / inn), high=np.sqrt(1 / inn), size=(inn, embed_size)).astype(np.float32)
        self.feature_embed.weight[1:].data.copy_(torch.tensor(init_embed))
        nn.init.normal_(self.W.weight[1:],0, np.sqrt(2.0 /inn))
        # kaiming_normal_
        #nn.init.kaiming_uniform_(self.feature_embed.weight[1:], mode='fan_in', nonlinearity='relu')  # sqrt(6/inn) 
        # normal
        #nn.init.normal_(self.feature_embed.weight[1:],0, np.sqrt(2.0 / (inn + embed_size)))  
        # nn.init.xavier_normal_(self.feature_embed.weight[1:])             # embedding 默认是N(0,1)。
        
        
    def forward(self,f_p,f_x): # B,n.  n是每个样本的原始特征数目
        """
        f_p: (B,N)  每个样本的原始特征，根据特征名（连续特征）/特征取值（离散特征）被映射到embedding上的位置。 N:原始特征数目
        f_x: (B,N)  每个样本的原始特征，对应的取值。 离散特征对应的是one-hot后的，所以在对应的f_p上取值为1
        """
        batch_size=f_p.shape[0]
        
        # 一阶score
        w=self.W(f_p).reshape(batch_size,-1)              # B,n   每个样本根据原始特征，找到对应位置处的w
        y_score1=self.w0 + torch.sum(torch.mul(w,f_x),1)  # B,1   wixi+b  B,n -->   B,1.  要是不sum，也可以作为n个特征。之后作为deepFM的改造
        
        # 二阶score
        embed=self.feature_embed(f_p)                     # B,n,d 每个样本根据原始特征，找到对应位置处的embedding
        
        embed=torch.mul(embed,f_x.unsqueeze(2))           # B,n,d 样本的每个向量乘上对应的xi： xi* embedding
                                                          #       tf.mul:按位置乘.广播 (B,n,d) * (B,n,1) ->(B,n,d) 
                                                          #       离散特征对应的xi是1.假设数值型已经归一化
        
        #embed=self.droplayer(embed)
        
        # 每个filed 向量22交互
        e_sum= torch.sum(embed,1)                         # B,d   每个样本。所有embedding对应维度元素相加，得到e_sum
        e_sum_square=torch.square(e_sum)                  # B,d   (e_sum)^2
        
        e_square=torch.square(embed)                      # B,n,d  平方后的
        e_square_sum=torch.sum(e_square,1)                # B,d    每个维度相加
        
        f =0.5*(e_sum_square-e_square_sum)                # B,d    n个embedding，每个维度元素22交互的结果（可作为新特征，拼接到后边）
    
        y_score2= torch.sum(f,1)                          # (B,)
        
        logits=self.weight[0]*y_score1+self.weight[1]*y_score2                          # (B,)   最终分数
        
        # TODO:loss加正则项（最后加）/dropout/grad-clip等. 看下train-loop
        
        return logits
