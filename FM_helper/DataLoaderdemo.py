# 建自己的dataset:
class Mydata(Dataset):
    def __init__(self,df):
        super(Mydata, self).__init__()
        self.df=df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        print("每个epoch下的样本index:",index)                     # 每个loader完整轮询一遍。index从0到len-1. 
                                                                 # 每个batch对应其中[0-B],[B,2B]...都是递增的
                                                                 # 多个epoch.又从index=0开始。
                                              # 可以借助这一特性，把大文件按样本顺序切n个小文件。每次只打开index对应的小文件
        return torch.tensor(self.df.iloc[index,:].values)
mydata=Mydata(df)
loader = DataLoader(mydata, shuffle=False, batch_size=32)
epoch_num=3
for epoch in range(epoch_num):
    for batch_id,x in enumerate(loader):                   # 把数据完整轮询一遍。对应一个epoch。 在外边加一个epoch的for循环。把数据遍历n次
        print("第{}个epoch,第{}个batch".format(epoch,batch_id))  # 内部先按照index顺序，依次调用batch_size个__getitem__。 再对这个batch进行处理
        continue

# 每个epoch下的样本index: 0
# 每个epoch下的样本index: 1
# 每个epoch下的样本index: 2
# 每个epoch下的样本index: 3
# 每个epoch下的样本index: 4
# 每个epoch下的样本index: 5
# 每个epoch下的样本index: 6
# 每个epoch下的样本index: 7
# 每个epoch下的样本index: 8
# 每个epoch下的样本index: 9
# 每个epoch下的样本index: 10
# 每个epoch下的样本index: 11
# 每个epoch下的样本index: 12
# 每个epoch下的样本index: 13
# 每个epoch下的样本index: 14
# 每个epoch下的样本index: 15
# 每个epoch下的样本index: 16
# 每个epoch下的样本index: 17
# 每个epoch下的样本index: 18
# 每个epoch下的样本index: 19
# 每个epoch下的样本index: 20
# 每个epoch下的样本index: 21
# 每个epoch下的样本index: 22
# 每个epoch下的样本index: 23
# 每个epoch下的样本index: 24
# 每个epoch下的样本index: 25
# 每个epoch下的样本index: 26
# 每个epoch下的样本index: 27
# 每个epoch下的样本index: 28
# 每个epoch下的样本index: 29
# 每个epoch下的样本index: 30
# 每个epoch下的样本index: 31
# 第0个epoch,第0个batch
# 每个epoch下的样本index: 32
# 每个epoch下的样本index: 33
# 每个epoch下的样本index: 34
# 每个epoch下的样本index: 35
# 每个epoch下的样本index: 36
# 每个epoch下的样本index: 37
# 每个epoch下的样本index: 38
# 每个epoch下的样本index: 39
# 每个epoch下的样本index: 40
# 每个epoch下的样本index: 41
# 每个epoch下的样本index: 42
# 每个epoch下的样本index: 43
# 每个epoch下的样本index: 44
# 每个epoch下的样本index: 45
# 每个epoch下的样本index: 46
# 每个epoch下的样本index: 47
# 每个epoch下的样本index: 48
# 每个epoch下的样本index: 49
# 每个epoch下的样本index: 50
# 每个epoch下的样本index: 51
# 每个epoch下的样本index: 52
# 每个epoch下的样本index: 53
# 每个epoch下的样本index: 54
# 每个epoch下的样本index: 55
# 每个epoch下的样本index: 56
# 每个epoch下的样本index: 57
# 每个epoch下的样本index: 58
# 每个epoch下的样本index: 59
# 每个epoch下的样本index: 60
# 每个epoch下的样本index: 61
# 每个epoch下的样本index: 62
# 每个epoch下的样本index: 63
# 第0个epoch,第1个batch
# ...
# 每个epoch下的样本index: 1568
# 每个epoch下的样本index: 1569
# 每个epoch下的样本index: 1570
# 每个epoch下的样本index: 1571
# 每个epoch下的样本index: 1572
# 每个epoch下的样本index: 1573
# 每个epoch下的样本index: 1574
# 每个epoch下的样本index: 1575
# 每个epoch下的样本index: 1576
# 每个epoch下的样本index: 1577
# 每个epoch下的样本index: 1578
# 每个epoch下的样本index: 1579
# 每个epoch下的样本index: 1580
# 每个epoch下的样本index: 1581
# 每个epoch下的样本index: 1582
# 每个epoch下的样本index: 1583
# 每个epoch下的样本index: 1584
# 每个epoch下的样本index: 1585
# 每个epoch下的样本index: 1586
# 每个epoch下的样本index: 1587
# 每个epoch下的样本index: 1588
# 每个epoch下的样本index: 1589
# 每个epoch下的样本index: 1590
# 每个epoch下的样本index: 1591
# 每个epoch下的样本index: 1592
# 每个epoch下的样本index: 1593
# 每个epoch下的样本index: 1594
# 每个epoch下的样本index: 1595
# 每个epoch下的样本index: 1596
# 每个epoch下的样本index: 1597
# 每个epoch下的样本index: 1598
# 第0个epoch,第49个batch
# 每个epoch下的样本index: 0
# 每个epoch下的样本index: 1
# 每个epoch下的样本index: 2
# 每个epoch下的样本index: 3
# 每个epoch下的样本index: 4
# 每个epoch下的样本index: 5
# 每个epoch下的样本index: 6
# 每个epoch下的样本index: 7
# 每个epoch下的样本index: 8
# 每个epoch下的样本index: 9
# 每个epoch下的样本index: 10
# 每个epoch下的样本index: 11
# 每个epoch下的样本index: 12
# 每个epoch下的样本index: 13
# 每个epoch下的样本index: 14
# 每个epoch下的样本index: 15
# 每个epoch下的样本index: 16
# 每个epoch下的样本index: 17
# 每个epoch下的样本index: 18
# 每个epoch下的样本index: 19
# 每个epoch下的样本index: 20
# 每个epoch下的样本index: 21
# 每个epoch下的样本index: 22
# 每个epoch下的样本index: 23
# 每个epoch下的样本index: 24
# 每个epoch下的样本index: 25
# 每个epoch下的样本index: 26
# 每个epoch下的样本index: 27
# 每个epoch下的样本index: 28
# 每个epoch下的样本index: 29
# 每个epoch下的样本index: 30
# 每个epoch下的样本index: 31
# 第1个epoch,第0个batch
# ...
