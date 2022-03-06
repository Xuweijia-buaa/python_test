import torch
from torch.utils.data import TensorDataset,DataLoader
class SimpleCustomBatch:
    # data:样本1,2,3。 每个样本含2个元素
    def __init__(self, data):
        # 每个样本的对应位置元素，合成一个tuple
        # tuple1：每个样本的第一个元素  (tensor([0., 1., 2., 3., 4.]), tensor([5., 6., 7., 8., 9.]), tensor([10., 11., 12., 13., 14.]))
        # tuple2：每个样本的第二个元素
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0) # 拼接成B,n . 每个样本的该位置元素拼接，增加Batch维
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=3, collate_fn=collate_wrapper,
                    pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())
