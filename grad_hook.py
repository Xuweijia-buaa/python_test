import torch
import torch.nn as nn
import torch.optim as optim

# 作为hook函数，修改某变量的梯度. 返回新的梯度。在计算梯度的时候执行（backward时），用新的梯度更新参数/传播
def grad_hook(grad):               # 输入该变量的原始梯度
    print("参数的原始梯度:",grad)
    
    grad.mul_(grad_mask)           # 就地修改grad.  该变量不需要计算梯度的位置乘0。导数永远为0
    
    print("参数被hook修改后的新梯度:",grad)    # 返回Tensor作为新梯度,或者就地修改。不反回或者返回None，原始梯度不变


x=torch.Tensor([1,2,3,4,5])       # 输入


n=5
model=nn.Linear(n,1)                            #模型
act=model.weight.register_hook(grad_hook)       #对某部分参数的导数进行操作。用返回后的导数反向传播。 可以mask/clip
#act.remove()                                   # 可以通过remove撤销该hook
                                                #该函数作用于计算梯度阶段（backward时）

grad_mask = torch.ones(1, 5)                    # 对w的mask.用在grad函数里 . 大小同weight/grad  model.weight.shape
grad_mask[0,0] = 0.0                            # 不计算梯度的，置成0(第一行第一个元素)。  weight 本身是2维 

    
optimizer = optim.SGD(model.parameters(), lr=1.0)
criterion = nn.MSELoss()


y=model(x)                         # 输出 B,C
target=torch.tensor(1.0)

optimizer.zero_grad()
print("原始参数",model.weight)                   # 原参数

loss = criterion(y[0], target)
loss.backward()                                # 计算梯度。此时会根据hook修改梯度
optimizer.step()                               # 更新

print("更新后参数",model.weight)                 # 更新后参数。 mask掉的参数梯度不变。
