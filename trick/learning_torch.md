## 随机数种子seed

```

import torch

torch.manual_seed(2)   #为CPU设置种子用于生成随机数，以使得结果是确定的 
print(torch.rand(2))


if args.cuda: 
    torch.cuda.manual_seed(args.seed) #为当前GPU设置随机种子；
    # 如果使用多个GPU，应该使用 torch.cuda.manual_seed_all()为所有的GPU设置种子。
```

## cat 、stack

**1 torch.cat()**

> `torch.cat`(*tensors*,*dim=0*,*out=None*)→ Tensor

torch.cat()对tensors沿指定维度拼接，但返回的Tensor的维数不会变

```python3
>>> import torch
>>> a = torch.rand((2, 3))
>>> b = torch.rand((2, 3))
>>> c = torch.cat((a, b))
>>> a.size(), b.size(), c.size()
(torch.Size([2, 3]), torch.Size([2, 3]), torch.Size([4, 3]))
```

可以看到c和a、b一样都是二维的。

2 torch.stack()

> `torch.stack`(*tensors*,*dim=0*,*out=None*)→ Tensor

torch.stack()同样是对tensors沿指定维度拼接，但返回的Tensor会多一维

```python3
>>> import torch
>>> a = torch.rand((2, 3))
>>> b = torch.rand((2, 3))
>>> c = torch.stack((a, b))
>>> a.size(), b.size(), c.size()
(torch.Size([2, 3]), torch.Size([2, 3]), torch.Size([2, 2, 3]))
```

可以看到c是三维的，比a、b多了一维。





## 函数作用：

函数stack()对序列数据内部的张量进行扩维拼接，指定维度由程序员选择、大小是生成后数据的维度区间。

## 存在意义：

在自然语言处理和卷及神经网络中， 通常为了保留–[序列(先后)信息] 和 [张量的矩阵信息] 才会使用stack。



手写过RNN的同学，知道在循环神经网络中输出数据是：一个list，该列表插入了seq_len个形状是[batch_size, output_size]的tensor，不利于计算，需要使用stack进行拼接，保留–[1.seq_len这个时间步]和–[2.张量属性[batch_size, output_size]]。
