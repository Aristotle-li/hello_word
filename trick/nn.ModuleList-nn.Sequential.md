1、nn.ModuleList的作用
    先来探究一下nn.ModuleList的作用，定义一个简单的只含有全连接的网络来看一下。当不使用ModuleList只用list来定义网络中的层的时候：

```haskell
import torch
import torch.nn as nn
 
class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.combine = []
        self.combine.append(nn.Linear(100,50))
        self.combine.append(nn.Linear(50,25))
 
net = testNet()
print(net)
```

可以看到结果并没有显示添加的全连接层信息。如果改成采用ModuleList：

```haskell
import torch
import torch.nn as nn
 
class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.combine = nn.ModuleList()
        self.combine.append(nn.Linear(100,50))
        self.combine.append(nn.Linear(50,25))
 
net = testNet()
print(net)
```

可以看到定义的时候[pytorch](https://so.csdn.net/so/search?q=pytorch)可以自动识别nn.ModuleList中的参数而普通的list则不可以。如果用Sequential也能达到和ModuleList同样的效果。

nn.ModuleList和nn.Sequential的区别
 如果给上面定义的网络一个输入，查看输出结果：
  

```haskell
import torch
import torch.nn as nn
 
class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.combine = nn.ModuleList()
        self.combine.append(nn.Linear(100,50))
        self.combine.append(nn.Linear(50,25))
 
testnet = testNet()
input_x = torch.ones(100)
output_x = testnet(input_x) 
print(output_x)
```

会报错NotImplementedError：

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgwatermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhdGVybWVsb24xMTIz,size_16,color_FFFFFF,t_70.png)

这是因为没有实现forward()方法，如果将forward()方法补全如下：

```haskell

```

发现仍旧会报NotImplementedError错误。这是因为nn.ModuleList是一个无序性的序列，他并没有实现forward()方法，我们不能通过直接调用x = self.combine(x)的方法来实现forward()。如果想要实现ModuleList的方法需要如下定义forward():
  

```python
import torch
import torch.nn as nn
 
class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.combine = nn.ModuleList()
        self.combine.append(nn.Linear(100,50))
        self.combine.append(nn.Linear(50,25))
    #补全forward（）
    def forward(self, x):
        x = self.combine(x)
        return x
testnet = testNet()
input_x = torch.ones(100)
output_x = testnet(input_x) 
print(output_x)
```

得到了正确的结果。如果替换成nn.Sequential定义：

```python
import torch
import torch.nn as nn
 
class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.combine = nn.Sequential(
            nn.Linear(100,50),
            nn.Linear(50,25),
        ) 
    def forward(self, x):
        x = self.combine(x)
        return x
testnet = testNet()
input_x = torch.ones(100)
output_x = testnet(input_x) 
print(output_x)
```

nn.Sequential定义的网络中各层会按照定义的顺序进行级联，因此需要保证各层的输入和输出之间要衔接。并且nn.Sequential实现了farward()方法，因此可以直接通过类似于x=self.combine(x)的方式实现forward。

而nn.ModuleList则没有顺序性要求，并且也没有实现forward()方法。



```
import torch
import torch.nn as nn
 
class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.combine = nn.Sequential(nn.Linear(100,50),
        nn.Linear(50,25))
     def forward(self,x):
     		x = self.combine(x)
     		return x
testnet = testNet()
input_x = torch.ones(100)
output_x = testnet(input_x) 
print(output_x)
```

