### action



用argparse模块让python脚本接收参数时，对于True/False类型的参数，向add_argument方法中加入参数action=‘store_true’/‘store_false’。
顾名思义，store_true就代表着一旦有这个参数，做出动作“将其值标为True”，也就是没有时，默认状态下其值为False。反之亦然，store_false也就是默认为True，一旦命令中有此参数，其值则变为False。



### Torch.gt



 pytorch中逐元素比较两个torch.tensor大小用到了如下的三个函数：

- **torch.gt(tensor1, tensor2, out=None) || tensor1.gt(tensor2,out=None)**：tensor1对应的元素大于tensor2的元素会返回True，否则返回False。参数out表示为一个数或者是与第一个参数相同形状和类型的tensor。
- **torch.lt(tensor1, tensor2, out=None) || tensor1.lt(tensor2,out=None)**：tensor1对应的元素小于tensor2的元素会返回True，否则返回False。参数out表示为一个数或者是与第一个参数相同形状和类型的tensor。
- **torch.eq(tensor1, tensor2, out=None) || tensor1.eq(tensor2,out=None)**：tensor1对应的元素等于tensor2的元素会返回True，否则返回False。参数out表示为一个数或者是与第一个参数相同形状和类型的tensor。

```python
import torch
Matrix_A = torch.tensor([1,2,3,4,5,3,1])
Matrix_B = torch.tensor([2,1,3,1,5,2,0])
print(torch.gt(Matrix_A,Matrix_B))
tensor([False,  True, False,  True, False,  True,  True])
print(torch.lt(Matrix_A,Matrix_B))
tensor([ True, False, False, False, False, False, False])
print(torch.eq(Matrix_A,Matrix_B))
tensor([False, False,  True, False,  True, False, False])
print(Matrix_A.gt(Matrix_B))
tensor([False,  True, False,  True, False,  True,  True])
print(Matrix_A.lt(Matrix_B))
tensor([ True, False, False, False, False, False, False])
print(Matrix_A.eq(Matrix_B))
tensor([False, False,  True, False,  True, False, False])
```



