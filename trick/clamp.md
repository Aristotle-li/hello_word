torch.clamp()



    torch.clamp(input, min, max, out=None) → Tensor

将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。

操作定义如下：

     				 | min, if x_i < min
      y_i =  | x_i, if min <= x_i <= max
             | max, if x_i > max




参数：

    input (Tensor) – 输入张量
    min (Number) – 限制范围下限
    max (Number) – 限制范围上限
    out (Tensor, optional) – 输出张量

示例：

![image-20211230185645998](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211230185645998.png)

    tensor.clamp()
    
    tensor.clamp(min, max, out=None) → Tensor

跟上面是一样的作用，tensor就是input
参数：

    min (Number) – 限制范围下限
    max (Number) – 限制范围上限
    out (Tensor, optional) – 输出张量

![image-20211230185632562](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211230185632562.png)