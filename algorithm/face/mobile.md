首发于[计算机视觉随便记](https://www.zhihu.com/column/c_1043101340913872896)

![不用重新训练，直接将现有模型转换为MobileNet](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-5faafaacb520d24d783a82cc97415d19_1440w.jpg)

# 不用重新训练，直接将现有模型转换为MobileNet

[![autocyz](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/img860e27ff8588772c8a70b063ce1e19d5_xs.jpg)](https://www.zhihu.com/people/chen-yong-zhi-41)

[autocyz](https://www.zhihu.com/people/chen-yong-zhi-41)

Hello, it's me

## 从MobileNet中的深度可分卷积（Depthwise Separable Convolution）讲起

看过MobileNet的都知道，MobileNet最主要的加速就是因为深度可分卷积（Depthwise Separable  Convolution）的使用。将原本一步到位的卷积操作，分为两个卷积的级联，分成的两个卷积相对于原始的卷积而言，参数大大减少，计算量大大减少。

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-35c2b269c661f0af95aa9aaac1ad4e2e_720w.jpg)

1、标准卷积：

> 假设卷积层的输入为： ![[公式]](https://www.zhihu.com/equation?tex=M\times+H\times+W) ，输出为： ![[公式]](https://www.zhihu.com/equation?tex=N\times+H+\times+W+) ，标准卷积的卷积核为： ![[公式]](https://www.zhihu.com/equation?tex=M\times+K+\times+K\times+N+) ，计算量为： ![[公式]](https://www.zhihu.com/equation?tex=M\times+H\times+W\times+K\times+K\times+N) 

2、深度可分卷积：

深度可分卷积是将原来的一个卷积，分解成两个不同的卷积，每个卷积的功能不一样。一个在feature map进行卷积，一个在通道上进行卷积。

> Depthwise Conv：首先，对输入的每个channel使用一个 ![[公式]](https://www.zhihu.com/equation?tex=K+\times+K+) 的卷积核进行卷积，因为输入共有 ![[公式]](https://www.zhihu.com/equation?tex=M) 个channel，所以此时卷积核的参数为： ![[公式]](https://www.zhihu.com/equation?tex=M\times+K\times+K+) ，输出为： ![[公式]](https://www.zhihu.com/equation?tex=M\times+H\times+W+) ，计算量为： ![[公式]](https://www.zhihu.com/equation?tex=M\times+W+\times+H+\times+K+\times+K) 
>
> Pointwise Conv：然后，使用 ![[公式]](https://www.zhihu.com/equation?tex=N+) 个 ![[公式]](https://www.zhihu.com/equation?tex=1\times+1) 的卷积核，对上一步输出的 ![[公式]](https://www.zhihu.com/equation?tex=M) 个channel的结果进行卷积，起到通道融合的作用。这里卷积核的参数为 ![[公式]](https://www.zhihu.com/equation?tex=M\times+1\times+1+\times+N) ，输出为 ![[公式]](https://www.zhihu.com/equation?tex=N\times+H+\times+W+) ，计算量为： ![[公式]](https://www.zhihu.com/equation?tex=N\times+W\times+H\times+1\times+1) 
>  
> 参数量的比值： ![[公式]](https://www.zhihu.com/equation?tex=\frac{M\times+K\times+K+\times+1+%2B+M\times+1\times+1\times+N}{M\times+K\times+K+\times+N}+%3D+\frac{1}{N}+%2B+\frac{1}{K^2}) 
> 计算量的比值： ![[公式]](https://www.zhihu.com/equation?tex=\frac{M\times+W\times+H\times+K\times+K+\times+1+%2B+M\times+W+\times+H+\times+1\times+1\times+N}{M\times+W+\times+H\times+K\times+K+\times+N}+%3D+\frac{1}{N}+%2B+\frac{1}{K^2}) 

从上面参数量的比值和计算量的比值可以看出，深度可分卷积相对于传统的标准卷积可以极大的减少参数量和计算量。例如，对于输出通道 ![[公式]](https://www.zhihu.com/equation?tex=N%3D64+) ，卷积核大小为 ![[公式]](https://www.zhihu.com/equation?tex=K%3D3) 的标准卷积，参数数量变成了原来的 ![[公式]](https://www.zhihu.com/equation?tex=\frac{1}{64}+%2B+\frac{1}{9}) .

3、深度可分卷积的pytorch实现：

```python3
def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)        
    )
```



## 将训练好的卷积模型转换为深度可分卷积

既然有了深度可分卷积，并且这个卷积方法可以在不改变特征输入输出大小的基础上极大减少参数量和计算量，那么我们当然会想着把已有模型中的标准卷积换成深度可分卷积。可是，在替换成深度可分卷积时，实际上模型由原来的一个卷积变成了两个卷积，模型结构已经发生了变化，所以我们需要对转换后的深度可分卷积模型重新训练。这就要求我们有足够的数据，足够的时间重新训练。

> 有没有什么方法能够不重新训练模型，直接将已有的训练好的模型，转换成深度可分卷积呢？

答案是：有！

接下来要讲的方法就是出自文章：

[DAC: Data-free Automatic Acceleration of Convolutional Networks](https://arxiv.org/abs/1812.08374?context=cs)

[arxiv.org/abs/1812.08374?context=cs](https://arxiv.org/abs/1812.08374?context=cs)

**文章流程图：**

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-6e3993e35f1288699ebef12f97b04bfb_720w.jpg)

**文章思想：**

文章思想很简单，通过使用作者提出的DAC方法，直接将一个标准卷积的权植，分解成两个卷积（ ![[公式]](https://www.zhihu.com/equation?tex=T_d) ， ![[公式]](https://www.zhihu.com/equation?tex=T_s) ）的级联。

> 将 ![[公式]](https://www.zhihu.com/equation?tex=n\times+k_w+\times+k_h\times+c) 的卷积分解成 ![[公式]](https://www.zhihu.com/equation?tex=rC\times+k_w\times+k_h+\times+1) 和 ![[公式]](https://www.zhihu.com/equation?tex=n\times+1\times+1\times+rC) 

其中 ![[公式]](https://www.zhihu.com/equation?tex=n) 是输出channel， ![[公式]](https://www.zhihu.com/equation?tex=c) 是输入channel， ![[公式]](https://www.zhihu.com/equation?tex=k_w%2C+k_h) 是kernel size， ![[公式]](https://www.zhihu.com/equation?tex=rC%3Dr\times+c) ， ![[公式]](https://www.zhihu.com/equation?tex=r+) 是一个分解因子，也是设定的SVD分解中去的主成分个数， ![[公式]](https://www.zhihu.com/equation?tex=r) （文章中 ![[公式]](https://www.zhihu.com/equation?tex=r%3D1%2C2%2C3%2C4%2C5) ）越大说明保留的信息越多。

**与MobileNet中可分离卷积的区别：**

乍一看这种分解出来的结果和可分离卷积形式一样，实则不然，具体区别有两点：

- MobileNet的深度可分卷积在depthwise和pointwise之间有BN和激活层，而DAC分解的则没有。
- MobileNet的深度可分卷积在depthwise阶段，每个输入channel，只对应一个卷积核，即depthwise阶段卷积核的大小为 ![[公式]](https://www.zhihu.com/equation?tex=c\times+k_w+\times+k_h) ，而DAC中每个输入channel对应了 ![[公式]](https://www.zhihu.com/equation?tex=r) 个卷积和，所以此阶段总的卷积核大小为 ![[公式]](https://www.zhihu.com/equation?tex=r\times+c\times+h_w\times+h_h) 
- 同理，在pointwise阶段，两种方法的卷积核大小也不一样。

**具体方法：**

作者在文章中给出了非常清晰的算法流程，中间用到了SVD分解的方法获取主成分。具体方法如下：

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-d9bcdd9dca527480106efb2148d79b5f_720w.jpg)

**计算量的减少：**

假设输入feature为 ![[公式]](https://www.zhihu.com/equation?tex=W\times+H\times+c) ，输出大小为 ![[公式]](https://www.zhihu.com/equation?tex=W\times+H\times+n) ，则计算量的比值为：

![[公式]](https://www.zhihu.com/equation?tex=\frac{W\times+H\times+k_w+\times+k_h\times+rC+%2BW\times+H\times+rC\times+n}{W\times+H\times+c\times+k_w+\times+k_h\times+n}+%3D+\frac{r}{n}+%2B+\frac{r}{k_wk_h}) 

**实验分析：**

作者做了两个探究性实验，探究了分解哪些层对模型的影响最大，并且在物体分类、物体检测、pose estimation上都做了迁移测试。这里讲一下两个有意思的探究性实验，其他迁移实验可以去文章细看。

***实验一：验证分解单个层并且使用不同的rank（即上面说的分解因子 ![[公式]](https://www.zhihu.com/equation?tex=r) ）对模型整体性能的影响。***

作者对CIFAR-VGG模型的不同卷积层进行DAC分解，每次只分解一个卷积层，得到如下结果：

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-25f7d3e7502ab44aaa8d75e7d5f11e01_720w.jpg)

结论：

1、使用较小的rank，比如rank=1时，分解前面的层（例如conv2d_*1）会导致模型精度损失特别厉害（93.6->18.6），而分解后面的层（conv2d_*13）则不会损失很多（93.6->92.9）.

2、越大的rank，精度损失的越小。比如在conv2d_1阶段，rank5 的精度远远高于rank1.

***实验二：验证分解前k个层和后k个层对模型精度的影响***

前k个层就是从第一个到第k个， 后k就是从最后一个到导数第k个

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-fa89655c4d95c23a6b3404bc83b515c9_720w.jpg)

结论：同等条件下（相同的rank），分解网络后面的层产生的精度损失会小于分解网络前面的层。



## 总结

方法非常简单使用，而且又很容易迁移到各种模型上去。最最主要的是，直接对训练好的model进行分解，不必再重新训练了。省时有省力，美滋滋。

> [1] Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional  neural networks for mobile vision applications[J]. arXiv preprint  arXiv:1704.04861, 2017.
>
> [2] Li X, Zhang S, Jiang B, et al. DAC:  Data-free Automatic Acceleration of Convolutional Networks[J]. arXiv  preprint arXiv:1812.08374, 2018.