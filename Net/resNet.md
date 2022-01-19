#### resNet

1、short cut 解决了梯度消失的问题

2、shot cut 耦合了不同感受野尺度特征图的表达能力





![image-20210118164037288](/Users/lishuo/Library/Application Support/typora-user-images/image-20210118164037288.png)

![image-20210118164745394](/Users/lishuo/Library/Application Support/typora-user-images/image-20210118164745394.png)

![image-20210118165210532](/Users/lishuo/Library/Application Support/typora-user-images/image-20210118165210532.png)

![image-20210118165830546](/Users/lishuo/Library/Application Support/typora-user-images/image-20210118165830546.png)



## 迁移学习

![image-20210118201206494](/Users/lishuo/Library/Application Support/typora-user-images/image-20210118201206494.png)

![image-20210118201259722](/Users/lishuo/Library/Application Support/typora-user-images/image-20210118201259722.png)

### 1、载入权重后全部重新训练

### 2、载入权重后，只保留feature的权重，delete classifier weights，只训练分类器

```python
#这里使用的是第二种迁移学习的方法，

net = MobileNetV2(num_classes=5) # 这里可以直接指定种类5，是因为后面的模型参数的导入，只导入了features层的权重，classifier层的权重没有导入
# load pretrain weights
# download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
model_weight_path = "./mobilenet_v2.pth"
assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
pre_weights = torch.load(model_weight_path)
# delete classifier weights
pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

# freeze features weights
for param in net.features.parameters():
    param.requires_grad = False
```

### 3、载入权重后保留所有权重，在此基础上再添加一层全连接层，只训练最后的全连接层

```python
net = resnet34()  #这里这mobileNet比起来，就没有先传入种类5，因为导入了全部的模型参数，是基于1000类的参数，先指定5类，导入1000类的参数肯定会报错
# load pretrain weights
# download url:
model_weight_path = "./resnet34-pre.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
# for param in net.parameters():
#     param.requires_grad = False
# change fc layer structure
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 5) # 在这里，接一个1000-5 的全连接层，完全导入参数
net.to(device)
```

## pytorch图像分类篇：6. ResNet网络结构详解与迁移学习简介

最近在b站发现了一个非常好的 **计算机视觉 + pytorch** 的教程，相见恨晚，能让初学者少走很多弯路。
 因此决定按着up给的教程路线：图像分类→目标检测→…一步步学习用pytorch实现深度学习在cv上的应用，并做笔记整理和总结。

up主教程给出了pytorch和tensorflow两个版本的实现，我暂时只记录pytorch版本的笔记。

参考内容来自：

- up主的b站链接：https://space.bilibili.com/18161609/channel/index
- up主将代码和ppt都放在了github：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
- up主的CSDN博客：https://blog.csdn.net/qq_37541097/article/details/103482003
- [ResNet网络结构详解与模型的搭建](https://blog.csdn.net/qq_37541097/article/details/104710784)

------

# 1. ResNet 详解

原论文地址：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)（作者是CV大佬何凯明团队）

ResNet 网络是在 **2015年** 由微软实验室提出，斩获当年ImageNet竞赛中分类任务第一名，目标检测第一名。获得COCO数据集中目标检测第一名，图像分割第一名。

在ResNet网络的创新点：

- 提出 **Residual** 结构（残差结构），并搭建超深的网络结构（可突破1000层）
- 使用 **Batch Normalization** 加速训练（丢弃dropout）

下图是ResNet34层模型的结构简图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719204103322.png?#pic_center)


## 1.1 Why residual?

在ResNet网络提出之前，传统的卷积神经网络都是通过将一系列卷积层与池化层进行堆叠得到的。

一般我们会觉得网络越深，特征信息越丰富，模型效果应该越好。但是实验证明，当网络堆叠到一定深度时，会出现两个问题：

1. **梯度消失或梯度爆炸**

   > 关于梯度消失和梯度爆炸，其实看名字理解最好：
   >  若每一层的误差梯度小于1，反向传播时，网络越深，梯度越趋近于0
   >  反之，若每一层的误差梯度大于1，反向传播时，网路越深，梯度越来越大

2. **退化问题**(degradation problem)：在解决了梯度消失、爆炸问题后，仍然存在深层网络的效果可能比浅层网络差的现象

总结就是，**当网络堆叠到一定深度时，反而会出现深层网络比浅层网络效果差的情况**。

如下图所示，20层网络 反而比 56层网络 的误差更小： ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719205325378.png?#pic_center)

- 对于梯度消失或梯度爆炸问题，ResNet论文提出通过数据的预处理以及在网络中使用 [**BN**（**Batch Normalization**）](https://blog.csdn.net/qq_37541097/article/details/104434557)层来解决。
- 对于退化问题，ResNet论文提出了 **residual结构**（**残差结构**）来减轻退化问题，下图是使用residual结构的卷积网络，可以看到随着网络的不断加深，效果并没有变差，而是变的更好了。（虚线是train error，实线是test error）
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200805095831298.png?#pic_center)



## 1.2 What is residual？

为了解决深层网络中的退化问题，可以人为地让神经网络某些层跳过下一层神经元的连接，隔层相连，弱化每层之间的强联系。这种神经网络被称为 **残差网络** (**ResNets**)。

残差网络由许多隔层相连的神经元子模块组成，我们称之为 **残差块** **Residual block**。单个残差块的结构如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200423164533200.png#pic_center)
 上图中红色部分称为 **short cut** 或者 **skip connection**（也称 捷径分支），**直接建立**                                        a                                   [                            l                            ]                                           a^{[l]}               a[l] 与                                        a                                   [                            l                            +                            2                            ]                                           a^{[l+2]}               a[l+2]之间的**隔层联系**。其前向传播的计算步骤为：

- ​                                             z                                       [                               l                               +                               1                               ]                                            =                                   W                                       [                               l                               +                               1                               ]                                                      a                                       [                               l                               ]                                            +                                   b                                       [                               l                               +                               1                               ]                                                 z^{[l+1]}=W^{[l+1]}a^{[l]}+b^{[l+1]}                  z[l+1]=W[l+1]a[l]+b[l+1]
- ​                                             a                                       [                               l                               +                               1                               ]                                            =                         g                         (                                   z                                       [                               l                               +                               1                               ]                                            )                              a^ {[l+1]}=g(z^{ [l+1]} )                  a[l+1]=g(z[l+1])
- ​                                             z                                       [                               l                               +                               2                               ]                                            =                                   W                                       [                               l                               +                               2                               ]                                                      a                                       [                               l                               +                               1                               ]                                            +                                   b                                       [                               l                               +                               2                               ]                                                 z ^{[l+2]} =W ^{[l+2]} a^{[l+1]}+b^{ [l+2]}                  z[l+2]=W[l+2]a[l+1]+b[l+2]
- ​                                             a                                       [                               l                               +                               2                               ]                                            =                         g                         (                                   z                                       [                               l                               +                               2                               ]                                            +                                   a                                       [                               l                               ]                                            )                              a^{ [l+2] }=g(z ^{[l+2]} +a^{ [l]} )                  a[l+2]=g(z[l+2]+a[l])

​                                       a                                   [                            l                            ]                                           a ^{[l]}               a[l] 直接隔层与下一层的线性输出相连，与                                        z                                   [                            l                            +                            2                            ]                                           z^{[l+2]}               z[l+2] 共同通过激活函数（ReLU）输出                                        a                                   [                            l                            +                            2                            ]                                           a^{[l+2]}               a[l+2]。

由多个 残差块 组成的神经网络就是 残差网络 。其结构如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200423165747663.png#pic_center)

实验表明，这种模型结构对于训练非常深的神经网络,效果很好。另外，为了便于区分，我们把 **非残差网络** 称为 **Plain Network。**



## 1.3 ResNet中的残差结构

实际应用中，残差结构的 short cut 不一定是隔一层连接，也可以中间隔多层，ResNet所提出的残差网络中就是隔多层。

跟VggNet类似，ResNet也有多个不同层的版本，而残差结构也有两种对应浅层和深层网络：

|          | ResNet           | 残差结构   |
| -------- | ---------------- | ---------- |
| 浅层网络 | ResNet18/34      | BasicBlock |
| 深层网络 | ResNet50/101/152 | Bottleneck |

下图中左侧残差结构称为 **BasicBlock**，右侧残差结构称为 **Bottleneck**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200805101305631.png?#pic_center)
 对于深层的 Bottleneck，1×1的卷积核起到降维和升维（特征矩阵深度）的作用，同时可以大大减少网络参数。

> 可以计算一下，假设两个残差结构的输入特征和输出特征矩阵的深度都是256维，如下图：（注意左侧结构的改动）
>
>  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200805110510978.png?#pic_center)
>  那么两个残差结构所需的参数为：
>
> - 左侧：                                        3                            ×                            3                            ×                            256                            ×                            256                            +                            3                            ×                            3                            ×                            256                            ×                            256                            =                            1                            ,                            179                            ,                            648                                  3 \times 3 \times 256 \times 256+3 \times 3 \times 256 \times 256=1,179,648                     3×3×256×256+3×3×256×256=1,179,648
> - 右侧：                                        1                            ×                            1                            ×                            256                            ×                            64                            +                            3                            ×                            3                            ×                            64                            ×                            64                            +                            1                            ×                            1                            ×                            64                            ×                            256                            =                            69                            ,                            632                                  1 \times 1 \times 256 \times 64+3 \times 3 \times 64 \times 64+1 \times 1 \times 64 \times 256=69,632                     1×1×256×64+3×3×64×64+1×1×64×256=69,632
>
>  注：CNN参数个数 = 卷积核尺寸×卷积核深度 × 卷积核组数 = 卷积核尺寸 × 输入特征矩阵深度 × 输出特征矩阵深度  
>
> 
> 明显搭建深层网络时，使用右侧的残差结构更合适。



## 1.4 降维时的 short cut

观察下图的 ResNet18层网络，可以发现有些残差块的 short cut 是实线的，而有些则是虚线的。

这些虚线的 short cut 上通过1×1的卷积核进行了维度处理（特征矩阵在长宽方向降采样，深度方向调整成下一层残差结构所需要的channel）。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200805143600781.png?#pic_center)
 下图是原论文给出的不同深度的ResNet网络结构配置，注意表中的残差结构给出了主分支上卷积核的大小与卷积核个数，表中 残差块×N 表示将该残差结构重复N次。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200805145611987.png?#pic_center)
 原文的表注中已说明，conv3_x, conv4_x, conv5_x所对应的一系列残差结构的**第一层残差结构**都是虚线残差结构。因为这一系列残差结构的第一层都有**调整输入特征矩阵shape**的使命（将特征矩阵的高和宽缩减为原来的一半，将深度channel调整成下一层残差结构所需要的channel）

需要注意的是，对于ResNet50/101/152，其实conv2_x所对应的一系列残差结构的第一层也是虚线残差结构，因为它需要调整输入特征矩阵的channel。根据表格可知通过3x3的max pool之后输出的特征矩阵shape应该是[56, 56,  64]，但conv2_x所对应的一系列残差结构中的实线残差结构它们期望的输入特征矩阵shape是[56, 56,  256]（因为这样才能保证输入输出特征矩阵shape相同，才能将捷径分支的输出与主分支的输出进行相加）。所以第一层残差结构需要将shape从[56, 56, 64] --> [56, 56, 256]。注意，这里只调整channel维度，高和宽不变（而conv3_x, conv4_x, conv5_x所对应的一系列残差结构的第一层虚线残差结构不仅要调整channel还要将高和宽缩减为原来的一半）。

下面是 ResNet 18/34 和 ResNet 50/101/152 具体的实线/虚线残差结构图：

- ResNet 18/34
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200805112731312.png?#pic_center)
- ResNet 50/101/152s
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200805112656263.png?#pic_center)

# 2. 迁移学习简介

迁移学习是一个比较大的领域，我们这里说的迁移学习是指神经网络训练中使用到的迁移学习。

在迁移学习中，我们希望利用源任务（Source Task）学到的知识帮助学习目标任务 (Target  Task)。例如，一个训练好的图像分类网络能够被用于另一个图像相关的任务。再比如，一个网络在仿真环境学习的知识可以被迁移到真实环境的网络。迁移学习一个典型的例子就是载入训练好VGG网络，这个大规模分类网络能将图像分到1000个类别，然后把这个网络用于另一个任务，如医学图像分类。

为什么可以这么做呢？如下图所示，神经网络逐层提取图像的深层信息，这样，预训练网络就相当于一个特征提取器。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200815095455312.png?#pic_center)

- **使用迁移学习的优势**：

1. 能够快速的训练出一个理想的结果
2. 当数据集较小时也能训练出理想的效果

   注意：使用别人预训练好的模型参数时，要注意别人的预处理方式。

- **常见的迁移学习方式**：

1. 载入权重后训练所有参数
2. 载入权重后只训练最后几层参数
3. 载入权重后在原网络基础上再添加一层全连接层，仅训练最后一个全连接层

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200815101406219.png?#pic_center)

# 3. pytorch搭建ResNet

> 可参考
>  [pytorch官方实现resnet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
>  [解读 pytorch对resnet的官方实现](https://www.cnblogs.com/wzyuan/p/9880342.html)

## 3.1 model.py

- 定义ResNet18/34的残差结构，即BasicBlock
- 定义ResNet50/101/152的残差结构，即Bottleneck
- 定义ResNet网络结构
- 定义resnet18/34/50/101/152

```python
import torch.nn as nn
import torch


# ResNet18/34的残差结构，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1  # 残差结构中，主分支的卷积核个数是否发生变化，不变则为1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):  # downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:  # 虚线残差结构，需要下采样
            identity = self.downsample(x)  # 捷径分支 short cut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# ResNet50/101/152的残差结构，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    expansion = 4  # 残差结构中第三层卷积核个数是第一/二层卷积核个数的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # 捷径分支 short cut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # block = BasicBlock or Bottleneck
    # block_num为残差结构中conv2_x~conv5_x中残差块个数，是一个列表
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])             # conv2_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # conv5_x
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # channel为残差结构中第一层卷积核个数
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        # ResNet50/101/152的残差结构，block.expansion=4
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899100101102103104105106107108109110111112113114115116117118119120121122123124125126127128129130131132133134135136137138139140141142143144145146147148
```

## 3.2 train.py

由于ResNet网络较深，直接训练的话会非常耗时，因此用迁移学习的方法导入预训练好的模型参数：
 在pycharm中输入`import torchvision.models.resnet`，ctrl+左键`resnet`跳转到pytorch官方实现resnet的源码中，下载预训练的模型参数：

```python
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
1234567891011
```

然后在实例化网络时导入预训练的模型参数。下面是完整代码：

```python
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from model import resnet34, resnet101
import torchvision.models.resnet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path

train_dataset = datasets.ImageFolder(root=image_path+"train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

net = resnet34()
# load pretrain weights
model_weight_path = "./resnet34-pre.pth"
missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
# for param in net.parameters():
#     param.requires_grad = False
# change fc layer structure
inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 5)
net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
save_path = './resNet34.pth'
for epoch in range(3):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')
123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899100101102103104105106107
```

## 3.3 predict.py

预测脚本跟之前的几章差不多，就不详细讲了

```python
import torch
from model import resnet34
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# load image
img = Image.open("../tulip.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = resnet34(num_classes=5)
# load model weights
model_weight_path = "./resNet34.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].numpy())
plt.show()
```