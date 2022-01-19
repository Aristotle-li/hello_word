<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210118143633733.png" alt="image-20210118143633733" style="zoom: 67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210118144638574.png" alt="image-20210118144638574" style="zoom:67%;" />



# pytorch图像分类篇：5. GoogLeNet结构详解与模型的搭建

最近在b站发现了一个非常好的 **计算机视觉 + pytorch** 的教程，相见恨晚，能让初学者少走很多弯路。
 因此决定按着up给的教程路线：图像分类→目标检测→…一步步学习用pytorch实现深度学习在cv上的应用，并做笔记整理和总结。

up主教程给出了pytorch和tensorflow两个版本的实现，我暂时只记录pytorch版本的笔记。

参考内容来自：

- up主的b站链接：https://space.bilibili.com/18161609/channel/index
- up主将代码和ppt都放在了github：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
- up主的CSDN博客：https://blog.csdn.net/qq_37541097/article/details/103482003

------

# 学习资料

1. [GoogLeNet网络详解](https://www.bilibili.com/video/BV1z7411T7ie)
2. [使用pytorch搭建GoogLeNet网络](https://www.bilibili.com/video/BV1r7411T7M5)

------

# 1. GoogLeNet网络详解

GoogLeNet在2014年由Google团队提出（与VGG网络同年，注意GoogLeNet中的L大写是为了致敬LeNet），斩获当年ImageNet竞赛中Classification Task (分类任务) 第一名。

- 原论文地址：[Going deeper with convolutions](https://arxiv.org/abs/1409.4842)
- GoogLeNet 的创新点：
  - 引入了 **Inception** 结构（融合不同尺度的特征信息）
  - 使用1x1的卷积核进行降维以及映射处理 （虽然VGG网络中也有，但该论文介绍的更详细）
  - 添加两个辅助分类器帮助训练
  - 丢弃全连接层，使用平均池化层（大大减少模型参数，除去两个辅助分类器，网络大小只有vgg的1/20）

## 1.1 inception 结构

传统的CNN结构如AlexNet、VggNet（下图）都是串联的结构，即将一系列的卷积层和池化层进行串联得到的结构。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200717120443698.png?#pic_center)

### inception原始结构

GoogLeNet 提出了一种并联结构，下图是论文中提出的inception原始结构，将特征矩阵**同时输入到多个分支**进行处理，并将输出的特征矩阵**按深度进行拼接**，得到最终输出。

- inception的作用：增加网络深度和宽度的同时减少参数。
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200717121029783.png?#pic_center)
   注意：每个分支所得特征矩阵的高和宽必须相同（通过调整stride和padding），以保证输出特征能在深度上进行拼接。

### inception + 降维

在 inception 的基础上，还可以加上降维功能的结构，如下图所示，在原始 inception 结构的基础上，在分支2，3，4上加入了**卷积核大小为1x1的卷积层**，目的是为了降维（减小深度），减少模型训练参数，减少计算量。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200717121920660.png?#pic_center)

- **1×1卷积核的降维功能**
   同样是对一个深度为512的特征矩阵使用64个大小为5x5的卷积核进行卷积，不使用1x1卷积核进行降维的 话一共需要819200个参数，如果使用1x1卷积核进行降维一共需要50688个参数，明显少了很多。
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200717122403870.png?#pic_center)
   注：CNN参数个数 = 卷积核尺寸×卷积核深度 × 卷积核组数 = 卷积核尺寸 × 输入特征矩阵深度 × 输出特征矩阵深度

## 1.2 辅助分类器（Auxiliary Classifier）

AlexNet 和 VGG 都只有1个输出层，GoogLeNet 有3个输出层，其中的两个是辅助分类层。

如下图所示，网络主干右边的 两个分支 就是 辅助分类器，其结构一模一样。
 在训练模型时，将两个辅助分类器的损失乘以权重（论文中是0.3）加到网络的整体损失上，再进行反向传播。

> 引用：[GoogLeNet(Inception V1)](https://www.cnblogs.com/itmorn/p/11230388.html)
>
> 辅助分类器的两个分支有什么用呢？
>
> - 作用一：可以把他看做inception网络中的一个小细节，它确保了即便是隐藏单元和中间层也参与了特征计算，他们也能预测图片的类别，他在inception网络中起到一种调整的效果，并且能防止网络发生过拟合。
> - 作用二：给定深度相对较大的网络，有效传播梯度反向通过所有层的能力是一个问题。通过将辅助分类器添加到这些中间层，可以期望较低阶段分类器的判别力。在训练期间，它们的损失以折扣权重（辅助分类器损失的权重是0.3）加到网络的整个损失上。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200717161450737.png?#pic_center)

## 1.3 GoogLeNet 网络参数

下面是原论文中给出的网络参数列表，配合上图查看

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200717163647478.png?#pic_center)
 对于Inception模块，所需要使用到参数有`#1x1`, `#3x3reduce`, `#3x3`, `#5x5reduce`, `#5x5`, `poolproj`，这6个参数，分别对应着所使用的卷积核个数。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200717163540388.png?#pic_center)

- `#1x1`对应着分支1上1x1的卷积核个数
- `#3x3reduce`对应着分支2上1x1的卷积核个数
- `#3x3`对应着分支2上3x3的卷积核个数
- `#5x5reduce`对应着分支3上1x1的卷积核个数
- `#5x5`对应着分支3上5x5的卷积核个数
- `poolproj`对应着分支4上1x1的卷积核个数。

# 2. pytorch搭建GoogLeNet

## 2.1 model.py

相比于 AlexNet 和 VggNet 只有卷积层和全连接层这两种结构，GoogLeNet多了 inception 和  辅助分类器（Auxiliary Classifier），而 inception 和 辅助分类器  也是由多个卷积层和全连接层组合的，因此在定义模型时可以将 **卷积**、**inception** 、**辅助分类器**定义成不同的类，调用时更加方便。

```python
import torch.nn as nn
import torch
import torch.nn.functional as F

class GoogLeNet(nn.Module):
	# 传入的参数中aux_logits=True表示训练过程用到辅助分类器，aux_logits=False表示验证过程不用辅助分类器
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Inception结构
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1) # 按 channel 对四个分支拼接  

# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x

# 基础卷积层（卷积+ReLU）
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899100101102103104105106107108109110111112113114115116117118119120121122123124125126127128129130131132133134135136137138139140141142143144145146147148149150151152153154155156157158159160161162163164165166167168169170171172
```

## 2.2 train.py

训练部分跟AlexNet和VGG类似，有两点需要注意：

1. 实例化网络时的参数

   ```python
   net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
   1
   ```

2. GoogLeNet的网络输出 loss 有三个部分，分别是主干输出loss、两个辅助分类器输出loss（权重0.3）

   ```python
   logits, aux_logits2, aux_logits1 = net(images.to(device))
   loss0 = loss_function(logits, labels.to(device))
   loss1 = loss_function(aux_logits1, labels.to(device))
   loss2 = loss_function(aux_logits2, labels.to(device))
   loss = loss0 + loss1 * 0.3 + loss2 * 0.3
   12345
   ```

## 2.3 predict.py

预测部分跟AlexNet和VGG类似，需要注意在**实例化模型时不需要 辅助分类器**

```python
# create model
model = GoogLeNet(num_classes=5, aux_logits=False)

# load model weights
model_weight_path = "./googleNet.pth"
12345
```

但是在加载训练好的模型参数时，由于其中是包含有辅助分类器的，需要设置`strict=False`

```python
missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)
```