# FCOS:一阶全卷积目标检测首发于[平凡而诗意](https://www.zhihu.com/column/sharetechlee)

![FCOS:一阶全卷积目标检测](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-309265c2ea3fea05f958c07203780be8_1440w.jpg)

# FCOS:一阶全卷积目标检测

[![Jackpop](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-3aff6dbd998567e60686173ce84e8a0b_xs.jpg)](https://www.zhihu.com/people/sharetechlee)

[Jackpop](https://www.zhihu.com/people/sharetechlee)[](https://www.zhihu.com/question/48510028)



哈尔滨工业大学 计算数学硕士

[红色石头](https://www.zhihu.com/people/red_stone_wl)等 

> 本文介绍一下近期比较热门的一个目标检测算法FCOS(FCOS: Fully Convolutional One-Stage Object  Detection)，该算法是一种基于FCN的逐像素目标检测算法，实现了无锚点（anchor-free）、无提议（proposal  free）的解决方案，并且提出了中心度（Center—ness）的思想，同时在召回率等方面表现接近甚至超过目前很多先进主流的基于锚框目标检测算法。此外，本算法目前已开源。

## **摘要**

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-75a00559139b3fe35fa433d156984256_720w.jpg)

本文提出一种基于像素级预测一阶全卷积目标检测(FCOS)来解决目标检测问题，类似于语音分割。目前大多数先进的目标检测模型，例如RetinaNet、SSD、YOLOv3、Faster R-CNN都依赖于预先定义的锚框。相比之下，本文提出的FCOS是anchor    box free，而且也是proposal  free，就是不依赖预先定义的锚框或者提议区域。通过去除预先定义的锚框，FCOS完全的避免了关于锚框的复杂运算，例如训练过程中计算重叠度，而且节省了训练过程中的内存占用。更重要的是，本文避免了和锚框有关且对最终检测结果非常敏感的所有超参数。由于后处理只采用非极大值抑制(NMS)，所以本文提出的FCOS比以往基于锚框的一阶检测器具有更加简单的优点。

------

## **锚框：Anchor box**

## **锚框介绍**

锚框首先在Faster R-CNN这篇文章中提出，后续再很多知名的目标检测模型中得到应用，例如SSD、YOLOv2、YOLOv3(YOLOv1是anchor free的)，在这里不多赘述，想要了解锚框相关内容，请查看我的另一篇文章【[Jackpop：锚框：Anchor box综述](https://zhuanlan.zhihu.com/p/63024247)】。

## **锚框缺点**

1. 检测表现效果对于锚框的尺寸、长宽比、数目非常敏感，因此锚框相关的超参数需要仔细的调节。
2. 锚框的尺寸和长宽比是固定的，因此，检测器在处理形变较大的候选对象时比较困难，尤其是对于小目标。预先定义的锚框还限制了检测器的泛化能力，因为，它们需要针对不同对象大小或长宽比进行设计。
3. 为了提高召回率，需要在图像上放置密集的锚框。而这些锚框大多数属于负样本，这样造成了正负样本之间的不均衡。
4. 大量的锚框增加了在计算交并比时计算量和内存占用。

------

## **FCOS详细介绍**

## **FCOS优势**

1. FCOS与许多基于FCN的思想是统一的，因此可以更轻松的重复使用这些任务的思路。
2. 检测器实现了proposal free和anchor  free，显著的减少了设计参数的数目。设计参数通常需要启发式调整，并且设计许多技巧。另外，通过消除锚框，新探测器完全避免了复杂的IOU计算以及训练期间锚框和真实边框之间的匹配，并将总训练内存占用空间减少了2倍左右。
3. FCOS可以作为二阶检测器的区域建议网络(RPN)，其性能明显优于基于锚点的RPN算法。
4. FCOS可以经过最小的修改便可扩展到其他的视觉任务，包括实例分割、关键点检测。

## **算法详细介绍**

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-bbec7cf563fc6c0bb981bef30bb9ca17_720w.jpg)

>  关键字： Backone、特征字塔、Center-ness

### **1.全卷积一阶检测器**

FCOS首先使用Backone CNN(用于提取特征的主干架构CNN)，另*s*为feature map之前的总步伐。

***与anchor-based检测器的区别\***

***第一点\***

- anchor-based算法将输入图像上的位置作为锚框的中心店，并且对这些锚框进行回归。
- FCOS直接对feature map中每个位置对应原图的边框都进行回归，换句话说FCOS直接把每个位置都作为训练样本，这一点和FCN用于语义分割相同。

>  FCOS算法feature map中位置与原图对应的关系，如果feature map中位置为![[公式]](https://www.zhihu.com/equation?tex=(x%2Cy)) ,映射到输入图像的位置是 ![[公式]](https://www.zhihu.com/equation?tex=(\lfloor+\frac{s}{2}+\rfloor%2Bxs%2C\lfloor+\frac{s}{2}+\rfloor%2Bys)) 。

***第二点\***

- 在训练过程中，anchor-based算法对样本的标记方法是，如果anchor对应的边框与真实边框(ground truth)交并比大于一定阈值，就设为正样本，并且把交并比最大的类别作为这个位置的类别。
- 在FCOS中，如果位置 ![[公式]](https://www.zhihu.com/equation?tex=(x%2Cy)) 落入**任何**真实边框，就认为它是一个正样本，它的类别标记为这个真实边框的类别。

>  这样会带来一个问题，如果标注的真实边框重叠，位置 ![[公式]](https://www.zhihu.com/equation?tex=(x%2Cy)) 映射到原图中落到多个真实边框，这个位置被认为是模糊样本，后面会讲到用**多级预测**的方式解决的方式解决模糊样本的问题。
>  

***第三点\***

- 以往算法都是训练一个多元分类器
- FCOS训练 ![[公式]](https://www.zhihu.com/equation?tex=C) 个二元分类器(*C*是类别的数目)

***与anchor-based检测器相似之处\***

与anchor-based算法的相似之处是FCOS算法训练的目标同样包括两个部分：位置和类别。

FCOS算法的损失函数为：

![[公式]](https://www.zhihu.com/equation?tex=\begin{aligned}+L\left(\left\{\boldsymbol{p}_{x%2C+y}\right\}%2C\left\{\boldsymbol{t}_{x%2C+y}\right\}\right)+%26%3D\frac{1}{N_{\text+{+pos+}}}+\sum_{x%2C+y}+L_{\text+{+cls+}}\left(\boldsymbol{p}_{x%2C+y}%2C+c_{x%2C+y}^{*}\right)+\\+%26%2B\frac{\lambda}{N_{\text+{+pos+}}}+\sum_{x%2C+y}+\mathbb{1}_{\left\{c_{x%2C+y}>0\right\}}+L_{\operatorname{reg}}\left(\boldsymbol{t}_{x%2C+y}%2C+\boldsymbol{t}_{x%2C+y}^{*}\right)+\end{aligned}) 

其中 ![[公式]](https://www.zhihu.com/equation?tex=L_{cls}) 是类别损失， ![[公式]](https://www.zhihu.com/equation?tex=L_{reg}) 是交并比的损失。

### **2.用FPN对FCOS进行多级预测**

首先明确两个问题：

1. 基于锚框的检测器由于大的步伐导致低召回率，需要通过降低正的锚框所需的交并比分数来进行补偿：在FCOS算法中表明，及时是大的步伐(stride)，也可以获取较好的召回率，甚至效果可以优于基于锚框的检测器。
2. 真实边框中的重叠可能会在训练过程中造成难以处理的歧义，这种模糊性导致基于fcn的检测器性能下降：在FCOSzhong ，采用多级预测方法可以有效地解决模糊问题，与基于锚框的模糊检测器相比，基于模糊控制器的模糊检测器具有更好的性能。

前面提到，为了解决真实边框重叠带来的模糊性和低召回率，FCOS采用类似FPN中的多级检测，就是在不同级别的特征层检测不同尺寸的目标。

***与基于锚框不同的地方\***

- 基于锚框的检测器将不同尺寸的锚框分配到不同级别的特征层
- FCOS通过直接限定不同特征级别的边界框的回归范围来进行分配

此外，FCOS在不同的特征层之间共享信息，不仅使检测器的参数效率更高，而且提高了检测性能。

### **3.Center-ness**

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='478' height='412'></svg>)

通过多级预测之后发现FCOS和基于锚框的检测器之间仍然存在着一定的距离，主要原因是距离目标中心较远的位置产生很多低质量的预测边框。

在FCOS中提出了一种简单而有效的策略来抑制这些低质量的预测边界框，而且不引入任何超参数。具体来说，FCOS添加单层分支，与分类分支并行，以预测"Center-ness"位置。

![[公式]](https://www.zhihu.com/equation?tex=cenerness^{*}%3D\sqrt{\frac{\min+\left(l^{*}%2C+r^{*}\right)}{\max+\left(l^{*}%2C+r^{*}\right)}+\times+\frac{\min+\left(t^{*}%2C+b^{*}\right)}{\max+\left(t^{*}%2C+b^{*}\right)}}) 

center-ness(可以理解为一种具有度量作用的概念，在这里称之为"中心度")，中心度取值为0,1之间，使用交叉熵损失进行训练。并把损失加入前面提到的损失函数中。测试时，将预测的中心度与相应的分类分数相乘，计算最终得分(用于对检测到的边界框进行排序)。因此，中心度可以降低远离对象中心的边界框的权重。因此，这些低质量边界框很可能被最终的非最大抑制（NMS）过程滤除，从而显着提高了检测性能。

## **实验结果**

**1.召回率**

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='454' height='281'></svg>)

在召回率方便表现接近目前最先进的基于锚框的检测器。

**2. 有无Center-ness的结果对比**

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='464' height='261'></svg>)

“None”表示没有使用中心。“中心度”表示使用预测回归向量计算得到的中心度。“中心度”是指利用提出的中心度分支预测的中心度。中心度分支提高了所有指标下的检测性能。

**3.与先进的一阶、二阶检测器效果对比**

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='943' height='407'></svg>)

与目前最主流的一些一阶、二阶检测器对比，在检测效率方面FCOS优于Faster R-CNN、YOLO、SSD这些经典算法。

## 开源代码

目前FCOS算法代码已经开源，

FCOS的实现基于Mask R-CNN，因此它的安装与原始的Mask R-CNN相同。安装的主要依赖如下：

- Pytorch 1.0
- torchvision
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV(可选)

安装方式有两种：

- 通过pip、conda、编译等一步一步安装
- 通过docker镜像安装

## 参考文献

------

