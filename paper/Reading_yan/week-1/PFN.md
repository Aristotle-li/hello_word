金字塔本来在目标检测领域应用很广泛，但是随着深度学习的发展，由于金字塔结构计算复杂浪费内存被放弃了，但是本文提出了基于卷积神经网络固有的多尺度金字塔层次结构来构造特征金字塔，不需要额外计算。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210401225637215.png" alt="image-20210401225637215" style="zoom:50%;" />

金字塔为什么有用：

因为大目标卷10层完美提取特征，小目标卷十层卷没了，也许五层的时候才是最完美的，当然不是这么简单因为还有top-down结构，接收高层语义信息。但是很容易理解，目标检测不仅要在一个尺度扫描，还要在不同尺度扫描，目标的比例变化需要通过移动其在金字塔中的高度来抵消。

目标检测模型通过在位置和金字塔等级上扫描来检测大范围尺度上的对象。



FPN：

结构：基于卷积神经网络固有的多尺度金字塔层次结构来构造特征金字塔，具有横向连接（传递更精确的特征位置）的自顶向下（很强的语义特征和很好的分辨率，位置并不精确）结构，用于构建各种尺度的高级语义特征图，作为一个通用的特征提取器使用。

同层具有相等的spatial size ，同层的bottom-up层 subsampled fewer times.所以定位准确，但是高层语义少，top-down层相反。

做法：merge同层的top-down和bottom-up。how to merge？

对于较粗分辨率的特征图，我们将空间分辨率提高了2倍（使用最近邻上采样）。然后，通过逐元素加法将上采样映射与相应的自底向上映射（该映射经历1×1卷积层以减少通道维数）合并。此过程将迭代，直到生成最精细的分辨率贴图。





细节：我们在每个merged map 上附加一个3×3卷积来产生最终的特征地图，以减少上采样的混叠效应。固定了所有feature map中的channel（通道数，表示为d）。在本文中，我们设置d=256，因此所有额外的卷积层都有256个通道输出。在这些额外的层中没有非线性，我们根据经验发现这些层的影响很小。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210401160638265.png" alt="image-20210401160638265" style="zoom: 50%;" />

优点：1、端到端训练，训练/测试时都能使用，图像金字塔在训练中不能用。2、和single-scale baseline相比，没增加测试时间性能大大提高

CNN：

1、能够表示更高层次的语义。2、卷积神经网络对尺度上的差异也更有鲁棒性，从而便于从单个输入尺度上计算的特征进行识别



featured image pyramid：

优点：每一层进行特征化的主要优点，它产生了一种多尺度特征表示，其中所有的层都是语义强的，包括高分辨率的层。

缺点：1）推理时间大大增加，2）train一个end to end 的deep nn就内存而言不够，3）如果train不用，test用造成推断时间不一致



 Pyramidal feature hierarchy：

不同：没有topdown

缺点：深度不同，引入较大语义鸿沟。高分辨率的feature map含有low-level features，损害了其对目标识别的表征能力



SSD：

缺点：使用了 Pyramidal feature hierarchy，但是为了防止low-level features，放弃了high-resolution maps，但是这对检测小目标很重要。



RPN、Fast R-CNN：

滑动窗口提议器（Region Proposal Network，简称RPN）[29]和基于区域的检测器（Fast R-CNN）

RPN[29]是滑动窗口类不可知对象检测器。在最初的RPN设计中，在一个单尺度卷积特征映射的基础上，在密集的3×3滑动窗口上对一个小的子网络进行评估，执行对象/非对象二值分类和边界盒回归。这是通过一个3×3卷积层和一对1×1卷积来实现的，用于分类和回归，我们称之为 network head。对象/非对象标准和边界框回归目标是相对于一组称为锚的参考框定义的[29]。锚定具有多个预定义的比例和纵横比，以便覆盖不同形状的对象。

我们在特征金字塔的每一层都附加一个相同设计的 network head（3×3 conv和一对1×1 conv），the head slides densely over all locations in all pyramid levels, it is not necessary to have multi-scale anchors on a specific level。本来RPN是在单一feature map上设置不同大小的anchor，用FPN在每一层都接一个RPN那么每一个只用一个大小就可。{322 , 642 , 1282 , 2562 , 5122 }pixel on {P2, P3, P4, P5, P6} respectively. use anchors of multiple aspect ratios {1:2, 1:1, 2:1} at each level. 所以金字塔有15个anchor。

共享参数：network head 的参数在所有要素金字塔级别之间共享（不共享参数方案区别不大），共享参数的良好性能表明我们金字塔的所有级别共享相似的语义级别。

RPN中anchor 的train label是通过计算与 ground-truth boxes的IoU来分配的。positive ：1、highest IoU 2、over 0.7  negative：lower than 0.3



Fast R-CNN：

在ROI pooling层使用FPN，将不同尺度的roi分配给金字塔层，we assign an RoI of width w and height h (on the input image to the network) to the level Pk of our feature pyramid by:<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210401190211799.png" alt="image-20210401190211799" style="zoom:50%;" />

 predictor heads attach 到所有level 的所有RoI，同样均共享参数，



rpn训练细节：

表1中的所有架构都是经过端到端培训的。调整输入图像的大小，使其较短的一侧有800个像素。我们在8个GPU上采用同步SGD训练。一个小批量包含每个GPU 2个图像和每个图像256个锚。我们使用0.0001的重量衰减和0.9的动量。前30k小批量的学习率为0.02，后10k小批量的学习率为0.002。对于所有RPN实验（包括基线），我们将图像外的锚定框包括在内进行训练，这与[29]中忽略这些锚定框的情况不同。其他实现细节如[29]所示。在8个GPU上用FPN训练RPN大约需要8个小时

Fast/Faster R-CNN训练细节：

调整输入图像的大小，使其较短的一侧有800个像素。采用同步SGD在8gpu上对模型进行训练。每个小批量涉及每个GPU 2个图像和每个图像512个ROI。我们使用0.0001的重量衰减和0.9的动量。前60k小批量的学习率为0.02，后20k小批量的学习率为0.002。在COCO数据集上用FPN训练Fast R-CNN大约需要10个小时。