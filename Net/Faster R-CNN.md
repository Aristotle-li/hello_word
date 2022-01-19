https://imlogm.github.io/深度学习/rcnn/

![image-20210125133744796](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125133744796.png)



![image-20210125133931650](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125133931650.png)

## RPN (大思路：将视域空间的操作，切换到特征空间，速度快)

图比较难画就借用原文了，如上，而rpn网络的功能示意图如下。

目标分数的预测和边界框回归参数的预测

anchorgenerator ：在原图中生成anchor

<img src="https://img-blog.csdnimg.cn/20181224222746443.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhY2tlcl9sb25n,size_16,color_FFFFFF,t_70" alt="img" style="zoom:50%;" />

最开始当我们想要去提取一些区域来做分类和回归时，那就穷举嘛，搞不同大小，不同比例的框，在图像中从左到右，从上到下滑动，如上图，一个就选了9种。这样计算量很大，所以selective search利用`语义信息`改进，避免了滑动，但还是需要在原图中操作，因为特征提取不能共享。

现在不是有了sppnet和roi pooling的框架，**把上面的这个在原图中进行穷举滑动的操作，换到了比原图小很多的特征空间（比如224\*224 --> 13\*13），还是搞滑动窗口，就得到了rpn，如下图。**

<img src="https://img-blog.csdnimg.cn/20181224222800580.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhY2tlcl9sb25n,size_16,color_FFFFFF,t_70" alt="img" style="zoom:50%;" />

![img](https://img-blog.csdnimg.cn/20181224222806337.png)

rpn实现了了上面的输入输出。不同与简单滑窗的是，网络会对这个过程进行学习，得到更有效的框。

剩下的就是一些普通cnn的知识，不用多说，有了上面的这些基础后，我们开始解读代码。

 

> 02 **py-faster-rcnn框架解读**

> 03 **网络分析**

下面我们开始一个任务，就来个猫脸检测吧，使用VGG CNN 1024网络，看一下网络结构图，然后我们按模块解析一下。

<img src="https://img-blog.csdnimg.cn/2018122422303848.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhY2tlcl9sb25n,size_16,color_FFFFFF,t_70" alt="img"  />

Faster R-CNN源代码的熟悉几乎是所有从事目标检测的人员必须迈过的坎，由于目录结构比较复杂而且广泛为人所用，涉及的东西非常多，所以我们使用的时候维持该目录结构不变，下面首先对它的目录结构进行完整的分析。

3*3的滑动窗口在feature map 上滑动，每滑动到一个位置，就生成一个一维的向量，再分别通过两个全连接层，一个输出目标概率，另一个输出边界框回归参数。

2k scores：针对每一个anchor生成两个概率，一个是背景概率，一个是前景的概率（在FPN中cls layer只判断有没有目标，不判断具体的种类）。

4k coordinates：每个anchor生成4个回归参数 


![image-20210125134152703](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125134152703.png)

![image-20210125134925656](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125134925656.png)

![image-20210125135447289](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125135447289.png)

![image-20210125135847347](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125135847347.png)

> 在这里，3*3的感受野虽然没有所给定的anchor大，但是也是可以预测的，因为类比人类的视野，只看到事物的一部分，也是可以预测物体的。

![image-20210125150910708](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125150910708.png)

![image-20210125151132668](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125151132668.png)

滑窗中心点对应原图的点感受野有大量的重叠，vggNet 最后3*3 feature map感受野228，所以achor （128、256、512）也有大量重叠，回归后因为感受野的大量重叠得到的proposal必然有大量重叠，（proposal大感受野小为什么可以检测更大的目标？原文说：类比人类视觉，看到车的一部分也能判断出来是车。）

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210416111139073.png" alt="image-20210416111139073" style="zoom:50%;" />

解决正负样本不均衡：

1、用AchorTargetCreator采样一部分achor用于训练rpnHead（正负比1：1）（IoU >0.7为+ <0.3为-），前景回归，背景不回归。

最后结果得到大约20k proposal，去掉跨越边界，还有6k，基于score 使用NMS IoU设为0.7将每张图片的proposal压缩到大约2k个。

2、由于RoIs 给出的2000个候选框，分别对应feature map不同大小的区域。利用ProposalTargetCreator 挑选出128个sample_rois, 然后使用了RoIPooling 将这些不同尺寸的区域全部pooling到同一个尺度（7×7）上，（IoU >0.5为+ <0.1为-）拿去训练roi Head。

（为什么要pooling成7×7的尺度？是为了能够共享权重。在之前讲过，除了用到VGG前几层的卷积之外，最后的全连接层也可以继续利用。）



测试的时候对所有的RoIs（大概300个左右，为什么变少了) 计算概率，并利用位置参数调整预测候选框的位置。

然后再用一遍极大值抑制（之前在RPN的ProposalCreator用过）

![image-20210125151653813](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125151653813.png)

![image-20210125152018325](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125152018325.png)

![image-20210125152650959](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125152650959.png)

![image-20210125153309591](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125153309591.png)

![image-20210125153740700](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125153740700.png)

> **官方实现就是用的二值交叉熵损失，用了k个值。**

![image-20210125155221092](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125155221092.png)

> **$ t_x,t_y,t_w,t_h $是通过回归层直接预测出来的，和人工标注的GT box 对应的参数经过 smooth L1损失函数，进行反向传播，更新权值，使得预测更接近GT box**

![image-20210125155432248](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125155432248.png)

![image-20210125155804501](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125155804501.png)

![image-20210323192154860](/Users/lishuo/Library/Application Support/typora-user-images/image-20210323192154860.png)

黄色框训练时才有







## 思考

FPN和FCN的范式：

FPN：精度高，计算量大，占内存，代表网络retinaNet，FCOS

FCN：速度快

高层语义信息到底是什么？ 首先高层的语义信息一定是感受野更大的，融合了一个物体范围更大的特征。

![image-20210414192554630](/Users/lishuo/Library/Application Support/typora-user-images/image-20210414192554630.png)

![image-20210414192606494](/Users/lishuo/Library/Application Support/typora-user-images/image-20210414192606494.png)

* 细长的目标既需要感受野中心命中它，这是由短边决定的；又需要多大的感受野将其覆盖，这是由长边决定的；也就是说既需要命中多，又需要感受野大。



* 不太理解的：FPN的好处，缓解了一方面由于目标检测一直存在的特征对齐问题，就是卷积核的感受野固定形状，另一方面就是分类和回归的低相关性，彼此独立又彼此冲突了，所以这也导致anchor的问题就是如何定义正负样本，如何去平衡，引来了一系列的改进。

  







# 【个人整理】faster-RCNN的训练过程以及关键点总结

 2019-06-02阅读 7.8K0

前言

**前言：**faster-RCNN是区域卷积神经网络（RCNN系列）的第三篇文章，是为了解决select search方法找寻region proposal速度太慢的问题而提出来的，整个faster-RCNN的大致框架依然是沿袭了fast-RCNN的基本能结构，只不过在region proposal的产生上面应用了专门的技术手段——区域推荐网络（region proposal network，即RPN），这是整个faster最难以理解的地方，本文也将以他为重点进行说明。鉴于篇幅较长，本次系列文章将分为3篇来说明：

第一篇：faster-RCNN的背景、结构以及大致实现架构

第二篇：faster-RCNN的核心构件——RPN区域推荐网络

**第三篇：faster-RCNN的训练以及补充**

本次为**系列文章第三篇。**



**四、RoIHead与Fast R-CNN的进一步训练**

​       RPN只是给出了2000个候选框，RoI Head在给出的2000候选框之上继续进行分类和位置参数的回归。其实所谓的ROIHead就是对生成的候选框进行处理，这个地方与前面的fast-RCNN是一样的。

**4.1 ROIHead的网络结构**

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210416105754311.png" alt="image-20210416105754311" style="zoom:50%;" />

​       由于RoIs给出的2000个候选框，分别对应feature map不同大小的区域。首先利用ProposalTargetCreator 挑选出128个sample_rois, 然后使用了RoIPooling 将这些不同尺寸的区域全部pooling到同一个尺度（7×7）上。下图就是一个例子，对于feature map上两个不同尺度的RoI，经过RoIPooling之后，最后得到了3×3的feature map.

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210416105843319.png" alt="image-20210416105843319" style="zoom:50%;" />

RoIPooling，其实这里的ROIPooling跟fast-RCNN里面的是一样的。

   RoI Pooling 是一种特殊的Pooling操作，给定一张图片的Feature map (512×H/16×W/16) ，和128个候选区域的座标（128×4），RoI Pooling将这些区域统一下采样到 （512×7×7），就得到了128×512×7×7的向量。可以看成是一个batch-size=128，通道数为512，7×7的feature map。

​         为什么要pooling成7×7的尺度？是为了能够共享权重。在之前讲过，除了用到VGG前几层的卷积之外，最后的全连接层也可以继续利用。当所有的RoIs都被pooling成（512×7×7）的feature map后，将它reshape 成一个一维的向量，就可以利用VGG16预训练的权重，初始化前两层全连接。最后再接两个全连接层，分别是：

FC 21 用来分类，预测RoIs属于哪个类别（20个类+背景）

FC 84 用来回归位置（21个类，每个类都有4个位置参数）

**4.2 训练**

​    前面讲过，RPN会产生大约2000个RoIs，这2000个RoIs不是都拿去训练，而是利用ProposalTargetCreator 选择128个RoIs用以训练。选择的**规则如下**：

（1）RoIs和gt_bboxes 的IoU大于0.5的，选择一些（比如32个）

（2）选择 RoIs和gt_bboxes的IoU小于等于0（或者0.1）的选择一些（比如 128-32=96个）作为负样本

（3）为了便于训练，对选择出的128个RoIs，还对他们的gt_roi_loc 进行标准化处理（减去均值除以标准差）

（4）对于分类问题,直接利用交叉熵损失. 而对于位置的回归损失,一样采用Smooth_L1Loss, 只不过只对正样本计算损失.而且是只对正样本中的这个类别4个参数计算损失。

**举例来说:**

​        一个RoI在经过FC 84后会输出一个84维的loc 向量. 如果这个RoI是负样本,则这84维向量不参与计算 L1_Loss。如果这个RoI是正样本,属于label K,那么它的第 K×4, K×4+1 ，K×4+2， K×4+3 这4个数参与计算损失，其余的不参与计算损失。

**4.3 生成预测结果**

​       测试的时候对所有的RoIs（大概300个左右) 计算概率，并利用位置参数调整预测候选框的位置。然后再用一遍极大值抑制（之前在RPN的ProposalCreator用过）。

**注意：**

在RPN的时候，已经对anchor做了一遍NMS，在RCNN测试的时候，还要再做一遍

在RPN的时候，已经对anchor的位置做了回归调整，在RCNN阶段还要对RoI再做一遍

在RPN阶段分类是二分类，而Fast RCNN阶段是21分类

**4.4 模型架构图**

最后整体的模型架构图如下：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210416105935387.png" alt="image-20210416105935387" style="zoom:50%;" />

需要注意的是： **蓝色箭头的线代表着计算图，梯度反向传播会经过。而红色部分的线不需要进行反向传播**（论文了中提到了ProposalCreator生成RoIs的过程也能进行反向传播，但需要专门的算法）。

**五、faster-RCNN里面的几个重要概念（四个损失三个creator）**

**5.1 四类损失**

​      虽然原始论文中用的4-Step Alternating Training 即四步交替迭代训练。然而现在github上开源的实现大多是采用近似联合训练（Approximate joint training），端到端，一步到位，速度更快。

在训练Faster RCNN的时候有四个损失：

（1）RPN 分类损失：anchor是否为前景（二分类）

（2）RPN位置回归损失：anchor位置微调

（3）RoI 分类损失：RoI所属类别（21分类，多了一个类作为背景）

（4）RoI位置回归损失：继续对RoI位置微调

四个损失相加作为最后的损失，反向传播，更新参数。

**5.2 三个creator**

**（1）AnchorTargetCreator ：** 负责在训练RPN的时候，从上万个anchor中选择一些(比如256)进行训练，以使得正负样本比例大概是1:1. 同时给出训练的位置参数目标。 即返回gt_rpn_loc和gt_rpn_label。

**（2）ProposalTargetCreator：** 负责在训练RoIHead/Fast R-CNN的时候，从RoIs选择一部分(比如128个)用以训练。同时给定训练目标, 返回（sample_RoI, gt_RoI_loc, gt_RoI_label）

**（3）ProposalCreator：** 在RPN中，从上万个anchor中，选择一定数目（2000或者300），调整大小和位置，生成RoIs，用以Fast R-CNN训练或者测试。

​        其中AnchorTargetCreator和ProposalTargetCreator是为了生成训练的目标，只在训练阶段用到，ProposalCreator是RPN为Fast R-CNN生成RoIs，在训练和测试阶段都会用到。三个共同点在于他们都不需要考虑反向传播（因此不同框架间可以共享numpy实现）