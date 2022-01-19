**联想到人脸识别问题：** 

18.19年的人脸识别论文，讨论的基本都是如何减小类内距， 增大类间距。 从center loss, contrastive loss到triplit loss，再到基于角度margin的各种loss.  人脸识别从训练上来说本质是一个大类别的分类网络， 往往采用的是Cross-Entropy loss。 但Cross-Entropy  loss只能使得类别可分，却不能使得类别具有强的区分度。因为，从定义上来将，Cross-Entropy,   聚焦的是使得标签对应的概率值越大越好。 而非标签的值，却并不会做任何处理（因为在one hot label中，这些值为0）。

再来看MSE loss， 他优化的目标是使得，标签对应的概率值，趋向于1； 非标签对应的概率值趋向于0，  这不就是在扩大类间距离吗！！！如此好的特性，为何不用？？！！由上面的分析可以知道，  训练的收敛还是第一位的，当出现梯度消失现象，网络无法收敛的时候，再好的设计都是rubbish.

所以， 人脸识别依然选用的是Cross-Entropy来计算loss.  在此基础上，为了更进一步，增加角度margin的限制，以此扩大类间距离，才有意义。

**总结：**

1. MSE   loss和 softmax不兼容， 类别越大，越水火不容；
2. 网络训练中，网络能够收敛是第一位的，否则无从谈起；
3. 如果把softmax去掉， 避免gradient vanishing,  是不是MSE  loss就可以work了.  如果可以，姑且叫他AL2 loss吧。

首发于[AI on chip](https://www.zhihu.com/column/c_1178097838814363648)

# 人脸检测工程上的一些小trick

[![言煜秋](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-fbb66a4ba07a08732d3cf8b417569f22_xs.jpg)](https://www.zhihu.com/people/waitsop)

[言煜秋](https://www.zhihu.com/people/waitsop)

## 1 前言

检测算法整体设计上可分为one/two stage。对于轻量化的需求，我是基于yolov3改进的，这期间yolov4的出现也提供了不少的思路。

我想聊的trick主要分4方面：网络设计，训练，数据处理，工程中的传统算法的refine。

## 2.1 网络设计

## 2.1.1 网络整体结构

[准确率超99.5%！滴滴开源防疫口罩识别技术，及视觉比赛进展](https://mp.weixin.qq.com/s?__biz=MzUzODkxNzQzMw==&mid=2247487240&idx=1&sn=46f80b3057e3164645036f4c5d227c98&chksm=fad1205ecda6a94890e22d16c1df8363a70bb2a956f4ad61da6a7c4ae997a42312f5a0d118fd&mpshare=1&scene=23&srcid=1126UUo4OuMJ6fzx9zjJfTYC&sharer_sharetime=1606368556151&sharer_shareid=d40b3cbde4320259b16d48bf0090e9e2#rd&ADUIN=3287963521&ADSESSION=1618881209&ADTAG=CLIENT.QQ.5803_.0&ADPUBNO=27129)

目前基于深度学习的目标检测算法分为两类：一类是双阶段目标检测算法，另外一类是单阶段目标检测算法。通常双阶段准，单阶段快。

常见的**双阶段**目标检测算法包括Faster R-CNN、R-FCN和FPN等。该类算法在基于特征提取的基础上，有独立的网络分支生成大量的候选区域，然后对这些候选区域进行分类和回归，确定目标的准确位置框和类别。

常见的**单阶段**目标检测算法：YOLOv3、SSD和RetinaNet等。该类算法直接生成候选区域的同时进行分类和回归。

目标检测通常检测多类别，人脸检测只检测人脸+背景两类。针对人脸类别的单一性，五官特殊性发展出了大量的人脸检测算法，包括MTCNN以三个级联网络实现快速人脸检测，并利用图像金字塔实现不同尺度人脸的检测、Face R-CNN基于Faster  R-CNN框架进行人脸检测、SSH提出了对不同深度的卷积层分别进行检测以实现多尺度、FAN提出了基于锚点级的注意力机制、PyramidBox利用人脸的上下文信息提高遮挡人脸检测，即结合人头、身体等信息。上述算法主要解决不同于其他领域的人脸多尺度、遮挡等问题。

## YOLOv4中的总结

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-cf9b8fb667c724e6b9f093d97fa7b6bc_720w.jpg)one stage和two stage检测框架汇总(two stage的检测网络，相当于在one stage的密集检测上增加了一个稀疏的预测器，或者说one stage网络是 two stage的 RPN部分，是它的一个特例或子集。)

通常来说，一个检测器=在ImageNet上预训练的backbone+预测类和框的head。后来的检测器中，经常在backbone和neck之间塞很多层，作用是用来收集不同阶段的特征层，我们称之为neck。（包含多个自上向下和自下向上的paths）

**Input** **：**Image，Patches，Images Pyramid 

**Backbones：**提取特征，图像的浅层特征类似，可迁移性强 

- **GPU：**VGG16，ResNet-50，SpineNet，EfficientNet-B0/B7，CSPResNeXt50（适合分类任务），CSPDarknet53 (27.6M) （适合检测任务）
- **轻量级：**MobileNetV123 , ShuffleNet12 

**Neck：**增强特征模块，针对任务 

- **Additional blocks：**SPP（扩大感受野），ASPP in deeplabV3+，RFB，SAM 
- **Path-aggregation blocks：**FPN，PAN（路径聚合模块），NAS-FPN，Fully-connected FPN，BiFPN，ASFF，SFAM 

**Heads：**bbox, classification, regression, segmentation, heatmap ...

- **Dense Predictions(one-stage)：** 

- - RPN，SSD，YOLO（和yolov3一致），RetinaNet （基于anchor） 
  - CornerNet，CenterNet，MatrixNet，FCOS（无anchor） 

- **Sparse Predictions(two-stages)：** 

- - Faster R-CNN，R-FCN，Mask R-CNN（基于anchor）
  - RepPoints（无anchor） 

PS：

yolov4使用CSPDarknet53 （backbone）+ SPP+PAN（Neck）+ YoloV3

## 2.1.2 网络设计trick



## 正则化方法

[如果你是面试官，你怎么去判断一个面试者的深度学习水平？](https://www.zhihu.com/question/41233373/answer/145404190)

[机器学习中常常提到的正则化到底是什么意思？](https://www.zhihu.com/question/20924039)

- DropOut：[Dropout](https://zhuanlan.zhihu.com/p/61398871)

- - 一般来说dropout rate 设为0.3-0.5即可（降低拟合力换取泛化能力）。注意rescale。
  - 多用于密集网络，全连接部分，有时可用于CNN第一层卷积（相当于加噪声的作用，卷积核小时效果较差）。dropout被广泛地用作全连接层的正则化技术，但是对于卷积层，通常不太有效。dropout在卷积层不work的原因可能是由于卷积层的特征图中相邻位置元素在空间上共享语义信息，所以尽管某个单元被dropout掉，但与其相邻的元素依然可以保有该位置的语义信息，信息仍然可以在卷积网络中流通。

- SpatialDropout：按channel随机扔

- Stochastic Depth：按res block随机扔

- DropConnect

- DropPath

- Cutout：在input层按spatial块随机扔

- DropConnect：只在连接处扔，神经元不扔。

- **DropBlock：**每个feature map上按spatial块随机扔，优势很大 [dilligencer：DropBlock](https://zhuanlan.zhihu.com/p/142299442)

- - 在卷积层进行dropout来防止网络过拟合（b）。
  - 在featuremap上去一块一块的找，进行归零操作，叫做dropblock（c）。
  - dropblock有三个比较重要的参数，一个是block_size，用来控制进行归零的block大小；一个是γ，用来控制每个卷积结果中，到底有多少个channel要进行dropblock；最后一个是keep_prob，作用和dropout里的参数一样。
  - 我们发现，除了卷积层外，在**skip connection（short cut）**中应用DropbBlock可以提高精确度。此外，在训练过程中，逐渐增加dropped unit的数量会导致更好的准确性和对超参数选择的鲁棒性。（keep_prob从1逐渐减小到目标值如0.9的线性方案）

<br> (二维码自动识别)

**类间过拟合**

- **Label smoothing 标签平滑**：softmax后和label计算交叉熵loss。把这个label从[0,0,1,0]变为[0.01，0.01，0.95，0.01]
- 细分类效果往往较好（如行人，车辆-->行人，自行车，轿车，货车）。

## 平衡正负样本

**Focal loss**

**OHEM(在线难分样本挖掘):**  

[醒了么：OHEM论文解读](https://zhuanlan.zhihu.com/p/58162337) 

我们知道，基于SVM的检测器，在训练时，使用hard example mining来选 择样本需要交替训练，先固定模型，选择样本，然后再用样本集更新模型， 这样反复交替训练直到模型收敛

作者认为可以把交替训练的步骤和SGD结合起来。之所以可以这样，作 者认为虽然SGD每迭代一次只用到少量的图片，但每张图片都包含上千 个RoI，可以从中选择hard  examples，这样的策略可以只在一个mini-batch中固定模型，因此模型参数是一直在更新的。

更具体的，在第t次迭代时，输入图片到卷积网络中得到特征图，然后把特征图和所有的RoIs输入到RoI网络中并计算所有RoIs的loss，把loss从高到低排序，然后选择B/N个RoIs。这里有个小问题，位置上相邻的RoIs通过RoI网络后会输出相近的损失，因此使用了NMS(非最大值抑制)算法（先把损失按高到低排序，然后选择最高的损失，并计算其他RoI这个RoI的IoU，移除IoU大于一定阈值的RoI，然后反复上述流程直到选择了B/N个RoIs。）

**TODO**：

OHEM只能用于two-stage模型，且较影响训练速度。能否在one-stage中找到一种近似的解决方案？

提取最后输出的特征图的置信度通道，和label相减，得到loss较大区域。

## loss

置信度loss+类别loss+回归框loss

class：原本YOLOV3的损失函数在obj loss部分应该用二元交叉熵损失的，但是作者在代码里直接用方差损失代替了。

## 回归loss改进

[目标检测回归损失函数小结IOU、GIOU、DIOU、CIOU_yx868yx的博客-CSDN博客](https://blog.csdn.net/yx868yx/article/details/106794233?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242)

- GIOU：考虑两框不重合时iou一直是0，不管距离多远没区别的情况。
- DIOU：考虑两个框距离
- **CIoU：长宽比**
- MSE



## 学习率

Cosine学习率：余弦退火的原理：因为我们的目标优化函数可能是多峰的，除了全局最优解之外还有多个局部最优解，在训练时梯度下降算法可能陷入局部最小值，此时可以通过突然提高学习率，来“跳出”局部最小值并找到通向全局最小值的路径。这种方式称为**带重启的随机梯度下降方法**。（可以先warm up再从下降开始余弦）

[机器学习入坑者：pytorch必须掌握的的4种学习率衰减策略](https://zhuanlan.zhihu.com/p/93624972)

## SAM

空间attention改为逐点attention。

**注意力机制**

SE等：[仿佛若有光：CVPR2021| 继SE,CBAM后的一种新的注意力机制Coordinate Attention](https://zhuanlan.zhihu.com/p/363327384)

CBAM：[注意力模型CBAM_年轻即出发，-CSDN博客](https://blog.csdn.net/qq_14845119/article/details/81393127)

SE：

- Glogalavgpooling+fc（压）+nonlinear+fc+sigmoid。
- 吃显存。
- 加在提取信息分支上（加在residual而不是shortcut）
- 加在分割效果一般。（空间注意力对分割效果还行）

## PAN 

[chaser：Path Aggregation Network 论文阅读](https://zhuanlan.zhihu.com/p/66413042)

[目标检测 - Neck的设计 PAN（Path Aggregation Network）](https://blog.csdn.net/flyfish1986/article/details/110520667)

PAN简单理解就是FPN多了一条Bottom-up path  augmentation（b）。FPN是从上向下（a），PAN包含了从上向下和从下向上的路径。自适应特征图池化，充分利用更多尺度特征进行proposals的detection，避免固定分配level带来的信息断层 * 不同方式得到的特征进行FC融合，预测出更好的mask。

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-cf3c576f99f2cf21a376c211eadf0589_720w.jpg)

下采样、卷积、element wise sum，然后卷积消除混叠效应。（add后的特征不连续，特征混乱）

yolov4改进：快捷链接通道的add改为concat。

## **跨Batch的批归一化（CmBN）**

BN - CBN - CmBN

BN：[王二的石锅拌饭：网络中BN层的作用](https://zhuanlan.zhihu.com/p/75603087)

- 正则化的一种方式（防过拟合），用了之后可以抛弃drop out、L2正则项参数了。
- 减少对初始化的强烈依赖。
- 允许更大的学习率，提速。
- 控制梯度爆炸防止梯度消失（网络前面不动了后几层一只学）。

## **增大感受野**

- SPP
- ASPP
- RFB

## 注意力机制

- Squeeze-and-Excitation (SE)， 增加2%计算量（但推理时有10%的速度），可以提升1%的ImageNet top-1精度。
- Spatial Attention Module (SAM)，增加0.1%计算量，提升0.5%的top-1准确率。

## 激活函数

- ReLU

- LReLU

- PReLU：难训练

- SELU：难训练

- ReLU6：为量化网络设计的

- **Mish**：x * tanh(ln(1+e^x))

- - 理论上对负值的轻微允许允许更好的梯度流，而不是像ReLU中那样的硬零边界。

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-5e3189067b966b544288c675901ef5f4_720w.jpg)

- Swish：x * sigmoid(x)
- hard-Swish



## 特征融合集成

- FPN
- SFAM
- ASFF
- BiFPN （也就是大名鼎鼎的EfficientDet）

## 后处理非最大值抑制算法

- soft-NMS
- **DIoU NMS**

## 跨阶段空间连接（CSP）

## 多输入权重残差连接

## SPP-block（multi input weighted residual connections）

[SPP(Spatial Pyramid Pooling)详解](https://www.cnblogs.com/zongfa/p/9076311.html)

可以提升模型的感受野、分离更重要的上下文信息、不会导致模型推理速度的下降。

原始spp：

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-1759d65c2f8a70afb0f77629ed1178f2_720w.jpg)

spp模块结构：

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-421f57c8b67c4ed58a149b33316bfafc_720w.jpg)



## 关于anchor

映射：对shift和scale的容忍度，yolov4的改进

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-0b331dde79cf947a28f3d17bdbc564d6_720w.jpg)

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-4c4c20ec320358ef639aa34a2c4100af_720w.jpg)

- 数据集的框映射到yolo输出shape上，有一点量化的思想。映射方式yolov4有更新：

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-681825e2072ce8d96f960fd60d61f89b_720w.jpg)

- 输出的loss计算，MSE没考虑到四个点之间的联系。
- xywh直接预测训练不稳，所以加平移（sigmoid）缩放（指数）。其中平移用sigmoid说明中心位置的定为更重要，scale用指数，重视预测偏小的情况。yolov4更新：
- 设定base的HW值：使用k-means对数据集或测试集聚类。
- 分配数量调节

## 2.2 训练

## 使用遗传算法选择最优超参数

## 自对抗训练  SAT （Self-Adversarial Training）

一种数据增强手段：在一张图上，让神经网络反向更新图像，对图像做改变扰动，然后在这个图像上训练。这个方法，是图像风格化的主要方法，让网络反向更新图像来风格化图像。

## 对新数据不敏感

[周翼南：深度挖坑6：关于人脸识别新数据finetune中的一个小trick](https://zhuanlan.zhihu.com/p/61587832)

实际在做训练模型的项目中，随着新数据的逐渐增加，对新的数据进行finetune是必不可少的一步，在实际的finetune过程中，有可能出现神经网络对新出现的数据不敏感的情况。（训不动）

造成这种现象的一个原因是weight-decay（[权重衰减（L2正则化）and学习率衰减](https://blog.csdn.net/program_developer/article/details/80867468)，[weight decay，momentum，normalization](https://www.zhihu.com/question/24529483)）。在训练baseline的过程中，我们会选择一个合适的weight-decay来增强我们模型的泛化能力。问题在于，weight-decay会将一些对于baseline数据集无关的kernel置为0，也就是我们常将的dead kernel。这些dead  kernel在finetune的时候是不会受到新数据的激活的（已经变成0了也就没有梯度了）。这样其实我们是在一个缩减版的网络上来进行我们新数据的finetune，效果会大打折扣。

解决方法：保持weight-decay的情况下不让这些kernel变成0就可以了。

原来我们的网络结构一般是conv(bias=false)+bn+scale+relu，那么weight-decay的问题作用在这个scale层里面的scale_factor上面（也就是variance），所以我们只需要将这个variance固定为1，就能够很好的解决这个问题了。

在mxnet  中的实现只需设置fix_gamma=true。对于pytorch，官方并没有提供这个实现，这里我们可以将网络重新设计为:bn+conv(bias=true)+relu，然后将bn中的affine参数（默认是True,即使用BN时加入可训练的参数gamma、beta）设为false即可（bn+scale两个等式合并后为ax+b的形式，其中a包含gamma,mean,var，b包含gamma,beta,mean,var）。

我认为这是一个实际工程中对于任何finetune问题都通用的一个小trick，希望对于从业人员能够有所帮助。当然不只是对于finetune，因为这个操作最大化的利用了kernel，所以训练出来的模型无法进行减枝等一系列操作，但是如果你不想在对设计好的小网络进行剪枝，那么这个功能将能够使你网络的capability最大化。



## 2.3 数据处理

## 数据增强

- 图像扰动
- 改变亮度、对比度、饱和度、色调
- 加噪声
- 随机缩放
- 随机裁剪（random crop）
- 翻转
- 旋转
- 随机擦除（random erase）

以下两种方式对遮挡问题有较好的效果，实际使用是可以自行设计，如将目标框内的部分擦除，补零或者使用背景的部分填充。

- Cutout：CutOut能够让CNN利用整幅图像的全局信息，而不是一些小特征组成的局部信息。
- Random Erasing：遮挡问题



- MixUp：0.3*图A+0.7*图B（半透明叠加）
- **CutMix**
- **Mosaic：**四图拼一图，变相增大了一次训练的图片数量，让最小批数量降低。（方便单GPU训练）PS：一种代替Multi-scale training的方式，如[SNIPER算法](https://www.cnblogs.com/kk17/p/9807571.html)，一般检测网络中会有多层检测，训练的时候统一把所有检测层的结果和ground truth进行计算loss，然后反传。SNIPER则是令不同的检测层分别负责不同尺度范围的ground  truth，不在自己负责范围内的ground truth则不参与计算loss。
- 对逆光：压暗人像（分割结果）亮度
- resize方式：保持HW比，pad填充128，**局部平均下采样**（**暗图有效**，平均采样有减少噪声影响的作用）。
- 直方图均衡/先随机调节亮度再直方图均衡：作用有两个：1.增强对比度。2.归一化图像亮度：用来使图像具有相同的亮度，如果数据集图像具有不同的亮度，那么需要在训练之前进行直方图均衡化，达到相同的亮度条件。

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-a6113867a6dfa5812307dba4315a10e6_720w.jpg)



## 2.4 工程上的传统算法refine

## 检测定位偏离

马尔科夫随机场

canny





## 流程改进

**两阶段检测**

- 检测低置信度（如阈值为0.5，选取置信度在0.2~0.5部分的框）所有候选框内平均亮度（各个框中心部分，然后根据框大小归一化），过低则抬亮原图进入第二轮检测，根据前后两阶段某框的置信度之比确定是否选择（抬亮后置信度提升大则保留）。
- 训练二分类网络，二次检测。（针对小物体检测对圆形的误检）

**维纳滤波**

时域上增强框的稳定性。

## 3 针对问题

## 小目标检测（尺度问题）

卷积网络具有较好的平移不变性，但是对尺度不变性有较差的泛化能力，现在网络具有的一定尺度不变性、平移不变性往往是通过网络很大的capacity来“死记硬背”，小目标物体难有效的检测出来，主要原因有：1.物体尺度变化很大，CNN学习尺度不变性较难。2.通常用分类数据集进行预训练，而分类数据集物体目标通常相对较大，而检测数据集则较小。这样学习到的模型对小物体更难有效的泛化，存在很大的domain-shift。3.有的CNN的卷积核stride过大，小尺度中的小物体很容易被忽略，下图可以很好的体现这一问题。

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-5d9400f720956520f25ab15477443e69_720w.jpg)

对小物体检测提升大，还能防止过拟合（误检降低）：

- mosaic：结合tf.data和Estimator（tf）/设置Dataloader中的num_workers和pin_memory（pytorch）的前处理和训练的并行技巧，基本不会额外增加多少训练耗时（不到3%）。
- Multi-scale training：SNIP/SNIPER，CSN等。

训练方式：

- 对yolov3的s_box分支的迁移训练：先训练l,m分支，再固定后训练s分支。

网络设计：

- 使用**空洞卷积**(dilated/atrous convolution)。相比普通降采样，不丢失分辨率，且仍然扩大感受野。还可以通过dilation rate的设置捕获多尺度上下文信息**。**
- 训练预测前处理按照一定比例**放大。**
- 使用CNN中不同层的特征分别进行预测，浅层负责检测小目标，深层负责检测大目标。
- 深浅层特征结合进行预测，使浅层特征结合深层的语义特征，如FPN。但当目标尺寸较小时特征金字塔生成的高层语义特征也可能对检测小目标帮助不大。



## 4 参考

[【尺度不变性】An Analysis of Scale Invariance in Object Detection - SNIP 论文解读](https://www.cnblogs.com/kk17/p/9807571.html)

[准确率超99.5%！滴滴开源防疫口罩识别技术，及视觉比赛进展](https://mp.weixin.qq.com/s?__biz=MzUzODkxNzQzMw==&mid=2247487240&idx=1&sn=46f80b3057e3164645036f4c5d227c98&chksm=fad1205ecda6a94890e22d16c1df8363a70bb2a956f4ad61da6a7c4ae997a42312f5a0d118fd&mpshare=1&scene=23&srcid=1126UUo4OuMJ6fzx9zjJfTYC&sharer_sharetime=1606368556151&sharer_shareid=d40b3cbde4320259b16d48bf0090e9e2#rd&ADUIN=3287963521&ADSESSION=1618881209&ADTAG=CLIENT.QQ.5803_.0&ADPUBNO=27129)

[周翼南：深度挖坑6：关于人脸识别新数据finetune中的一个小trick](https://zhuanlan.zhihu.com/p/61587832)

[如何评价新出的YOLO v4 ？](https://www.zhihu.com/question/390191723?rf=390194081)

[目标检测存在的问题总结_lzg13663472636的博客-CSDN博客_目标检测存在的问题](https://blog.csdn.net/lzg13663472636/article/details/103195319)

[收藏！YOLOv4全文解读与翻译总结！（附思维导图和论文译文）](https://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247489140&idx=1&sn=7ec16d43b182dd56b2d975bb3cd05adf&chksm=fcb70acbcbc083ddbab8be612388135e4cc5ebee00b301755404c66f3da51f7ebd6a17f89eba&mpshare=1&&srcid=&sharer_sharetime=1587900548327&sharer_shareid=8a987be85d15ba23c8b37882f59430b7&from=timeline&scene=2&subscene=1&clicktime=1587901287&enterid=1587901287#rd)





