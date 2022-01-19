首发于[AI on chip](https://www.zhihu.com/column/c_1178097838814363648)

# MTCNN实时人脸检测从理论到嵌入式工程实现

[![言煜秋](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-fbb66a4ba07a08732d3cf8b417569f22_xs-20211222215343604.jpg)](https://www.zhihu.com/people/waitsop)

[言煜秋](https://www.zhihu.com/people/waitsop)

## 前言

个人认为MTCNN在人脸检测中的地位和Mask R-CNN在分割中的地位差不多。大浪淘沙，有的paper在刷了几个点的榜之后没几个月就销声匿迹，有的却在“过时”后还能为人铭记。

学习经典的算法还有一个好处，就是大家都已经把它吃透了，github上基于各个框架的复现，工程实现等等可以说是目前人脸检测算法中最多的。亲手跑一遍MTCNN，可以分成四个阶段：

1 看论文，跑一下别人的test代码，或者用别人封装好的库。稍作添改使之可以测试单张图，批量跑图并测速，视频测试，摄像头实时测试。

2 自己下数据集训练，了解数据集是怎么划分并训练的，熟悉如何进行多阶段的训练（预处理，分阶段训练，loss等细节）。

3 徒手复现：如果想要学习某个没用过的框架可以对着自己熟悉的框架复现，因为有各种开源的代码所以不用担心没有参考。理想目标是可以看着论文复现（当然效果可能会下降）。

4 调参：虽然很多新手经常被调侃成调参侠，但是懂的都懂，炼丹不是真的随缘，其功力是随着一个人的数感和实战经验提升的。真到了随便调调就总是可以超越论文作者给出的point，那“调参侠”也就成为褒义词了。

5 工程落地 AI落地中最困难的是落地到移动端嵌入式设备中，要考虑的不只是效果和速度，针对某平台的功耗也要综合考虑进去。需要具备的能力有：模型量化压缩，C++编程，inference引擎优化加速，部署（嵌入式设备测试or服务器高并发）等。

## MTCNN（Multi-task Cascaded Convolutional Networks）

**paper：**

[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://kpzhang93.github.io/MTCNN_face_detection_alignment/)

[kpzhang93.github.io/MTCNN_face_detection_alignment/![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-7256752aa9ad9d4c75a6a73c0be8ae14_180x120.jpg)](https://kpzhang93.github.io/MTCNN_face_detection_alignment/)

**code：**放一些我看过的，其中很多是没有训练代码的，还有一些是写好的库

官方matlab：   [kpzhang93/MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)

all platforms:    https://github.com/imistyrain/MTCNN

tensorflow ：   https://github.com/AITTSMD/MTCNN-Tensorflow

https://github.com/LeslieZhoa/tensorflow-MTCNN

https://github.com/ipazc/mtcnn

caffe：              https://github.com/DuinoDu/mtcnn

pytorch：         https://github.com/kuaikuaikim/DFace

ZQCNN：         https://github.com/zuoqing1988/ZQCNN-MTCNN-vs-libfacedetection

https://github.com/zuoqing1988/ZQCNN

**PS：**为了讲的细点又让人看着不那么枯燥，采用由浅入深的方式，慢慢补充细节。

## **概括**

为了解决人脸检测和对齐提出的级联结构，又快又准，但训练慢。

## **流程**

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-22f88f92f62f0dfc347909a7049d0c8a_720w.jpg)MTCNN流程

首先流程很简单： **Image → (Cut when training/Image pyramid when inference) → Pnet → Rnet → Onet** 

推理过程：先生成图像金字塔，依次进三个网络，每个网路都**输出 classification, bbox, landmark 三种信息**。

训练过程：见下文训练细节1

从名字也可以看出： P-Net（Proposal Network) 生成大量候选框，R-Net（Refinement  Network）对其精修，O-Net（Output Network）输出最终bbox和landmarks。整体是一个由粗到细的过程。  [如果比作倒水，那Pnet是汤勺，Rnet是小勺，Onet是吸管]

三个网络看着差不多，具体区别还是要看一下网络内部：

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-372ca1ff70a36c1dfc5b146df4df8541_720w.jpg)三个网络的结构

## Image pyramid(推理)

输入图片按照某比例resize（官方的缩放因子选择0.709） N次,得到了N+1层金字塔。最上面也就是最小的一层要 ≥ 12x12（Pnet的输入）

## Pnet: 12x12x3 → 5x5x10 → 3x3x16 →（ 1x1x2+1x1x4+1x1x10）

上述金字塔的每一层都要进Pnet，每12x12x3作为一个bbox输入，输出是1x1x32（2分类+4bbox回归（四个偏移量）+10landmark（5个点的坐标））

此过程先根据分类得分筛一次，再根据预测框IoU进行NMS再筛一次（此处是重复进行的）。最后得到x个候选框，每个候选框映射到原图上切片（实际会多按照bbox最大边取正方形，为了和后面的1：1输入对应，防止形变影响），然后resize到24x24，作为Rnet的输入。

## Rnet: 24x24x3 → 11x11x28 → 4x4x48 → 3x3x64→ FC128 2+4+10 

和Pnet一样，输出（2分类+4回归+10landmark），同样是筛两次（根据分类得分一次+IOU NMS多次），最后把得到的切片resize到48x48作为Onet的输入。

## Onet: 48x48x3 → 23x23x32 → 10x10x64 → 4x4x64 → 3x3x128 → FC 256(2+4+10)

再次重复上述流程，得到最终的坐标和landmark点。

## 损失函数

看起来三个网络差不多，除了网络中间卷积池化的操作不同，还有一个重要的区别就是损失函数。

首先分类，回归，landmark 三个误差都有对应的损失函数：

分类的cross-entropy loss:

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-d3dbd8cf5df98cc2c6ec160b3aa42163_720w.png)

回归的Euclidean loss：

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-dd6c3940517c378b68ce51c2d8958293_720w.png)

landmark的Euclidean loss：

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-3272064c95ee1f4406a2eeebf4206154_720w.png)

PRO每个网络都是综合考虑三种损失函数，但各自考虑三种损失的权重不同，采用以下公式：

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-e95f32858369177f9e583c832318c4b8_720w.png)

α：表示对应网络三种损失的权重。

β：判断feed给网络的图片的“成分”，neg只贡献分类损失，part贡献分类+回归损失，pos贡献分类+回归+landmark三种损失（关于neg part pos 后面会有介绍）。

min：此处的min在原文中被命名为“Online Hard sample mining”，对每个batch的图片只取分类损失在前70%的训练（这部分样本就是“hard  samples”。（前人也有提出舍弃easy samples的思想，但本文的是在反向传播前就舍弃了这部分easy  samples，这也是“online”的含义），这样做节省训练时间的同时获得了更好的效果。其思想很简单：当你教一个两岁小孩辨认男女的时候，会用特征很明显的男人或女人的照片还是用长相很中性的人的照片？

## 训练细节

\1. 训练数据由四部分组成：pos, part, neg, landmark，比例为1：1：3：1。

此过程是在image pyramid之前对训练数据的处理，pos,part,neg是人脸数据集（[WIDER FACE](http://shuoyang1213.me/WIDERFACE/)+[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)+[Deep Convolutional Network Cascade for Facial Point Detection](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)）的图片随机裁剪得到的，裁剪部分与ground truth的最大的IoU值大于0.65的为pos图像，0.4和0.65之间的为part图像，小于0.3的为neg图像，landmark截取的是带有关键点的部分。

PS1：很多图片中人脸占比较小，纯随机截取会导致pos和part数量远少于neg，所以在原本的人脸ground truth进行微调截取以得到更多的pos和part。

PS2：每类数据的 label 不同

pos 和 part 的 label 有两种：

- 类别：‘1’,‘-1’（上文提到的β就是通过这个label来判断图片类别，比如此label是-1，表示是part，则β输出（1 ，1， 0），即三种损失只计算分前两种）
- 偏移量：截出的pos，part截图相对于实际人脸框label（ground truth）的左上角（x1,y1）和右下角(x2,y2)的偏移量（偏移量最后要除以截图的大小做了归一化）。

neg 的 label 只含一种：

- 类别：‘0’

landmark 的 label 有两种：

- 类别：‘-2’ 
- 偏移量：5个关键点的坐标偏移量（同样也要做归一化）。



2.把图片输入模型的时候要对每个像素做(x – 127.5)/128的归一化操作 → 加快收敛速度。

3.分类只用pos和neg训练，bbox回归用pos和part，landmark用landmark。这就是"MT"Multi-task的本质思想。

4.实际上每次bbox都是沿用之前网络的预测结果，而landmark不是，仅在最后的Onet做预测。


 

编辑于 2019-11-20 17:54

「真诚赞赏，手留余香」

还没有人赞赏，快来当第一个赞赏的人吧！

[人脸识别](https://www.zhihu.com/topic/19559196)

[图像识别](https://www.zhihu.com/topic/19588774)

[嵌入式开发](https://www.zhihu.com/topic/19610823)

### 文章被以下专栏收录

- [![AI on chip](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-4092259e0fc66d3a7d2fc134848cf24e_xs-20211222215343863.jpg)](https://www.zhihu.com/column/c_1178097838814363648)

- ## [AI on chip](https://www.zhihu.com/column/c_1178097838814363648)

- 深度学习算法部署嵌入端

### 推荐阅读



[![[树莓派\]人脸识别+活体检测 加载2800+人脸数据还能达到20FPS！](https://pic2.zhimg.com/v2-bdb9f65f3e44db27be40cd2198ce1884_250x0.jpg?source=172ae18b)[树莓派]人脸识别+活体检测 加载2800+人脸数据还能达到20FPS！阿尔姆斯基](https://zhuanlan.zhihu.com/p/166283422)[人工智能-OpenCV+Python实现人脸识别（视频人脸检测）上期文章我们分享了opencv识别图片中的人脸， OpenCV图片人脸检测，本期我们分享一下如何从视频中检测到人脸 视频人脸检测 OpenCV打开摄像头特别简单，只需要如下一句代码 capture = cv2.Vi…人工智能研究所](https://zhuanlan.zhihu.com/p/146284417)[![如何用OpenCV在Python中实现人脸检测](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-1d107a7db6623f4d901c508c65b220dc_250x0.jpg)如何用OpenCV在Python中实现人脸检测机器之心发表于机器之心](https://zhuanlan.zhihu.com/p/67548563)[人脸识别完整项目实战课程试看 案例驱动 实战开发 源码剖析 项目概述篇：系统介绍人脸识别项目的系统架构设计、项目关键技术说明、项目业务需求分析、项目业务流程设计； 环境部署篇：提供C++和Python两种编程语…菲满树](https://zhuanlan.zhihu.com/p/93497251)

## 还没有评论

评论区功能升级中