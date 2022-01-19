## day 1

adBoost+

随机森林

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210307231749792.png" alt="image-20210307231749792" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210307231942056.png" alt="image-20210307231942056" style="zoom:50%;" />



最难的部分是对于数据本身的探索性分析，所以他会先从头开始理解数据集，先探索数据本身，思考数据的产生以及收集过程、训练集和测试集的划分、各个表项目的具体含义等等，并不着急建模。

### 2、

数据分析

特征工程

模型训练

线下验证

### 3、

赛题介绍：全球小麦检测

目标检测：框选出小麦头的位置

难点：

1、密集的小麦植物经常重叠

2、风会使照片模糊

3、外观会因为成熟度，颜色基因型和头部方向而异

4、数据样本分析：训练集样本bbox分布、拍摄环境不同导致光线有亮有暗

map 评价指标

gtx1080

rtx2080Ti

5、baseline：就选最先进的模型就好了

efficientDet ， YoloV5

6、基础数据增强

质量：HSV通道颜色变换，亮度，对比度变换

数量：水平垂直翻转，转灰度图，随机裁剪

进阶：

马赛克

Mixup

cutout：随机把样本部分区域cut，

cutmix

训练策略：

K-fold训练：数据集分成k份，k-1份训练，1份验证，做k轮

自适应调整学习率

LambdaLR:将每一个参数的学习率设置为学习率lr的某个函数倍

```text
export PATH="/home1/hehao/cuda-10.1/bin:$PATH"
export LD_LIBRARY_PATH="/home1/hehao/cuda-10.1/lib64:$LD_LIBRARY_PATH"
```

## day 2

1、目标检测通用trick

离线增强:直接对数据集进行处理，数据的数目会变成增强因子x原数据集的数目，这种方法常用于数据集很小的时候

在线增强：这种增强的方法用于，获得batch数据之后，然后对这个batch的数据进行增强，如旋转、平移、翻折等相应的变化，由于有些数据集不能接受线性级别的增长，这种方法常用于大的数据集，很多机器学习框架已经支持了这种数据增强方式，并且可以使用GPU优化计算。

空间几何变换：翻转（水平和垂直）、随机裁剪、旋转、视觉变换（四点透视变换）、分段放射

像素颜色变换类; CoarseDropout、SimplexNoiseAlpha、FrequencyNoiseAlpha、ElasticTransformation

HSV对比度变换

RGB颜色扰动

随机擦除

超像素法

边界检测

锐化与浮雕

warmup：学习率先小后大

label smoothing：增加泛化能力，防止过拟合（与one-hot对应）

K-Fold 交叉验证：

NMS（非极大值抑制）：一个物体可能有好几个框，使用非极大值抑制只保留最优的一个，抑制的过程是迭代-遍历-消除

soft NMS 不要粗暴删除所有IOU大于阈值的框，而是降低其置信度。

DiouNMS：考虑了两框中心点的信息
$$
s_i=\left\{ \begin{array}{c}
	s_{i\,\,},IoU-\mathcal{R}_{DIoU}\left( \mathcal{M},\mathcal{B}_i \right) <\varepsilon\\
	0,IoU-\mathcal{R}_{DIoU}\left( \mathcal{M},\mathcal{B}_i \right) \ge \varepsilon\\
\end{array} \right.
$$


2、baseline trick

mosaic : 利用四张图片拼在一起，当前图片加随机抽出三张图片，丰富检测物体的背景

mixup：用于解决过拟合问题的数据增强方式

Kmeans 聚类 anchor

Bbox  regression loss : smooth L1 loss、 IOU loss 、GIoU loss、DIoU loss、CIoU loss

重叠问题：https://arxiv.org/abs/1711.07752

检测网络之前接一个分类网络，把没有目标的图像分出来，单独处理，以免空白图像在检测网络检测出来目标，扣分严重

3、测试阶段trick

plabel：Pseudo-Labeling 伪标签半监督学习，（根据不同的测试集，选择不同的模型来制作伪标签）                    

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210307213625879.png" alt="image-20210307213625879" style="zoom: 25%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210307213750501.png" alt="image-20210307213750501" style="zoom: 50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210307213927785.png" alt="image-20210307213927785" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210307214247896.png" alt="image-20210307214247896" style="zoom:50%;" />



modelEnsemble：模型融合

单个模型的融合：不同epoch进行融合、不同fold融合、不同参数融合

多个模型融合：例如efficient和YOLO5，取并集防止漏检(WBF，去重)，取交集，防止误检，提高精度

TTA（test time augmentation）：水平翻转，垂直翻转，90度翻转，测试时简单增强

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210307221140445.png" alt="image-20210307221140445" style="zoom:50%;" />

OOF：平衡精确率和召回率的阈值

WBF：加权和融合算法，对框去重

![image-20210307212234801](/Users/lishuo/Library/Application Support/typora-user-images/image-20210307212234801.png)

## Day 3

