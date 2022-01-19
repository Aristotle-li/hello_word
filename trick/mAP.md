TP：IoU大于0.5的检测框数量（同一个GT只计算一次）

FP：IoU小于0.5的检测框数量（检测到同一个GT多余的检测框数量）

FP：没有检测到的GT的数量

precision（查准率）：预测的所有目标中，预测正确的比例

recall（查全率）：所有正例中，预测正确的比例

AP：P-R曲线的线下面积

mAP：各类别AP的平均值





## 1.3 Precision精度

  **正确预测占全部预测的百分比**：
          ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200519162126447.png)

## 1.4 Recall查全率

  **正确预测占所有真值框的百分比**：
          ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200519162356954.png)



- [一、准确率、召回率、F1](https://www.p-chao.com/2016-11-05/信息检索（ir）的评价指标-precision、recall、f1、map、roc、auc/#F1)
- [二、AP和mAP（mean Average Precision）](https://www.p-chao.com/2016-11-05/信息检索（ir）的评价指标-precision、recall、f1、map、roc、auc/#APmAPmean_Average_Precision)
- [三、ROC和AUC](https://www.p-chao.com/2016-11-05/信息检索（ir）的评价指标-precision、recall、f1、map、roc、auc/#ROCAUC)
- [四、PR与ROC](https://www.p-chao.com/2016-11-05/信息检索（ir）的评价指标-precision、recall、f1、map、roc、auc/#PRROC)

![image-20210324184404729](/Users/lishuo/Library/Application Support/typora-user-images/image-20210324184404729.png)



coco数据集中：AP=mmAP 、 $ AP_{IoU=0.5}=mAP $











## 一文搞懂目标检测的常用指标（Pascal VOC与COCO）

## 检测框预测正确的标准是什么？

在探究目标检测模型的性能评估指标之前，我们首先需要搞清楚，在什么情况下才认为一个检测框是正确的。

这里主要涉及两个阈值，IoU阈值和置信度（confidence）阈值。

首先是置信度阈值，所谓置信度即模型认为检测框中存在目标的确信度。

对于一个检测框，会首先使用置信度阈值进行过滤，当检测框的置信度大于该阈值时才认为检测框中存在目标（即为正样本，positive），否则认为不存在目标（即为负样本，negative）。

其次是IoU阈值，所谓IoU，就是交并比，是用来衡量两个矩形框重合程度的一种度量指标，如下所示：

![IMG_7E0103C4A1AD-1](http://activepony.com/shen-du-xue-xi/mu-biao-jian-ce/mu-biao-jian-ce-ping-jie-zhi-biao/IMG_7E0103C4A1AD-1.jpeg)

IoU就等于上图中两个矩形框重合部分的面积与两个矩形框的面积之和的比例。

而IoU阈值的作用就是，当模型给出的检测框和真实的目标框之间的IoU大于该阈值时，才认为该检测框是正确的，即为True Positive，否则为False Positive。

![IMG_91B4CCC3F4F8-1](http://activepony.com/shen-du-xue-xi/mu-biao-jian-ce/mu-biao-jian-ce-ping-jie-zhi-biao/IMG_91B4CCC3F4F8-1.jpeg)

如上图所示，其中粉色框为真实标定，黄色框为检测器给出的预测。如果设定IoU阈值为0.8，则人对应的预测框会被判定为TP，而狗对应的预测框就会被判定为FP。相应的，如果降低IoU阈值至0.3，则人和狗对应的预测框都会被判定为TP。

只有当检测框的置信度和与真实标定框的IoU分别都大于置信度阈值和IoU阈值时，才认为该检测框为正确的。

接下来将要讨论的AP、mAP等指标就是在设定不同的置信度阈值或IoU阈值的情况下进行统计得出的。

## AP的计算

### Precision x Recall曲线（PR曲线）

通过改变置信度confidence的值，可以针对每一个类别画出一条precision-recal曲线。通过设置不同的confidence，可以得到不同的precision和recall的对应关系。

观察某一个目标检测模型关于某一类别的PR曲线，如果随着recall的增高，其precision仍旧保持较高的值（无论如何设置confidence的阈值，precision和recall都能保持较高的值），那么我们就可以认为对于该类别来说，该模型具有比较好的性能。

判断目标检测模型性能好坏的另外一种方式是：看该模型是否只会识别出真实的目标（False Positives的个数为0，即高precision），同时能够检测出所有的真实目标（False Negatives的个数为0，即高recall）。

而一个性能比较差的模型要想检测出所有的真实目标（高recall），就需要增加其检测出的目标的个数（提高False  Positive），这会导致模型的precision降低。在实际测试中，我们会发现PR曲线在一开始就有较高的precision，但随着recall的增高，precision会逐渐降低。

### Average Precision

另外一种表征目标检测模型性能的方式是计算PR曲线下的面积（area under the curve,  AUC）。因为PR曲线总是呈Z字型上升和下降，因而我们很难将多个模型的PR曲线绘制在一起进行比较（曲线会相互交叉）。这也是我们常使用AP这一具有具体的数值的度量方式的原因。

实际上，可以将AP看作precision以recall为权重的加权平均。

### Pascal VOC中AP的计算方式

明白了AP的定义为PR曲线下的面积，那么问题就转换为如何求解PR曲线下的面积。曲线的面积实际上可以转换为积分的求解。但在计算机中，我们是无法精确计算积分值的。只能采取一些近似的方法。常用的积分求解方法就是插值法。

在2010年以前，Pascal VOC在计算每一个类的AP时所采用的方式在PR曲线中取11个插值点，使用这11个插值点来计算积分值，最终作为该类别的AP值。但在2010年以后，Pascal VOC摒弃了这一计算方式，转而使用所有的数据点来计算AP值。

![image-20210324170433182](/Users/lishuo/Library/Application Support/typora-user-images/image-20210324170433182.png)

其中*p*(*r*̂ )

表示recall *r*̂ 

对应的precision。

在进行插值的过程中，每一个插值点*r*

所对应的precision值为所有大于*r*+1

的recall所对应的precision中的最大值。

按照上述两种AP的计算方式的定义，给出示意图如下：

![IMG_712C7E8C37FD-1](http://activepony.com/shen-du-xue-xi/mu-biao-jian-ce/mu-biao-jian-ce-ping-jie-zhi-biao/IMG_712C7E8C37FD-1.jpeg)

### 可视化分析

为了更好地理解上述的两种计算方式，给出如下的例子（[图来源](https://github.com/Cartucho/mAP)）。

![img](http://activepony.com/shen-du-xue-xi/mu-biao-jian-ce/mu-biao-jian-ce-ping-jie-zhi-biao/samples_1_v2.png)

在上述7张图片中总共有15个真实目标框（绿色），24个检测到的目标框（红色）。在每一个检测框的下面都有其置信度分数。

如下表所示，给出了在**IoU阈值为30%**的情况下，各个检测框被判定为TP和FP的情况（[图来源](https://github.com/Cartucho/mAP)）。

![img](http://activepony.com/shen-du-xue-xi/mu-biao-jian-ce/mu-biao-jian-ce-ping-jie-zhi-biao/table_1_v2.png)

在上述图片中，某些真实目标框和不只一个检测框重合，此时具有最高的IoU的检测框被判定为TP，其余的被判定为FP（这一规则来自于Pascal VOC2012，如果有5个检测框和1个真实目标框重合，则只有1个被判定为TP,，剩余的都将被判定为FP）。

为了画出PR曲线，需要对TP和FP的个数求累积和，进而得到不同的recall和precision。为了达到这一目的，首先需要按照各个检测框的置信度分数的大小从大到小进行排序。接着，依次计算不同置信度阈值所对应的recall和precision（只有当检测框的置信度分数高于置信度阈值时，才会被判定为存在目标）。计算过程如下表所示（[图来源](https://github.com/Cartucho/mAP)）：

![img](http://activepony.com/shen-du-xue-xi/mu-biao-jian-ce/mu-biao-jian-ce-ping-jie-zhi-biao/table_2_v2.png)

观察上表可以发现，当设置置信度阈值为0.95时，只有一个检测框被认为存在目标，此时TP个数为1，被判定为存在目标的框的个数为1，因而此时precission=1.0，而recall只有0.0666。随着置信度阈值的降低，TP的个数在增加，但同时FP的个数也在增加，且FP个数的增加速度要高于TP，所导致的结果就是precision逐渐降低，recall逐渐增大。

将上表中precision和recall的变化趋势转换为PR曲线，如下图所示（[图来源](https://github.com/Cartucho/mAP)）：

![img](http://activepony.com/shen-du-xue-xi/mu-biao-jian-ce/mu-biao-jian-ce-ping-jie-zhi-biao/precision_recall_example_1_v2.png)

接下来，使用之前所说的两种不同的AP计算方式来计算上述PR曲线对应的AP。

#### 11点插值法

正如之前所说的，在11点插值法中，每一个插值点所对应的precision值为所有大于等于该插值点所对应的precision的最大值（[图来源](https://github.com/Cartucho/mAP)）。

![img](http://activepony.com/shen-du-xue-xi/mu-biao-jian-ce/mu-biao-jian-ce-ping-jie-zhi-biao/11-pointInterpolation.png)

按照公式，AP的计算方式如下：

![1590464812795](http://activepony.com/shen-du-xue-xi/mu-biao-jian-ce/mu-biao-jian-ce-ping-jie-zhi-biao/1590464812795.png)

最终的AP=0.2684。

#### 使用所有点进行插值

通过使用所有的数据点进行插值，可以将AP解释为PR曲线的AUC的近似值。采用这一计算方式是为了降低PR曲线的扰动所带来的影响。在之前的公式中，可以看出每一个插值点的precision取该插值点之后的所有recall点所对应的precision中的最大值。如下图所示（[图来源](https://github.com/Cartucho/mAP)）：

![img](http://activepony.com/shen-du-xue-xi/mu-biao-jian-ce/mu-biao-jian-ce-ping-jie-zhi-biao/interpolated_precision_v2.png)

按照公式中的计算方式，可以将整个PR曲线的面积近似划分为如下的四个部分（[图来源](https://github.com/Cartucho/mAP)）：

![img](http://activepony.com/shen-du-xue-xi/mu-biao-jian-ce/mu-biao-jian-ce-ping-jie-zhi-biao/interpolated_precision-AUC_v2-20200510190828940.png)

写出数值计算过程如下：

![image-20200510190920559](http://activepony.com/shen-du-xue-xi/mu-biao-jian-ce/mu-biao-jian-ce-ping-jie-zhi-biao/image-20200510190920559.png)

可以看出，按照上述两种不同的计算方式可以得到不同的AP值。

这里要注意的一点是，在上述计算过程中，将检测框与目标框的IoU阈值设定为30%。



## mAP的计算

![image-20210324170543246](/Users/lishuo/Library/Application Support/typora-user-images/image-20210324170543246.png)



> AP is averaged over all categories. Traditionally, this is called  “mean average precision” (mAP). We make no distinction between AP and  mAP (and likewise AR and mAR) and assume the difference is clear from  context.

![image-20210324170613169](/Users/lishuo/Library/Application Support/typora-user-images/image-20210324170613169.png)