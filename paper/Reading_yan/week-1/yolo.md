







学习网络：第一步看讲解、第二步看paper、第三步看源码（网络的搭建、数据预处理、损失的计算、正负样本均衡）



## 和rcnn系列不同：

### 解决的问题：

目标检测实时性的问题

### Idea：

one-stage network：将目标检测作为一个回归问题，直接对每个grid cell 回归边界框坐标和相关的类概率，分类和回归一次完成。the whole detection pipeline is a single network，it can be optimized end-to-end directly on detection performance.（不需要像faster-RCNN那样接两个全连接层，分别预测类别和回归参数，多任务联合训练）

### 缺点：

1、YOLO makes more localization errors

2、预测bbox是从数据中学的，所以很难泛化到新的长宽比目标

### 优点：

1、YOLO检测物体非常快（45-155FPS）

2、YOLO可以很好的避免背景错误，产生false positives（可以看到全局图像，有上下文信息）

3、泛化性能更好，（learns generalizable representations of objects.）

### 发展：

DPM：使用sliding-window，通过在每个窗口重用classifier，实现detection

RCNN：系列使用region proposal method，本质上仍是sliding-window，只不过是在feature map上执行，生成proposals， then run a classifier on these proposed boxes

YOLO：我们将目标检测重构为一个单一的回归问题，直接从图像像素到边界框坐标和类概率。（在整个feature map 上直接训练预测，这种结构毫无疑问优点是更全局且泛化性能更好但是缺点是定位精度会欠缺）

### 改进：

1、融合底层的feature map，因为底层细节信息更丰富

2、引入achor ，应对新的比例目标就好一些

### 词句：

1、Frame  n. 框架；结构；画面vt. 设计；陷害；建造；使…适合responsive = 反应灵敏的 is responsible for 负责 conditional class proba- bilities 条件类概率

regardless of 不考虑，不管  deviation=偏离  relatively coarse features  相对粗糙的特征

2、Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities.

3、YOLO is simple and straightforward.

4、thresholds the resulting detections by the model’s confidence.根据模型的置信度来确定检测结果的阈值。

5、This unified model has several benefits over traditional methods of object detection.

6、This means we can process streaming video in real-time with less than 25 milliseconds of latency.这意味着我们可以在不到25毫秒的延迟时间内实时处理流视频

7、YOLO reasons globally about the full image and all the objects in the image. 对图像全局推理

8、We unify the separate components of object detection into a single neural network.我们将目标检测的各个组成部分统一到一个神经网络中

9、our system divides the input image into an S × S grid

10、Detection often requires fine-grained visual information 检测常需要细粒度信息

11、however it does not perfectly align with our goal of maximizing average precision.符合、对齐

12、It weights localization er- ror equally with classification error 它将定位误差与分类误差平均加权



> 先前的fater-rcnn 分类和回归共用同一套网络，重新利用了分类器网络去做检测任务。
>
> YOLO将目标检测作为一个回归问题，回归到空间上分离的边界框和相关的类概率。
>
> A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since , 
>
> 一个单一的神经网络预测边界框和类概率直接从完整的图像在一次评估。由于整个检测管道是一个单一的网络，因此可以直接对检测性能进行端到端的优化。





## YOLO-v1

> 题目：You Only Look Once: Unified, Real-Time Object Detection
>
> 作者：Joseph Redmon∗, Santosh Divvala、Facebook AI Research
>
> 来源：CVPR 2016 
>
> 性能：45fps 480*480 63.4mAP
>
> SSD
>
> 74.3mAP
>
> 59fps 300*300
>
> Faster-RCNN 
>
> 73.2mAP
>
> 思想：
>
> 将图像分成grid cell，对坐标直接回归

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210419223133652.png" alt="image-20210419223133652" style="zoom: 67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210419223307001.png" alt="image-20210419223307001" style="zoom: 67%;" /><img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210419223427423.png" alt="image-20210419223427423" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210419223307001.png" alt="image-20210419223307001" style="zoom: 67%;" /><img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210419223427423.png" alt="image-20210419223427423" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210420091925726.png" alt="image-20210420091925726" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210420092026162.png" alt="image-20210420092026162" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210420092041266.png" alt="image-20210420092041266" style="zoom:67%;" />

which gives us class-specific confidence scores for each box. These scores encode both the probability of that class appearing in the box and how well the predicted box fits the object.

给了我们每个bbox的对应特定类别的置信度分数，这些分数的大小既反映了类别的概率，也反映了bbox和ground-Truth的IoU。

confidence (置信度)：预测有目标等于IoU ,没有目标等于0.

This pushes the “confidence” scores of those cells towards zero, often overpowering the gradient from cells that do contain objects. This can lead to model instability, causing training to diverge early on.

有目标的grid cell 比没有目标的少的多，造成了正负样本不均衡，没有目标的cell 将 confidence score 推向 0，这部分的占比比包含目标cell的梯度要强得多

常规解决思路就是降低负样本的权重

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210419232543557.png" alt="image-20210419232543557" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210419232702212.png" alt="image-20210419232702212" style="zoom:67%;" />



为什么w 和 h 要开根号

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210419233049501.png" alt="image-20210419233049501" style="zoom:67%;" />



题目：YOLO9000- Better, Faster, Stronger

作者：Joseph Redmon Ali Farhadi

来源：CVPR 2017

性能：67 fps 76.8mAP

## YOLO-v2

> 题目：YOLO9000- Better, Faster, Stronger
>
> 作者：Joseph Redmon Ali Farhadi
>
> 来源：CVPR 2017
>
> 性能：67 fps 76.8mAP
>

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210420095604471.png" alt="image-20210420095604471" style="zoom: 50%;" />

使用k-means聚类来得到achor的尺寸



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210420220944598.png" alt="image-20210420220944598" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210420223322965.png" alt="image-20210420223322965"  />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210420223417169.png" alt="image-20210420223417169" style="zoom:67%;" />



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210421163521261.png" alt="image-20210421163521261" style="zoom:50%;" />



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210421163537531.png" alt="image-20210421163537531" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210421163608890.png" alt="image-20210421163608890" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210420092715337.png" alt="image-20210420092715337" style="zoom:67%;" />

### 解决的问题：

1、检测数据集太小，能检测的类别有限，利用现有的分类数据集

2、提高recall（看该模型是否只会识别出真实的目标（False Positives的个数为0，即高precision），同时能够检测出所有的真实目标（False Negatives的个数为0，即高recall））

3、减小localization errors

### idea：

1、Hierarchical classification：a language database that structures concepts and how they relate ，WordNet is structured as a directed graph,building a hierarchical tree from the concepts in ImageNet.

构建一个WordNet，计算所有属于相同概念下义词的同义词集的softmax，然后通过将类别数据集映射到树中的同义词集的方式 merge datasets together

2、联合训练算法能够在检测和分类数据上训练目标检测器：利用标记检测图像来学习精确定位目标，同时使用分类图像来增加词汇表和鲁棒性

3、tricks + achor + feature  hierarchy + Multi-Scale 

### 细节：

1、Batch Normalization

2、High Resolution Classifier

3、Convolutional With Anchor Boxes ：放弃直接预测coordinates，引入achor预测offset

 （Predicting offsets instead of coordinates simplifies the problem and makes it easier for the network to learn.）

4、Dimension Clusters.（we run k-means clustering on the training set bounding boxes to automatically find good priors.）

5、使用achor 但是不去sliding-window，那么对每个cell的achor回归就可能offset到其他cell，造成不稳定，所以对offset做出了限制，输入一个sigmoid函数。让每个achor 只去预测目标中心落在某个grid cell 区域内的目标 （faster-RCNN没有这个问题是因为采用了sliding-window，那么滑动到背景不做回归，滑动到目标做回归）

6、Fine-Grained Features：通过 passthrough layer  把 low level 26 分辨率的feature map 堆叠临近的features 到不同的channel   到 13 ，使用 concatenate 和 high level feature map 结合

7、Multi-Scale Training：

### 改进方向：

构建WordNet效率太低，能不能用一种算法自动划分，比如聚类或者随机森林

### 词句

1、harness 、employ 利用  framerate 帧率  custom network 定制的网络 traverse the tree down 向下遍历树

2、YOLO suffers from a variety of shortcomings relative to state-of-the-art detection systems.和先进的检测系统相比，YOLO有很多的缺点

3、Computer vision generally trends towards larger, deeper networks . Better performance often hinges on training larger networks or ensembling multiple models together.    hinge on 取决于

4、We pool a variety of ideas from past work with our own novel concepts我们将过去工作中的各种想法与我们自己新颖的概念结合起来

5、YOLO’s convolutional layers downsam- ple the image by a factor of 32YOLO的卷积层对图像进行了32倍的降采样

6、The passthrough layer concatenates the higher resolution features with the low resolution features by stacking adjacent features into different channels instead of spatial locations, similar to the identity mappings in ResNet. 

7、 so that it has access to fine grained features. 

8、We propose a mechanism for jointly training on classi- fication and detection data.

9、we compute the softmax over all synsets that are hyponyms of the same concept(Using WordTree we perform multiple softmax operations over co-hyponyms.  )同义句





## YOLO-v3

> 题目：YOLOv3: An Incremental Improvement
>
> 作者：Joseph Redmon Ali Farhadi
>
> 来源：CVPR 2018
>
> 性能：

### 解决的问题：

小目标检测更好

### idea：继承自yolo v2

1、是分类和置信度损失从平方误差损失改到二值交叉熵损失

使用softmax，假定只有一个类别，且所有类别概率和为1。

但是如果同时有 person 和 women那么可能同时属于两个标签，所以采用二值交叉熵损失，每个类别的概率是相互之间独立的 ，多标签方法可以更好地对数据建模。

2、同时加入skip-layer，在三个层分别进行预测，层1 不融合，层2，3通过skiplayer融合前面层的细粒度信息。（v2只融合一次，只在融合后的层预测）

3、darknet-19结合resnet 创造处darknet-53，特点是去掉了maxpooling，加入了Residual



### 改进方向：

原文已说明：检测出物体的能力很强，但是，随着IOU阈值的增加，性能会显着下降，这表明YOLOv3难以使框与对象完美对齐。







### 词句： 

1、Using a softmax imposes the assumption that each box has exactly one class which is often not the case. A multilabel approach better models the data.

2、This method allows us to get more meaningful semantic information from the upsampled fea- tures and finer-grained information from the earlier feature map. 

3、We perform the same design one more time to predict boxes for the final scale.

4、 Thus our predictions for the 3rd scale benefit from all the prior computation as well as fine- grained features from early on in the network.

5、Darknet-53 performs on par with state-of-the-art classifiers but with fewer floating point operations and more speed. 

6、performance drops significantly as the IOU threshold increases indicating YOLOv3 struggles to get the boxes perfectly aligned with the object.

7、hybrid 混合的

![image-20210421205349484](/Users/lishuo/Library/Application Support/typora-user-images/image-20210421205349484.png)

## yolov3spp

![yolov3spp](/Volumes/Macintosh HD/paper/yolov3spp.png)

![image-20210421211936722](/Users/lishuo/Library/Application Support/typora-user-images/image-20210421211936722.png)

![image-20210421212544153](/Users/lishuo/Library/Application Support/typora-user-images/image-20210421212544153.png) 

![image-20210421212652500](/Users/lishuo/Library/Application Support/typora-user-images/image-20210421212652500.png)

![image-20210421213008560](/Users/lishuo/Library/Application Support/typora-user-images/image-20210421213008560.png)

![image-20210421215225020](/Users/lishuo/Library/Application Support/typora-user-images/image-20210421215225020.png)

使用二值交叉熵损失每个类别的概率是相互之间独立的 



$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],$

$\ell(x, y) = \begin{cases}
    \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
    \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
\end{cases}$





![image-20210421230432737](/Users/lishuo/Library/Application Support/typora-user-images/image-20210421230432737.png)