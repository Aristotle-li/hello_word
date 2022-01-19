首发于[Dive into 深度学习](https://www.zhihu.com/column/c_1394966951895113729)

# YOLOX

[![物格意诚](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-03eea73897fcaf59237122a053ed826a_xs.jpg)](https://www.zhihu.com/people/dahaiyidi)

[物格意诚](https://www.zhihu.com/people/dahaiyidi)[](https://www.zhihu.com/question/48510028)



哈尔滨工业大学 工程硕士

这项工作是旷世的。

号称：Exceeding YOLO Series in 2021

github地址:

[Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

[github.com/Megvii-BaseDetection/YOLOX![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-e50a06483cf7f360e19f8dfe76191c55_180x120.jpg)](https://github.com/Megvii-BaseDetection/YOLOX)

Paper 地址：

[https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)

[arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)

具体有多强大？见下图。又小又好用。YOLOX-Nano 只有 0.91M Parameters .

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-6a3faffeb2d548a39580f257a0388a4c_720w.jpg)



此外，还提供了各种落地源码：

1. [ONNX export and an ONNXRuntime](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime)
2. [TensorRT in C++ and Python](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/TensorRT)
3. [ncnn in C++ and Java](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ncnn)
4. [OpenVINO in C++ and Python](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/OpenVINO)

最近两年，有一些新的技术没有集成到yolo中，

Anchor-free detectors, advanced label assignment strategies, and end-to-end (NMS-free) detectors.



## 主要贡献

**1.Anchor -free**

基于anchor的检测有很多问题：

- 需要在训练前，通过聚类决定一系列的anchor，这些anchor具有很强的领域性，泛化差。
- anchor机制增加了detection heads 的复杂性，增加了预测数量。这在边缘AI 系统中，是一个瓶颈。

将anchor-free引入yolo的做法是：

在每个位置只预测一次，预测四个值——左上角xy坐标的偏移，高、宽。

将中心3x3=9的区域设置为正样本，称作，center sampling。





**2.Decoupled head** 

classification和 regression 是有冲突的，但是yolo系列工作还是将两者放在一个head里面。

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-bb3a48cb9e401f43296803c4436453ce_720w.jpg)



**Strong data augmentation**

使用Mosaic 和 MixUp,但是在最后的15个epoch中关闭。

作者发现，使用了strong data augmentation 后，ImageNet pre-training , 不能带来更多益处。



**3. SimOTA(the leading label assignment strategy)**

 OTA 将label  assignment 视作Optimal Transport (OT) problem。

外加动态top-k策略，形成SimOTA方法。

ground truth ![[公式]](https://www.zhihu.com/equation?tex=g_i+) 和prediction ![[公式]](https://www.zhihu.com/equation?tex=p_j+) 的损失为：

![[公式]](https://www.zhihu.com/equation?tex=c_{ij}%3DL_{ij}^{cls}%2B\lambda+L_{ij}^{reg}) 

第一项为分类损失，第二项为回归损失。

对于ground truth ![[公式]](https://www.zhihu.com/equation?tex=g_i+)，选取位于center region 的top k 个prediction（loss最小的k个prediction）作为![[公式]](https://www.zhihu.com/equation?tex=g_i+)的正样本。注意，k对于不同的ground truch是不一样的。

其他的作为负样本。

**4. end-to-end (NMS-free) detectors**

**实施细节**

- 共训练300 epochs with 5 epochs warmup on COCO train2017
- SGD， momentum 0.9
- lr * batch_size / 64, 初始lr为0.01， cosine lr schedule
- weight decay 0.0005
- The input size is evenly drawn from 448 to 832 with 32 strides.
- 还有个很重要的细节，最后的10-15epochs是要停止数据增强的！