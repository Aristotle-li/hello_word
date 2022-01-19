## object detection

### one-stage:SSD 、YOLO

基于anchors直接进行分类以及调整边界框

### Two-stage:Faster-RCNN

1、通过专门模块去生成候选框（RPN），寻找前景以及调整边界框（基于anchors）

2、基于之前生成的候选框进一步分类以及调整边界框（基于proposals）

目标检测、图像分割、人物关键点检测

 ![image-20210122142358976](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122142358976.png)



![image-20210122142418509](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122142418509.png)

![image-20210122144331552](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122144331552.png)



![image-20210122144731314](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122144731314.png)









## 理解感受野

当盯着某个点时，

* 视野范围很大

* 只有焦点附近是清晰的

* 周围模糊

  人类进化到这个阶段，有如此结构必然是为了生存，那么技术的路线也不一定是要严格按照人类的进化走。

  