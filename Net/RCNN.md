## pytorch目标检测篇：1.1 RCNN（Region with CNN feature）

第一阶段的[图像分类篇](https://blog.csdn.net/m0_37867091/category_10165686.html)基本结束了，接下来开始目标检测篇。

学习资料来自：

- b站视频链接：https://space.bilibili.com/18161609/channel/index
- github代码和PPT：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
- up主的CSDN博客：https://blog.csdn.net/qq_37541097/article/details/103482003

------

## 1. RCNN简介

原论文：《Rich feature hierarchies for accurate object detection and semantic segmentation》，发表于 CVPR 2014

R-CNN可以说是利用深度学习进行目标检测的开山之作。作者Ross Girshick多次在PASCAL VOC的目标检测竞赛中折桂，曾在2010年带领团队获得终身成就奖。

------

## 2. RCNN算法流程

1. 使用 Selective Search 方法，对一张图像生成1000~2000个候选区域
2. 对每个候选区域，使用深度网络提取特征
3. 特征送入每一类的SVM分类器，判断是否属于该类
4. 使用回归器精细修正候选框的位置

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200915163530487.png?#pic_center)



## 2.1 Region Proposal(Selective Search)

**候选区域的生成**

使用 [Selective Search (SS)](https://www.cnblogs.com/zyly/p/9259392.html) 算法，对一张图像生成1000~2000个候选区域，然后使用一些合并策略将这些区域合并，得到一个层次化的区域结构，而这些结构就包含可能需要的

如下图所示，假如我们要检测花，那么先用SS算法生成可能是花的候选区域：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916211302927.png?#pic_center)

> 关于SS算法，可以简单理解为将图像分割成很多很多的小块，计算每两个相邻的区域的相似度，然后每次合并最相似的两块，最后得到的每一块区域都是SS算法所认为的一个完整的物体，如下图所示：
>  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916212525732.png#pic_center)



## 2.2 Feature Extraction(CNN)

**CNN提取候选区域的特征**

将2000候选区域 resize 到227×227pixel，接着将候选区域输入事先训练好的AlexNet CNN网络获取4096维的特征，最后得到2000×4096维的特征矩阵。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020091621284962.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3ODY3MDkx,size_16,color_FFFFFF,t_70#pic_center)


## 2.3 Classification(SVM)

**SVM 判定候选区域所属类别**

将 2000 个候选区域的特征向量，即 2000×4096 维特征矩阵送入 20 个SVM分类器，获得 2000×20 维的概率矩阵，每一行代表一个候选区域归为每个目标类别的概率。(红点代表每个候选框对应最大概率的类别)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916214903991.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3ODY3MDkx,size_16,color_FFFFFF,t_70#pic_center)
 还是以花这幅图像为例，由于图中只有花这一个类别目标，因此 2000 个候选框的分类最大概率都应该是花这个类别（不排除某些候选框被分到其他类别的可能），只是每个候选框的最大概率不同。



> SVM分类器为二分类器，以 PASCAL VOC 有20个类别为例，即每个类别都有一个SVM分类器
>  
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/2020091621545882.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3ODY3MDkx,size_16,color_FFFFFF,t_70#pic_center)

分别对上述 2000×20 维矩阵中每一列，即每一类进行 **非极大值抑制（Non-Maximum Suppression：NMS）** 剔除重叠的候选框，保留高质量的候选框。

那么肯定有人会好奇，如果图片中有好几个人脸，你这选取一个最大的，那第二个人脸怎么办呢。

实际上这是一个迭代的过程，第一步的非极大值抑制就是选取了某一个最大的得分，然后删除了他周边的几个框，第二次迭代的时候在剩下的框里面选取一个最大的，然后再删除它周围iou区域大于一定阈值的，这样不停的迭代下去就会得到所有想要找到的目标物体的区域。

假设有ABCDEF这么多个得分框（已经按照得分从小到大排序）。

- 寻找分类概率最高的候选框 F

- 计算其他候选框与该候选框的IOU，假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。

- 删除所有IOU值大于给定阈值的候选框，从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。
  
- 一直重复这个过程，找到所有曾经被保留下来的矩形框。
  

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916222146240.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3ODY3MDkx,size_16,color_FFFFFF,t_70#pic_center)

IOU(Intersection over Union)，交并比

>  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916223857247.png#pic_center)
>
> <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210122152315647.png" alt="image-20210122152315647" style="zoom:33%;" />

## 2.4 Bounding-box Regression

**边框回归：回归器修正候选框位置**

对 NMS 处理后剩余的建议框进一步筛选，即用20个**回归器**对上述20个类别中剩余的建议框进行回归操作，最终得到每个类别的修正后的得分最高的 bounding-box

![image-20210122153504073](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122153504073.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200922162715279.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3ODY3MDkx,size_16,color_FFFFFF,t_70#pic_center)

------

## 3. RCNN框架总结

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916225314485.png?#pic_center)

------

## 4. RCNN存在的问题

- **测试速度慢**：一张图像内候选框之间存在大量重叠，提取特征操作冗余。
- **训练速度慢**：过程极其繁琐
- **训练所需空间大**：对于SVM和bbox回归训练，需要从每个图像中的每个目标候选框提取特征，并写入磁盘。对于非常深的网络，如VGG16，从VOC07训练集上的5k图像上提取的特征需要数百GB的存储空间。
- ![image-20210122153719220](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122153719220.png)

