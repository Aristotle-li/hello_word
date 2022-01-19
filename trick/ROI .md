​                           

## 详解 Mask-RCNN 中的 “RoIAlign” 作用 / 双线性插值的方法

### 文章目录

- [1. RoIAlign的产生背景](https://blog.csdn.net/qq_42902997/article/details/105087407#1_RoIAlign_1)
- [2. RoI Pooling](https://blog.csdn.net/qq_42902997/article/details/105087407#2_RoI_Pooling_9)
- [3. RoI Align](https://blog.csdn.net/qq_42902997/article/details/105087407#3_RoI_Align_13)
- [4. 双线性插值](https://blog.csdn.net/qq_42902997/article/details/105087407#4__32)



## 1. RoIAlign的产生背景

首先设想一个场景，假设从一个图像中锚定了一个人的边界框
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/2020032509234760.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTAyOTk3,size_16,color_FFFFFF,t_70)
 *图1 锚定一个目标人物的边界框*

这个时候，要提取边界框中的人的特征，显然应该用 CNN 网络来做这个工作。CNN 处理之后会形成一个特征图。按照一般的处理方式，会使用  RoI Pooling 来缩小特征图的尺寸。但是在 Mask-RCNN 中提出了一个 RoIAlign  的方式，使得得到的小特征图可以更加的精确和信息完整。

## 2. RoI Pooling

举例来说，如果我们现在得到了一个特征图，特征图尺寸是  5×7，要求将此区域缩小为2×2。此时，因为 $ {5}/{2}=2.5$ 是个非整数，所以此时系统会对其进行取整的分割，即将 “5 ”分割成 “ 3+2 ”，将 “ 7 ” 分割成 “ 3+4 ”，然后取每个区域的最大值作为本区域的值，整个过程如下所示：![在这里插入图片描述](https://img-blog.csdnimg.cn/20200325102909797.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTAyOTk3,size_16,color_FFFFFF,t_70)
 *图2 整个RoI Pooling 的过程*

## 3. RoI Align

可以知道，使用RoIPooling 存在一个很大的问题：很粗糙地选用了一个值来代替一个区域的值，而且每个区域的尺寸还有很大的差距。  

RoIAlign 的过程如下：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200325111347800.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTAyOTk3,size_16,color_FFFFFF,t_70)
 *图3 整个RoI Align 的过程*

- 因为最终要将这个5×7 的特征图处理成2×2 的特征图。所以先将要进行 RoIAlign 的过程转换成2×2 个相同规模的范围，这个过程中不做任何量化处理。
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200325110453993.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTAyOTk3,size_16,color_FFFFFF,t_70)
- 将这  4 个模块（①，②，③，④）内部同样进行这样的处理，再细分成 4 个规模相同的区域（图中虚线表示）。
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200325110537871.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTAyOTk3,size_16,color_FFFFFF,t_70)
- 然后对于每一个**最小的区域**（包含不止一个像素点）确定其中心点（图中的红色× ）然后使用双线性插值法得到这个× 号所在位置的值作为 **最小格子区域** 的值。
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200325111233640.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTAyOTk3,size_16,color_FFFFFF,t_70)
- 对于每一个 **小区域**（①，②，③，④）都会有4 个这样的值，把这4 个值取他们的最大值作为每个 **小区域（①，②，③，④）** 的值。这样最终就可以得到                                    4 个小区域的4 个值，作为最终的特征图输出结果。 RoIAlign 提出的这种方式可以避免过程中丢失原特征图的信息，中间过程全程不量化来保证最大的信息完整性。

## 4. 双线性插值


 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200325172951743.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTAyOTk3,size_16,color_FFFFFF,t_70)
