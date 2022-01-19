# MobileNet(v1、v2)网络详解与模型的搭建

​         

首先给出三个链接：
 [1. MobileNet（v1，v2）网络详解视频](https://www.bilibili.com/video/BV1yE411p7L7)
 [2. 使用pytorch搭建mobilenet v2并基于迁移学习训练视频](https://www.bilibili.com/video/BV1qE411T7qZ)
 [3. 使用tensorflow2搭建mobilenet v2并基于迁移学习训练视频](https://www.bilibili.com/video/BV1NE411K7tX)

在之前的文章中讲的AlexNet、VGG、GoogLeNet以及ResNet网络，它们都是传统卷积神经网络（都是使用的传统卷积层），缺点在于内存需求大、运算量大导致无法在移动设备以及嵌入式设备上运行。而本文要讲的MobileNet网络就是专门为移动端，嵌入式端而设计。
 

### MobileNet v1



MobileNet网络是由google团队在2017年提出的，专注于移动端或者嵌入式设备中的轻量级CNN网络。相比传统卷积神经网络，在准确率小幅降低的前提下大大减少模型参数与运算量。(相比VGG16准确率减少了0.9%，但模型参数只有VGG的1/32)。
 要说MobileNet网络的优点，无疑是其中的Depthwise  Convolution结构(大大减少运算量和参数数量)。下图展示了传统卷积与DW卷积的差异，在传统卷积中，每个卷积核的channel与输入特征矩阵的channel相等（每个卷积核都会与输入特征矩阵的每一个维度进行卷积运算）。而在DW卷积中，每个卷积核的channel都是等于1的（每个卷积核只负责输入特征矩阵的一个channel，故卷积核的个数必须等于输入特征矩阵的channel数，从而使得输出特征矩阵的channel数也等于输入特征矩阵的channel数）
 ![dw卷积与普通卷积](https://img-blog.csdnimg.cn/20200426162556656.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center)
  刚刚说了使用DW卷积后输出特征矩阵的channel是与输入特征矩阵的channel相等的，如果想改变/自定义输出特征矩阵的channel，那只需要在DW卷积后接上一个PW卷积即可，如下图所示，其实PW卷积就是普通的卷积而已（只不过卷积核大小为1）。通常DW卷积和PW卷积是放在一起使用的，一起叫做Depthwise Separable Convolution（深度可分卷积）。
 ![DW卷积和PW卷积](https://img-blog.csdnimg.cn/20200426163946303.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center)
 那Depthwise Separable  Convolution（深度可分卷积）与传统的卷积相比有到底能节省多少计算量呢，下图对比了这两个卷积方式的计算量，其中Df是输入特征矩阵的宽高（这里假设宽和高相等），Dk是卷积核的大小，M是输入特征矩阵的channel，N是输出特征矩阵的channel，卷积计算量近似等于卷积核的高 x 卷积核的宽 x 卷积核的channel x 输入特征矩阵的高 x  输入特征矩阵的宽（这里假设stride等于1），在我们mobilenet网络中DW卷积都是是使用3x3大小的卷积核。所以理论上普通卷积计算量是DW+PW卷积的8到9倍（公式来源于原论文）：
 ![计算量对比](https://img-blog.csdnimg.cn/20200426164254206.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center)
 在了解完Depthwise Separable Convolution（深度可分卷积）后在看下mobilenet  v1的网络结构，左侧的表格是mobileNetv1的网络结构，表中标Conv的表示普通卷积，Conv  dw代表刚刚说的DW卷积，s表示步距，根据表格信息就能很容易的搭建出mobileNet  v1网络。在mobilenetv1原论文中，还提出了两个超参数，一个是α一个是β。α参数是一个倍率因子，用来调整卷积核的个数，β是控制输入网络的图像尺寸参数，下图右侧给出了使用不同α和β网络的分类准确率，计算量以及模型参数：
 ![mobilenet v1](https://img-blog.csdnimg.cn/20200426165207571.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center)



### MobileNet v2



在MobileNet  v1的网络结构表中能够发现，网络的结构就像VGG一样是个直筒型的，不像ResNet网络有shorcut之类的连接方式。而且有人反映说MobileNet v1网络中的DW卷积很容易训练废掉，效果并没有那么理想。所以我们接着看下MobileNet v2网络。
 MobileNet  v2网络是由google团队在2018年提出的，相比MobileNet V1网络，准确率更高，模型更小。刚刚说了MobileNet  v1网络中的亮点是DW卷积，那么在MobileNet v2中的亮点就是Inverted residual  block（倒残差结构），如下下图所示，左侧是ResNet网络中的残差结构，右侧就是MobileNet  v2中的到残差结构。在残差结构中是1x1卷积降维->3x3卷积->1x1卷积升维，在倒残差结构中正好相反，是1x1卷积升维->3x3DW卷积->1x1卷积降维。为什么要这样做，原文的解释是高维信息通过ReLU激活函数后丢失的信息更少（注意倒残差结构中基本使用的都是ReLU6激活函数，但是最后一个1x1的卷积层使用的是线性激活函数）。

![倒残差结构](https://img-blog.csdnimg.cn/20200426171005366.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center)
  在使用倒残差结构时需要注意下，并不是所有的倒残差结构都有shortcut连接，只有当stride=1且输入特征矩阵与输出特征矩阵shape相同时才有shortcut连接（只有当shape相同时，两个矩阵才能做加法运算，当stride=1时并不能保证输入特征矩阵的channel与输出特征矩阵的channel相同）。

![倒残差shortcut](https://img-blog.csdnimg.cn/20200426171906852.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center)
 下图是MobileNet  v2网络的结构表，其中t代表的是扩展因子（倒残差结构中第一个1x1卷积的扩展因子），c代表输出特征矩阵的channel，n代表倒残差结构重复的次数，s代表步距（注意：这里的步距只是针对重复n次的第一层倒残差结构，后面的都默认为1）。

![MobileNet v2结构](https://img-blog.csdnimg.cn/20200426172803520.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center)
 关于MobileNet V2模型的搭建与训练代码放在我的github中，大家可自行下载使用：

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

pytorch版本在pytorch_learning文件夹中，tensorflow版本在tensorflow_learning文件夹中





![](/Users/lishuo/Library/Application Support/typora-user-images/image-20210121171402841.png)