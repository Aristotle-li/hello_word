### 感受野(receptive field)



计算：



![0DC8EE89026AE700B6CA9770511C494D](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/img0DC8EE89026AE700B6CA9770511C494D.png)

![A754EDAFCBB41EFEEEB559E11AF1D235](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgA754EDAFCBB41EFEEEB559E11AF1D235.png)

![C528CD80F9D93F56D10D8BA7066B74D0](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgC528CD80F9D93F56D10D8BA7066B74D0.png)



分类中的一个trick：

10-crop evaluation：将原图按一定规则裁剪出10个不同位置的子图像，分跑网络，最后平均概率。

道理是有效感受野其实是一个类似于高斯函数的区域，中间位置的特征表达是最好的。





anchor-base的方法其实不科学，感受野是天然的anchor



分割网络设计的重点就是如何更加高效耦合更多尺寸感受野特征



设计原理：

 FCOS检测头共享减少参数量，提高检测性能：

不同尺度的物体在不同特征层上对于head来说是一致的；

权值共享解决了物体在不同尺度下分布不均衡的问题，扩充了数据量，使得head能得到更充分的训练

为什么可以权重共享：因为物体的尺度与当前特征层感受野的比例是对等的。







one-stage anchor free 

关键认知，本质：

1、速度快，内存占用少，好落地

2、backbone、neck、head

2.1、backbone 

VGG、ResNet、ResNeXt、EfficientNet：

提供若干种感受野大小和中心步长的组合，以满足对不同尺度和类别的目标检测



目的感受野中心命中目标：

步长决定了在学习的时候正样本的数量，即到底有多少感受野中心能够命中这个目标，因为命中了都会被当做正样本。



2.2、neck

NaiveNeck(SSD)、FPN、BiFPN、PANet、NAS-FPN

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgC72453ED9528E2C4BEAE11B97B1C23C0.png" alt="C72453ED9528E2C4BEAE11B97B1C23C0" style="zoom:50%;" />

将来自backbone的不同尺度feature map特征融合，宽度对其，以供权重共享的head使用

2. 3、head

   RetinaNet-Head（anchor-base，no-quality）、FCOS-Head（anchor-free，quality）

   划分：有无anchor、有无quality分支

   quality：预测每个point在预测定位时的质量优劣

   不使用BN，不用Norm，或使用GN

   

   <img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgCB66E15DBC2E624CE257146BB5BA5B60.png" alt="CB66E15DBC2E624CE257146BB5BA5B60" style="zoom:67%;" />



3.FCN和FPN

![image-20220106211729063](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20220106211729063.png)

FCN：SSD、YOLOv1v2

FPN：RetinaNet、FCOS

精度：FPN>FCN

速度：FCN更快

内存：FCN更少

FCN 选用的feature map同等步长下，感受野比FPN小。

所以对于长条的物体，感受野命中需要考虑其短边，检测到需要具备多大的感受野考虑其长边，所以要求步长小，且感受野大。

要想在步长比较小的时候达到比较大的感受野，需要更多的卷积层。 此时FPN能将高层较大感受野的信息引入底层，缓解这个问题。



![image-20220106212326925](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20220106212326925.png)





head 中为什么不用BN

因为共享head接受来自不同尺度的特征度，到了推理阶段，为推理而累积计算的均值和方差在不同head之间摇摆，根本不行。 有可能起作用的是为每个head保存一套不同的均值和方差



![image-20220106214900259](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20220106214900259.png)



4. Centerness

加入了quality 分支

point越靠近目标中心，分数越高。



5、FPN：

![image-20220106222455143](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20220106222455143.png)

![image-20220106222531671](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20220106222531671.png)

以短边长度作为measure+FPN，很好检测条状物

但是不适用于自然场景下的人脸检测，人脸遮挡比较多。



人脸检测有没有FPN差距不明显？