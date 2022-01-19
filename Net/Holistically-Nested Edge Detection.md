## Holistically-Nested Edge Detection



边缘检测任务：precisely localizing edges in natural images involves visual perception of various “levels”

精确定位边缘涉及不同层次的视觉感知。

边缘检测的任务是提取图像内每个对象的边界，而排除对象的纹理。HED 被设计以解决两个问题：（1）对图像整体的训练和预测，End-to-end；（2）多尺度的特征提取。

### 整体嵌套边缘检测：

abstract：我们设计了什么算法，解决了什么问题，痛点，我们给他命名**，他的结构，他的作用重要性应用，他在数据集的表现。

introduction：我的工作是解决了啥，这个工作是很重要的，在很多领域应用广泛啊，比如。

### 名称：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210411184146711.png" alt="image-20210411184146711" style="zoom:33%;" />

我们使用术语“整体”：因为HED，尽管没有`显式地建模 `结构化输出，但其目的是以图像到图像的方式训练和预测边缘。也就是说

在“嵌套”中：我们强调继承的和逐步细化的边缘图（edge map）作为side outputs（With “nested”, we emphasize the inherited and progressively refined edge maps produced as side outputs）

嵌套多尺度特征学习：在不同层次对特征的集成学习与以前的多尺度方法不同。

不同尺度下的Canny边缘不仅没有直接的联系，而且还表现出空间偏移和不一致性。

### 语义分割、正负样本不均衡-损失函数设计

像素级别的分类正负样本很不均衡：怎么办？

因为是像素级别的分类，正负样本很不均衡，90%是non-edge ，为了平衡正负样本，A cost-sensitive loss function is proposed in （[19] J.-J. Hwang and T.-L. Liu. Pixel-wise deep learning for con- tour detection. In ICLR, 2015. 1, 2, 4, 7）, with additional trade-off parameters introduced for biased sampling.用于有偏采样的附加权衡参数

目标检测里面也有正负样本不均衡的问题，怎么做的？

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210411162433407.png" alt="image-20210411162433407" style="zoom:50%;" />

和语义分割是不同的！在前景和背景像素大致一样时，使用交叉熵损失是有用的。矩形框分类是不均衡的思路是使用IoU可以达到目的。

但是边缘检测不可以使用IoU，如果一个边缘检测结果是 GT 平移了几个像素，那它仍可称得上好，但它的 IoU 却会骤降至和随机结果差不了多少。如果对边缘检测问题用 IoU 做优化对象，恐怕在优化时根本找不到可以下降的梯度方向。



涉及损失函数的原则：在人类感知几乎完美预测的情况下，函数不给惩罚

### 结构：



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210328152215266.png" alt="image-20210328152215266" style="zoom: 67%;" /><img src="https://img-blog.csdn.net/20171218103557861?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2FuZ2t1bjEzNDAzNzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center" alt="img" style="zoom:50%;" />

backbone为VGG16，根据尺寸分为5个stage。再通过一个1x1x1的conv，将每个输出进行反卷积到原来尺寸，然后进行相加。再经过一个1x1x1的conv降低维度得到结果图。同时loss也与每张图的loss有关。

结构包含一个有多个side outputs单流的深度网络，每个网络层都被认为是一个单子网络，负责生成一定比例的edge map。做出预测的每一条path 普遍存在与每一个edge map（we intend to show that the path along which each prediction is made is common to each of these edge maps）

Our architecture comprises a single- stream deep network with multiple side outputs.

在边缘检测中，获得多个预测然后将edge map组合在一起通常是有利的（并且确实是普遍的）。





<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210411105850980.png" alt="image-20210411105850980" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210411105914591.png" alt="image-20210411105914591" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210411154955837.png" alt="image-20210411154955837" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210411155016398.png" alt="image-20210411155016398" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210411155030537.png" alt="image-20210411155030537" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210411155042526.png" alt="image-20210411155042526" style="zoom: 50%;" />

a）我们将我们的侧输出层连接到每个阶段的最后一个卷积层，分别是conv1 2、conv2 2、conv3 3、conv4 3、conv5 3。每个卷积层的感受野大小与相应的边输出层相同；（b）我们切割VGGNet的最后一级，包括第5池层和所有完全连接的层。“修剪”VGNET的原因有两个方面。首先，因为我们期望不同尺度的有意义的边输出，一个跨距为32的层产生一个太小的输出平面，其结果是插值预测图太模糊，无法利用。

Our final HED network architecture has 5 stages, with strides 1, 2, 4, 8 and 16, respectively, and with different receptive field sizes, all nested in the VGGNet. See Table 1 for a summary of the configurations of the receptive fields and strides.

我们最终的HED网络结构分为5个阶段，分别为跨距1、2、4、8和16，感受野大小不同，都嵌套在VGGNet中。关于感受野和感受器的配置，

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210411130924303.png" alt="image-20210411130924303" style="zoom:50%;" />



### deep supervision

如果没有deep supervision，cnn网络的高层是细节缺失和甚至空间位置偏移的高层语义信息，所以如果没有对每一个stage进行监督只有weighted-fusion output layer，那么网络就会往那个方向学习，而对每一层使用groud truth进行规范，那么参数就会往：每一层side output 输出边缘逐渐变粗，并且更加“全局”，同时保留关键对象边界的方向去学习。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210411133721049.png" alt="image-20210411133721049" style="zoom:50%;" />

### 损失函数

HED 将边缘检测任务归纳为对每个像素点的二分类任务——“边缘”和“非边缘”。对于 HED 的单个输出而言，其损失函数为所有像素点的二分类损失函数的和，另外，由于边缘占的像素总数一般都会少于非边缘，所以实际是边缘的像素提供的二分类损失函数会乘以一个更大的权重，以进行正负样本平衡。HED 整体的损失函数就是它所有输出的损失函数的加权和。

转化成数学语言就是：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210411185316910.png" alt="image-20210411185316910" style="zoom:67%;" />

其中![\mathbf{W}](https://math.jianshu.com/math?formula=%5Cmathbf%7BW%7D)指特征提取网络（VGG）的权重，![\mathbf{w}^{(m)}](https://math.jianshu.com/math?formula=%5Cmathbf%7Bw%7D%5E%7B(m)%7D)指 HED 第$m$层输出的输出层权重，![\alpha _{m}](https://math.jianshu.com/math?formula=%5Calpha%20_%7Bm%7D)为平衡每层输出为最终损失贡献的系数，$\beta = |Y_{-}|/(|Y_{-}|+|Y_{+}|)$为平衡正负样本的系数，$Y_{+}$和![Y_{-}](https://math.jianshu.com/math?formula=Y_%7B-%7D)分别指代边缘像素和非边缘像素，![y_{j}](https://math.jianshu.com/math?formula=y_%7Bj%7D)为像素$j$输出的置信度。

#### 融合输出

上面的损失函数是针对每个侧输出进行优化，HED 的最终输出是每个侧输出按照一定的权重加总得到的融合输出，这些权重是通过训练学习到的，而非人为设定的。

融合输出的损失函数如下：

$\mathcal{L}_{\text {fuse }}(\mathbf{W}, \mathbf{w}, \mathbf{h})=\operatorname{Dist}\left(Y, \hat{Y}_{\text {fuse }}\right)\\$

其中融合输出$\hat{Y}_{\text {fuse }} \equiv \sigma\left(\sum_{m=1}^{M} h_{m} \hat{A}_{\text {side }}^{(m)}\right)$，![h_{m}](https://math.jianshu.com/math?formula=h_%7Bm%7D)是每个侧输出在融合时的权重，$Dist()$计算输出和 GT 之间的距离，这里采用交叉熵函数。

整个模型在训练时的优化目标权重为：$(\mathbf{W}, \mathbf{w}, \mathbf{h})^{\star}=\operatorname{argmin}\left(\mathcal{L}_{\text {side }}(\mathbf{W}, \mathbf{w})+\mathcal{L}_{\text {fuse }}(\mathbf{W}, \mathbf{w}, \mathbf{h})\right)\\$。可以看到，最终的损失函数中存在一定的冗余，由于融合输出是由侧输出得到的，侧输出似乎被不止一次地惩罚了。不过，先不论这种冗余是不是必要的，据作者言，只对融合输出进行惩罚得到的效果是不够好的，因为模型总会区域学习更大尺度上的特征。

#### 对损失函数的想法

HED 的损失函数是一种很直接的思路，不过任然有这样的问题：当一个被预测为“边缘”的像素点实际上是“非边缘”时，不管它和 GT 离得有多近，体现在损失函数上，都和一个差 GT 十万八千里的像素点没有区别。这种设计——就我个人的理解——会让损失函数的梯度出现大面积的平坦区域，梯度下降难以工作。但目前的工作似乎都是在用交叉熵作为损失函数，虽然今年也提出了 G-IoU、D-IoU 等将 IoU 调整后作为损失函数的方法，但是限于数学表达上的困难，目前只能应用于矩形边界框，而不能应用于像素集分割。



