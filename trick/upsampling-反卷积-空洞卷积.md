## 【学习笔记】浅谈Upsampling

学机器学习的同学都接触过一个词，叫做**上采样（upsampling）**，这其实是一种统称。上采样是指将图像上采样到更高*分辨率（resolution）*，是一种把低分辨率图像采样成高分辨率图像的技术手段。

```text
常见的分辨率：
1080p ( 1920 × 1080 )
720p ( 1280 × 720 ）
4k ( 3840 × 2160 )
分辨率是指横向有多少个像素*纵向有多少个像素
```

那么一般常见的上采样家族都有那些呢？这里简要的进行归类和介绍：

## 1.Unpooling

国内有的学者称之为上池化，有的学者称之为解池化、反池化等，英文字面顾名思义，是pooling池化的逆运算，与pooling运算相反，但切记，池化是不可逆的过程，反池化只是一种近似。如下图，在pooling后得到的结果，可以通过unpooling家族的以下几种形式进行。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210413100445765.png" alt="image-20210413100445765" style="zoom:50%;" />

图1：经过池化的feature

### 1.1 复制式的unpooling

紫色透明方块代表一个2*2的unpooling filter，我们以stride=1，在如图所示位置演示复制式的反池化过程。

![img](https://pic1.zhimg.com/80/v2-23134e2573c6607843fbb570d7a00c70_1440w.jpg)图2：复制式的反池化

国内也有学者把这种复制式的池化称为unsampling，目的是和1.3中的索引式unpooling区分，但1.2和1.3大类上都是补零式的unpooling，所以有读者在阅读论文的时候看到类似术语，还需要仔细判别。

### 1.2 补零式的unpooling

![img](https://pic4.zhimg.com/80/v2-6a0c0582218d3cfbdb3fc54799492443_1440w.jpg)图3：补零式的反池化

这种池化有称之为Bed of Nails，其过程是，针对逐个被反池化元素，在左上角保持元素，其余位置填充0

### 1.3 索引式的unpooling

![img](https://pic4.zhimg.com/80/v2-31183605ec7a5b06ca2ecf1aba2032af_1440w.jpg)

图4：索引式的反池化

根据索引提示，反池化feature，相当于max pooling的逆操作。这种索引反池化是需要对称的池化层提供索引信息的，相当于在网络结构中加入了对称的skip connection。

## 2.Transposed Convolution

这一节主要讲另一种上采样方式，Transposed Convolution，有国内学者翻译为反卷积、逆卷积、转置卷积。

------

这里先引入一个关于Deconvolution和Transposed  Convolution名字争议的小故事：很多网站博主都会在介绍反卷积的时候开篇加入这句话：“逆卷积(Deconvolution)这个名字很容易引起误会，转置卷积(Transposed  Convolution)是一个更为合适的叫法”.甚至有些博主在学习笔记中建议机器学习交流中，不要使用Deconv以免造成误会，这到底是为什么呢？他们有什么区别呢？——然而他们都没有深入的提

在搜遍全网后，我在torch官方论坛[[1\]](https://zhuanlan.zhihu.com/p/359762840?utm_source=qq&utm_medium=social&utm_oi=793775383441985536#ref_1)了解到一个比较深刻的解释，供大家参考：

> A true deconvolution would be the mathematical inverse of a convolution.  The weights of the deconvolution would accept a 1 x 1 x in_channels  input, and output a kernel x kernel x out_channels output.
> A transposed convolution does not do this.

感谢这位mattrobin，原来Deconv英文本意所指是一种纯数学层面的卷积inverse 变换，是一种信号还原，并非机器学习中所指的这种upsampling方式。

------

言归正传，进入正题。

> 因为接下来要引用一些经典动图，所以这里要预先推荐一项Vincent Dumoulin大佬著名的tech repo工作[[2\]](https://zhuanlan.zhihu.com/p/359762840?utm_source=qq&utm_medium=social&utm_oi=793775383441985536#ref_2)，*A guide to convolution arithmetic for deep learning*，链接放在reference中，希望了解更多或者感兴趣的同学可以在文章末尾自取。

<img src="https://pic3.zhimg.com/v2-705305fee5a050575544c64067405fce_b.jpg" alt="img" style="zoom:50%;" />



图5：这是一个无stride无padding的卷积过程演示，蓝色是被卷输入，绿色是卷积输出

如图所示，是一个4x4的Input（蓝色），Conv Kernel为3x3,Conv Output为2x2（绿色），stride=0，padding=0. 我们针对这种卷积情况，讨论他的Transposed Convlution

### 2.1 外围padding式的Transposed Convlution

<img src="https://pic4.zhimg.com/v2-286ac2cfb69abf4d8aa06b8eeb39ced3_b.jpg" alt="img" style="zoom:50%;" />



图6：这个是反卷积过程演示，蓝色对应的是上图中的卷积输出（上图绿色），该图绿色是反卷积的输出


  反卷积的思想是，对feature进行padding，然后再进行卷积，这样可以达到Input为2X2但是output却为4X4的upsampling效果。可以看到，虽然名字叫反卷积，其本质还是卷积那一套东西，只不过对feature用padding做了扩张处理。

### 2.2 嵌入padding式的Transposed Convlution

<img src="https://pic3.zhimg.com/v2-2e99f89da83f29131f1a4c7593a6d5fa_b.webp" alt="img" style="zoom:50%;" />

[首页](https://www.freesion.com) [联系我们](mailto:freesion@gmx.com) [版权申明](https://www.freesion.com/copyright.html) [隐私政策](https://www.freesion.com/privacy.html)   [搜索](javascript:void(0))

## 3.空洞卷积与HDC

**空洞卷积**，**说白了就是把kernel变成筛状**。

两张图看明白。图源：

> https://www.jianshu.com/p/f743bd9041b3

**传统卷积核：**

![img](https://images2.freesion.com/324/39/393a2e0b589d5bd532320cda578f7344.png)
 **空洞卷积核：**
![img](https://images1.freesion.com/2/5f/5f73980c38d8ef5e57a3f505ee280622.png)
 而**HDC又是对空洞卷积的一种改进**，可以说他是**空洞卷积的一种特殊结构**

### 3.1.为何采用空洞卷积？

一句话：为了**增大感受野**，从而**抵消**一部分**池化层造成的信息丢失**

由此引出两个问题：
 （1）什么是感受野？
 （2）为什么池化层信息会丢失？

### 3.1.0.形象理解感受野

目前看网络上的一些帖子都采用的论文图，开始理解起来有一些困难，这里放上我个人的理解，如果有不对的地方欢迎大家指正。

首先，以**3x3传统卷积**为例，我们回顾一下图片卷积的工作模式。

**正着说，就是（靠上层中的）一片3x3区域被压缩为（靠下层中的）一个点**

![img](https://images1.freesion.com/99/df/dfee99c9554505228e92d742232ef6eb.png) 
 **那逆过来，就是（靠下层中的）一个点等效于（靠上层中的）一片3x3区域**
 那我就可以说，**这片3x3区域是相对于临近靠下层feature map中某个点的感受野**
 ![img](https://images1.freesion.com/375/a9/a92e535450b2f296b54de9cfec1e5b3f.png)

那如果靠下层中有多个这样的点组成一片3x3区域呢？再往下推一层，就是这样的。其中**黄色部分可以理解成上一张图的等效**。
 ![在这里插入图片描述](https://images1.freesion.com/407/8d/8dda1d5f46d1fd524a7134efd97ff2bf.png)

**以此类推**。推广到多层，就是这样的：
 ![在这里插入图片描述](https://images1.freesion.com/528/98/98c4c17492ccf9094b22c912c3640668.png)
 图源：

> https://blog.csdn.net/program_developer/article/details/80958716

### 3.1.1.如何通过空洞卷积增大感受野

我们从传统卷积说起。还是回到刚才这张图：可以发现，**我从第二层随便挑一个点，它的感受野都是5x5。**
 ![在这里插入图片描述](https://images1.freesion.com/691/ea/ea63ae28ecba0ab507b9964dc13c0eeb.png)
 现在我们将**第一层到第二层**的卷积方式改为**空洞卷积**：
 ![在这里插入图片描述](https://images1.freesion.com/241/f8/f8defe2cd12f96e7779032390a775269.png)
 同时，**保持输入层到第一层为普通卷积**：
 ![在这里插入图片描述](https://images1.freesion.com/585/b9/b981eeaf25b57eeb3ceeaf0e6333e549.png)
 那最终**等效出来**就是一个**感受野为7x7**大小的区域：
 ![在这里插入图片描述](https://images1.freesion.com/46/87/87c979daad99af1852df9b3c83ff67b6.png)
 **综上所述，适当采用空洞卷积可以增大感受野。**

 这里引入了一个新的超参数 d，(d - 1) 的值则为塞入的空格数，假定原来的卷积核大小为 k，那么塞入了 (d - 1) 个空格后的卷积核大小 n 为：

![img](https://img2018.cnblogs.com/blog/1365470/201903/1365470-20190307164837810-385722401.png) 

进而，假定输入空洞卷积的大小为 i，步长 为 s ,空洞卷积后特征图大小 o 的计算公式为
![img](https://img2018.cnblogs.com/blog/1365470/201903/1365470-20190307164858018-1858925034.png)

### 3.1.2.通过增大感受野，抵消一部分池化层带来的信息丢失

由池化（pooling）本身的原理来看，其实就是对feature map进行了采样，采样就要损失一部分信息。如图中，粉色部分采样后除了“6”以外的数据全部丢失。如果你的网络结构采用了多个pooling层，到了FC层门口，信息丢失的也差不多了。
 ![在这里插入图片描述](https://images1.freesion.com/907/f4/f40c4171daeca98b04d4f9822845db03.png)

对于保留下来的这个“6”来说，我们假设它是第二层feature map中的一点，那么在采用传统卷积的情况下，它对应到输出图像上的感受域是5x5大小：
 ![在这里插入图片描述](https://images1.freesion.com/871/df/dfb2c2aa76f1846552578af8fa72488f.png)
 但如果你在第一层到第二层的过程中采用了空洞卷积，情况是这样的：
 ![在这里插入图片描述](https://images1.freesion.com/689/ab/ab29d61cdd2ec35787c3c18a38a15369.png)

**现在一共4个袋子，池化层说，你只能带走一个。如果你用了空洞卷积，你带走的袋子里就有7x7=49块糖，如果你用的是普通卷积，对不起你只能带走5x5=25块糖**。

一句话：**同样被池化层剥离掉一部分（而且是相当一部分）原始信息，空洞卷积允许保留下来的元素携带更多信息量。**

### 3.2.为何采用HDC？

因为空洞卷积有一个**致命的问题**：**卷积核不连续**。

### 3.2.1.空洞卷积的缺陷

刚刚我们分析的情况都是只有第一层到第二层采用空洞卷积，那么如果从输入图像到第一层也采用空洞卷积呢？得到的是一个9x9的区域，重点是，**感受域不连续**。
 ![在这里插入图片描述](https://images1.freesion.com/179/c9/c9a516f70c833935e9107b6dbc617a2b.png)
 如果这个9x9区域里包含的细节较少还好，我们拿猫耳朵举例，所有黄色点拼在一起没准还能凑合看出来是只猫耳朵。（实际操作上猫耳朵已经算是非常细节、非常高级的特征了，这里只是打个比方便于理解。）
 ![在这里插入图片描述](https://images1.freesion.com/870/8b/8baac968c1cf8f9fff9e1fde440e18b6.png)
 **但如果这个区域包含了大量细节信息，卷积核get到的特征可能是紊乱的，甚至是错误的**。比如对于整只猫来说，仅仅用黄色点的信息就很难表示这真的是一只猫。
 ![在这里插入图片描述](https://images1.freesion.com/679/1d/1d64b50d5c849bbcd0ba1b6348100f6f.png)

### 3.2.2.HDC结构

HDC结构，全称Hybrid Dilated Convolution，直译过来就是**混合空洞卷积**，就是为了避免这种情况发生。

具体定义这里不放了，直白来讲就是**避免在所有层中采用相同间隔的空洞卷积**。

**所谓间隔，就是两个元素之间的距离，用rate表示。**

下图中（a）是连续做rate=2空洞卷积的结果（你可以把左1想象成原图取点的过程），而（b）是rate分别为1/2/3空洞卷积的结果。
 ![在这里插入图片描述](https://images1.freesion.com/378/06/069c0472cb606a8fa9213b7913c3d47a.png)
 （b）的牛逼之处在于，它一开始就保留了完整连续的3x3区域，之后几层的rate设置又刚好保证拼起来感受域的连贯性，**即使有所重叠，也密不透风**。

### 3.3.实现代码

在tensorflow里已经封装好了空洞卷积专用的函数：**tf.nn.atrous_conv2d**

使用起来也很简单，无论你想尝试错误的空洞卷积结构还是HDC结构均可。

```python
conv=tf.nn.atrous_conv2d(X,W,rate,padding)#空洞卷积
1
```

附上w3cschool的教程链接：

> https://www.w3cschool.cn/tensorflow_python/tf_nn_atrous_conv2d.html

**注意：rate=1的时候完全等效于普通卷积。**

### 3.4.参考文献

https://www.jianshu.com/p/f743bd9041b3

https://blog.csdn.net/program_developer/article/details/80958716

https://blog.csdn.net/zsf10220208/article/details/93995482






## 重新思考卷积： 

Rethinking Convolution在赢得其中一届ImageNet比赛里VGG网络的文章中，他最大的贡献并不是VGG网络本身，而是他对于卷积叠加的一个巧妙观察。

>  This (stack of three 3 × 3 conv layers) can be seen as imposing a regularisation on the 7 × 7 conv. filters, forcing them to have a decomposition through the 3 × 3 filters (with non-linearity injected in between).

这里意思是 7 x 7 的卷积层的正则等效于 3 个 3 x 3 的卷积层的叠加。而这样的设计不仅可以大幅度的减少参数，其本身带有正则性质的 convolution map 能够更容易学一个 generlisable, expressive feature space。这也是现在绝大部分基于卷积的深层网络都在用小卷积核的原因。![img](https://pic1.zhimg.com/80/v2-ee6f0084ca22aa8dc3138462ee4c24df_1440w.jpg?source=1940ef5c)

然而 Deep CNN 对于其他任务还有一些致命性的缺陷。较为著名的是 up-sampling 和 pooling layer 的设计。这个在 Hinton 的演讲里也一直提到过。

* 主要问题有：

  1、Up-sampling / pooling layer (e.g. bilinear interpolation) is deterministic. (a.k.a. not learnable)

  2、内部数据结构丢失；空间层级化信息丢失。

  3、小物体信息无法重建 (假设有四个pooling layer 则 任何小于 2^4 = 16 pixel 的物体信息将理论上无法重建。)

  

  在这样问题的存在下，语义分割问题一直处在瓶颈期无法再明显提高精度， 而 dilated convolution 的设计就良好的避免了这些问题。

  ### 空洞卷积的拯救之路：

  Dilated Convolution to the Rescue题主提到的这篇文章 [MULTI-SCALE CONTEXT AGGREGATION BY DILATED CONVOLUTIONS](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1511.07122.pdf) 可能是第一篇尝试用 dilated convolution 做语义分割的文章。后续图森组和 Google Brain 都对于 dilated convolution 有着更细节的讨论，推荐阅读：[Understanding Convolution for Semantic Segmentation](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1702.08502) [Rethinking Atrous Convolution for Semantic Image Segmentation](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.05587) 。

* 对于 dilated convolution， 我们已经可以发现他的优点，

  即内部数据结构的保留和避免使用 down-sampling 这样的特性。但是完全基于 dilated convolution 的结构如何设计则是一个新的问题。

### 潜在问题 1：The Gridding Effect

假设我们仅仅多次叠加 dilation rate 2 的 3 x 3 kernel 的话，则会出现这个问题：![img](https://pic1.zhimg.com/80/v2-478a6b82e1508a147712af63d6472d9a_1440w.jpg?source=1940ef5c)

我们发现我们的 kernel 并不连续，也就是并不是所有的 pixel 都用来计算了，因此这里将信息看做 checker-board 的方式会损失信息的连续性。这对 pixel-level dense prediction 的任务来说是致命的。

### 潜在问题 2：Long-ranged information might be not relevant.

我们从 dilated convolution 的设计背景来看就能推测出这样的设计是用来获取 long-ranged information。然而光采用大 dilation rate 的信息或许只对一些大物体分割有效果，而对小物体来说可能则有弊无利了。如何同时处理不同大小的物体的关系，则是设计好 dilated convolution 网络的关键。

### 通向标准化设计：Hybrid Dilated Convolution (HDC)

> 对于上个 section 里提到的几个问题，图森组的文章对其提出了较好的解决的方法。他们设计了一个称之为 HDC 的设计结构。
>
> 第一个特性是，叠加卷积的 dilation rate 不能有大于1的公约数。比如 [2, 4, 6] 则不是一个好的三层卷积，依然会出现 gridding effect。
>
> 第二个特性是，我们将 dilation rate 设计成 锯齿状结构，例如 [1, 2, 5, 1, 2, 5] 循环结构。
>
> 第三个特性是，我们需要满足一下这个式子： ![[公式]](https://www.zhihu.com/equation?tex=M_i%3D%5Cmax%5BM_%7Bi%2B1%7D-2r_i%2CM_%7Bi%2B1%7D-2%28M_%7Bi%2B1%7D-r_i%29%2Cr_i%5D)其中 ![[公式]](https://www.zhihu.com/equation?tex=r_i) 是 i 层的 dilation rate 而 ![[公式]](https://www.zhihu.com/equation?tex=M_i) 是指在 i 层的最大dilation rate，那么假设总共有n层的话，默认 ![[公式]](https://www.zhihu.com/equation?tex=M_n%3Dr_n) 。
>
> 假设我们应用于 kernel 为 k x k 的话，我们的目标则是 ![[公式]](https://www.zhihu.com/equation?tex=M_2+%5Cleq+k) ，这样我们至少可以用 dilation rate 1 即 standard convolution 的方式来覆盖掉所有洞。一个简单的例子: dilation rate [1, 2, 5] with 3 x 3 kernel (可行的方案） 

![img](https://pic2.zhimg.com/80/v2-3e1055241ad089fd5da18463903616cc_1440w.jpg?source=1940ef5c)

而这样的锯齿状本身的性质就比较好的来同时满足小物体大物体的分割要求(小 dilation rate 来关心近距离信息，大 dilation rate 来关心远距离信息)。

这样我们的卷积依然是连续的也就依然能满足VGG组观察的结论，大卷积是由小卷积的 regularisation 的 叠加。

以下的对比实验可以明显看出，一个良好设计的 dilated convolution 网络能够有效避免 gridding effect.

![img](https://pic1.zhimg.com/80/v2-b2b6f12a4c3d244c4bc7eb33814a1f0d_1440w.jpg?source=1940ef5c)

多尺度分割的另类解：Atrous Spatial Pyramid Pooling (ASPP)在处理多尺度物体分割时，我们通常会有以下几种方式来操作：

![img](https://pic2.zhimg.com/80/v2-0510889deee92f6290b5a43b6058346d_1440w.jpg?source=1940ef5c)



然而仅仅(在一个卷积分支网络下)使用 dilated convolution 去抓取多尺度物体是一个不正统的方法。比方说，我们用一个 HDC 的方法来获取一个大（近）车辆的信息，然而对于一个小（远）车辆的信息都不再受用。假设我们再去用小 dilated convolution 的方法重新获取小车辆的信息，则这么做非常的冗余。

基于港中文和商汤组的 PSPNet 里的 Pooling module （其网络同样获得当年的SOTA结果），ASPP 则在网络 decoder 上对于不同尺度上用不同大小的 dilation rate 来抓去多尺度信息，每个尺度则为一个独立的分支，在网络最后把他合并起来再接一个卷积层输出预测 label。

这样的设计则有效避免了在 encoder 上冗余的信息的获取，直接关注与物体之间之内的相关性。

### 总结

Dilated Convolution 个人认为想法简单，直接且优雅，并取得了相当不错的效果提升。他起源于语义分割，大部分文章也用于语义分割，具体能否对其他应用有价值姑且还不知道，但确实是一个不错的探究方向。有另外的答主提到WaveNet, ByteNet 也用到了 dilated convolution 确实是一个很有趣的发现，因为本身 sequence-to-sequence learning 也是一个需要关注多尺度关系的问题。则在 sequence-to-sequence learning 如何实现，如何设计，跟分割或其他应用的关联是我们可以重新需要考虑的问题。