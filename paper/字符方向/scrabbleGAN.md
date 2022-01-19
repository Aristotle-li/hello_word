## motivation



手写文本识别 (HTR) 与OCR相比还有很大提升空间，主要是HTR缺少通用的，有注释的手写文本数据集，获取成本很大

## idea

拼字游戏GAN：逐个字符生成序列。

对于HTR，训练词汇和风格丰富同样重要，只通过旋转，剪切，mask，拉伸，压缩的数据增广没有增加词汇量只丰富了风格，而基于GAN的方法可增加训练词汇和风格。



label：？是一个字符的one-hot向量，后期改进可以对对应的偏旁部首对应编码

sample：？





完全卷积手写文本生成算法，该算法有三层结构

1、我们提出了一种新颖的全卷积手写文本生成体系结构，它允许任意长的输出，它学习字符嵌入，而不需要字符级注释。

2、我们展示了如何在半监督的情况下训练这个生成器，使其能够适应一般的未标记数据，特别是测试时图像

3、明训练词汇对HTR训练的重要性不亚于丰富的风格





使用完全卷积生成器的另一大好处是不需要使用循环网络学习整个单词的嵌入，

学习相邻字符之间的依赖关系允许网络根据其相邻字符创建同一字符的不同变体

### G：生成手写体



和2相比，改进主要是生成器：不是像 [2] 中那样从整个单词表示中生成图像，而是单独生成每个字符，使用 CNN 的重叠感受野的特性来考虑附近字母的影响。换句话说，我们的生成器可以看作是相同类条件生成器 [33] 的串联，其中每个类都是一个字符。这些生成器中的每一个都会生成一个包含其输入字符的补丁。每个卷积上采样层都会加宽感受野，以及两个相邻字符之间的重叠。这种重叠允许相邻字符交互，并创建平滑过渡。图 2 左侧说明了单词“meet”的生成过程





图 2 左侧说明了单词“meet”的生成过程。对于每个字符，从与字母表一样大的过滤器组 F 中选择过滤器 f⋆，例如 F = {fa, fb,..., fz} 代表小写英文。四个这样的过滤器在图 2 中连接起来（fe 使用了两次），并乘以噪声向量 z，后者控制文本样式。

此外，学习相邻字符之间的依赖关系允许网络根据其相邻字符创建相同字符的不同变体。

每个图像的风格由作为网络输入的噪声向量 z 控制。为了为整个单词或句子生成相同的样式，这个噪声向量在输入中所有字符的生成过程中保持不变。

### D：分别提高样式



1、D的作用也是基于手写输出风格来区分这些图像。鉴别器结构必须考虑到生成图像的不同长度，因此也被设计为卷积的：鉴别器本质上是具有重叠感受野的独立“真/假”分类器的串联。

2、和cGAN不同，该结构使用字符级的编码，但却没有字符级的注释，所以我们不能对这些分类器中的每一个使用类监督。

3、这样做的一个好处是我们现在可以使用未标记的图像来训练 D，池化层将所有分类器的分数聚合到最终的鉴别器输出中。





### R： Localized text recognizer，用于提升准确度。数据保真度：

和2一样，R只在真实的、有标签的手写样本上训练。大多数识别网络使用循环模块，通常是双向LSTM[19]，该模块通过利用先前和后续图像块中的信息来读取当前图像块中的字符。



不能让R太强了，不然写的不好都可以识别出来了！

是因为CRNN学到的是一种隐式的语言模型，即使写的不清楚也可以识别出来，因此为了迫使R仅根据视觉特征做出决定，the recurrent head is removed



虽然在手写识别模型中通常需要这种质量，但在我们的例子中，这种质量可能会导致网络正确读取生成器没有清晰书写的字符。因此，我们选择不使用重复出现的识别网络的“头”，这使得这种质量，并保持只有卷积骨干。具体分析见补充资料。

### loss

梯度在量级可能有很大变化：使用如下公式解决

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210727202839138.png" alt="image-20210727202839138" style="zoom:50%;" />



### 结构：

G

每个filter的大小为 32*8192，n 个字符的单词通过连接n个filter实现，将它们与 32 维噪声向量 z1 相乘，得到在 n * 8192 矩阵中。接下来，后一个矩阵被reshape为 512* 4 *4n，这时每个字符spatial size 是4\*4，随后张量被送入三个residual block，这些块对空间分辨率进行上采样，创建上述感受野重叠，并导致最终图像大小为 32 *16n，

**条件实例归一化层**[11]用于使用三个额外的32维噪声向量z2、z3和z4调制剩余块。最后，使用具有tanh激活的卷积层来输出最终图像。



传统bigGAN是一个类用于整个图像，而这里图像的不同区域取决于不同的类（字符）。在第一层中施加这种空间条件比较容易，因为不同字符之间没有重叠。然而，由于感受野重叠，将该信息直接馈送到以下块中的CBN层更加困难。因此，我们仅使用噪声矢量z2到z4，而不对CBN层进行类别调整。有关第一层输入的更多详细信息，请参见本文第4.1节的实现细节。



D、R

鉴别器网络D的灵感来源于BigGAN[8]：4个residual block，后跟一个含有一个输出的线性层。为了处理不同宽度的图像生成，D也是完全卷积的，基本上是处理水平重叠的图像块。 The final prediction is the average of the patch predictions, which is fed into a GAN hinge-loss [30].。

识别网络R的灵感来自CRNN[38]。网络的卷积部分包含六个卷积层和五个池层，所有这些层都具有ReLU激活。**最后，使用线性层输出每个窗口的类别分数，并与使用连接主义时间分类（CTC）损失的ground truth  annotation 进行比较[15]**。



### datasets

**RIMES**数据集包含来自法语的单词，涵盖约6万张由1300位不同作者撰写的图像。

**IAM**数据集包含约10万张英语单词图像。数据集分为657位不同作者编写的单词。

训练、测试和验证集包含互斥作者编写的单词。

**CVL**数据集由七个手写文档组成，其中我们只使用六个英文文档。这些文档由大约310名参与者编写，产生了大约83k个单词，分为训练集和测试集。

所有图像都被调整到32像素的固定高度，同时保持原始图像的纵横比。

对于GAN训练的特定情况，并且仅当使用标签时（监督情况），我们额外水平缩放图像，使每个字符与合成字符的宽度大致相同，即每个字符16个像素。这样做是为了通过使真实样本与合成样本更相似来挑战鉴别器。

我们使用两个通用的标准度量来评估我们的方法。

首先，单词错误率（WER）是测试集中单词数量中的误读单词数量。

其次，标准化编辑距离（NED）是通过预测词和真实词之间的编辑距离来测量的，真实词长度标准化了预测词和真实词之间的编辑距离。只要有可能，我们将重复五次，并报告其平均值和标准偏差。

### 风格：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210812193421333.png" alt="image-20210812193421333" style="zoom:50%;" />

This image provides a good example of character interaction: while all repeti- tions of a character start with identical filters fi, each final instantiation might be different depending on the adjacent characters.

此图像提供了一个很好的 character interaction示例：虽然相同字符的filter是相同的，但是根据相邻字符的不同，每个人字符也不同。



虽然的所有重复都以相同的过滤器fi开始，但根据相邻角色的不同，每个最终实例化可能会有所不同。

### GB梯度平衡

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210812205824931.png" alt="image-20210812205824931" style="zoom:50%;" />

从左侧仅使用'R'开始训练，到右侧仅使用'D'进行训练

随着$\alpha$ 的增加，字符之间关联越弱，左边越大字符越分开越容易检测，内容越重要，右边越小风格越重要

### Handwriting text recognition

RCNN：

An end- to-end trainable neural network for image-based sequence recognition and its application to scene text recognition.

Poznanski等人[35]使用CNN来估计图像的n-grams轮廓，并将其与字典中现有单词的轮廓进行匹配。

Cnn-n-gram for hand- writing word recognition. 

Suerias等人[40]使用了一种受序列到序列[41]启发的架构，在这种架构中，他们使用的是注意力解码器，而不是使用CRNN

Offline continuous handwriting recognition using sequence to sequence neural networks



### Handwriting text generation

第一篇风格与内容解耦

Deepwriting: Making digital ink editable via deep generative modeling 

笔迹模仿，但对于每个新的数据样本，都需要一个耗时的字符级注释过程

My Text in Your Handwriting

42：他们以端到端的方式将这个过程融入到任务损失中。

Low-shot learning from imaginary data.



使用对抗式学习的低资源手写体识别。这种方法不能在给定的词典之外生成单词，这是一个关键的属性

Handwriting recognition in low-resource scripts using adversarial learning



Krishanan el al[28]提出了一种利用合成数据进行单词识别的方法，同时不依赖于合成数据的特定来源（例如，可以使用我们的方法生成的数据）

Word spotting and recognition using deep embedding



在[2]中提出的网络使用LSTM将输入字嵌入到一个固定长度的表示中，这个表示可以输入到BigGAN[8]体系结构中，启发自42

Adversarial generation of handwritten text images conditioned on sequences

（与我们允许可变字长和图像长度的方法相反，这个生成器只能在所有字长上输出固定宽度的图像。使用完全卷积生成器的另一大好处是不需要使用循环网络学习整个单词的嵌入，相反，我们可以直接学习每个字符的嵌入，而不需要字符级注释。）

生成用于文本识别的合成数据：上面提到的大多数 HTR 方法使用某种随机参数空间失真来放大数据的视觉可变性。

 Generating syn- thetic data for text recognition

Are multidimensional recurrent lay- ers really necessary for handwritten text recognition?

### 问题？

1、

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210727192825876.png" alt="image-20210727192825876" style="zoom:50%;" />

阿隆索等人。 [2] 提出了一个新的 HTG 模型，让人想起 [42] 中的工作，这反过来启发了我们的方法。 [2] 中介绍的网络使用 LSTM 将输入词嵌入到固定长度的表示中，该表示可以输入 BigGAN [8] 架构。与我们的方法不同，它允许可变的字长和图像长度，这个生成器只能输出所有字长的固定宽度的图像。使用完全卷积生成器的另一个巨大好处是无需使用循环网络学习整个单词的嵌入，我们可以直接学习每个字符的嵌入，而无需字符级注释。？



2、G、D、R的结构感觉不太优美。G1、G2，分别生成风格和文本特征

