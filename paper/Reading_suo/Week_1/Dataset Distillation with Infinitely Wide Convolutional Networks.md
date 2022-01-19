> 题目：Dataset Distillation with Infinitely Wide Convolutional Networks
>
> 来源：NeurIPS 2021
>
> 作者DeepMind† Google Research, Brain Team 



### motivation

问题：

将大型数据集压缩为更小但性能更高的数据集的数据集提取方法在训练效率和有用的特征提取方面将变得很有价值





prior work

Some direct approaches to this include choosing a representative subset of the dataset (i.e. a coreset) or else performing a low-dimensional projection that reduces the number of features. However, such methods typically introduce a tradeoff between performance and dataset size, since what they produce is a coarse approximation of the full dataset.

以前的方式是：实现这一点的一些直接方法包括选择数据集的代表子集（即核心集）或执行低维投影，以减少特征的数量。然而，这些方法通常会在性能和数据集大小之间进行权衡，因为它们产生的是完整数据集的粗略近似值。

this paper：dataset distillation

dataset distillation is to synthesize datasets that are more informative than their natural counterparts when equalizing for dataset size.

Such resulting datasets will not arise from the distribution of natural images but will nevertheless capture features useful to a neural network.

相比之下，数据集精馏的方法是合成数据集，当数据集大小相等时，它们比它们的自然对应物更具信息性[王等人，2018，Bodal等人，2020，NuyAn等人，2021，赵和BelEN，2021 ]。这样的结果数据集不会产生于自然图像的分布，但会捕获对神经网络有用的特征，



好处：

increasing the effectiveness of replay methods in continual learning [Borsos et al., 2020] and helping to accelerate neural architecture search [Zhao et al., 2021, Zhao and Bilen, 2021].



idea：

为此，我们采用了一种新的基于分布式内核的元学习框架，以实现使用无限宽卷积神经网络的数据集提取的最新结果。

Specifically, we apply the algorithms KIP (Kernel Inducing Points) and LS (Label Solve), first developed in Nguyen et al. [2021], to infinitely wide convolutional networks by implementing a novel, distributed meta-learning framework that draws upon hundreds of accelerators per training.

具体而言，我们通过实现一个新的分布式元学习框架（每次训练使用数百个加速器），将首次在Nguyen等人[2021]中开发的算法KIP（内核诱导点）和LS（标签求解）应用于无限宽的卷积网络。



不懂：

，随着中间层中隐藏单元的数量接近无穷大，贝叶斯和梯度下降训练的神经网络收敛到高斯过程（GP）



#### structure



设置：The central infinite-width model we consider is a simple 3-layer ReLU CNN with average pooling layers that we refer to as ConvNet throughout the text，This is the default model used by Zhao and Bilen [2021], Zhao et al. [2021],2 and was chosen for ease of baselining.

我们考虑的中心无限宽模型是一个简单的3层ReLU CNN，具有平均池化层，在整个文本中我们称为ConvNet。这是Zhao和Bilen[2021]、Zhao等人[2021]使用的默认模型，选择它是为了便于基线。