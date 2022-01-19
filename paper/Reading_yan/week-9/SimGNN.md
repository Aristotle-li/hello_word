

> 题目：SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
>
> 来源：WSDM 2019
>
> 作者：University of California, Los Angeles, CA, USA Purdue University, IN, USA Zhejiang University, China



### motivation

图的相似度计算应用十分广泛，比如找到与query分子到相似的结构，通常用**图编辑距离(GED)**或者**最大共同子图(MCS)**来衡量图的相似度，然而这两个指标的计算复杂度都是很高的（NP-complete）。这篇文章提出了一种基于图神经网络的方法来解决这一问题。

### Main idea

采用图神经网络，使用基于learning的方法，学习的对象是从输入 **一对图（a pair of graphs）**到输出 **两个图的相似度分数** 的映射，因此是一种有监督的学习，需要知道输入图对相似度的ground truth。

本文采用两个策略，，一、将图送入GCN得到node-Embedding，对其使用注意力模块aggregate每个图最相关的部分生成graph-level  Embeding，再计算相似度。二、a pairwise node comparison method to supplement the graph-level embeddings 

### Network structure

它试图学习一个函数来将一对图映射成一个相似性分数。一个简单直接的思想就是：给定一对图，我们需要将图进行向量表示，再计算graph embedding的计算内积来计算相似度。

本文采用两个策略：

一、（1）节点嵌入阶段，将图的每个节点转化为一个向量，对其特征和结构性质进行编码，本文使用GCN。(2） 图嵌入阶段，通过对前一阶段生成的节点嵌入进行基于注意的聚合，为每个图生成一个嵌入(3） 图-图交互阶段，接收两个图级嵌入，返回表示图相似度的交互分数，本文使用NTN计算相似性向量；(4)最后的图相似度得分计算阶段，进一步将 interaction scores vector减少为一个最终的相似度得分，本文通过一个全连接神经网络实现。

二、在此基础上，考虑到只利用 graph embedding 可能忽略了局部节点的差异性，因此作者进一步考虑了两个图中节点之间的相关性或者是差异性 （ comparing two sets of node-level embeddings）。



![image-20210714145353441](/Users/lishuo/Library/Application Support/typora-user-images/image-20210714145353441.png)

#### strategy1：Graph embedding

1、采用GCN因为满足表示不变性、inductive、learnable，得到节点嵌入：

首先对不同类型的节点进行 one-hot encoding，如下图有两类共8个节点，就对应可以得到 8 个 one-hot 向量

![img](https://pic4.zhimg.com/80/v2-7ee18817c1bd9e40efd5597a62fe527b_1440w.png)

通过 Graph convolutional network 进行特征提取（聚合就是邻居特征一起加权在经过激活函数变换一下）

![[公式]](https://www.zhihu.com/equation?tex=conv%28u_n%29%3Df_1%28%5Csum_%7Bm%5Cin+N%28n%29%7D+%5Cfrac%7B1%7D%7B%5Csqrt%7Bd_n+d_m%7D%7Du_m+W_1%5E%7B%28l%29%7D%2Bb_1%5E%7B%28l%29%7D%29)

经过几层图卷积后可以得到节点的embedding表示了，比如刚才上图的 one-hot 向量就变成下图这样稠密的向量了

![img](https://pic4.zhimg.com/80/v2-d120a3013b614e8d5d51899dfc216d63_1440w.png)

2、可以简单的直接取平均或加权平均得到图嵌入，但文中采用了注意力机制。(4) AttDegree uses the natural log of the degree of a node as its attention weight，(5) AttLearnableGC utilize the global graph context to compute the attention weights,但前者不适用于可学习权值的非线性变换，而后者则适用。

* 加权平均得到 a global graph context c，权值矩阵可学习
* 计算每个节点嵌入和c的内积输入sigmoid函数得到注意力系数a，内积直觉是，类似于全局上下文的节点应该获得更高的注意权重。
* 计算u的加权得到h。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210714144805598.png" alt="image-20210714144805598" style="zoom:50%;" />

也就是一个图现在变成一个向量表示了

![img](https://pic3.zhimg.com/80/v2-c1c86b77017881ad8a668d4a807ae8d2_1440w.png)

3、计算两个entity的相似性，最简单的方法是计算对应嵌入向量的内积。文中使用了 Neural Tensor Network(NTN)，来计算两个图的相似度。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210714145138985.png" alt="image-20210714145138985" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210714153030996.png" alt="image-20210714153030996" style="zoom:50%;" />

该网络的输入就是两个 graph embedding，输出为一个 K 维的相似度向量。K是一个超参数，用于控制模型为每个图嵌入对生成的交互（相似性）分数的数量。

4、loss：using the following mean squared error loss function:

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210714151824789.png" alt="image-20210714151824789" style="zoom:50%;" />

#### strategy2：Pairwise Node Comparison

启发于自然语言处理，基于sentence embedding的匹配，加入fine-grained word-level information后，性能得到进一步提升。

为了进一步考虑局部节点的细粒度信息，作者提出了个成对节点比较的方式，说白点其实就是把前面得到的节点embedding分别内积一下（维度不够的补0），得到一个相关性矩阵

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210714153144895.png" alt="image-20210714153144895" style="zoom:50%;" />

考虑如果利用这个相关性矩阵，这里作者的方式是将其转化为直方图特征。

why：。图节点之间通常没有自然的排序。同一个图的不同初始节点排序会导致不同的S和vec(s)，To ensure the model is invariant to the graph representations ，可以使用节点排序方案，但是本文使用直方图特征。

#### 相似度计算

所以将 神经张量网络 输出的相似度向量 与 pairwise node comparison 得到的归一化后的直方图特征进行concat，然后经过几层全连接层进行维度压缩得到最终的图相似度

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210714153215006.png" alt="image-20210714153215006" style="zoom:50%;" />

## **不足之处与可以改进的地方**

- 没有利用到边的特征（edge feature）
- 模型的泛化能力
- 策略二无法反向传播，histogram is not continuous differential function。

