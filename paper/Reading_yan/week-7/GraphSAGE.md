> 题目：Inductive Representation Learning on Large Graphs
>
> 来源：NeurIPS 2019
>
> 作者：William L. Hamilton

### motivation

现在大多数方法都是直推式学习， 不能直接泛化到未知节点。

这些方法是在一个固定的图上直接学习每个节点embedding，但是大多情况图是会演化的，当网络结构改变以及新节点的出现，直推式学习需要重新训练（复杂度高且可能会导致embedding会偏移），很难落地在需要快速生成未知节点embedding的机器学习系统上。

本文提出归纳学习—GraphSAGE框架，通过训练聚合节点邻居的函数（卷积层），使GCN扩展成归纳学习任务，对未知节点起到泛化作用。

> **直推式(transductive)学习**：从特殊到特殊，仅考虑当前数据。在图中学习目标是学习目标是直接生成当前节点的embedding，例如DeepWalk、LINE，把每个节点embedding作为参数，并通过SGD优化，又如GCN，在训练过程中使用图的拉普拉斯矩阵进行计算，
> **归纳(inductive)学习**：平时所说的一般的机器学习任务，从特殊到一般：目标是在未知数据上也有区分性。

### idea

 GraphSAGE：与基于矩阵分解的嵌入方法不同，

1、我们利用节点特征（例如，文本属性、节点配置文件信息、==节点度数==）来学习可推广到不可见节点的嵌入函数。通过在学习算法中加入节点特征，我们==同时学习每个节点邻域的拓扑结构以及邻域中节点特征的分布==。

2、没有为每个节点训练不同的嵌入向量，而是训练一组==聚合器函数==，这些函数学习==从节点的局部邻域聚合特征信息==。

本文提出GraphSAGE框架的核心是如何聚合节点邻居特征信息，本章先介**绍GraphSAGE前向传播过程**（生成节点embedding），**不同的聚合函数**设定；然后介绍**无监督和有监督的损失函数**以及**参数学习。**

#### 2.GraphSAGE框架

#### 2.1 前向传播

**a. 伪代码:**

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210605220108509.png" alt="image-20210605220108509" style="zoom:50%;" />

4-5行是核心代码，介绍卷积层操作：聚合与节点v相连的邻居（采样）k-1层的embedding，得到第k层邻居聚合特征 ![[公式]](https://www.zhihu.com/equation?tex=h%5Ek_%7BN%28v%29%7D) ，与节点v第k-1层embedding ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bk-1%7D_v) 拼接，并通过全连接层转换，得到节点v在第k层的embedding ![[公式]](https://www.zhihu.com/equation?tex=h%5Ek_v) 。

**b. Neighborhood definition**

将算法 1 扩展到小批量设置，统一采样一个固定大小的邻域集，而不是在算法1中使用完整的邻域集，以保持每个批的计算量是固定的

我们在每次迭代k中抽取不同的均匀样本，

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210605222633636.png" alt="image-20210605222633636" style="zoom: 67%;" />

**c. 可视化例子：**下图是GraphSAGE 生成目标节点（红色）embededing并供下游任务预测的过程：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210605223405927.png" alt="image-20210605223405927" style="zoom:67%;" />

1. 先对邻居随机采样，降低计算复杂度（图中一跳邻居采样数=3，二跳邻居采样数=5）
2. 生成目标节点emebedding：先聚合2跳邻居特征，生成一跳邻居embedding，再聚合一跳邻居embedding，生成目标节点embedding，从而获得二跳邻居信息。（后面具体会讲）。
3. 将embedding作为全连接层的输入，预测目标节点的标签。

#### 2.2 聚合函数

 如何从节点的局部邻域（例如，附近节点的度或文本属性）聚合特征信息：

伪代码第5行可以使用不同聚合函数，本小节介绍五种满足排序不变量的聚合函数：平均、GCN归纳式、LSTM、pooling聚合器。（因为邻居没有顺序，聚合函数需要满足排序不变量的特性，即输入顺序不会影响函数结果）

**a.平均聚合：**先对邻居embedding中每个维度取平均，然后与目标节点embedding拼接后进行非线性转换。
$$
h_{N(v)}^k=mean({\{h^{k-1}_u,u\in N(v)\}})\\
h^k_v=\sigma(W^k\cdot CONCAT(h_v^{k-1},h^k_{N(v)}))
$$
**b. 归纳式聚合：**直接对目标节点和所有邻居emebdding中每个维度取平均（替换伪代码中第5、6行），后再非线性转换：
$$
h^k_v=\sigma (W^k\cdot mean({\{h^{k-1}_v}\}\cup {\{h_u^{k-1},\forall u\in N(v)\}})
$$


**c. LSTM聚合：**LSTM函数不符合“排序不变量”的性质，需要先对邻居随机排序，然后将随机的邻居序列embedding ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Bx_t%2C+t+%5Cin+N%28v%29%5C%7D) 作为LSTM输入。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210605223539926.png" alt="image-20210605223539926" style="zoom:67%;" />

**d. Pooling聚合器:**先对每个邻居节点上一层embedding进行非线性转换（等价单个全连接层，每一维度代表在某方面的表示（如信用情况）），再按维度应用 max/mean pooling，捕获邻居集上在某方面的突出的／综合的表现 以此表示目标节点embedding。
$$
h_{N(V)}^k=max(\{{\sigma(W_{pool}h_ui^k+b)},\forall u_i \in N(v)\})\\
h^k_v=\sigma(W^k\cdot CONCAT(h_v^{k-1},h^k_{N(v)}))
$$
![[公式]](https://www.zhihu.com/equation?tex=h_%7BN%28v%29%7D%5Ek%3Dmax%28%5C%7B%5Csigma%28W_%7Bpool%7Dh_%7Bui%7D%5Ek%2Bb%29%5C%7D%2C%5Cforall+u_i+%5Cin+N%28v%29%29+%5C%5C+h_v%5Ek%3D%5Csigma%28W%5Ek%5Ccdot+CONCAT%28h_v%5E%7Bk-1%7D%2Ch_%7BN%28u%29%7D%5E%7Bk-1%7D%29%29)

#### 2.3 无监督和有监督损失设定

损失函数根据具体应用情况，可以使用**基于图的无监督损失**和**有监督损失**。

**a. 基于图的无监督损失：**希望节点u与“邻居”v的embedding也相似（对应公式第一项），而与“没有交集”的节点 ![[公式]](https://www.zhihu.com/equation?tex=v_n) 不相似（对应公式第二项)。

![img](https://pic2.zhimg.com/80/v2-f74c0b25dc4cb406035659966d0e8691_1440w.png)

- ![[公式]](https://www.zhihu.com/equation?tex=z_u) 为节点u通过GraphSAGE生成的embedding。
- 节点v是节点u固定长度随机游走访达“邻居”。
- ![[公式]](https://www.zhihu.com/equation?tex=v_n+%5Csim+P_n%28u%29) 表示负采样：节点 ![[公式]](https://www.zhihu.com/equation?tex=v_n) 是从节点u的负采样分布 ![[公式]](https://www.zhihu.com/equation?tex=P_n) 采样的，Q为采样样本数，即负样本的数量。
- embedding之间相似度通过向量点积计算得到
- 通过随机梯度下降聚合函数的参数

**b. 有监督损失：**无监督损失函数的设定来学习节点embedding 可以供下游多个任务使用，若仅使用在特定某个任务上，则可以替代上述损失函数符合特定任务目标，如交叉熵。

#### 2.4 参数学习

通过前向传播得到节点u的embedding ![[公式]](https://www.zhihu.com/equation?tex=z_u) ,然后梯度下降（实现使用Adam优化器） **进行反向**传播优化参数 ![[公式]](https://www.zhihu.com/equation?tex=W%5Ek) 和聚合函数内参数。

### 3.实验

#### 3.1 实验目的

1. 比较GraphSAGE 相比baseline 算法的提升效果；
2. 比较GraphSAGE的不同聚合函数。

#### 3.2 数据集及任务

1. Citation 论文引用网络（节点分类）
2. Reddit web论坛 （节点分类）
3. PPI 蛋白质网络 （graph分类）

#### 3.3 比较方法

1. 随机分类器
2. 手工特征（非图特征）
3. deepwalk（图拓扑特征）
4. deepwalk+手工特征
5. GraphSAGE四个变种 ，并无监督生成embedding输入给LR 和 端到端有监督

(分类器均采用LR)

#### 3.4 GraphSAGE 设置

- K=2，聚合两跳内邻居特征
- S1=25，S2=10： 对一跳邻居抽样25个，二跳邻居抽样10个
- RELU 激活单元
- Adam 优化器
- 对每个节点进行步长为5的50次随机游走
- 负采样参考word2vec，按平滑degree进行，对每个节点采样20个。
- 保证公平性，：所有版本都采用相同的minibatch迭代器、损失函数、邻居抽样器。

#### 3.5 运行时间和参数敏感性

1. **计算时间：**下图A中GraphSAGE中LSTM训练速度最慢，但相比DeepWalk，GraphSAGE在预测时间减少100-500倍（因为对于未知节点，DeepWalk要重新进行随机游走以及通过SGD学习embedding）
2. **邻居抽样数量：**下图B中邻居抽样数量递增，边际收益递减（F1），但计算时间也变大。 平衡F1和计算时间，将S1设为25。
3. **聚合K跳内信息**：在GraphSAGE， K=2 相比K=1 有10-15%的提升；但将K设置超过2，边际效果上只有0-5%的提升，但是计算时间却变大了10-100倍。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210605223606409.png" alt="image-20210605223606409" style="zoom:67%;" />

#### 3.6 效果

1. GraphSAGE相比baseline 效果大幅度提升
2. GraphSAGE有监督版本比无监督效果好。
3. LSTM和pool的效果较好

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210605223624344.png" alt="image-20210605223624344" style="zoom:67%;" />

### 改进

聚合函数还是太粗糙，如何既能保持邻居节点的信息，又可以聚合，cnn中是靠卷积实现的。