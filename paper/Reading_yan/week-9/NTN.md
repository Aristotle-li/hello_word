



> 题目：Reasoning With Neural Tensor Networks for Knowledge Base Completion
>
> 来源：
>
> 作者：Richard Socher∗, Danqi Chen*, Christopher D. Manning, Andrew Y. Ng
>
> Computer Science Department, Stanford University, Stanford, CA 94305, USA

### motivation

WordNet[1]、Yago或Google知识图，they suffer from incompleteness and a lack of reasoning capability.

### Main idea

1、本文目标是能够说明两个实体$（e_1，e_2）$是否处于某种关系R中。例如，关系$（e_1,R,e_2）$=（孟加拉虎，有部分，尾巴）是否真实。Single Layer Model 建模了非线性关系。Bilinear Model 建模了线性关系，本文结合二者提出神经张量网络。

2、以前的工作将实体表示为离散的单元或用单个实体向量表示，本文发现当实体被表示为它们的组成词向量的平均值时，允许描述每个实体的词之间共享统计强度，性能可以得到改善。

我们通过考虑给定知识库子集的实体之间预测额外真实关系的问题来评估模型。



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210715202228525.png" alt="image-20210715202228525" style="zoom:50%;" />



### model structure

神经张量网络（NTN）用一个双线性张量层来代替标准的线性神经网络层，双线性张量层将两个实体向量直接关联到多个维度上。该模型通过以下基于NTN的函数计算两个实体处于特定关系的可能性得分：



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210715205910536.png" alt="image-20210715205910536" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210715224748681.png" alt="image-20210715224748681" style="zoom:50%;" />

它的主要优点是可以将两个输入相乘地联系起来，而不是像标准的神经网络那样只通过非线性隐式地连接实体向量。我们可以看到张量的每个片段负责一种类型的实体对或关系的实例化。另一种解释每个张量切片的方法是，它以不同的方式调解两个实体向量之间的关系。

loss：$e_c$ 是sampled randomly from the set of all entities that can appear at that position in that relation.

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210716103122082.png" alt="image-20210716103122082" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210716103101414.png" alt="image-20210716103101414" style="zoom:50%;" />



### Related model

Distance Model. ：使用特定于关系的映射矩阵将左实体和右实体映射到公共空间，并测量两者之间的L1距离来对关系进行评分。

该模型的主要问题是两个实体向量的参数不相互作用，而是独立地映射到一个公共空间

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210715213032230.png" alt="image-20210715213032230" style="zoom:33%;" />

Single Layer Model.：通过标准单层神经网络的非线性隐式连接实体向量，W 是关系R的评分函数的参数。

虽然这是对距离模型的改进，但非线性只提供了两个实体向量之间的弱相互作用，而代价是更难的优化问题。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210715213055109.png" alt="image-20210715213055109" style="zoom:33%;" />



Hadamard Model.：并通过多个矩阵积和Hadamard积解决了弱实体-向量相互作用的问题，

因为它将每个关系简单地表示为一个向量$e_R$，该向量通过几个线性积与实体向量交互，所有这些线性积都由相同的参数矩阵W参数化。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210715225223373.png" alt="image-20210715225223373" style="zoom:50%;" />

Bilinear Model. ：该模型在表达能力和参数数量上受到词向量的限制。双线性形式只能模拟线性交互作用，不能拟合更复杂的评分函数。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210716102502212.png" alt="image-20210716102502212" style="zoom:50%;" />



### 改进

没有明确处理多义词：和每个词包含多个词向量的想法结合起来，采用注意力机制分配权重。

self-attention计算相关性

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210728185146061.png" alt="image-20210728185146061" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210728195648857.png" alt="image-20210728195648857" style="zoom:50%;" />

