

> 题目：GRAPH ATTENTION NETWORKS
>
> 来源： ICLR 2018
>
> 作者：Yoshua Bengio Petar Velicˇkovic ́∗Guillem Cucurull∗

### motivation

图神经网络在处理非结构化数据有巨大优势，人们提出了基于RNN的方法以及变种，但是人们始终想把卷积引入图网络，而后提出了一系列基于图拉普拉斯矩阵分解的谱域方法，在所有谱方法中，学习滤波器依赖于拉普拉斯特征基，而拉普拉斯特征基依赖于图的结构。因此，在特定结构上训练的模型不能直接应用于具有不同结构的图。

基于以上缺陷，空域的方法提出，直接在空间近邻上操作。这些方法的挑战之一是如何定义一个能处理不同大小邻域并保持CNNs权重共享特性的算子。

注意力机制现今已经成为了处理序列问题的范式，好处：1、它们允许处理大小不一的输入，将注意力集中在输入中最相关的部分来做出决定。当注意机制用于计算单个序列的表示时，通常称为自我注意或内部注意。

### Main idea

注意力机制应用十分广泛，由于节点邻居无序且数目不一致，基于LSTM： feeding randomly-ordered sequences to the LSTM. 和GraphSAGE： samples a fixed-size neighborhood of each node的解决都不尽如人意，受注意力机制启发，注意机制有一个好处，它们允许处理大小不一的输入，将注意力集中在输入中最相关的部分来做出决定。其思想是计算图中每个节点的隐藏表示，通过关注其邻居，遵循自我注意策略。

它的主要创新点在于**利用了注意力机制(Attention Mechanism)来自动的学习和优化节点间的连接关系**，这一作法有以下几个优点：

1. 克服了GCN只适用于直推式学习的缺陷(在训练期间需要测试时的图数据)，可以应用于我们熟悉的归纳式学习任务(在训练期间不需要测试时的图数据)。
2. 使用注意力权重代替了原先非0即1的节点连接关系，即两个节点间的关系可以被优化为连续数值，从而获得更丰富的表达
3. 利用掩蔽的自我注意层对图形结构数据进行操作。在这些网络中使用的图形注意层在计算上是高效的（不需要昂贵的矩阵运算，并且attention值的计算是可以在图形中的所有节点间并行进行），允许（隐式地）在处理不同大小的邻域时将不同的重要性分配给邻域中的不同节点，并且不依赖于预先了解整个图形结构，从而解决了以前基于谱的方法的许多理论问题。

### model structure

##### GRAPH ATTENTION LAYER的输入输出：

作为一层网络，图注意力层的输入为$h=\{h_{1},...,h_{N}\}, h_{i} \in \mathbb{R}^{F}$这里的N是图的节点个数，$h_{i}$表示节点的特征向量，F表示特征维度。图注意力层的输出为${h}'=\{{h}'_{1},...,{h}'_{N}\}, {h}'_{i} \in \mathbb{R}^{{F}'}$


同样的，${F}$表示输出的特征维度。从图注意力层的输入输出可以看出，其本质上也是对特征的一种变换，和其余的网络层功能是类似的。

##### GRAPH ATTENTION LAYER：

首先需要定义一个特征变换矩阵$W \in \mathbb{R}^{F \times {F}'}$ 用于每一个节点从输入到输出的变换。

1. GAT中的attention机制被称为self-attention，记为$f$，其功能如下：

   $e_{ij}=f(Wh_{i},Wh_{j})$

   如图所示，该式表示了self-attention利用节点$i$和节点$j$的特征作为输入计算出了$e_{ij}$, 而$e_{ij}$则表示了节点$j$对于节点$i$的重要性。

2. 这里的节点$j$是节点$i$的First-order近邻，而节点$i$可能是拥有多个近邻的，因此就有了下面的softmaxsoftmax归一化操

   $a_{ij}=softmax(e_{ij})=\frac{exp(e_{ij})}{\sum_{k \in N_{i} }exp(e_{ik})}$$N_{i}$是节点$i$的近邻集合。

3. self-attention机制，也就是我们一开始提到的$a(Wh_{i},Wh_{j})$计算

   $f(Wh_{i},Wh_{j}) = LeakyReLU(a[Wh_{i} || Wh_{j}])$这里的$a \in \mathbb{R}^{2{F}'}$表示需要训练的网络参数，$||$表示的是矩阵拼接操作，LearkyReLu则是一种激活函数，是ReLu的一种改进。

4. 最后给出图感知层的定义，即${h}'_{i}=\sigma(\sum_{j \in N_{i}}a_{ij}Wh_{j})$

上面就是GAT的attention计算方法了，其中会有两个知识点会影响理解

1. self-attention机制为什么可以表示节点间的重要性
2. LearkyReLuLearkyReLu的定义

对于上面这两点，如果知道的话，再结合对GCN的理解，可以很容易的get到GAT的点和含义，如果不清楚的话可能会有点迷糊。

1. attention机制实际上是在有监督的训练下计算两个向量的匹配程度，从而揭示其重要性和影响，
2. LearkyReLuLearkyReLu的定义如下：$y=\left\{\begin{matrix} x & if x >=0 \\ ax & else \end{matrix}\right.$即引入了一个系数a来取消ReLU的死区。

##### 多头ATTENTION机制：

1. 为了稳定self−attention的学习过程，GAT还采用了一种多头机制，即独立的计算K个attention，然后将其获得的特征拼接起来，获得一个更全面的表述，表示如下${h}'_{i}=||^{K}_{k=1} \sigma(\sum_{j \in N_{i}}a^{k}_{ij}W^{k}h_{j})$这里的$ ||$ 表示矩阵拼接的操作，其余的符号和上面描述的一致。
2. 同时，考虑到在网络的最后一层输出层如果还采用这种拼接的方式扩大特征维度，可能不合理，因此，GAT又为输出层定义了平均的操作${h}'_{i}= \sigma(\frac{1}{K}\sum^{K}_{k=1}\sum_{j \in \chi_{i}}a^{k}_{ij}W^{k}h_{j})$
3. 多头attention机制如图所示
   ![在这里插入图片描述](https://www.freesion.com/images/345/ac3da683b229c0c352f05b9c0e5dabb1.png)



# 二、GAN的PYTHON复现

模型的核心代码如下

```python
import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits
```

