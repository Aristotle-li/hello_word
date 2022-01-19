> 题目：Graph Embedding Techniques, Applications, and Performance: A Survey
>
> 作者：Palash Goyal and Emilio Ferrara



### challenges：

可伸缩性、维度的选择、要保留的特性

### approaches：

factorization methods, random walks, and deep learning,

### 应用：

a）节点分类，

节点分类旨在基于其他标记的节点和网络拓扑来确定节点（也称为顶点）的标签。

对于节点分类，大致有两类方法：使用随机游走传播标签的方法，以及从节点提取特征并在其上应用分类器的方法

b）链接预测

链接预测是指预测丢失的链接或将来可能发生的链接的任务。链接预测的方法包括基于相似性的方法，最大似然模型和概率模型

c）聚类

聚类用于查找相似节点的子集并将它们分组在一起。聚类方法包括基于属性的模型和直接最大化（或最小化）集群间（或集群内）距离的方法。

d）可视化

可视化有助于提供对网络结构的见解。



通常，为解决基于图的问题而定义的模型：

* 可以在原始图邻接矩阵上

* 在派生的矢量空间上运行。

将embeding作为模型的输入特征，根据训练数据学习参数。这样就无需使用直接应用于图形的复杂分类模型。

但是也有挑战：

 Choice of property：A “good” vector representation of nodes 应该保留图的结构和各个节点之间的连接



 Scalability：很难找到表示的最佳维度。例如，维数越高，重建精度越高，但时间和空间复杂度越高。

### 理解二阶邻接矩阵:

邻接矩阵中：边缘权重sij通常被视为节点vi和vj之间的相似性度量。

一阶邻近：边权重si j也称为节点vi和vj之间的一阶近似，因为它们是两个节点之间相似性的首要度量。反映的是节点自己的邻居的链接情况

二阶邻近：比较两个节点的邻域，如果它们有相似的邻域，则将它们视为相似。反映的是节点的邻居的链接情况

实现原理：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210528191001106.png" alt="image-20210528191001106" style="zoom:50%;" />

根据矩阵乘法，$a*a_{:,j}$ 就是将与第$j$个节点相连的节点对应列累加（因为相连为1），不相连的置零（因为不相连为1），并将$j$列自己链接置零（对角线=0可以起到这个作用）

```
import numpy as np
a = np.array([[0,1,0,0],[1,0,0,1],[0,0,0,1],[0,1,1,0]])
b= np.dot(a,a)
print(b)
print(a)
结果：
a= 
[[0 1 0 0]
 [1 0 0 1]
 [0 0 0 1]
 [0 1 1 0]]
b= 
[[1 0 0 1]
 [0 2 1 0]
 [0 1 1 0]
 [1 0 0 2]]
```



嵌入会将每个节点映射到低维特征向量，并尝试保留顶点之间的连接强度。

嵌入保留一阶近似可通过最小化：<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210528193641880.png" alt="image-20210528193641880" style="zoom:33%;" />获得，

让两个节点对( vi，vj )和( vi，vk )与连接强度相关联，假如Sij>Sik。在这种情况下，vj将被映射到嵌入空间中比vk的映射更接近vi的点。

**2.1 Locally Linear Embedding (LLE)** 

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210528201546174.png" alt="image-20210528201546174" style="zoom: 50%;" />

最小$tr(Y^TLY)$可以对应于寻找$D^{-1/2}LD^{-1/2}$最小的d个eig对应的eidvectors，$u_1,u_2...u_d$，取$U^{n*d}=[u_1\ u_2...u_d]$各行即对应$y_1,y_2...y_n$



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210528204348255.png" alt="image-20210528204348255" style="zoom:50%;" />

使用柯西距离，好处：

拉普拉斯特征图对嵌入之间的距离使用二次惩罚函数，也就是说目标函数更倾向于保留节点间的dissimilarity，可能会使图失去局部的拓扑结构，

我的理解是平方惩罚当节点：

相似时，边权重很大，最小化目标函数，那么$||y_i-y_j||^2$是最小化，所以当相似时，在嵌入空间可以实现距离压缩。

但是不相似时，边权重很小，最小化目标函数，那么$||y_i-y_j||^2$

改用柯西距离后更关注similarity：

相似时，边权重很大，最大化目标函数，那么$\frac{1}{||y_i-y_j||^2+\sigma^2}$是最大化，$||y_i-y_j||^2+\sigma^2$最小化， 所以当相似时，在嵌入空间可以实现距离压缩。

不相似时，边权重很小，最大化目标函数，那么$\frac{1}{||y_i-y_j||^2+\sigma^2}$是最大化

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210528211804967.png" alt="image-20210528211804967" style="zoom:50%;" />



结构保留嵌入（SPE）:

**2.5. Graph Factorization (GF)**

GF对图的邻接矩阵进行因式分解，以最小化以下损失函数：

![img](https://pic1.zhimg.com/80/v2-4f9cb5da38e46728811568d3118ec6d0_1440w.jpg)

其中，λ是一个正则化系数。注意，求和是在观察到的边上，而不是所有可能的边上。这是一个考虑到可伸缩性的近似值，因此可能会在解决方案中引入噪声。注意，由于邻接矩阵通常不是半正定的，即使嵌入的维数为|v|，损失函数的最小值也大于0。

### 3、基于随机游走的方法：适用于大型图，可以获得局部的特征



**3.1. DeepWalk** 

DeepWalk方法受到word2vec的启发，首先选择某一特定点为起始点，做随机游走得到点的序列，然后将这个得到的序列视为句子，用word2vec来学习，得到该点的表示向量。DeepWalk通过随机游走去可以获图中点的局部上下文信息，因此学到的表示向量反映的是该点在图中的局部结构，两个点在图中共有的邻近点（或者高阶邻近点）越多，则对应的两个向量之间的距离就越短。



![img](https://pic2.zhimg.com/80/v2-b07e7401ec18955d247db8295fa5a539_1440w.jpg)

以一个节点为基准，有很多条随机游走的路径，优化其中一条，



优化目标变成了这样：

<img src="https://pic1.zhimg.com/80/v2-001ac943d4bc016c24771dfc3ce3712c_1440w.jpg" alt="img" style="zoom: 50%;" />

其中概率部分的意思是，在一个随机游走中，当给定一个顶点 ![[公式]](https://www.zhihu.com/equation?tex=v_i) 时，出现它的w窗口范围内顶点的概率。

![image-20210529105607966](/Users/lishuo/Library/Application Support/typora-user-images/image-20210529105607966.png)

然后：performs the optimization over sum of log-likelihoods for each random walk.

直观理解：最大化随机行走同时经过root和邻接w个节点的likelihood

损失函数：$\mathcal{L} =\sum_{v_i\in{V}}\sum_{v_{i-w},v_{i+w}\in{N^R}}-log \ pr({v_{i-w},v_{i+w}}|\Phi(v_i))$



**3.2. node2vec**

与DeepWalk相似，node2vec通过最大化随机游走得到的序列中的节点出现的概率来保持节点之间的高阶邻近性。与DeepWalk的最大区别在于，node2vec采用有偏随机游走，在广度优先（bfs）和深度优先（dfs）图搜索之间进行权衡，从而产生比DeepWalk更高质量和更多信息量的嵌入。



![img](https://pic4.zhimg.com/80/v2-a7cb8277b728f351269f3eb1e412f4c3_1440w.jpg)



**3.3. Hierarchical representation learning for networks (HARP)** 

   DeepWalk和node2vec随机初始化节点嵌入以训练模型。由于它们的目标函数是非凸的，这种初始化很可能陷入局部最优。HARP引入了一种策略，通过更好的权重初始化来改进解决方案并避免局部最优。为此，HARP通过使用图形粗化聚合层次结构上一层中的节点来创建节点的层次结构。然后，它生成最粗糙的图的嵌入，并用所学到的嵌入初始化精炼图的节点嵌入（层次结构中的一个）。它通过层次结构传播这种嵌入，以获得原始图形的嵌入。因此，可以将HARP与基于随机行走的方法（如DeepWalk和node2vec）结合使用，以获得更好的优化函数解。

### **4、基于深度学习的方法**

**4.1. Structural deep network embedding (SDNE)** 

   SDNE建议使用深度自动编码器来保持一阶和二阶网络邻近度。它通过联合优化这两个近似值来实现这一点。该方法利用高度非线性函数来获得嵌入。模型由两部分组成：无监督和监督。前者包括一个自动编码器，目的是寻找一个可以重构其邻域的节点的嵌入。后者基于拉普拉斯特征映射，当相似顶点在嵌入空间中彼此映射得很远时，该特征映射会受到惩罚。

![img](https://pic1.zhimg.com/80/v2-59510892dc36cd3bfc7190a9e415b7a4_1440w.jpg)



**4.2. Deep neural networks for learning graph representations (DNGR)**

   DNGR结合了随机游走和深度自动编码器。该模型由3部分组成：随机游走、正点互信息（PPMI）计算和叠加去噪自编码器。在输入图上使用随机游走模型生成概率共现矩阵，类似于HOPE中的相似矩阵。将该矩阵转化为PPMI矩阵，输入到叠加去噪自动编码器中得到嵌入。输入PPMI矩阵保证了自动编码器模型能够捕获更高阶的近似度。此外，使用叠加去噪自动编码器有助于模型在图中存在噪声时的鲁棒性，以及捕获任务（如链路预测和节点分类）所需的底层结构。



**4.3. Graph convolutional networks (GCN)**

   上面讨论的基于深度神经网络的方法，即SDNE和DNGR，以每个节点的全局邻域（一行DNGR的PPMI和SDNE的邻接矩阵）作为输入。对于大型稀疏图来说，这可能是一种计算代价很高且不适用的方法。图卷积网络（GCN）通过在图上定义卷积算子来解决这个问题。该模型迭代地聚合了节点的邻域嵌入，并使用在前一次迭代中获得的嵌入及其嵌入的函数来获得新的嵌入。仅局部邻域的聚合嵌入使其具有可扩展性，并且多次迭代允许学习嵌入一个节点来描述全局邻域。最近几篇论文提出了利用图上的卷积来获得半监督嵌入的方法，这种方法可以通过为每个节点定义唯一的标签来获得无监督嵌入。这些方法在卷积滤波器的构造上各不相同，卷积滤波器可大致分为空间滤波器和谱滤波器。空间滤波器直接作用于原始图和邻接矩阵，而谱滤波器作用于拉普拉斯图的谱。



**4.4. Variational graph auto-encoders (VGAE)** 

  VGAE采用了图形卷积网络（GCN）编码器和内积译码器。输入是邻接矩阵，它们依赖于GCN来学习节点之间的高阶依赖关系。他们的经验表明，与非概率自编码器相比，使用变分自编码器可以提高性能。



### **5、其他**

**LINE**

LINE适用于任意类型的信息网络：无向、有向和无权、有权。该方法优化了精心设计的目标函数，能够保留局部和全局网络结构。此外，LINE中还提出了边缘采样算法，解决了经典随机梯度下降的局限性，提高了算法的有效性和效率。具体来说，LINE明确定义了两个函数，分别用于一阶和二阶近似，并最小化了这两个函数的组合。一阶邻近函数与图分解（GF）相似，都是为了保持嵌入的邻接矩阵和点积接近。区别在于GF通过直接最小化两者的差异来实现这一点。相反，LINE为每对顶点定义了两个联合概率分布，一个使用邻接矩阵，另一个使用嵌入。然后，LINE最小化了这两个分布的Kullback–Leibler（KL）散度。这两个分布和目标函数如下：

![img](https://pic2.zhimg.com/80/v2-3f9b44fc91da1791858dabd17cd35a85_1440w.jpg)

作者用和上面相似的方法定义了二阶近似的概率分布和目标函数：

![img](https://pic2.zhimg.com/80/v2-983d790fd777365806a5edd1fdb25ffd_1440w.jpg)

![img](https://pic3.zhimg.com/80/v2-83ccaeb69d906f5222336def769ace22_1440w.jpg)

为简单起见，将λi设置为顶点i的度数，即λi= di。同样采用KL散度作为距离函数， 用KL散度代替d（·，·）。再省略一些常数，得到：

![img](https://pic2.zhimg.com/80/v2-e59cbe51bc352201c4c7a76a4703f909_1440w.jpg)