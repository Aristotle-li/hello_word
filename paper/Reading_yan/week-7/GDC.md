> 题目：Diffusion Improves Graph Learning
>
> 来源：NeurIPS 2019
>
> 作者：Johannes Klicpera Stefan Weißenberger   Technical University of Munich

### motivation

一种受频谱方法启发的执行消息传递的新技术：图扩散卷积（GDC）。图扩散卷积（GDC）。 GDC 不是只聚合来自1-hop 邻居的信息，而是聚合来自更大邻域的信息。这个邻域是通过一个新的图来构建的，这个新图是通过对图扩散的广义形式进行稀疏化而生成的。我们展示了图扩散如何

### idea

与DCNN区别：DCNN是将H矩阵concat在一起，取每一个节点的各阶信息,GDC是将H矩阵加在一起，类似于谱分析中的多项式矩阵

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210604194214806.png" alt="image-20210604194214806" style="zoom:50%;" />                 <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210604194244203.png" alt="image-20210604194244203" style="zoom: 25%;" />

广义图扩散：$S=\sum_{k=0}^{∞}\theta_kT^k$

* 用加权系数 $θ_k$ 和广义转移矩阵 $T$ 

* walk transition matrix $T_rw = AD^{−1} $， $T_{rw}$ is column-stochastic

* the symmetric transition matrix $T_{sym} = D^{−1/2}AD^{−1/2}$

* a dense matrix S，S 中的值代表所有节点对之间的影响
* D is the diagonal matrix of node degrees，$$D_{ii} $$= 􏰄$\sum_{j=1}^{N}\ A_{ij}$

扩散卷积 (GDC) 就是将邻接矩阵 A 用稀疏矩阵 S 代替，频谱域不提供任何局部性概念。空间定位允许我们简单地截断 S 的小值并恢复稀疏，得到矩阵 $\hat{S}$。

稀疏化方法：

1、top-k，Use the k entries with the highest mass per column,

2、阈值：将 ε 以下的条目设置为零。

稀疏化仍然需要在预处理期间计算密集矩阵 S。然而，许多流行的图扩散可以在线性时间和空间中有效且准确地近似

 there are fast approximations for both PPR  and the heat kernel , with which GDC achieves a linear runtime O(N)

### 不足：

GDC 基于同质性假设

### 谱分析：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210604172137059.png" alt="image-20210604172137059" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210604172153137.png" alt="image-20210604172153137" style="zoom:50%;" />

文献中 g 的常见选择是多项式J阶滤波器，因为它是局部的并且参数数量有限

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210604172253561.png" alt="image-20210604172253561" style="zoom: 33%;" />

多项式滤波器与广义图扩散的有密切关系：<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210604172627150.png" alt="image-20210604172627150" style="zoom: 33%;" />

