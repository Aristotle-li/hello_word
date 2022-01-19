

> 题目：Deep Convolutional Networks on Graph-Structured Data
>
> 作者：Yann LeCun  Joan Bruna

### motivation

SCNN提出的方法，对于图结构已知时

推广到图形结构未知的环境中。在这种情况下，学习图结构相当于估计相似度矩阵，其复杂性为O（N2）

每个特征图 g，我们需要卷积核通常被限制为具有较小的空间支持，与输入像素的数量 N 无关，这使模型能够学习与 N 无关的多个参数。 为了恢复类似的学习由于频谱域的复杂性，因此有必要将频谱乘法器的类别限制为对应于局部滤波器的那些。



### idea

1、

平滑核：$\mathcal{K}\in \mathbb{R}^{N\times N0} $

$w_g=\mathcal{K}\tilde{w}_g$

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210608165504101.png" alt="image-20210608165504101" style="zoom:50%;" />

2、

使用分层图聚类进行池化

3、

Graph Construction

* Unsupervised Graph Estimation

  $X=R^{N\times F}$ N是节点个数，F是特征维数

  构建图：计算两个特征之间的距离 $d(i,j)=||X_i-X_j||^2$ ，$X_i$是$X$ 的第$i$ 列，

