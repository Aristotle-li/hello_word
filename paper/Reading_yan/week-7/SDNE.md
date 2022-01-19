> 题目：Structural Deep Network Embedding
>
> 来源：SIGKDD 2016
>
> 作者：Tsinghua University

### 解决的问题：

底层网络结构复杂，浅层模型无法捕捉高度非线性的网络结构，导致网络表示不理想

 ### idea：

##### 结合深度学习解决graph embeding：

first propose a semi-supervised deep model，exploit the first-order and second-order proximity jointly to preserve the network structure.无监督组件重构二阶邻近度来捕获全局网络结构，一阶邻近度用监督组件中的监督信息以保留本地网络结构。

##### 为什么用二阶邻近度（二阶邻近度是指一对顶点的邻域结构的相似程度）：

现实世界的数据集通常非常稀疏，具有二阶邻近性的顶点对的数目比具有一阶邻近性的顶点对的数目大得多，以至于观察到的链接只占一小部分，存在许多彼此相似但没有任何边连接，但是有很多共同邻居的的顶点。因此，通过引入二阶邻近度，能够表征全局网络结构并缓解稀疏问题

##### 为什么encode-decode结构对应二阶邻近度：

虽然最小化重建损失并没有显示的保留样本之间的相似性，但重建准则可以平滑地捕获数据流形，从而保留样本之间的相似性。如果我们使用邻接矩阵 S 作为自编码器的输入，即 xi = si，由于每个实例 si 表征顶点 vi 的邻域结构，重建过程将使具有相似邻域结构的顶点具有类似的潜在表征。

##### 为什么使用B矩阵：

由于网络的稀疏性，S 中非零元素的数量远远少于零元素的数量。那么如果我们直接使用 S 作为传统自编码器的输入，更容易重构 S 中的零元素，我们对非零元素的重构误差施加了比零元素更多的惩罚。

结果具有相似邻域结构的顶点将被映射到表示空间后距离更近：
$$
\mathcal{L}_{2nd}=\sum_{i=1}^n||(\hat{x_i}-x_i)\odot b_i||_2^2\\
=||(\hat{X}-X)\odot B||_F^2
$$
<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210531163857729.png" alt="image-20210531163857729" style="zoom:50%;" />

##### 一阶近邻度：借用了拉普拉斯特征映射思想，s=0映射到距离远不产生惩罚，s=1映射到距离越远惩罚越大。

与Laplacian Eigenmaps不同，Laplacian Eigenmaps：最小$tr(Y^TLY)$可以对应于寻找$D^{-1/2}LD^{-1/2}$最小的$d$个eig对应的eidvectors，$u_1,u_2...u_d$，取$U^{n*d}=[u_1\ u_2...u_d]$各行即对应$y_1,y_2...y_n$，这里的$y$并不是特征值分解得到的，是通过神经网络学到的特征表示。


$$
\mathcal{L}_{1st}\sum_{i,j=1}^n A_{i,j}||y_i^{(K)}-y_j^{(K)}||_2^2
$$
<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210531164536130.png" alt="image-20210531164536130" style="zoom:50%;" />

##### 总的loss：第三项是防止过度拟合的 L2-norm 正则项

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210531164916762.png" alt="image-20210531164916762" style="zoom:50%;" /><img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210531165152402.png" alt="image-20210531165152402" style="zoom:50%;" />

##### model：

* 整体采用AutoEncoder的思路，由Encoder和Decoder组成
* 输入的$x_i=s_i$，表示结点$i$的邻接特征$ s_i=\{s_{i,j}\}^n_{j=1}$
* 在非监督的部分，通过重建每个节点的邻居结构，来**保留二阶相似性**，提取全局特征，即 $y_i^{(K)}$到 $\hat{x}_i $ 的过程
* 监督部分，通过使相邻节点在表示空间中尽可能相近来实现**保留一阶相似性**，提取局部特征

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210531155705422.png" alt="image-20210531155705422" style="zoom:67%;" />

##### algorithm：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210531165515620.png" alt="image-20210531165515620" style="zoom:50%;" />

### 词句



 1、This is motivated by the recent success of deep learning, which has been demonstrated to have a powerful representation ability to learn complex structures of the data  and has achieved substantial success in dealing with images  text and audio data

这是由最近深度学习的成功所激发的，深度学习已被证明具有强大的表示能力来学习数据的复杂结构，并在处理图像文本和音频数据方面取得了实质性的成功。

2、although minimizing the reconstruction loss does not explicitly preserve the similarity between samples, the recon- struction criterion can smoothly capture the data manifolds and thus preserve the similarity between samples.

尽管最小化重建损失并不能明确保留样本之间的相似性，但是重建准则可以平滑地捕获数据流形，从而保留样本之间的相似性。

3、However, real-world datasets are often so sparse that the observed links only account for a small portion.

然而，现实世界的数据集通常非常稀疏，以至于观察到的链接只占一小部分

4、However, since the underlying network structure is complex, shallow models cannot capture the highly non-linear network structure, resulting in suboptimal network representations

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210531170016781.png" alt="image-20210531170016781" style="zoom:67%;" />

