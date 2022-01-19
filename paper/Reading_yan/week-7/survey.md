相关工作深入学习图形。最早尝试将神经网络推广到我们已知的图形是由于Scarselli等人[16，31]。这项工作几乎没有被注意到，直到最近才被重新发现[24，37]。在Bruna等人[11,18]的开创性工作之后，计算机视觉和机器学习界对非欧几里德深度学习的兴趣最近激增，在这些工作中，作者在谱域的图上建立了类似CNN的[23]深度神经结构，利用经典傅里叶变换和投影到拉普拉斯算子特征基上的类比[33]。在后续工作中，Defferrard等人[13]提出了一种高效的滤波方案，该方案不需要使用递归切比雪夫多项式显式计算拉普拉斯特征向量。Kipf和Welling[21]进一步简化了这种方法，使用简单的过滤器操作图的1跳邻域。在[3]和[15]中提出了类似的方法。最后，在网络分析领域，受Word2Vec技术[27]的启发，几部著作构建了图嵌入[28，38，12，17，44]方法。对[11，18，13]等谱方法的一个关键批评是卷积的谱定义依赖于傅里叶基（拉普拉斯本征基），而傅里叶基又依赖于域。这意味着在一个图上学习的光谱CNN模型不能简单地转换到另一个具有不同傅里叶基的图上，因为它可以用“不同的语言”来表示。



缺点

1、谱滤波器系数是基相关的，因此，在一个图上学习的谱CNN模型不能应用于另一个图。

2、没有类似以FFT 的快速算法，前向和逆向图傅立叶变换的计算会导致复杂度 O(n2) 矩阵乘法

3、不能保证在频谱域中表示的滤波器在空间域中是局部的；假设使用了拉普拉斯算子的 k = O(n) 个特征向量，光谱卷积层需要 pqk = O(n) 个参数来训练。



深入学习流形。在计算机图形学界，我们可以注意到一种并行的工作，即将深度学习体系结构推广到建模为流形（曲面）的三维形状。Masci等人[26]提出了流形上卷积神经网络的第一个内在版本，将滤波器应用于以测地极坐标表示的局部面片。Boscaini等人[7]使用各向异性热核[1]作为在流形上提取内在面片的替代方法。在[6]中，同一作者使用加窗傅里叶变换形式提出了一种空频域的CNNtype结构[34]。Sinha等人[35]使用几何图像表示来获得可以应用标准CNN的3D形状的欧几里德参数化。空间技术的主要优点是它们可以跨不同的领域推广，这是计算机图形学应用中的一个关键特性（CNN模型可以在一个形状上训练，然后应用到另一个形状上）。然而，尽管各向异性热核等空间结构在流形上有明确的几何解释，但它们在一般图上的解释却有些难以捉摸。





时空图卷积神经网络比较好的帖子：
 1、https://www.cnblogs.com/shyern/p/11262926.html#_label4_0
 2、https://zhoef.com/2019/08/24/14_ST-Gcn/







在每一层节点 ν接收来自所有相邻节点w的消息并根据这些消息更新其嵌入。

第一层之前的节点嵌入通常是从一些给定的节点特征中获得的。在引用图中，论文通过引用连接起来，这些特征通常是每篇论文摘要的词袋向量。



小的本征值对应 密集链接和大的cluster

大的本征值对应 小尺度的结构和震荡



GDC 通常充当低通过滤器。换句话说，GDC 放大了大型且连接良好的cluster，并抑制了与小规模结构相关的信号

这直接解释了为什么 GDC 可以帮助执行节点分类或聚类等任务：它放大了与图形中最占主导地位的结构相关的信号，即（希望）我们感兴趣的少数大型类或群集。

low values correspond to eigenvectors that define tightly connected, large communities





## 摘要

随着计算机和互联网技术的不断发展，深度卷积神经网络彻底改变了很多机器学习任务，从图像分类，视频处理到语音识别，自然语言处理等，但是通常来说，这些任务的数据都是网格化、序列化欧式数据。现实中，很多数据，如交通网络、社交网络、引文网络等是位于非欧空间的图结构。将卷积神经网络迁移到图数据分析处理中的核心在于图卷积算子的构建和图池化算子的构建，本文对图卷积神经网络进行综述，首先介绍两类经典方法：谱域图卷积和空域图卷积。由于图结构中节点周围邻居既没有循序，也没有固定数量，无法定义类似CNN中具有平移不变性的卷积算子，谱方法借助卷积定理在谱域定义图卷积，而空间方法通过在空域定义消息传递函数和聚合函数来实现图卷积。此外，本文介绍了图卷积神经网络的相关应用，包括推荐系统领域、交通预测领域等；最后 本文对图卷积神经网络的发展趋势进行了总结和展望。

关键字：图卷积神经网络；卷积；池化；非欧空间

## 1、引言

​	卷积神经网络(CNN)能够提取出多尺度的局部空间特征，并将它们进行组合来构建高层语义信息。可以发现CNN的核心特点在于：局部连接(local connection)、权重共享(shared weights)和多层叠加(multi-layer)。这些同样在图问题中非常试用，因为图结构是最典型的局部连接结构，其次，共享权重可以减少计算量，另外，多层结构是处理分级模式(hierarchical patterns)的关键。然而，CNN只能在欧几里得数据(Euclidean data)，比如图像和序列数据上进行处理，因为这些领域的数据具有平移不变性。

​	以图像数据为例，一张图片可以表示为欧氏空间的网格，像素点规则散布其中，平移不变性则表示以任意像素点为中心，可以获取相同尺寸的局部结构(包括邻居的排列顺序和数量)，基于此，卷积神经网络通过在输入空间学习在个像素点全局共享的卷积核来建模局部连接，进而为图片学习到意义丰富的隐层表示，从而定义卷积神经网络。而这些具有平移不变性的数据只是图结构的特例而已，对于一般的图结构，可以发现很难将CNN中的卷积核(convolutional filters)和池化操作(pooling operators)迁移到图的操作上。

​	图数据是一种表达能力很强的数据结构，如交通网络和社交网络等。不同于图像和文本数据，图数据中每个节点的局部结构各异，这使得平移不变性不再满足。在深度学习之前，大家对图的研究主要集中在图嵌入(graph embeding)并取得了不错的效果，结合CNN和graph embeding的工作，研究人员设计了图卷积神经网络(GCN)，标准的神经网络比如CNN和RNN不能够适当地处理图结构输入，因为它们都需要节点的特征按照一定的顺序进行排列，但是，对于图结构而言，并没有天然的顺序而言，如果使用顺序来完整地表达图的话，那么就需要将图分解成所有可能的序列，然后对序列进行建模，显然，这种方式非常的冗余以及计算量非常大，与此相反，GCN采用在每个节点上分别传播(propagate)的方式进行学习，由此忽略了节点的顺序，相当于GNN的输出会随着输入的不同而不同。另外，图结构的边表示节点之间的依存关系，然而，传统的神经网络中，依存关系是通过节点特征表达出来的，也就是说，传统的神经网络不是显式地表达中这种依存关系，而是通过不同节点特征在网格的位置来间接地表达节点之间的关系。通常来说，GCN通过邻居节点的加权求和来更新节点的隐藏状态。

​	借助于卷积神经网络对局部结构的建模能力及图上普遍存在的节点依赖关系，图卷积神经网络成为其中最活跃的研究方向之一，近期陆续涌现出一些文章探索图上深度学习再此对其进行综述，本篇文章中整理总结了图卷积神经网络的发展历程及其未来趋势。





### 1、1 现有方法及分类



​	图数据建模的历史由来已久。起初，研究人员关注统计分析的方法，这个时期没有机器学习模型的参与，如网页排序的常用算法PageRank等。此外，研究人员也借用图谱理论的知识，如用拉普拉斯矩阵的特征值和特征向量做社区分析或者人群聚类[12]等。随着深度学习的崛起，研究人员关注如何考虑把深度学习的模型引入到图数据中，代表性的 [13]研究工作是网络嵌入(NetworkEmbeding)，即对图的节点、边或者子图(subgraph)学习得到一个能反映其近邻性低维的向量表示，传统的机器学习方法通常基于人工特征工程来构建特征，在解决具体的问题时，研究人员通常将其建模为两阶段问题，以节点分类为例，第一阶段为每个节点学习统一长度的表达，第二阶段将节点表达作为输入，训练分类模型。但是这种方法受限于灵活性不足、表达能力不足以及工程量过大的问题，图嵌入常见模型有DeepWalk、Node2Vec、SDNE等，然而，这些方法方法有两种严重的缺点，首先就是节点编码中权重未共享，导致权重数量随着节点增多而线性增大**，**另外就是直接嵌入方法缺乏泛化能力，意味着无法处理动态图以及泛化到新的图。

​	近年来，研究人员对图数据建模的关注逐渐转移到如何将深度学习的模型迁移到图数据上，进行端到端的建模，而图卷积神经网络则是其中最活跃的一支.。在建模图卷积神经网络时，研究人员关注如何在图上构建卷积算子。Bruna等人[17]在2013年提出 第一个图卷积神经网络，他们基于图谱理论从卷积定理出发，在谱空间定义图卷积。这一支后来发展为图卷积领域的谱方法。最初的谱方法具有时空复杂度较高的弊端，ChebNet[18]和GCN[19]对谱方法中的卷积核函数进行级数展开，大大降低了时空复杂度。这两个方法虽然被归为谱方法，但已经开始从空间角度定义节点的权重矩阵。

​	在这两个方法的启发下，空间方法应用而生，开始考虑在节点域用注意力机制、序列化模型等建模节点间的权重。如何训练更高效的图卷积神经网络也受到广泛关注。研究人员开始试图训练更深层的图卷积神经网络，以增强模型的泛化能力。同时，模型到大规模图的可扩展性以及训练的速度也是图卷积神经网络中非常重点的研究方向。表1展示了图卷积神经网络的主要方法。

池化算子作为卷积神经网络的主要组成部分， 作用是扩大感受野，降低参数。近期，也有研究开始关注图上池化算子的构建。图上池化算子主要用于图分类问题，目的是学习到图的层级结构。如何建模图上的高阶信息，并分别针对边上带特征的图、异质图的建模也引起了研究人员的兴趣。



| 方法               |                           相关论文                           |
| ------------------ | :----------------------------------------------------------: |
| 图嵌入             |                DeepWalk、LINE、node2vec、SDNE                |
| 谱域图卷积神经网络 |                SCNN、smoothSCNN、chebNet、GCN                |
| 空域图卷积神经网络 | MoNet、DCNN、GDC、MPNNS、NN4G、GIN、GraphSAGE、GAT、GAAN、PATCHY-SAN、LGCN、FastGCN、AS-GCN、StoGCN、Cluster-GCN、DCRNN、STGCN |

### 2 任务 

图数据建模所针对的应用场景非常广泛，这也使得图数据建模所处理的任务多样.我们将下游任务分为节点级别的任务和图级别的任务，节点级别的任务包括节点分类，链接预测等，如引文网络中的文章分类，推荐系统中用户对商品的偏好推断.图级 2 符号定义 别的任务包括图生成，图分类等，如药物网络生成，蛋白质网络中的蛋白质分类.

### 3章节组织

本文第2节给定全文的符号定义；第3节回顾 初期图卷积网络中卷积算子和池化算子的定义，其中卷积算子定义部分又包括了空间方法和谱方法 第4节主要关注图卷积神经网络的一些新进展，包括图特性建模和训练优化两个部分；第5节我们详 细介绍了图卷积神经网络在现有各种典型应用下图 的构造及卷积的定义;在第6节和第7节中，我们给 出图卷积神经网络的未来发展展望，并总结全文.

##    2、符号定义

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210610130739364.png" alt="image-20210610130739364" style="zoom:50%;" />



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210610131323697.png" alt="image-20210610131323697" style="zoom:50%;" />

用$G=\{{V,E,\Lambda}\}$ 表示无向图，其中V表示节点集合，$|V|=n$ 表示图上共有$n$ 个节点，$E$ 表示边集合，$A$ 表示邻接矩阵，定义节点之间的相互连接，且在无向图中$A_{i,j}=A_{j,i}$ 。$L=D-A$ 表示图上的拉普拉斯矩阵，其中$D$ 是对角矩阵，$D_{i，i}$ 表示第$i$ 个节点的度且$D_{i,i}=\sum_j A_{i,j}$  ，归一化后的拉普拉斯矩阵定义为$L= I_n-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ ，其中$I_n\in R^{n\times n}$ 是单位阵。$L$ 是实对称矩阵，对$L$ 做特征值分解得到$L=U\Lambda U^T$ 。其中$U={\{u_i}\}^n_{i=1}$ 表示$n$ 个相互正交的特征向量。$\Lambda = diag(\lambda_1\cdots\lambda_n)$ 是对角阵，$\lambda_i$ 表示$u_i$ 对应的特征值。

$h_v^k$ 表示第k层v节点的特征表示。

## 3、图卷积神经网络

图卷积神经网络主要包括卷积算子和池化算子的构建，其中卷积算子的目的是刻画节点的局部结构，而池化算子的目的是学到网络的层级化表示，降低参数。在解决节点级别的任务时，研究人员更关注如何给每个节点学到更好的表达，此时池化算子并不必要，因此前期大量的工作仅关注图上卷积算子的构建，而池化算子通常用在图级别的任务上。在这 一章节中，我们将详细介绍图上卷积算子和池化算子的构建.

### 31 图卷积算子的构建 

本节介绍关注卷积算子构建的图卷积神经网络方法。现有的图卷积神经网络分为谱方法和空间方法两类，谱方法利用图上卷积定理从谱域定义图卷积，而空间方法从节点域出发，通过定义聚合函数来 聚合每个中心节点和其邻近节点。

#### 3.1.1 图卷积神经网络谱方法

图上平移不变性的缺失，给在节点域定义卷积神经网络带来困难。谱方法利用卷积定理从谱域定义图卷积。我们首先给出卷积定理的背景知识。

(1)图信号处理 

卷积定理:



图上的傅里叶变换写作：
$$
\hat{X}_k= \left< \boldsymbol{X}, \boldsymbol{u_k}\right>
$$
$\boldsymbol{X}=(X_1,…,X_n)$是由节点信息构成的n维向量。做个类似的解释，即特征值$\lambda_k$(频率)下，$\boldsymbol{X}$的Graph傅里叶变换（振幅）等于$\boldsymbol{X}$与$\lambda_k$对应的特征向量$\boldsymbol{u_k}$的内积。

推广到矩阵形式，**图傅里叶变换**：
$$
\hat{\boldsymbol{X}} = \boldsymbol{U}^T \boldsymbol{X}
$$
其中，$\hat{\boldsymbol{X}}=(\hat{X}_1, \hat{X}_2,…, \hat{X}_n)$，即图傅里叶变换，即不同特征值(频率)下对应的振幅构成的向量。$\boldsymbol{X}=(X_1,..,X_n)$是由节点信息构成的n维向量。

迁移到Graph上的逆傅里叶变换推广到矩阵形式，
$$
\boldsymbol{X} = \boldsymbol{U} \boldsymbol{\hat{X}}
$$

个人理解，**Graph傅里叶变换是为了将Graph从Spatial Domain转换到Spectural Domain，使得Graph在Spectural Domain有向量化表示，卷积更方便。这就类比于，传统傅里叶变换为了将Function从Time Domain转换到Frequency Domain，使得Function在Frequency Domain有向量化表示。**

**类比到Graph上并把傅里叶变换的定义带入**，$\boldsymbol{X}=(X_1…,X_n)$为Graph节点信息向量，$h$为卷积核，则$X$和$h$**在Graph上的卷积可按下列步骤求出：**

- $\boldsymbol{X}$的Graph傅里叶变换：$\hat{\boldsymbol{X}} = \boldsymbol{U}^T \boldsymbol{X}$
- 卷积核$\boldsymbol{h}$的Graph傅里叶变换为：$\hat{\boldsymbol{h}}=(\hat{h}_1,…,\hat{h}_n)$，其中，$\hat{h}_k=\langle h, \phi_k \rangle, k=1,2…,n$。实际上，$\hat{\boldsymbol{h}}=\boldsymbol{U}^T \boldsymbol{h}$.
- 求图傅里叶变换向量$\hat{\boldsymbol{X}} \in \mathbb{R}^{N \times 1}$和$\hat{\boldsymbol{h}} \in \mathbb{R}^{N \times 1}$ 的element-wise乘积，等价于将$\hat{\boldsymbol{h}}$组织成对角矩阵的形式，即$diag[\hat{h}(\lambda_k)] \in \mathbb{R}^{N \times N}$，再求$diag[\hat{h}(\lambda_k)] $和$\hat{\boldsymbol{f}}$矩阵乘法（稍微想一下就可以知道）。
- 求上述结果的逆傅里叶变换，即左乘$\boldsymbol{U}$。

则：图上的卷积定义为：
$$
(\boldsymbol{X} * \boldsymbol{h})_\mathcal{G}=\boldsymbol{U} \text{diag}[\hat{h}(\lambda_1),…,\hat{h}(\lambda_n)] \boldsymbol{U}^T \boldsymbol{X} \tag{1}
$$

(2)基于卷积定理的图卷积神经网络

谱卷积神经网络(SpectralCN)[17]是最早提出在图上构建卷积神经网络的方法，该方法利用卷 积定理在每一层定义图卷积算子，在损失函数指导 下通过梯度反向回传学习卷积核，并堆叠多层组成 神经网络.谱卷积神经网络的结构如下

Deep Learning中的**Convolution就是要设计含有trainable**共享参数的kernel。**从公式1看很直观：graph convolution**的参数就是$\text{diag}[\hat{h}(\lambda_1),…,\hat{h}(\lambda_n)]$，也就是说，简单粗暴的将其变成了卷积核 $diag [\theta_1,…,\theta_n]$。这也是为什么我们前面不把卷积核$\boldsymbol{h}$的Graph傅里叶变换表示为$\hat{\boldsymbol{h}}=\boldsymbol{U}^T \boldsymbol{h}$的原因，我们要把$\boldsymbol{h}$变换后的结果$\boldsymbol{\hat{h}}$直接作为参数向量$\boldsymbol{\theta} \rightarrow \boldsymbol{\hat{h}}$进行学习。

可以得到第一代的GCN，([Spectral Networks and Locally Connected Networks on Graphs](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1312.6203))：
$$
\boldsymbol{y}_{output} = \sigma(\boldsymbol{U} \boldsymbol{g}_{\theta} \boldsymbol{U}^T \boldsymbol{X})=\sigma(\boldsymbol{U} \text{diag}[\theta_1,…,\theta_n] \boldsymbol{U}^T \boldsymbol{X})
$$
**$\boldsymbol{X}$就是graph上对应于每个顶点的feature构成的向量$\boldsymbol{X}=(X_1,X_2,…,X_n)$，目前仍然是每个节点信息是标量（即单通道），后续推广到向量很方便（多通道）。** $\boldsymbol{y}_{output}$是该节点卷积后的输出。所有的节点都要经过该操作，得到各自的输出。再$\sigma$激活后，传入下一层。$\boldsymbol{g}_{\boldsymbol{\theta}}=\text{diag}[\theta_1,…,\theta_n]$。相当于拿这个卷积核每个节点卷一遍。

这个问题在于，

- 复杂度太高，需要对拉普拉斯矩阵进行谱分解求$\boldsymbol{U}$，Graph很大时复杂度较高。每次前向传播都要计算矩阵乘积，复杂度$O(n^2)$， $n$为Graph节点数。
- 卷积核的参数为$n$，当Graph很大时，$n$非常大。
- 卷积核的spatial localization不好。（相对于下一代GCN而言）

第二代的GCN，[Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](http://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering)

为了解决上述问题，首先回顾一下，**图傅里叶变换**是关于特征值(频率)的函数$F(\lambda_1), …,F(\lambda_n)$, 即，$F(\boldsymbol{\Lambda})$，因此可以将上述卷积核$\boldsymbol{g}_{\theta}$写作$\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{\Lambda})$。接着，将$\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{\Lambda})$定义成如下**k阶多项式**形式：
$$
\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}}(\boldsymbol{\Lambda}) \approx \sum_{k=0}^K \theta_k^{\prime} \boldsymbol{\Lambda}^k \\
\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}}(\boldsymbol{\Lambda})类比成频域滤波器，假如=e^x，那么级数就是1+x+\frac{x^2}{2}+\frac{x^3}{3}+\cdots,对应高通滤波器
$$
代入可以得到：
$$
\begin{aligned}
\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}} * \boldsymbol{x} &\approx \boldsymbol{\Phi} \sum_{k=0}^K \theta_k^{\prime} \boldsymbol{\Lambda}^k \boldsymbol{\Phi}^T \boldsymbol{x} \\
&= \sum_{k=0}^K \theta_k^{\prime} (\boldsymbol{\Phi} \boldsymbol{\Lambda}^k \boldsymbol{\Phi}^T) \boldsymbol{x} \\
&= \sum_{k=0}^K \theta_k^{\prime} (\boldsymbol{\Phi} \boldsymbol{\Lambda}\boldsymbol{\Phi}^T)^k \boldsymbol{x} \\
& = \sum_{k=0}^K \theta_k^{\prime} \boldsymbol{L}^k \boldsymbol{x}
\end{aligned}
$$
上述推导第三步应用了特征分解的性质。

上式为第二代的GCN。不需要做特征分解了，直接对拉普拉斯矩阵进行变换。可以事先把$\boldsymbol{L}^{k}$计算出来，这样前向传播的时候，就只需要计算矩阵和相邻的乘法。复杂度为$O(Kn^2)$。如果使用稀疏矩阵（$L$比较稀疏）算法，复杂度为$O(k|E|)$.

那么上式是如何体现localization呢？我们知道，矩阵的$k$次方可以用于求连通性，即1个节点经过$k$步能否到达另一个顶点，矩阵$k$次方结果中对应元素非0的话可达，为0不可达。因此$L$矩阵的$k$次方的含义就是代表$\text{k-hop}$之内的节点。进一步，根据拉普拉斯算子的性质。可以证明，如果两个节点的最短路径大于$K$的话，那么$L^{K}$在相应位置的元素值为0。因此，实际上只利用到了节点的K-Localized信息。

另外，作者提到，可以引入切比雪夫展开式来近似$\boldsymbol{L}^k$，因为**任何k次多项式都可以使用切比雪夫展开式来近似**。(类比泰勒展开式对函数进行近似）。

引入切比雪夫多项式（Chebyshev polynomial) $T_k(x)$的$K$阶截断获得对$\boldsymbol{L}^k$的近似，进而获得对$\boldsymbol{g}_{\theta}(\boldsymbol{\Lambda})$的近似，来降低时间复杂度。
$$
\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}}(\boldsymbol{\Lambda}) \approx \sum_{k=0}^K \theta_k^{\prime} T_k(\tilde{\boldsymbol{\Lambda}})
$$
其中，$\tilde{\boldsymbol{\Lambda}}=\frac{2}{\lambda_{max}}\boldsymbol{\Lambda}-\boldsymbol{I}_n$为经图拉普拉斯矩阵$L$的最大特征值（即谱半径）缩放后的特征向量矩阵（防止连乘爆炸）。$\boldsymbol{\theta}^{\prime} \in \mathbb{R}^{K}$表示一个**切比雪夫向量**，$\theta_k^{\prime}$是第$k$维分量。切比雪夫多项式$T_k(x)$使用递归的方式进行定义：$T_k(x)=2xT_{k-1}(x)-T_{k-2}(x)$，其中$T_0(x)=1,T_1(x)=x$。

此时，可以使用近似的$\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}}$替换原来的$\boldsymbol{g}_{\theta}$，可以得到：
$$
\begin{aligned}
\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}} * \boldsymbol{X} &\approx \boldsymbol{U} \sum_{k=0}^K \theta_k^{\prime} T_k(\tilde{\boldsymbol{\Lambda}}) \boldsymbol{U}^T \boldsymbol{x} \
&\approx \sum_{k=0}^K \theta_k^{\prime} (\boldsymbol{U} T_k(\tilde{\boldsymbol{\Lambda}}) \boldsymbol{U}^T) \boldsymbol{X} \\
&=\sum_{k=0}^K \theta_k^{\prime} T_k(\tilde{\boldsymbol{L}}) \boldsymbol{X}
\end{aligned}
$$
其中，$\tilde{\boldsymbol{L}}=\frac{2}{\lambda_{max}} \boldsymbol{L}- \boldsymbol{I}_n$。

因此有，
$$
\boldsymbol{y}_{output} = \sigma(\sum_{k=0}^K \theta_k^{\prime} T_k(\tilde{\boldsymbol{L}}) \boldsymbol{X})
$$
参数向量$\boldsymbol{\theta}^{\prime} \in \mathbb{R}^{k}$，需要通过反向传播学习。时间复杂度也是$O(K|E|)$。


第三代的GCN对上式进一步简化，在图上半监督学习场景下，带标签的数据非常少， 为了避免模型过拟合，Kipf等人约束$θ=θ_0=-θ_1$来 降低模型参数，并对权重矩阵做归一化处理，最终得 到如下的一阶图卷积神经网络

[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

- 取$K=1$，此时模型是1阶的first-order proximity。即每层卷积层只考虑了直接邻域，类似CNN中3*3的卷积核。

- 深度加深，宽度减小。即，若要建立多阶 proximity，只需要使用多个卷积层。

- 并加了参数的一些约束，如: $\lambda_{max}\approx2$，引入renormalization trick，大大简化了模型。

  具体推导，首先$K=1,\lambda_{max}=2$代入，

$$
\begin{aligned}
\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}} * \boldsymbol{x} &\approx \theta_0^{\prime} \boldsymbol{x} + \theta_1^{\prime}(\boldsymbol{L}- \boldsymbol{I}_n) \boldsymbol{x} \\
&= \theta_0^{\prime} \boldsymbol{x} - \theta_1^{\prime}(\boldsymbol{D}^{-1/2} \boldsymbol{A} \boldsymbol{D}^{-1/2}) \boldsymbol{x}
\end{aligned}
$$

上述推导利用了归一化拉普拉斯矩阵$\boldsymbol{L}=\boldsymbol{D}^{-1/2}(\boldsymbol{D}-\boldsymbol{A})\boldsymbol{D}^{-1/2}=\boldsymbol{I_n}-\boldsymbol{D}^{-1/2} \boldsymbol{A} \boldsymbol{D}^{-1/2}$。此时只有两个参数，即每个卷积核只有2个参数，$\boldsymbol{W}$是邻接矩阵。

进一步简化，假设$\theta_0^{\prime}=-\theta_1^{\prime}$，则此时单个通道的单个卷积核参数只有1个$\theta$：
$$
\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}} * \boldsymbol{x} = \theta(\boldsymbol{I_n} + \boldsymbol{D}^{-1/2} \boldsymbol{W} \boldsymbol{D}^{-1/2}) \boldsymbol{x}
$$

$\boldsymbol{I_n} + \boldsymbol{D}^{-1/2} \boldsymbol{W} \boldsymbol{D}^{-1/2}$谱半径$[0,2]$太大，使用renormalization trick（关于这个trick，参考[从 Graph Convolution Networks (GCN) 谈起](https://zhuanlan.zhihu.com/p/60014316)，讲的非常好！），
$$
\boldsymbol{I_n} + \boldsymbol{D}^{-1/2} \boldsymbol{A} \boldsymbol{D}^{-1/2} \rightarrow \tilde{\boldsymbol{D}}^{-1/2}\tilde{\boldsymbol{A}} \tilde{\boldsymbol{D}}^{-1/2}
$$
其中，$\tilde{\boldsymbol{A}}=\boldsymbol{A}+\boldsymbol{I}_n$(相当于加了**self-connection**，本来$\boldsymbol{W}$对角元素为0) , $\tilde{\boldsymbol{D}}_{i,i}=\sum_{j} \boldsymbol{\tilde{A}}_{ij}$。

则：
$$
\underbrace{\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}} * \boldsymbol{x}}_{\mathbb{R}^{n \times 1}} = \theta(\underbrace{\tilde{\boldsymbol{D}}^{-1/2}\tilde{\boldsymbol{A}} \tilde{\boldsymbol{D}}^{-1/2}}_{\mathbb{R}^{n \times n}}) \underbrace{\boldsymbol{x}}_{\mathbb{R}^{n \times 1}}
$$

推广到**多通道**(单个节点的信息是向量，对比图像上3个通道RGB的值构成3维向量)和**多卷积核**(每个卷积核只有1个参数)，即，
$$
\boldsymbol{x} \in \mathbb{R}^{N \times 1} \rightarrow \boldsymbol{X} \in \mathbb{R}^{N \times C}
$$
其中，$N$是节点的**数量**，$C$是通道数，或者称作表示节点的**信息维度数。** $\boldsymbol{X}$是节点的feature矩阵。

相应的卷积核参数变化：
$$
\theta \in \mathbb{R} \rightarrow \boldsymbol{\Theta} \in \mathbb{R}^{C \times F}
$$
其中，$F$为卷积核数量。

则卷积结果写作矩阵形式如下：
$$
\underbrace{\boldsymbol{Z}}_{\mathbb{R}^{N \times F}} = \underbrace{\tilde{\boldsymbol{D}}^{-1/2}\tilde{\boldsymbol{A}} \tilde{\boldsymbol{D}}^{-1/2}}_{\mathbb{R}^{N \times N}} \underbrace{\boldsymbol{X}}_{\mathbb{R}^{N \times C}} \ \ \underbrace{\boldsymbol{\Theta}}_{\mathbb{R}^{C \times F}}
$$
最终得到的卷积结果$\boldsymbol{Z} \in \mathbb{R}^{N \times F}$。即，每个节点的卷积结果的维数等于卷积核数量。

上述操作可以叠加多层，对$\boldsymbol{Z}$激活一下，然后将激活后的$Z$作为下一层的节点的feature矩阵。



## 感悟

> 看到这里感觉十分离谱：一代GCN将图投影到了频率域，也就是说每一个频率代表的不再是具体的邻居，而是总体的链接情况，低频对应大的链接优良的团簇，但是面对海量图要进行奇异值分解计算量太大了，于是人们想出了用级数把频域函数分解，不再正交分解这样做确实降低了计算量，还用了切比雪夫不等式，但是这同样也破坏了从频域成分分析的可能，级数中的一次方代表邻居，二次方代表邻居的邻居，物理意义又变成时域的了，K=1，其实就是对应某种归一化的邻接矩阵的一次方，和x乘起来当然对应x邻居的信息了。
>
> 总结：对频域滤波器级数分解，是开历史的倒车，正交分解提取出的东西，又换还回去了，和扩散图卷积其实是一个东西，挂了个频域的招牌。分别对应归一化拉普拉斯矩阵和随机游走拉普拉斯矩阵



第三代GCN特点总结：

- 复杂度为$O(E)$ (稀疏矩阵优化的话)
- 只考虑1-hop，若要建模多hop，通过叠加层数，获得更大的感受野。（联想NLP中使用卷积操作语句序列时，也是通过叠加多层来达到获取长依赖的目的）。



图热核网络(GraphHeat)[2]从滤波器的角度 对以上谱方法进行分析，指出谱卷积神经网络是非 参滤波器，而切比雪夫网络和一阶图卷积神经网络 都是高通滤波器，但这与图半监督学习的任务中的 平滑性先验不一致，基于此，图热核网络利用热核函 数参数化卷积核，进而实现低通滤波器.图热核网络 的思想如图2所示.

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210610160824632.png" alt="image-20210610160824632" style="zoom:50%;" />





基于个性化PageRank的图卷积神经网络[23] [24] (PNP)和简明一阶图卷积神经网络(SGC)则是对一阶图卷积神经网络方法进行分析，并提出 了一些简化和变体.PNP从深层网络的搭建出发，

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210610162530844.png" alt="image-20210610162530844" style="zoom:50%;" /><img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210610162611322.png" alt="image-20210610162611322" style="zoom:50%;" />





### 3.1.2 图卷积神经网络空间方

上述方法都是从卷积定理出发在谱域定义图卷积,空间方法旨在从节点域出发,通过定义聚合函数来聚合每个中心节点和其邻近节点.上文中的切比雪夫网络和一阶图卷积网络可以看作以拉普拉斯矩阵或其变体作为聚合函数. 在此启发下,近期出现一些工作通过注意力机制或递归神经网络等直接从节点域学习聚合函数,此外,也有一些工作从空间角度定义了图卷积神经网络的通用框架并解释图卷积神



(1)通用框架
通用框架的定义指出图卷积网络的核心问题,同时给已有工作提供一个对比分析的平台. 近期现两篇文章旨在定义图卷积网络的通用框架,其中混合卷积网络(MoNet) [25] 着眼于图上平移不变性的缺失,通过定义映射函数将每个节点的局部结构映射为相同尺寸的向量,进而在映射后的结果上学习共享的卷积核;而消息传播网络(MPNNS)[26]立足于节点之间的信息传播聚合,通过定义聚合函数的通用形式提出框架.

平移不变性的缺失给图卷积神经网络的定义带来因难,混合卷积网络在图上定义坐标系,并将节点之间的关系表示为新坐标系下的一个低维向量. 同时,混合卷积网络定义一簇权重函数,权重函数作用在以一个节点为中心的所有邻近节点上,其输入为节点间的关系表示(一个低维向量),输出为一个标量值.通过这簇权重函数,混合卷积网络为每个节点获得相同尺寸的向量表示:

$D_j(x)f=\sum_{v\ \in N(x)}w_j(u(x,y))f(y),j=1,\cdots ,J$



其中，$N(x)$ 表示$x$ 的邻近节点集合，$f(y)$ 表示节点$y$ 在信号$f$ 上的取值，$u(x,y)$ 指坐标系$u$ 下节点关系的低维向量表示，$w_j$ 表示第$j$ 个权重函数，$J$ 表示权重函数的个数，该公式使得每个节点都得到一个$J$ 维融合了节点局部信息的表示，MoNet正是在此表示上执行卷积操作：

$(f_G^*)(x)=\sum_{j=1}^Jg(j)D_j(x)f$

其中，${\{g(j)\}_{j=1}^J}$ 指卷积核。



### Mixture Model Network (MoNet)

混合模型网络MoNet(Geometric deep learning on graphs and manifolds using mixture model cnns ,CVPR 2017)采用了一种不同的方法来为一个节点的邻居分配不同的权值。它引入节点伪坐标来确定一个节点与其邻居之间的相对位置。一旦知道了两个节点之间的相对位置，权重函数就会将相对位置映射到这两个节点之间的相对权重。通过这种方式，可以在不同的位置共享图数据过滤器的参数。在MoNet框架下，现有的几种流形处理方法，如Geodesic CNN (GCNN)、Anisotropic CNN (各向异性CNN，ACNN)、Spline CNN，以及针对图的处理方法GCN、DCNN等，都可以通过构造非参数权函数将其推广为MoNet的特殊实例。此外，MoNet还提出了一个具有可学习参数的高斯核函数来自适应地学习权值函数。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210616160645195.png" alt="image-20210616160645195" style="zoom:50%;" />

![image-20210616121606685](/Users/lishuo/Library/Application Support/typora-user-images/image-20210616121606685.png)

| Method | Pseudo-coordinates                 | $u(x,y)$                                      | Weight function $w_j(u),j=1,\cdots,j $                       |
| ------ | ---------------------------------- | --------------------------------------------- | ------------------------------------------------------------ |
| CNN    | Local Euclidean                    | $ \mathrm{x}(x,y)=\mathrm x(y)-\mathrm{x}(x)$ | $\delta(u-\bar{u_j})$                                        |
| GCNN   | Local polar geodesic               | $\rho(x,y),\theta(x,y)$                       | $exp(-\frac{1}{2}(u-\bar{u}_j)^T)\begin{pmatrix} \bar{\sigma}_{\rho}^2  &  \\  & \bar{\sigma}_{\theta}^2\\ \end{pmatrix}^{-1}(u-\bar{u}_j)$ |
| ACNN   | Local polar  geodesic              | $\rho(x,y),\theta(x,y)$                       | $exp(-\frac{1}{2}(u^TR_{\bar{\theta}_j})\begin{pmatrix} \bar{\alpha} &  \\  & 1\\ \end{pmatrix}R_{\bar{\theta}_j}^Tu)$ |
| GCN    | Vertex degree                      | $deg(x),deg(y)$                               | $(1-|1-\frac{1}{\sqrt{u_1}}|)(1-|1-\frac{1}{\sqrt{u_2}}|)$   |
| DCNN   | Transition probability in $r$ hops | $p^0(x,y),\cdots ,p^{r-1}(x,y)$               | $id(u_j)$                                                    |





<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210615230036656.png" alt="image-20210615230036656" style="zoom:50%;" /> <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210615230050039.png" alt="image-20210615230050039" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210615230113852.png" alt="image-20210615230113852" style="zoom:50%;" />



##### Neural Network for Graphs (NN4G)

**NN4G是第一个提出的基于空间的ConvGNNs**。NN4G通过直接累加节点的邻域信息来实现图的卷积。它还应用剩余连接和跳跃连接来记忆每一层的信息。因此，NN4G的下一层节点状态为：

$h_v^{k}=\sigma(X_vA^{(k-1)}+\sum_{i=1}^{k-1}\sum_{u\in N(v)}h_v^{k-1}\Theta^{k-1})$ 

可以看出来，形式和GCN类似，主要区别在于NN4G使用了非标准化的邻接矩阵，可能会导致数值不稳定的问题。



### 扩散CNN（DCNN）

 Atwood 和 Towsley [2] 提出了一种不同的空间域方法，他们考虑了图上的扩散（随机游走）过程。图上随机游走的转移概率由$ P =D^{-1}W$ 给出。通过应用不同长度的扩散（功率$P^0,\cdots ,P^{r-1}$）产生不同的特征，

Diffusion Convolutional Neural Network (DCNN)扩散卷积神经网络
DCNN将图卷积看作一个扩散过程。它假设信息以一定的转移概率从一个节点转移到相邻的一个节点，使信息分布在几轮后达到均衡。DCNN将扩散图卷积定义为



对于 Hidden state 0，计算出与所选节点距离为1的特征的和的平均，得到各个节点的特征![H^0](https://math.jianshu.com/math?formula=H%5E0)[![H^0](https://math.jianshu.com/math?formula=H%5E0)是由Hidden state 0所有节点组成的矩阵]

对于 Hidden sate 1，计算出与所选节点距离为2的特征的和的平均，得到各个节点的特征![H^1](https://math.jianshu.com/math?formula=H%5E1)[![H^1](https://math.jianshu.com/math?formula=H%5E1)是由Hidden state 1所有节点组成的矩阵]

以此类推，可以得到多个上述的矩阵

将得到的矩阵按照如图形式表示

$H^{(k)}=\sigma(w^{(k)}\bigodot P^kX)$

概率转移矩阵$P=D^{-1}A \in R^{n\times n}$ ，在DCNN中，隐层表示矩阵$H^{(k)}$ 和输入特征矩阵有相同的维度，不是前一层隐层表示矩阵$H^{(k-1)}$ 的函数，DCNN将$H^{(1)},\cdots ,H^{(k)}$ concat起来作为模型最后的输出。

##### Diffusion Graph Convolution(DGC) 扩散图卷积

由于DCNN扩散过程的平稳分布是概率转移矩阵的幂级数的总和，因此扩散图卷积(Diffusion Graph Convolution, DGC)将每个扩散步骤的输出相加，而不是拼接。它定义扩散图卷积为：

$H=\sum_{k=0}^K\sigma(P^kXW^{(k)})$

利用转移概率矩阵的幂意味着，遥远的邻居对中心节点提供的信息非常少。

### Message Passing Neural Networks (MPNN) 信息传递神经网络

信息传递神经网络(MPNNs)(Neural message passing for quantum chemistry,ICML 2017)概述了基于空间的卷积神经网络的一般框架。它把图卷积看作一个信息传递过程，信息可以沿着边直接从一个节点传递到另一个节点。MPNN运行K-step消息传递迭代，让信息进一步传播。定义消息传递函数(即空间图卷积)为：

$h_v^{(k-1)}=U_k(h_v^{(k-1)},\sum_{u\in N(v)}M_k(h_v^{k-1}，h_u^{(k-1)},e_{u\to v}^{(k-1)}))$ 

$M_k$：the message function

$\sum$：reduce function，可以表示任何函数，不一定是求和

$U_k$：update function，可以学习参数

在得到每个节点的隐含表示后，可以将$h_v^{k}$ 传递给输出层执行节点预测任务，或者传递给readout函数来执行图级别的预测任务。readout函数基于节点隐含表示生成整个图的表示。通常定义为：$h_G=R(h_v^{(K)}|v\in G)$  

### Graph Isomorphism Network (GIN) 图同构网络

然而，图同构网络(GIN)发现，MPNN框架下的方法不能根据生成的图embedding来区分不同的图结构。为了修正这个缺点，GIN通过一个可学习的参数$\epsilon^{(k)}$调整中心节点的权重。它通过下式来计算图卷积

然而，图同构网络(GIN)发现，MPNN框架下的方法不能根据生成的图embedding来区分不同的图结构。为了修正这个缺点，GIN通过一个可学习的参数$ \epsilon^{(k)}$ 调整中心节点的权重。它通过下式来计算图卷积：

$ \mathbf{h}_{v}^{(k)}=\sigma\left(\left(1+\epsilon^{(k)}\right) \mathbf{h}_{v}^{(k-1)}+\sum_{u \in N(v)} \mathbf{h}_{u}^{(k-1)} \mathbf{W}^{(k-1)}\right)$


### GraphSage

由于一个节点的邻居数量可能从1个到1000个甚至更多，因此获取一个节点邻居的完整大小是低效的。GraphSage(Inductive representation learning on large graphs，NIPS 2017)引入聚合函数的概念定义图形卷积。聚合函数本质上是聚合节点的邻域信息，需要满足对节点顺序的排列保持不变，例如均值函数，求和函数，最大值函数都对节点的顺序没有要求。图的卷积运算定义为:



$ \mathbf{h}_{v}^{(k)}=\sigma\left(\mathbf{W}^{(k)} \cdot f_{k}\left(\mathbf{h}_{v}^{(k-1)},\left\{\mathbf{h}_{u}^{(k-1)}, \forall u \in S_{\mathcal{N}(v)}\right\}\right)\right)$
$ \mathbf{h}_{v}^{(0)}=\mathbf{x}_{v}$

$ f_{k}(\cdot)$是一个聚合函数，聚合函数应该不受节点排序(如均值、和或最大值)的影响
$S_{\mathcal{N}(v)}$ 表示节点$v$ 的邻居的一个随机采样



GraphSage没有更新所有节点上的状态，而是提出了一种batch训练算法，提高了大型图的可扩展性。GraphSage的学习过程分为三个步骤。首先，对一个节点的K-跳邻居节点取样，然后，通过聚合其邻居节的信息表示中心节点的最终状态，最后，利用中心节点的最终状态做预测和误差反向传播。如图所示k-hop,从中心节点跳几步到达的顶点。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210610165236161.png" alt="image-20210610165236161" style="zoom:50%;" />

假设在第t-hop取样的邻居个数是$s_t$ ，GraphSage一个batch的时间复杂度是$O(\prod_{t=1}^Ts_t)$。因此随着$t$ 的增加计算量呈指数增加，这限制了GraphSage朝深入的框架发展。但是实践中，作者发现$t=2$ 已经能够获得很高的性能。

### Graph Attention Network (GAT)图注意力网络

图注意网络(GAT)(ICLR 2017,Graph attention networks)假设相邻节点对中心节点的贡献既不像GraphSage一样相同，也不像GCN那样预先确定(这种差异如图4所示)。GAT在聚合节点的邻居信息的时候使用注意力机制确定每个邻居节点对中心节点的重要性，也就是权重。定义图卷积操作如下：

$ \mathbf{h}_{v}^{(k)}=\sigma\left(\sum_{u \in \mathcal{N}(v) \cup v} \alpha_{v u} \mathbf{W}^{(k-1)} \mathbf{h}_{u}^{(k-1)}\right)$
$ \mathbf{h}_{v}^{(0)}=\mathbf{x}_{v}$
$ \alpha_{v u}$表示节点$ v$ 和它的邻居节点$u$ 之间的连接的权重，通过下式计算

$ \alpha_{v u}=\operatorname{softmax}\left(g\left(\mathbf{a}^{T}\left[\mathbf{W}^{(k-1)} \mathbf{h}_{v} \| \mathbf{W}^{(k-1)} \mathbf{h}_{u}\right]\right)\right)$
$ g(·)$是一个LeakReLU激活函数
$ \mathbf{a}$ 是一个可学习的参数向量

softmax函数确保节点$v$ 的所有邻居的注意权值之和为1。GAT进一步使用multi-head注意力方式，并使用concat方式对不同注意力节点进行整合，提高了模型的表达能力。这表明在节点分类任务上比GraphSage有了显著的改进。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210610204412758.png" alt="image-20210610204412758" style="zoom:50%;" />

图4展示了GCN和GAN在聚合邻居节点信息时候的不同。

（a）图卷积网络GCN(2017,Semi-supervised classification with graph convolutional networks)在聚集过程中很清楚地分配了一个非参数的权重$ a_{ij}=\frac{1}{\sqrt{deg(v_i)deg(v_j)}}$a 

（b）图注意力网络GAT(ICLR 2017,Graph attention networks)通过端到端的神经网络结构隐式地捕获$ a_{ij}$的权重，以便更重要的节点获得更大的权重。

### Gated Attention Network (GAAN)-门控注意力网络

GAAN(Gaan:Gated attention networks for learning on large and spatiotemporal graphs,2018)也利用multi-head注意力的方式更新节点的隐层状态。与GAT为各种注意力设置相同的权重进行整合的方式不同，GAAN引入self-attention机制对每一个head，也就是每一种注意力，计算不同的权重，规则如下:

$$h_i^t = \phi_o(x_i\oplus\|_{k=1}^Kg_i^k\sum_{j\in N_i}\alpha_k(h_i^{t-1},h_j^{t-1})\phi_v(h_j^{t-1}))$$

其中，$ \phi_o(\cdot)$和$\phi_v(\cdot)$ 表示前馈神经网络，$g_i^k$ 表示第$k$ 个注意力head的权重。

GeniePath除了在空间上应用了图注意力之外，还提出了一种类似于LSTM的门控机制来控制跨图卷积层的信息流。还有其他一些有趣的图注意模型（Graph classification using structural attention，KDD 18）、（Watch your step: Learning node embeddings via graph attention，NeurIPS,2018）。但是，它们不属于ConvGNN框架。



PATCHY-SAN
另一种不同的工作方式是根据特定的标准对节点的邻居进行排序，并将每个排序与一个可学习的权重关联起来，从而实现跨不同位置的权重共享。PATCHY-SAN(Learning convolutional neural networks for graphs，ICML 2016)根据每个节点的图标签对邻居排序，并选择最上面的q个邻居。图标记实质上是节点得分，可以通过节点度、中心度、Weisfeiler-Lehman颜色等得到。由于每个节点现在有固定数量的有序邻居，因此可以将图形结构的数据转换为网格结构的数据。PATCHY-SAN应用了一个标准的一维卷积滤波器来聚合邻域特征信息，其中该滤波器的权值的顺序对应于一个节点的邻居的顺序。PATCHY-SAN的排序准则只考虑图的结构，这就需要大量的计算来处理数据。GCNs中利用标准CNN能过保持平移不变性，仅依赖于排序函数。因此，节点选择和排序的标准至关重要。PATCHY-SAN中，排序是基于图标记的，但是图标记值考虑了图结构，忽略了节点的特征信息。

Large-scale Graph Convolution Networks (LGCN)
LGCN(Large-scale learnable graph convolutional networks,SIGKDD 2018)提出了一种基于节点特征信息对节点的邻居进行排序的方法。对于每个节点，LGCN集成其邻居节点的特征矩阵，并沿着特征矩阵的每一列进行排序，排序后的特征矩阵的前k行作为目标节点的输入网格数据。最后LGCN对合成输入进行1D-CNN得到目标节点的隐含输入。PATCHY-SAN中得到图标记需要复杂的预处理，但是LGCN不需要，所以更高效。LGCN提出一个子图训练策略以适应于大规模图场景，做法是将采样的小图作为mini-batch。

GraphSAGE
通常GCN这样的训练函数需要将整个图数据和所有节点中间状态保存到内存中。特别是当一个图包含数百万个节点时，针对ConvGNNs的full-batch训练算法受到内存溢出问题的严重影响。为了节省内存，GraphSage提出了一种batch训练算法。它将节点看作一棵树的根节点，然后通过递归地进行K步邻域扩展，扩展时保持采样大小固定不变。对于每一棵采样树，GraphSage通过从下到上的层次聚集隐含节点表示来计算根节点的隐含表示。

FastGCN
FastGCN（fast learning with graph convolutional networks via importance sampling，ICLR 2018）对每个图卷积层采样固定数量的节点，而不是像GraphSage那样对每个节点采样固定数量的邻居。它将图卷积理解为节点embedding函数在概率测度下的积分变换。采用蒙特卡罗近似和方差减少技术来简化训练过程。由于FastGCN对每个层独立地采样节点，层之间的连接可能是稀疏的。

AS-GCN
AS-GCN（Adaptive Sampling Towards Fast Graph Representation Learning，NeurIPS 2018）提出了一种自适应分层采样方法，其中下层的节点根据上层节点为条件进行采样。与FastGCN相比，该方法以采用更复杂的采样方案为代价，获得了更高的精度。

StoGCN（或VR-GCN）
StoGCN（Stochastic training of graph convolutional networks with variance reduction，VR-GCN，ICML 2018）的随机训练使用历史节点表示作为控制变量，将图卷积的感知域大小降低到任意小的范围。即使每个节点有两个邻居，StoGCN的性能也不相上下。但是，StoGCN仍然必须保存所有节点中间状态，这对于大型图数据来说是消耗内存的。

注：VR-GCN是ClusterGCN中的叫法。

Cluster-GCN
Cluster-GCN使用图聚类算法对一个子图进行采样，并对采样的子图中的节点执行图卷积。由于邻域搜索也被限制在采样的子图中，所以Cluster-GCN能够同时处理更大的图和使用更深层次的体系结构，用更少的时间和更少的内存。值得注意的是，Cluster-GCN为现有的ConvGNN训练算法提供了时间复杂度和内存复杂度的直接比较。

复杂度对比
下表是ConvGNN训练算法时间和内存复杂度的对比，GCN是进行full-batch训练的baseline方法。GraphSage以牺牲时间效率为代价来节省内存。同时，随着K KK和r rr的增加，GraphSage的时间复杂度和内存复杂度呈指数增长，其中，Sto-GCN的时间复杂度最高，内存瓶颈仍然没有解决。然而，Sto-GCN可以用非常小的r rr实现令人满意的性能。由于没有引入冗余计算，因此Cluster-GCN的时间复杂度与baseline方法相同。在所有的方法中，Cluster-GCN实现了最低的内存复杂度。

![image-20210610165538171](/Users/lishuo/Library/Application Support/typora-user-images/image-20210610165538171.png)



$n$ 是所有节点的数量
$m$ 是所有边的数量
$K$ 是网络层数
$s$ 是batch size
$r$ 是每一个节点采样的邻居的数量
为简单起见，节点隐含特征的维数保持不变，用$d$ 表示





5、应用

 图卷积神经网络自提出以来，引起了研究人员的浓烈兴趣，主要集中在以下几个领域：网络分析、推荐系统、生物化学、交通预测、计算机视觉、自然语言处理。其应用范围包括计算机科学、信号处理等传统机器学习领域，也包括物理、生物、化学、社会科学等跨学科领域的应用。不同的研究领域具有不同的图数据，节点和边的关系也差异性很大，如何结合各自领域的实际问题恰当的为网络建模是能否落地的关键。

![image-20210617121110065](/Users/lishuo/Library/Application Support/typora-user-images/image-20210617121110065.png)

![image-20210617121017560](/Users/lishuo/Library/Application Support/typora-user-images/image-20210617121017560.png)

![image-20210617120916917](/Users/lishuo/Library/Application Support/typora-user-images/image-20210617120916917.png)

![image-20210617120542559](/Users/lishuo/Library/Application Support/typora-user-images/image-20210617120542559.png)

![image-20210617120207739](/Users/lishuo/Library/Application Support/typora-user-images/image-20210617120207739.png)

![image-20210617120259049](/Users/lishuo/Library/Application Support/typora-user-images/image-20210617120259049.png)

![image-20210617120439898](/Users/lishuo/Library/Application Support/typora-user-images/image-20210617120439898.png)
