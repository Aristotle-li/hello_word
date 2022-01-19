## 斯坦福2021版图机器学习课程完结，视频、PPT全放送，大神主讲

在整体课程中，Jure Leskovec团队将其划分为了20个单元，分别是：

图机器学习简介、图机器学习传统方法、节点嵌入、link分析（PageRank）、节点分类（标签传播算法）、图神经网络（GNN模型、空间设计）、图神经网络的应用、图神经网络的原理、知识图嵌入、知识图推理、用GNNs进行频繁的子图挖掘、网络中的社区结构（Community Structure in Networks）、传统的图形生成模型、.图的传统生成模型、图的深度生成模型、GNN进阶、演讲1（计算生物学的GNN）、演讲2（GNN的工业应用）、科学的GNN。





目前，课程视频已经有六节课放到B站上了，每节课10~30分钟不等。更新仍在继续，每周更新两讲。课程 PPT 已经全部放出，在课程主页亦可点击下载。



YouTube地址：

https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn



课程主页：

http://web.stanford.edu/class/cs224w/



B站链接：

https://www.bilibili.com/video/BV1FV411J74L?from=search&seid=1135846809245117647



### **修课必知**



想要修习斯坦福CS224W，必须满足以下条件：

- 基础计算机科学原理知识，能够写出合理的计算机程序；
- 了解基本概率论知识；
- 了解基本线性代数知识。



上课的第一周，教授也会带学习概括复习以下上述必备知识。



 此外，下列书籍可作为扩展阅读书目：

- Graph Representation Learning 作者：William L. Hamilton；
- https://www.cs.mcgill.ca/~wlh/grl_book/
- Networks, Crowds, and Markets: Reasoning About a Highly Connected World 作者：David Easley、Jon Kleinberg
- http://www.cs.cornell.edu/home/kleinber/networks-book/
- Network Science 作者：Albert-László Barabási
- http://networksciencebook.com/





## 图神经网络 | 近期必读的五篇硬核的ICML'21’研究论文

原创 GNN [深度学习与图网络](javascript:void(0);) *今天*

收录于话题

\#图表示最新研究进展41

\#ICML20211







![图片](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw2ibrPJY1icr02tic9Z3DEOhtIyvOicGdXff4ibLR6we17VYnuco1ibgNkzQXz0bU7QLVUC1SoMzvkkOgTg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

据twitter官方账号，ICML 2021今日发布论文接收情况，今年一共有5513篇有效投稿，其中1184篇论文被接收，接收率为21.48% 。另外在这1184篇被接收论文中，有166篇长presentations和1018篇短presentations。

![图片](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw2ibrPJY1icr02tic9Z3DEOhtIlGRcUjsQGpufibLnVsgtjQ5vjibN5SJ629P1rvGPHfvsbtU4kXbu6CKw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

ICML是 International Conference on Machine Learning的缩写，即国际机器学习大会。今年第37届ICML原定于2020年**7月18-24日**在线举行。

从下表可以看出，在投稿量增长523篇（10%增长）的情况下，今年21.4%的接收率仍为**近五年最低。**

![图片](https://mmbiz.qpic.cn/mmbiz_png/cNFA8C0uVPsqzUibYPDVu0SMFEyfrpAwicKO08HicazxHNu6ebdmzDAv2eibHefhibBUJ4WcDr8Z9h7sqnS29UrzvAg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

下面简单学习一下五篇相关的研究论文：

1. Lipschitz Normalization for Self-Attention Layers with Application to Graph Neural Networks
2. Directional Graph Networks
3. Optimization of Graph Neural Networks: Implicit Acceleration by Skip Connections and More Depth
4. GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training
5. Information Obfuscation of Graph Neural Networks

1

自注意力层的Lipschitz归一化及其在图神经网络中的应用



Lipschitz Normalization for Self-Attention Layers with Application to Graph Neural Networks





















































下面简单学习一下五篇相关的研究论文：

1. Lipschitz Normalization for Self-Attention Layers with Application to Graph Neural Networks
2. Directional Graph Networks
3. Optimization of Graph Neural Networks: Implicit Acceleration by Skip Connections and More Depth
4. GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training
5. Information Obfuscation of Graph Neural Networks



1、自注意力层的Lipschitz归一化及其在图神经网络中的应用



Lipschitz Normalization for Self-Attention Layers with Application to Graph Neural Networks



Noah’s Ark Lab, Huawei Technologies France等

George Dasoulas  Kevin Scaman  Aladin Virmaux 

https://arxiv.org/pdf/2103.04886.pdf



基于注意力的神经网络在许多应用中都是最新技术。然而，当层数增加时，它们的性能趋于下降。在这项工作中，我们表明通过标准化注意力得分来加强Lipschitz连续性可以显着改善深度注意力模型的性能。首先，我们表明，对于深图注意力网络（GAT），在训练过程中会出现梯度爆炸，从而导致基于梯度的训练算法的性能较差。为了解决这个问题，我们对注意力模块的Lipschitz连续性进行了理论分析，并介绍了LipschitzNorm，这是一种用于自注意力机制的简单且无参数的归一化方法，可将模型强制为Lipschitz连续性。然后，我们将LipschitzNorm应用于GAT和Graph Transformers，并显示它们在深层设置（10到30层）中的性能得到了显着改善。更具体地说，我们证明，使用LipschitzNorm的深层GAT模型可实现具有长期依赖性的节点标签预测任务的最新结果，同时在基准节点分类任务中显示出与未归类的同类任务相比具有一致的改进。





2、有向图网络Directional Graph Networks

https://arxiv.org/abs/2010.02863

Directional Graph Networks

图神经网络(GNN)中缺乏各向异性核极大地限制了其表达能力，导致了一些众所周知的问题，如过度平滑。**为了克服这个限制，作者提出了第一个全局一致的各向异性核GNN，允许根据拓扑导出的方向流定义图卷积。**首先，通过在图中定义矢量场，提出了一种方法应用方向导数和平滑投影节点特定的信息到场。然后，用拉普拉斯特征向量作为这种向量场。在Weisfeiler-Lehman 1-WL检验方面，证明了该方法可以在n维网格上泛化CNN，并证明比标准的GNN更有分辨力。



![图片](https://mmbiz.qpic.cn/mmbiz_png/ExPPKXgNVtfCIReQKeZ5mmVSkP4xNOPoibH3OiaHFtZJZPcakSpqCvbdfjXahz5gOZJKSFuuKdiaWOjd1TcQwcIFw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



在不同的标准基准上评估了我们的方法，发现在CIFAR10图数据集上相对误差减少了8%，在分子锌数据集上相对误差减少了11%到32%，在MolPCBA数据集上相对精度提高了1.6%。这项工作的重要成果是，它使图网能够以一种无监督的方式嵌入方向，从而能够更好地表示不同物理或生物问题中的各向异性特征。



3、图神经网络的优化



Optimization of Graph Neural Networks: Implicit Acceleration by Skip Connections and More Depth

https://arxiv.org/abs/2105.04550

作者：Keyulu Xu, Mozhi Zhang, Stefanie Jegelka, Kenji Kawaguchi



GNN的表示能力和泛化能力得到了广泛的研究。但是，它们的优化其实研究的很少。通过研究GNN的梯度动力学，我们迈出分析GNN训练的第一步。具体来说，首先，我们分析线性化（linearized）的GNN，并证明了：尽管它的训练不具有凸性，但在我们通过真实图验证的温和假设下，可以保证以线性速率收敛到全局最小值。其次，我们研究什么会影响GNN的训练速度。我们的结果表明，通过跳过（skip）连接，更深的深度和/或良好的标签分布，可以隐式地加速GNN的训练。实验结果证实，我们针对线性GNN的理论结果与非线性GNN的训练行为一致。我们的结果在优化方面为具有跳过连接的GNN的成功提供了第一个理论支持，并表明具有跳过连接的深层GNN在实践中将很有希望。



4、GraphNorm：加快图神经网络训练



GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training

https://arxiv.org/abs/2009.03294

作者：Tianle Cai, Shengjie Luo, Keyulu Xu, Di He, Tie-Yan Liu, Liwei Wang



众所周知，Normalization有助于深度神经网络的优化。不同的体系结构需要专门的规范化方法。在本文中，我们研究什么归一化对图神经网络（GNN）有效。首先，我们调整并评估其他领域到GNN的现有方法。与BatchNorm和LayerNorm相比，InstanceNorm可以实现更快的收敛。我们通过显示InstanceNorm充当GNN的前提条件来提供解释，但是由于图数据集中的大量批处理噪声，BatchNorm的这种预处理效果较弱。其次，我们证明InstanceNorm中的移位操作会导致GNN的表达性下降，从而影响高度规则的图。我们通过建议GraphNorm以可学习的方式解决此问题。根据经验，与使用其他规范化的GNN相比，具有GraphNorm的GNN收敛更快。GraphNorm还改善了GNN的泛化，在图分类基准上实现了更好的性能。



5、图神经网络的信息混淆



Information Obfuscation of Graph Neural Networks

https://arxiv.org/abs/2009.13504

code: https://github.com/liaopeiyuan/GAL



尽管图神经网络（GNN）的出现大大改善了许多应用中的节点和图表示学习，但邻域聚合方案向试图提取有关敏感属性的节点级信息的adversaries 暴露了其他漏洞。在本文中，我们研究了在学习图结构化数据时通过信息混淆保护敏感属性的问题。我们提出了一个框架，可以通过对抗训练以总变化量和Wasserstein距离在本地过滤出预定的敏感属性。我们的方法创建了强大的防御机制来抵御推理攻击，而只在任务性能上造成很小的损失。从理论上讲，我们针对最坏情况的adversaries 分析了我们框架的有效性，并在最大程度地提高预测准确性和最小化信息泄漏之间进行了固有的权衡。来自推荐系统，知识图和量子化学的多个数据集的实验表明，该方法可为各种图结构和任务提供强大的防御能力，同时还能为下游任务提供具有竞争力的GNN编码器.

## 2021年3月\图学习\综述论文，19页pdf概述图信号处理、矩阵分解、随机游走和深度学习算法





图是连接数据网络结构的一种常用表示形式。图数据可以在广泛的应用领域中找到，如社会系统、生态系统、生物网络、知识图谱和信息系统。随着人工智能技术的不断渗透发展，图学习(即对图进行机器学习)越来越受到研究者和实践者的关注。图学习对许多任务都非常有效，如分类，链接预测和匹配。图学习方法通常是利用机器学习算法提取图的相关特征。**在这个综述中，我们提出了一个关于图学习最全面的概述。**特别关注四类现有的图学习方法，包括图信号处理、矩阵分解、随机游走和深度学习。分别回顾了这些类别下的主要模型和算法。我们研究了诸如文本、图像、科学、知识图谱和组合优化等领域的图学习应用。此外，我们还讨论了该领域几个有前景的研究方向。



https://arxiv.org/pdf/2105.00696.pdf

真实的智能系统通常依赖于机器学习算法处理各种类型的数据。尽管图数据无处不在，但由于其固有的复杂性，给机器学习带来了前所未有的挑战。与文本、音频和图像不同，图数据嵌入在一个不规则的领域，使得现有机器学习算法的一些基本操作不适用。许多图学习模型和算法已经被开发出来解决这些挑战。**本文系统地综述了目前最先进的图学习方法及其潜在的应用。**这篇论文有多种用途。首先，它作为不同领域(如社会计算、信息检索、计算机视觉、生物信息学、经济学和电子商务)的研究人员和从业者提供图学习的快速参考。其次，它提供了对该领域的开放研究领域的见解。第三，它的目的是激发新的研究思路和更多的兴趣在图学习。



图，又称网络，可以从现实世界中丰富的实体之间的各种关系中提取。一些常见的图表已经被广泛用于表达不同的关系，如社会网络、生物网络、专利网络、交通网络、引文网络和通信网络[1]-[3]。图通常由两个集合定义，即顶点集和边集。顶点表示图形中的实体，而边表示这些实体之间的关系。由于图学习在数据挖掘、知识发现等领域的广泛应用，引起了人们的广泛关注。由于图利用了顶点[4]，[5]之间的本质和相关关系，在捕获复杂关系方面，图学习方法变得越来越流行。例如，在微博网络中，通过检测信息级联，可以跟踪谣言的传播轨迹。在生物网络中，通过推测蛋白质的相互作用可以发现治疗疑难疾病的新方法。在交通网络中，通过分析不同时间戳[6]的共现现象，可以预测人类的移动模式。对这些网络的有效分析很大程度上取决于网络的表示方式。



一般来说，图学习是指对图进行机器学习。图学习方法将图的特征映射到嵌入空间中具有相同维数的特征向量。图学习模型或算法直接将图数据转换为图学习体系结构的输出，而不将图投影到低维空间。由于深度学习技术可以将图数据编码并表示为向量，所以大多数图学习方法都是基于或从深度学习技术推广而来的。图学习的输出向量在连续空间中。图学习的目标是提取图的期望特征。因此，图的表示可以很容易地用于下游任务，如节点分类和链接预测，而无需显式的嵌入过程。因此，图学习是一种更强大、更有意义的图分析技术。





在这篇综述论文中，我们试图以全面的方式检验图机器学习方法。如图1所示，我们关注现有以下四类方法:基于图信号处理(GSP)的方法、基于矩阵分解的方法、基于随机游走的方法和基于深度学习的方法。大致来说，GSP处理图的采样和恢复，并从数据中学习拓扑结构。矩阵分解可分为图拉普拉斯矩阵分解和顶点接近矩阵分解。基于随机游动的方法包括基于结构的随机游动、基于结构和节点信息的随机游动、异构网络中的随机游动和时变网络中的随机游动。基于深度学习的方法包括图卷积网络、图注意力网络、图自编码器、图生成网络和图时空网络。基本上，这些方法/技术的模型架构是不同的。本文对目前最先进的图学习技术进行了广泛的回顾。



传统上，研究人员采用邻接矩阵来表示一个图，它只能捕捉相邻两个顶点之间的关系。然而，许多复杂和不规则的结构不能被这种简单的表示捕获。当我们分析大规模网络时，传统的方法在计算上是昂贵的，并且很难在现实应用中实现。因此，有效地表示这些网络是解决[4]的首要问题。近年来提出的网络表示学习(NRL)可以学习低维表示[7]-[9]的网络顶点潜在特征。当新的表示被学习后，可以使用以前的机器学习方法来分析图数据，并发现数据中隐藏的关系。



当复杂网络被嵌入到一个潜在的、低维的空间中时，结构信息和顶点属性可以被保留[4]。因此，网络的顶点可以用低维向量表示。在以往的机器学习方法中，这些向量可以看作是输入的特征。图学习方法为新的表示空间中的图分析铺平了道路，许多图分析任务，如链接预测、推荐和分类，都可以有效地解决[10]，[11]。网络的图形化表现方式揭示了社会生活的各个方面，如交流模式、社区结构和信息扩散[12]，[13]。根据顶点、边和子图的属性，可以将图学习任务分为基于顶点、基于边和基于子图三类。图中顶点之间的关系可以用于分类、风险识别、聚类和社区检测[14]。通过判断图中两个顶点之间的边的存在，我们可以进行推荐和知识推理。基于子图[15]的分类，该图可用于聚合物分类、三维可视化分类等。对于GSP，设计合适的图形采样方法以保持原始图形的特征，从而有效地恢复原始图形[16]具有重要意义。在存在不完整数据[17]的情况下，可以使用图恢复方法构造原始图。然后利用图学习从图数据中学习拓扑结构。综上所述，利用图学习可以解决传统的图分析方法[18]难以解决的以下挑战。



