# 图像检索技术综述（从SIFT到CNN）

这篇综述比较长，所以分为三个部分。本文是第一个部分，SIFT和CNN的综述

------

## SIFT Meets CNN:A Decade Survey of Instance Retrieval

​    **摘要：**在早期，基于内容的图像检索(CBIR)研究具有全局特征。自2003年以来，基于局部描述符(如SIFT)的图像检索由于SIFT在处理图像变换方面的优势而得到了十多年的广泛研究。最近，基于卷积神经网络(CNN)的图像表示方法引起了越来越多的关注，并显示出令人印象深刻的性能。鉴于这个快速发展的时代，本文对过去十年的实例检索（Instance  Retrieval）进行了全面的调查，提出了基于SIFT的和基于CNN的两大类方法。对于前者，根据码本（codebook）大小，我们将文献组织成使用大型/中型/小型码本。对于后者，我们讨论了三种方法，即使用预先训练、微调的CNN模型，以及混合方法。前两种方法执行图像到网络的单次传递（single-pass），而最后一种方法使用基于补丁的（patch-based）特征提取方案。本调查展示了现代实例检索的里程碑，回顾了在不同类别中广泛使用的以前的工作，并提供了关于SIFT和基于CNN的方法之间的联系的见解。通过对多个数据集上不同类别检索性能的分析和比较，讨论了面向通用和专用实例检索的发展方向。

## **1 Introduction绪论**

​    基于内容的图像检索(CBIR)在20世纪90年代初才真正开始研究。根据图像的纹理、颜色等视觉信号对图像进行检索（visual  cues），提出了多种算法和图像检索系统。一个简单的策略是提取全局描述符（global descriptors  ）。这个想法在20世纪90年代和21世纪初主导了图像检索领域。然而，一个众所周知的问题是全局特征（global  signatures）可能无法满足图像变化的不变性期望（invariance  expectation），比如光照、平移、遮挡和截断（truncation）。这些差异影响了检索的准确性，限制了全局描述符的应用范围。该问题催生了基于局部特征的图像检索方法的出现。

​    本研究的重点是实例级图像检索（instance-level image  retrieval）。在这个任务中，给定一个描述特定对象/场景/体系结构的查询图像，目标是检索包含相同对象/场景/体系结构的图像，这些图像可能在不同视图、光照或遮挡下捕获。实例检索不同于类检索（class retrieval），后者的目的是用查询检索同一个类的图像。接下来，如果没有指定，我们将交替使用图像检索和实例检索。

​    图1展示了过去几年实例检索的里程碑，其中重点介绍了基于SIFT和基于CNN的方法。大多数传统的方法可以认为是在2000年结束的。2003年将词袋(BoW，Bag-of-Words)模型引入图像检索领域，并于2004年应用于基于SIFT描述符的图像分类。从那时起，图像检索社区已经见证了BoW模型在过去十多年中的发展，在此期间提出了许多改进意见。2012年，Krizhevsky等人使用AlexNet在ILSRVC 2012中实现了最先进的识别精度，大大超过了以往的最佳结果。从那时起，研究的重点开始转向基于深度学习的方法，尤其是卷积神经网络(CNN)。

![img](https://img-blog.csdnimg.cn/20190507144021506.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RvbXhpYW9kYWk=,size_16,color_FFFFFF,t_70)

*图1为实例检索的里程碑。在Smeulders等人对2000年前的方法进行调查后，Sivic和Zisserman在2003年提出了视频谷歌（Video  Google），标志着BoW模型的开始。然后，分别由Stewenius、Nister和Philbin等人提出分层k均值和近似k均值，标志着大型码本在检索中的使用。在2008年，Jegou等人提出了汉明嵌入（Hamming  Embedding），这是使用中型代码本的一个里程碑。然后，Perronnin等人和Jegou等人在2010年提出了用于检索的紧凑视觉表示（compact visual  representations）。虽然基于SIFT的方法仍在不断发展，但在Krizhevsky等人的开创性工作之后，基于CNN的方法开始逐渐占据主导地位。2014年，Razavian等人提出了一种从图像中提取多个CNN特征的混合方法。Babenko等人是第一个对CNN模型进行微调以实现通用实例检索的人。两者都采用了来自预先训练好的CNN模型的列特征（column features），并启发了后来最先进的方法。这些里程碑是本次调查中分类方案的代表性工作。*

 

​    基于SIFT的方法大多依赖于BoW模型。BoW最初是为分析文档而提出的，因为文本可以自然地解析为单词。它通过将单词响应累积到一个全局向量中，为文档构建单词直方图。在图像领域，引入尺度不变特征变换(SIFT)使得BoW模型可行。SIFT最初由检测器（detector）和描述符（descriptor）组成，但现在已单独使用，在这个综述中，如果没有指定，SIFT通常指的是128维的描述符，这在社区中是一种常见的实践。通过预先训练的代码本(词汇表)，局部特征被量化为视觉单词（visual words）（译者注：可以理解为BOW中聚类得到的聚类中心）。因此，图像可以以类似于文档的形式表示，并且可以利用经典的权重和索引模式。

​    近年来，基于SIFT模型的流行似乎已经被卷积神经网络(CNN)所取代，在许多视觉任务中，卷积神经网络的层次结构表现得比手工制作的特征更出色。在检索方面，与BoW模型相比，即使使用短的CNN向量也具有更好的效果。基于CNN的检索模型通常计算紧凑表示（compact representations），并采用欧氏距离或近似最近邻(ANN，approximate nearest neighbor  )搜索方法进行检索。现在文献多会直接用预训练的CNN模型或者对于特定的检索任务采用模型微调（fine-tuning）。这些方法中的大多数只向网络提供一次图像来获得描述符。有些是基于补丁（patches），这些补丁被多次传递到网络，类似于SIFT;在这次调查中，我们将它们分为混合方法。

## 2 CATEGORIZATION METHODOLOGY分类方法

​    根据不同的视觉表征，本研究将检索文献分为基于SIFT和基于CNN两大类。基于SIFT的方法进一步分为三类:使用大型、中型或小型代码本。我们注意到代码本的大小与编码方法的选择密切相关。基于CNN的方法分为使用预训练或微调的CNN模型，以及混合方法。表1总结了它们的异同点。

![img](https://img-blog.csdnimg.cn/20190507143951770.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RvbXhpYW9kYWk=,size_16,color_FFFFFF,t_70)

*表1:不同类型的实例检索模型之间的主要区别。对于基于SIFT的方法，提取手工制作的局部不变特征，并根据码本大小使用不同的编码和索引策略。基于CNN的方法主要有预训练、微调的CNN模型和混合方法;利用近似最近邻(ANN)方法生成定长紧凑的向量。*

​    基于SIFT的方法在2012年之前就已经得到了广泛的研究(近年来也出现了一些优秀的研究成果)。这一行方法通常使用一种类型的检测器，例如Hessian仿射，以及一种类型的描述符，例如SIFT。编码将局部特征映射到向量。根据编码过程中使用的码本大小，我们将基于SIFT的方法分为以下三类：

1. 使用小码本。视觉单词（visual words）不到几千字。压缩向量（Compact vectors）是在降维和编码之前生成的
2. 使用中型码本。由于BoW的稀疏性和视觉词的识别能力较低，采用了倒排索引（inverted index）（译者注：inverted  index保存了某个单词与该单词一个文档或者一组文档中的存储位置的映射，它是文档检索系统中最常用的数据结构）和二进制特征。精度和效率之间的权衡是一个主要的影响因素
3. 使用大的码本。考虑到稀疏的BoW直方图和较高的视觉单词识别能力，采用了倒排索引和便于记忆的特征。在码本的生成和编码中使用了近似方法

​    基于CNN的方法利用CNN模型提取特征。通常构建紧凑的(固定长度)表示。有三个类

1. 混合方法。将图像patch多次输入CNN进行特征提取。编码和索引类似于基于SIFT的方法
2. 使用预先训练的CNN模型。使用CNN对一些大型数据集(如ImageNet)进行预训练，一次提取特征。使用了压缩编码/池化技术
3. 使用微调的CNN模型。CNN模型(例如，在ImageNet上进行预训练)在自己的训练集（与目标数据集具有类似的分布）上进行微调。CNN模型可以实现端到端的提取特征。视觉表征显示出更好的辨别能力。

![img](https://img-blog.csdnimg.cn/20190507143850627.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RvbXhpYW9kYWk=,size_16,color_FFFFFF,t_70)

*图2 基于SIFT和CNN的检索模型的通用流程。SIFT是计算从手工制作特征，CNN是利用大量的过滤器或图像补丁提取特征。在这两种方法中，在小码本下，使用编码/池化来生成紧凑的向量。在基于SIFT的方法中，大/中型码本下需要反向索引。CNN的特性也可以使用微调的CNN模型以端到端方式计算。*