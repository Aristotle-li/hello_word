> 题目：Neural Architecture Search: A Survey
>
> 来源：Journal of Machine Learning Research 20 (2019) 1-21
>
> 作者：Thomas Elsken, Jan Hendrik Metzen,Frank Hutter

## NAS(neural architecture search)：                               

搜索策略从预定义的搜索空间  中选择架构A。该架构被传递给性能估计策略，该策略将A的估计性能返回给搜索策略。

## Search Space：

一般会引入先验知识，但是同时引入了偏见，不利于发现超出人类认知的架构。通常大小是exponentialy甚至Unbounded

Search Space：搜索空间定义了NAS方法所能发现的所有架构。

### A:chain-structured neural networks可以参数化：

1、the number of layers

2、the type of operation every layer executes，e.g., pooling,convolutions,depthwise separable convolutions,dilated convolutions

3、hyperparameters associated with the operation，e.g., number of filters, kernel size, strides,number of units for fully-connected-networks.

### B:Cell-based search space is better than searching for whole architecture：

1、the size of the search space is reduced 

2、more easily be transferred or adapted to other data sets

3、repeating building block is useful

优化： accompanied by cell-based search space，a new design-choice arises: how to choose the macro-architecture, Ideally, both the macro-architecture and the micro-architecture should be optimized jointly. 

the search space based on a single cell with fixed macro- architecture, the optimization problem remains (i) non-continuous and (ii) relatively high- dimensional

the hierarchical search space consists of several levels of motifs.

  

改进方向：

common search spaces are also based on predefined building blocks, such as different kinds of convolutions and pooling, but do not allow identifying novel building blocks on this level.

## Search Strategy

说明怎样搜索，exploration-exploitation trade-off

快速找到够好的架构和防止落入次优区域的矛盾

### evolutionary algorithm + SGD:

#### Framework：

use gradient-based methods for optimizing weights and solely use evolutionary algorithms for optimizing the neural architecture itself.

#### Detail：

In the context of NAS, mutations are local operations, such as adding or removing a layer, altering the hyperparameters of a layer, adding skip connections, as well as altering training hyperparameters. Neuro-evolutionary methods differ in how they sample parents, update populations, and generate offsprings.

##### sample parents：

tournament selection、inverse density、 remove the oldest/worst individual or not

##### generate offsprings：

initialize child networks randomly、Lamarckian inheritance

#### discrete search space：

RL、evolutionary algorithm、Bayesian Optimization(derive kernel functions)、tree-based models(tree Parzen estimators、random forests、Monte Carlo Tree Search)、hill climbing algorithm(greedily)

#### continuous relaxation：

gradient-based optimization、

## Performance Estimation Strategy

NAS目标是找到在unseen data上的性能指标好的架构，最简单的方法是跑一遍，但是费时且昂贵且制约了探索更多架构。

 

### lower fidelities：

shorter training times , training on a subset of the data, on lower-resolution images, or with less filters per layer and less cells

这种方法通过打残每个模型让他们在同一起跑线得到更快的评估，那么打残到什么地步才是一个最优的近似呢？

### learning curve extrapolation：

extrapolate initial learning curves and terminate those predicted to perform poorly

Weight Inheritance/ Network Morphisms：network morphisms（它允许在修改架构的同时保持网络所代表的功能不变）

### One-Shot Architecture Search：

treats all architectures as different subgraphs of a supergraph (the one-shot model) and shares weights between architectures that have edges of this supergraph in common. Only the weights of a single one-shot model need to be trained (in one of various ways), and architectures can then be evaluated without any separate training.

#### 不足：

it is currently not well understood which biases they introduce into the search if the sampling distribution of architectures is optimized along with the one-shot model instead of fixing it

 

#### Detail：

Different one-shot NAS methods differ in how the one-shot model is trained：

##### ENAS

##### DARTS：

optimizes all weights of the one-shot model jointly with a continuous relaxation of the search space, obtained by placing a mixture of candidate operations on each edge of the one-shot model

##### SNAS：

Instead of optimizing real-valued weights on the operations as in DARTS, SNAS (optimizes a distribution over the candidate operations.

##### BinaryConnect

the combination of weight sharing and a fixed (carefully chosen) distribution might (perhaps surprisingly) be the only required ingredients for one-shot NAS.

通常采用 One-shot NAS + cell-based search space 

##### meta-learning：

Related to these approaches is meta-learning of hypernetworks that generate weights for novel architectures and thus requires only training the hypernetwork but not the architectures themselves

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210419162110971.png" alt="image-20210419162110971" style="zoom:50%;" /><img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210419162121923.png" alt="image-20210419162121923" style="zoom: 67%;" /> 

Once the one-shot model is trained, its weights are shared across different architectures



Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: Differentiable architecture search. In International Conference on Learning Representations, 2019b.

Sirui Xie, Hehui Zheng, Chunxiao Liu, and Liang Lin. SNAS: stochastic neural architecture search. In International Conference on Learning Representations, 2019.

Han Cai, Ligeng Zhu, and Song Han. ProxylessNAS: Direct neural architecture search on target task and hardware. In International Conference on Learning Representations, 2019.

Richard Shin, Charles Packer, and Dawn Song. Di↵erentiable neural network architecture search. In International Conference on Learning Representations Workshop, 2018.

Karim Ahmed and Lorenzo Torresani. Maskconnect: Connectivity learning by gradient descent. In European Conference on Computer Vision (ECCV), 2018.

Gabriel Bender, Pieter-Jan Kindermans, Barret Zoph, Vijay Vasudevan, and Quoc Le. Understanding and simplifying one-shot architecture search. In International Conference on Machine Learning, 2018.

Andrew Brock, Theodore Lim, James M. Ritchie, and Nick Weston. SMASH: one-shot model architecture search through hypernetworks. In NIPS Workshop on Meta-Learning, 2017.

Chris Zhang, Mengye Ren, and Raquel Urtasun. Graph hypernetworks for neural architecture search. In International Conference on Learning Representations, 2019.

Sirui Xie, Hehui Zheng, Chunxiao Liu, and Liang Lin. SNAS: stochastic neural architecture search. In International Conference on Learning Representations, 2019.

applying NAS to searching for architectures that are more robust to adversarial examples

Ekin D. Cubuk, Barret Zoph, Samuel S. Schoenholz, and Quoc V. Le. Intriguing Properties of Adversarial Examples. In arXiv:1711.02846, November 2017.

cosine an- nealing learning rate schedule

Loshchilov and F. Hutter. Sgdr: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations, 2017.

data augmentation by CutOut

Terrance Devries and Graham W. Taylor. Improved regularization of convolutional neural networks with cutout. arXiv preprint, abs/1708.04552, 2017.

MixUp

Hongyi Zhang, Moustapha Ciss ́e, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint, abs/1710.09412, 2017.

a combination of factors

Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V. Le. Au- toAugment: Learning Augmentation Policies from Data. In arXiv:1805.09501, May 2018.

regularization by Shake-Shake regularization

Xavier Gastaldi. Shake-shake regularization. In International Conference on Learning Representations Workshop, 2017.

regularization by ScheduledDropPath

Zhao Zhong, Zichen Yang, Boyang Deng, Junjie Yan, Wei Wu, Jing Shao, and Cheng- Lin Liu. Blockqnn: Efficient block-wise neural network architecture generation

分步学习：先用人类最优的architecture作为label，学到一些东西后，再进行学习。



基于One-Shot的结构搜索是目前的主流方法，该方法将搜索空间定义为超级网络（supernet），全部网络结构都被包含其中。这个方法最显著的特征就是在一个过参数化的大网络中进行搜索，交替地训练网络权重和模型权重，最终只保留其中一个子结构，上面提到的DARTS和ENAS就是这一类方法的代表。该类方法的本质其实是对网络结构进行排序，然而不同的网络共享同一权重这一做法虽然大大提高搜索效率，却也带来了严重的偏置。显然，不同的神经网络不可能拥有相同的网络参数，在共享权重时，网络输出必定受到特定的激活函数和连接支配。ENAS和DARTS的搜索结果也反应了这一事实，如图6所示，其中ENAS搜索出来的激活函数全是ReLU和tanh，而DARTS搜索出来激活函数的几乎全是ReLU。此外，DARTS等方法在搜索时计算了全部的连接和激活函数，显存占用量很大，这也是它只能搜索较小的细胞结构的原因。

![img](https://pic2.zhimg.com/v2-05456311a59108565396e8d5db1a3a5d_r.jpg)

图6 ENAS（左）和DARTS（右）在PTB上搜索的RNN模型

最近的一些工作着眼于解决共享权重带来的偏置问题和超级图的高显存占用问题，并将新的搜索目标如网络延时、结构稀疏性引入NAS中。商汤研究院提出的随机神经网络结构搜索（SNAS）通过对NAS进行重新建模，从理论上绕过了基于强化学习的方法在完全延迟奖励中收敛速度慢的问题，直接通过梯度优化NAS的目标函数，保证了结果网络的网络参数可以直接使用。旷视研究院提出的Single Path One-Shot NAS与MIT学者提出的ProxylessNAS类似，都是基于One-Shot的方法，与DARTS相比，它们每次只探索一条或者两条网络路径，大大减少了显存消耗，从而可以搜索更大的网络。其中，SNAS将结构权重表示为一个连续且可分解的分布，而ProxylessNAS将二值化连接引入NAS中。这些方法的涌现还标志着NAS正在朝着多任务、多目标的方向前进。

### 词句总结:



1、例句：Deep Learning has enabled remarkable progress over the last years on a variety of tasks, such as

​    提取：@@ has enabled remarkable progress over the last years on a variety of tasks.

​    在过去几年里，@@在多种任务中取得了显著进展。

2、One crucial aspect 一个关键方面 novel 新的perceptual tasks 感知任务

Incorporating prior knowledge about@@  结合@@先验知识

premature convergence 过早收敛

illustrate：说明，阐明

illustrate with cuts/example 附图说明，举例说明。

as illustrated in Figure 2 (left). 如图2（左）所示。

be written as  …被写成…

…corresponds to … …对应

hyperparameters associated with the operation 与操作相关的超参数

Employing such a function results in singnificantly degree of freedom. 利用该函数 

arbitrarily任意地

gives rise to 引起、导致

Proximal Policy Optimization 近端策略优化

REINFORCE policy gradient algorithm 强化策略梯度算法

contemporary neural architectures 当代神经架构

morphisms 态射：表示object之间的关系

3、Currently employed architectures have mostly been developed manually by human experts, which is a time-consuming and error- prone process.费时且容易出错

4、an overview of existing work in this field of research 这一研究领域的当前工作综述

5、The success of @@ in @@ is largely due to. @@在@@的成功很大程度由于@@

6、This success has been accompanied, however, by a rising demand for architecture engineering, where increasingly more complex neural architectures are designed manually. 伴随着

7、We categorize methods for NAS according to three dimensions

我们根据三个维度对NAS方法进行分类

8、从整体说到部分的结构怎么表达，用where，在这里翻译成其中

A chain-structured neural network architecture A can be written as a sequence of n layers, where the i’th layer Li receives its input from layer i-1 and its output serves as the input for layer i + 1  .server as 作为

9、Different layer types are visualized by different colors.

10、hence the parametrization of the search space is not fixed-length but rather a conditional space. 不是… 而是…

11、RL and evolution perform equally well in terms of final test accuracy,

12、achieve state-of-the-art performance on a wide range of problems 

13、However, training each architecture to be evaluated from scratch frequently yields computational demands in the order of thousands of GPU days for NAS

从头开始   产生  大约数千个GPU天

14、which we will now discuss.我们将在下文中进行讨论

15、While these low-fidelity approximations reduce the computational cost, they also introduce bias in the estimate as performance will typically be underestimated.

尽管降低了计算成本，但由于性能通常会被低估，因此也会在估计中引入偏差。

16、allows modifying an architecture while leaving the function represented by the network unchanged允许在修改架构的同时保持网络所代表的功能不变

17、This can be attenuated by employing approximate network morphisms that allow shrinking architectures这可以通过采用近似网络形态来减少体系结构来减弱

18、It is therefore conceivable that 因此可以想象



> 题目：Neural Architecture Search: A Survey
>
> 来源：Journal of Machine Learning Research 20 (2019) 1-21
>
> 作者：Thomas Elsken, Jan Hendrik Metzen,Frank Hutter





<font face="黑体">我是黑体字</font>
<font face="微软雅黑">我是微软雅黑</font>
<font face="STCAIYUN">我是华文彩云</font>
<font color=blue size=5 face="黑体">黑体</font>
<font color=#0ffff size=3>null</font>
<font color=gray size=5>gray</font>





DARTS : 例子：

![image-20210425174429150](/Users/lishuo/Library/Application Support/typora-user-images/image-20210425174429150.png)

实际的去优化一下这个问题，

![image-20210425182310238](/Users/lishuo/Library/Application Support/typora-user-images/image-20210425182310238.png)

![image-20210425182256326](/Users/lishuo/Library/Application Support/typora-user-images/image-20210425182256326.png)