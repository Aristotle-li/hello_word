## DARTS(一)



### 引言

NAS 非常耗时，之前有一些加速方案，如 RL、进化算法等，但问题没有得到根本解决。

本文提出 DARTS，将离散的参数变为连续变量，并给予梯度下降进行优化。

主要贡献：

- 可微分的网络架构搜索方法，可应用于 CNN 或 RNN
- DARTS 结果不错
- 效率提升，耗费几个 GPU 天
- 可迁移到其他模型

### 可微架构搜索

### 搜索空间

以 cell 结构搜索为例来说明。同 ENAS 不同，这里的 node 代表表征，边代表操作。如下图，可选的操作有3 种。搜索完成后选择最大可能性的操作，生成最终的结构。

![img](https://pic1.zhimg.com/v2-f1cfe1cc222856a61321f7a25bc63500_r.jpg)

### 连续松弛和优化

原来的操作有就是有，没有就是没有，相当于 0-1 离散选择。DARTS 将其改为连续型的，并基于 softmax 来计算每一种操作的权重。



![img](https://pic3.zhimg.com/v2-4c6bb89d30042de654a5239391a070d2_r.jpg)

最终算法为：



![img](https://pic3.zhimg.com/v2-2cb8e78d7cee6e66932bc2451b19c0be_r.jpg)



### 梯度估计

梯度计算是一个非常耗时的操作，因此这里使用近似方式估计。



![img](https://pic4.zhimg.com/v2-32dfc58ddcc97b32889bbf872001a9cb_r.jpg)



### 生成离散结构

最终结果，每个 node 的取得，都保留了 top-k 个最优的操作。这里 k 可以为 1 或者 2等。



### 思考

DARTS 同 ENAS 相比，效率要低一些，另外会占用较多内存和计算资源。但相比 ENAS，最终得到的模型可能会更好。









![网络搜索之DARTS, GDAS, DenseNAS, P-DARTS, PC-DARTS](https://pic2.zhimg.com/v2-5380f419f330f14b67ba7357768a37ad_1440w.jpg?source=172ae18b)



## DARTS(二)

### 网络搜索之DARTS, GDAS, DenseNAS, P-DARTS, PC-DARTS





最近项目需要，看了下 NAS，这里总结一下，以便后续查阅。

由于小弟不懂啥 RL 也不懂啥进化算法，谷歌爸爸那套是follow不了，故看的都是 gradient based 的方法，先总结下几篇文章都做了什么：

1.**DARTS [1]**：第一个能work的 end2end 基于梯度反传的 NAS 框架，当然你也可选择ENAS（重点是开源了，而且代码写得易懂，后面几个文章都是基于这个做的）。

2.**GDAS [2]**：百度出品，提出了可微的operation sampler，故每次只需优化采样到的部分子图，故特点就是一个字：快 （4 GPU hours）。

3.**DenseNAS [3]**：地平线出品，提出了可以同时搜 block的宽度和空间分辨率的可微分NAS，story讲得还行，实验部分有点虚。

4.**P-DARTS [4]**：华为出品，致力于解决在proxy训练与target测试的模型depth gap问题，参考李飞飞 PNAS 的思路用在block间。

5.**PC-DARTS [5]**：华为出品，针对现有DARTS模型训练时需要 large memory and computing问题，提出了 channel sampling 和 edge normalization的技术，故两个字：更快更好 （0.1 GPU-days）。

它们之间的联系：

无疑 [2], [3], [4], [5] 都是基于 DARTS 来进行拓展；[2] 和 [5] 都是希望加快搜索速度，故采取的sampling的策略和目标也不同；[3] 和 [4] 分别就模型 宽度 和 深度 方面进行拓展。更准确地来说，[3] 是针对移动端来做的，更类似应该是 ProxylessNAS, FBNet 。

\----------------------------------------------------------

**一、DARTS [1]，ICLR2019**

很早提出的文章，不知道为啥才中的ICLR。

DARTS 思想是直接把整个搜索空间看成supernet，学习如何sample出一个最优的subnet。这里存在的问题是子操作选择的过程是离散不可导，故DARTS 将单一操作子选择 松弛软化为 softmax 的所有操作子权值叠加。

下面来看看它主要做了什么：

![img](https://pic1.zhimg.com/80/v2-bd776d9edb2d0b3d8711c6c9e9d082d4_1440w.jpg)DARTS总览

由上图得：

(a) 定义的一个cell单元，可看成有向无环图，里面4个node，node之间的edge代表可能的操作（如：3x3 sep 卷积），初始化时unknown

(b) 把搜索空间连续松弛化，每个edge看成是所有子操作的混合（softmax权值叠加）

(c) 联合优化，更新子操作混合概率上的edge超参（即架构搜索任务）和 架构无关的网络参数 

(d) 优化完毕后，inference 直接取概率最大的子操作即可

故文章的重点就放在了 (b) 和 (c) 部分，分别可以由下面的一条公式和算法描述表示：

![img](https://pic3.zhimg.com/80/v2-69c9217d8e44179855d4d0edcd101946_1440w.png)

Softmax操作，每对nodes ![[公式]](https://www.zhihu.com/equation?tex=%28i%2Cj%29) 的子操作混合权重为 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha%5E%7Bi%2Cj%7D) ，维度为 ![[公式]](https://www.zhihu.com/equation?tex=%7CO%7C) 即node间子操作的总个数； ![[公式]](https://www.zhihu.com/equation?tex=o%28%29+) 表示当前子操作。

![img](https://pic1.zhimg.com/80/v2-e424529beb44f9838c0750f03edb2f64_1440w.jpg)

算法实现上就是轮流更新，在训练集上更新网络参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Comega) ，在验证集上更新网络架构超参 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha)

\--------------------------------------------------------------

**二、GDAS [2]，CVPR2019**

全称叫 Gradient-based search using Differentiable Architecture Sampler (GDAS)，顾名思义，是一种 gradient-based 的方法，且使用了可微采样器，那问题来了，**为啥要采样**呢？

\1. 直接在整个supernet更新整图太耗时了，每次iteration都要更新所有参数，比如DARTS，需要耗时一天。

2.同时优化不同的子操作会导致训练不稳定，因为有些子操作会有相反的值，是竞争关系，直接相加不科学。

故文中就提出了采样器来得到sub-graph，这样每次iteration优化更新只针对sub-graph操作即可。

**Differentiable Sampler：**

那么问题来了，怎么设计这个sampler？如果sampler不可微，怎么进行梯度反传更新呢？

首先对于node ![[公式]](https://www.zhihu.com/equation?tex=%28i%2Cj%29) , 从离散概率分布 ![[公式]](https://www.zhihu.com/equation?tex=T_%7Bi%2Cj%7D) 中采样一个变换函数（即子操作），故在搜索时通过下式计算可得每个node值：

![img](https://pic3.zhimg.com/80/v2-16cc71913882c2455ac55c32d75c336e_1440w.png)

其中 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bi%2Cj%7D) 是从离散概率分布 ![[公式]](https://www.zhihu.com/equation?tex=T_%7Bi%2Cj%7D) 采样得到， ![[公式]](https://www.zhihu.com/equation?tex=W_%7Bf_%7Bi%2Cj%7D%7D) 为其参数。离散概率分布可通过下述的可学习概率质量函数表示：

![img](https://pic1.zhimg.com/80/v2-4f639d6f6464c0ea702ab6737dc6be74_1440w.jpg)

其中 ![[公式]](https://www.zhihu.com/equation?tex=A_%7Bi%2Cj%7D%5E%7Bk%7D) 是 K 维可学习 vector ![[公式]](https://www.zhihu.com/equation?tex=A_%7Bi%2Cj%7D%5Cin+R%5E%7BK%7D) 中的第 k 个元素。

由于上述操作不可导，故作者使用 Gumbel-Max trick 来将公式(3)重写为：

![img](https://pic1.zhimg.com/80/v2-47332e5045db77571f9a64be09ff167c_1440w.jpg)

将 arg max 松弛软化为softmax形式，就可导了：

![img](https://pic1.zhimg.com/80/v2-3d2724bb827710f200cd34341ebda33c_1440w.png)

**定制化Reduction cell：**

为了减少搜索空间的大小和减缓联合搜 normal & reduction cell 的优化难度，文中根据专家经验自定义了 reduction cell：

![img](https://pic1.zhimg.com/80/v2-576c23ffbda7f4421b5b7d2a787902b8_1440w.jpg)reduction cell

**实验结果：**

来看看GDAS与DARTS在CIFAR和ImageNet的性能对比：

1.识别率基本与DARTS持平的情况下，搜索时间比它快5倍以上。

2.GDAS(FRC) 是基于上图定制化的reduction cell 来只搜normal cell的，性能更好，参数更少，搜索时间更短。

![img](https://pic1.zhimg.com/80/v2-f3a7c8bc2c343e68183515221d19db74_1440w.jpg)Search on CIFAR10, test on CIFAR10 &amp;amp;amp;amp;amp;amp; CIFAR100

![img](https://pic2.zhimg.com/80/v2-7f925d4fe082e3732323ca1078907765_1440w.jpg)Search on CIFAR10, test on ImageNet

文中3.2节有一段值得思考的话：

其实文章初衷是设计跑得快的NAS，估计是想用于直接跑target大数据集，比如ImageNet，不过最后发现搜出来的性能堪忧。哈哈，可以思考下why。（花絮，后面PC-DARTS里有句话解释了这个现象）

\-----------------------------------------------------------------

**三、DenseNAS [3]，2019**

由于NAS一般需要很强的专家经验来设计中间block的channel数以及何时进行downsampling，故人工设定的可能不是最优的，DenseNAS做的就是让网络通过搜索自行决定渐进的block的宽度及何时downsampling。

如何实现这个功能呢？如下图所示，定义一个密集连接的搜索空间，里面block间的宽度渐渐增加，stride慢慢下降，而每个block有一定概率与同分辨率及stride=2于当前的block进行连接（如灰色连接线所示）；由于这一套是基于stack mobile convolution block，故中间不设skip-connect操作， 最后训练完毕后，按照特定算法选择路径（红色实线）即可。

![img](https://pic1.zhimg.com/80/v2-8acb2646127b81400259783261b6c5b8_1440w.jpg)密集连接的搜索空间

**那么问题来了**：

1.不同block的宽度和分辨率可能不同，怎么进行连接？

2.block之间多了这么多path，怎么进行端到端梯度反向传播？

**文中给出的方案是**：

1.如下图所示，Block模块里，针对每个输入的previous block，都有各自的head layers，功能是转化为相同宽度和分辨率的tensor，这样才能权值叠加

2.在layer-level，跟DARTS一样，node之间该怎么反传就怎么反传；在Block-level， 也仿照layer-level 来松弛软化使用softmax来当成多个head layers后的权值叠加。

![img](https://pic3.zhimg.com/80/v2-90d2e7b30f09b4bd237fcdae2962ebd2_1440w.jpg)Layer, Block, Network 

最后当搜索完成，layer-level直接选择最大概率的子操作即可；对于 Network-level，使用Viterbi algorithm 来选择最高转移概率的block进行连接。

由于本文是做移动端搜索，故latency也作为优化的目标之一：

![img](https://pic3.zhimg.com/80/v2-9b2331fa87df5d7cf9d572c9ac84db3e_1440w.png)

**实验结果：**

只在ImageNet搜测了实验结果，感觉亮点不够突出呀。

![img](https://pic3.zhimg.com/80/v2-8c4db31e0a56f87a9b8ed04668162aa2_1440w.jpg)

\-----------------------------------------------------------------

**四、P-DARTS [4]，2019**

由于NAS受限于memory和计算消耗，一般都会在proxy集进行较浅的initial channel及layer depth搜索，然后把搜好的模型再扩充较大的channel和depth放到target集上重训重测。

那么问题来了：**怎么优化这个 depth gap 问题**？

![img](https://pic2.zhimg.com/80/v2-23c437099332f270356075fa893a87a9_1440w.jpg)

如上图所示，在DARTS中，搜索时候是以 8 cells with 50 epochs 来进行的，而evaluate时却以 20 cells，这bias造成了精度大幅度下降；而 P-DARTS 以渐进的方式 5 cells, 11 cells, 17 cells 分别 25 epochs 来进行，这样更能接近evaluate时的情况，故性能也更好。

OK，你可能会问，为什么不直接以20 cells 来进行搜索呢？好问题，理论上应该是可行的，就是太耗memory且容易网络架构过拟合；那17 cells也很深，memory够吗？这也是好问题，P-DARTS其实就是在解决这个问题：

![img](https://pic4.zhimg.com/80/v2-327a112a6594ba9ebdaa972f50d57a7b_1440w.jpg)P-DARTS pipeline

如上图所示，

(a) cells=5时，每个node间有5个candidate，当训练好了25 epochs后，会有对应的softmax置信度。

(b) 接着进行 cells=11的搜索，虽然深度加了一倍多，但这时每个node间operation candidate将会减少接近一半，即把(a)中最后置信度较低的operation直接delete掉。

(c) 同样的流程，最后进行 cells=17的搜索，再砍掉置信度低的一半opeartion。通过这样的方式来tradeoff depth及memory。

\------------------------------------

文中还探讨了另外一个问题，就是搜索空间里skip-connect的影响，从实际效果来看，过多的skip-connect（无可学习参数）会使其表征能力下降，测试时性能欠佳；但往往因为搜索时 skip-connect收敛得更快，故如何设计基于**skip-connect的约束**项呢？

\1. 在每个skip-connect操作后插入operation-level dropout，然后训练时逐渐减少Dropout rate。

2.加个超参 M=2 来限制最后cell内部 skip-connect的最大总数。

**实验结果：**

来看看在CIFAT上的实验结果，明显在搜索时间和性能上都碾压DARTS了。

不过有趣的是，最后两行(large) 是指将 initial channel 从 36 增加到 64，性能增加；故要不要搞个progressive channel 版本的搜索呢？哈哈

![img](https://pic1.zhimg.com/80/v2-30570221daaf248657e5052ddbc83d98_1440w.jpg)

最后我们来看看三个stage的memory消耗情况，尽管已经17 cells了，一个P100还OK：

![img](https://pic4.zhimg.com/80/v2-f8f9bbcc5adfaf8403d52f0c3c7199e3_1440w.jpg)

\-----------------------------------------------------------------

**五、PC-DARTS [5]，2019**

接着上面的P-DARTS来看，尽管上面可以在17 cells情况下单卡完成搜索，但妥协牺牲的是operation的数量，这明显不是个优秀的方案，故此文 Partially-Connected DARTS，致力于大规模节省计算量和memory，从而进行快速且大batchsize的搜索。

**PC-DARTS的贡献有两点**：

1.设计了基于channel的sampling机制，故每次只有小部分1/K channel的node来进行operation search，减少了(K-1)/K 的memory，故batchsize可增大为K倍。

2.为了解决上述channel 采样导致的不稳定性，提出了 edge normalization，在搜索时通过学习edge-level 超参来减少不确定性。（这跟DenseNAS中的head layer权值叠加有点像）

![img](https://pic3.zhimg.com/80/v2-d4e54706d8c424d9d346d1b0a343bb0a_1440w.jpg)PC-DARTS

**A、部分通道连接**：

如上图的上半部分，在所有的通道数K里随机采样 1/K 出来，进行 operation search，然后operation 混合后的结果与剩下的 (K-1)/K 通道数进行 concat，公式表示如下：

![img](https://pic4.zhimg.com/80/v2-6da2059a6332898796fbc9f018fbd61b_1440w.png)

**B、边缘正规化**：

上述的“部分通道连接”操作会带来一些正负作用：

**正作用**：能减少operations选择时的biases，弱化无参的子操作（Pooling, Skip-connect）的作用。文中3.3节有这么一句话：当proxy dataset非常难时（即ImageNet），往往一开始都会累积很大权重在weight-free operation，故制约了其在ImageNet上直接搜索的性能。

所以可以解释为啥GDAS直接在ImageNet搜效果不行，看回用GDAS在CIFAR10搜出来的normal cell，确实很多是Skip-connect，恐怕在ImageNet上搜，都是skip-connect。。。

**副作用**：由于网络架构在不同iterations优化是基于随机采样的channels，故最优的edge连通性将会不稳定。

为了克服这个副作用，提出边缘正规化（见上图的下半部分），即把多个PC后的node输入softmax权值叠加，类attention机制：

![img](https://pic3.zhimg.com/80/v2-1e12bec34e0f4ca87b0537464ffda6d2_1440w.png)

由于 edge 超参 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_%7Bi%2Cj%7D) 在训练阶段是共享的，故学习到的网络更少依赖于不同iterations间的采样到的channels，使得网络搜索过程更稳定。当网络搜索完毕，node间的operation选择由operation-level和edge-level的参数相乘后共同决定。

**实验结果**：

来看看直接在ImageNet搜出来的结果，只需要 3.8 GPU-days，有点牛逼。

不过对比下，P-DARTS（CIFAR10）的也牛逼，性能和搜索时间很好的tradeoff。

![img](https://pic4.zhimg.com/80/v2-9fc025f91895d0f2aab256bc71ab5e2b_1440w.jpg)

最后的最后，来看个有趣的消融实验：

直接看第二行，加了EN后，普通的DARTS性能也能提升；也就是说EN是可以用到所有的DARTS框架中，这个很不错。

![img](https://pic2.zhimg.com/80/v2-df9737e7c116ac2d58a05dccff466701_1440w.jpg)

\----------------------------------------------------

**总结与展望：**

小memory，少人工定义的参数(initial channel, layer数啥的就很恶心)，更任务相关的search space，架构跨数据集泛化能力。

在各个领域用起来，才能有各种新坑，和各种新的改进。

**Reference:**

[1] Hanxiao Liu et al., DARTS: DIFFERENTIABLE ARCHITECTURE SEARCH, ICLR2019

[2] Xuanyi Dong et al., Searching for A Robust Neural Architecture in Four GPU Hours, CVPR2019

[3] Jiemin Fang et al. ,Densely Connected Search Space for More Flexible Neural Architecture Search

[4] Xin Chen et al. ,Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation

[5] Yuhui Xu et al. ,PC-DARTS: Partial Channel Connections for Memory-Efficient Differentiable Architecture Search

编辑于 2019-07-23

- 



## NAS: One-Shot（三）

## NAS概述

当前使用的网络结构大部分是由人手工设计的，但是手工设计网络是耗时的且容易出错。因此，neural architecture search(NAS)变的流行起来。

NAS的工作可以根据三个方面进行划分：**search space**(搜索空间)、**search strategy**(搜索策略)、**performance estimation strategy**(性能评估策略)。

![img](https://pic2.zhimg.com/v2-a3805500d8c9e4d22447263b27636885_r.jpg)

**Search Space**：搜索空间定义为在一定原则下可以表示的结构。引入适合任务的先验知识能够减少搜索空间并且简化搜索。然而，这也会引入人为偏差，阻碍寻找到新颖的超出当前人类知识的结构构建模块。

**Search Strategy**：搜索策略详细说明如何探索搜索空间。它包含了典型的探索-开发的权衡，一方面，希望能够快速找到优良的结构，另一方面，希望避免提前收敛到结构次优的区域。

**Performance Estimation Strategy**：NAS典型目标是找到在未知数据上能够得到高性能的结构。性能评估是涉及评估性能的过程：最简单的方式是在数据上执行标准的训练和验证，不幸的是该过程计算昂贵并且能够探索的结构数量受限制。

**Search Strategy**目前有三种流行的搜索方法：**Reinforcement Learning(RL)、Evolutionary Algorithm(EA)、Gradient Based(GB)**

列一下相关论文，不具体细讲了。

**RL**

[Neural Architecture Search with Reinforcement Learning](https://link.zhihu.com/?target=https%3A//openreview.net/pdf%3Fid%3Dr1Ue8Hcxg)

[Learning Transferable Architectures for Scalable Image Recognition](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1707.07012)

[Progressive Neural Architecture Search](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1712.00559)

**EA**

[Large-Scale Evolution of Image Classifiers](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1703.01041)

[Regularized Evolution for Image Classifier Architecture Search](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1802.01548)

[Reinforced Evolutionary Neural Architecture Search](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1808.00193)

**GB**

[DARTS: Differentiable Architecture Search](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1806.09055)

[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1812.00332)

[FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1812.03443)



在NAS过程中，最为耗时的其实就是对于候选模型的训练。而初版的NAS因为对每个候选模型都是从头训练的，因此会相当耗时。一个直观的想法是有没有办法让训练好的网络尽可能重用。

一种思路是利用网络态射从小网络开始，然后做加法。

另一种思路是从大网络开始做减法，如One-Shot Architecture Search方法。

本文主要讲一下最近比较火的**One-Shot**(主要涉及到的论文有：**SMASH**、**ENAS**、**One-Shot**、**DARTS**、**ProxylessNAS**、**FBNet**、**SPOS**、**Single-Path NAS**、**FairNAS**)。

## One-Shot

**Example**

![img](https://pic1.zhimg.com/v2-2c6fc6cc3952e14a94792f4229dd50a0_r.jpg)

One-Shot模型的一个例子如图所示，我们可以在网络的特定位置使用3x3卷积、5x5卷积或最大池层三种操作的一种。不同于训练三个单独的模型，one-shot方法可以训练包含三种操作的单个模型。在评估时，可以选择将其中两种操作进行置零来确定哪种操作的预测精度最高。

### 1.SMASH

SMASH的目标是根据每个配置的验证性能来对一组神经网路配置进行相对排序，这个过程通过一个辅助网络产生的权值来完成。在每个训练阶段，我们随机采样一个网络架构(权值由HyperNet产生)，并且通过BP端到端训练。当模型完成训练时，我们采样一些随机架构(权值由HyperNet产生)并且在验证集上评估性能。我们随后选择性能最佳的架构并且训练它的权值。伪码如下：

![img](https://pic4.zhimg.com/v2-f10cbae636b4b678f006305ec617cb5f_r.jpg)

**Defining Variable Network Configuration**

![img](https://pic4.zhimg.com/v2-34f70515cb432bfde90c5a37404f4d2b_r.jpg)

不同于将网络看作一系列应用于前向传播信号的操作，我们将网络看作一系列可读可写的存储体(初始张量用0填充)。每层可以看成一个从存储子集中读取数据，修改数据，并且将结果写到另一个存储子集的操作。

![img](https://pic4.zhimg.com/v2-887cc68c0c8c8a9cc9020e325af1d2ef_r.jpg)

我们基础网络结构由多个模块组成，对于给定的空间分辨率下每个模块有一系列存储体，与大多数CNN架构一样，空间分辨率不断减半。下采样通过一个1x1卷积和平均池化完成，1x1卷积和全连接输出层的权值自由学习。

当下采样一个架构时，每个模块中，存储体的数量和每个存储体通道数是随机采样的。当定义模块的每层时，我们随机选择读写模式并且在读取数据时执行OP。当从多个存储体中读取时，我们沿着通道维度concat读取的张量，当写入存储体时，我们把当前每个存储体的张量加起来。

每个OP由1x1卷积(减少输入通道数)和数量不等带非线性的卷积组成，如图2(a)。我们随机选择四个卷积哪个是激活的，以及它们的滤波器大小、膨胀因子、组数和输出单元数(即层大小)。1x1 conv的输出通道数是op输出通道数的“瓶颈比”。

**Learning to map architectures to weights**

![img](https://pic3.zhimg.com/v2-8c0a8067366c09a3c0b0f296bf2f81a2_r.jpg)

**我们提出一个Dynamic Hypernet的变体，能基于主要网络结构** ![[公式]](https://www.zhihu.com/equation?tex=c) **编码的张量产生权值** ![[公式]](https://www.zhihu.com/equation?tex=W) 。我们的目标是去学习一个映射 ![[公式]](https://www.zhihu.com/equation?tex=W%3DH%28c%29) ，对于任何给定的输入该映射合理的接近最优 ![[公式]](https://www.zhihu.com/equation?tex=W) ，因此我们能够基于验证误差排序每个 ![[公式]](https://www.zhihu.com/equation?tex=c) 。因此，我们采用了一种 ![[公式]](https://www.zhihu.com/equation?tex=c) 的排布策略，以便能够对拓扑结构进行采样，并与标准库中的工具箱兼容，并使 ![[公式]](https://www.zhihu.com/equation?tex=c) 的维度尽可能具有可解释性。

我们的HyperNet是全卷积的，以至于输出张量 ![[公式]](https://www.zhihu.com/equation?tex=W) 的维度随着输入 ![[公式]](https://www.zhihu.com/equation?tex=c) 的维度变化，我们得到标准格式BCHW的4D张量，批量大小为1，这样没有输出元素是完全独立的。这允许我们通过增加c的高度或宽度来改变主要网络的深度和宽度。根据这一策略， ![[公式]](https://www.zhihu.com/equation?tex=W) 的每一片空间维度对应于 ![[公式]](https://www.zhihu.com/equation?tex=c) 的一个特定子集。OP的信息通过 ![[公式]](https://www.zhihu.com/equation?tex=W) 子集嵌入在通道维度相应的 ![[公式]](https://www.zhihu.com/equation?tex=c) 片来描述的。

### 2.ENAS

参考：[王佐：ENAS的原理和代码解析](https://zhuanlan.zhihu.com/p/33958314)

神经网络架构的搜索空间可以表示成有向无环图(DAG)，一个神经网络架构可以表示成DAG的一个子图。

![img](https://pic3.zhimg.com/v2-9109dbfc287e781a074bbb46ed10bdd6_r.jpg)

上图是节点数为5的搜索空间，红色箭头连接的子图表示一个神经网络架构。图中节点表示计算，边表示信息流。**ENAS使用一个RNN（称为controller）决定每个节点的计算类型和选择激活哪些边。**ENAS中使用节点数为12的搜索空间，计算类型为tanh，relu，identity，sigmoid四种激活函数。

controller工作流程：

以节点数为4的搜索空间为例。

![img](https://pic4.zhimg.com/80/v2-a7d2ef28e202e80ec66287d3d96b3783_1440w.png)

controller选择节点1的计算类型为tanh（节点1的前置节点是输入）；选择节点2的前置节点为1，计算类型为ReLU；选择节点3的前置节点为2，计算类型为ReLU；选择节点4的前置节点为1，计算类型为tanh。

![img](https://pic4.zhimg.com/v2-f57a1741c22a3e624ac0b3fab3f4e6e3_r.jpg)

便得到如下的RNN神经网络架构：节点3和节点4是叶子节点，他们输出的平均值作为RNN神经网络架构的输出。该神经网络架构的参数由![[公式]](https://www.zhihu.com/equation?tex=w%5E%7B1%2C2%7D%2C+w%5E%7B2%2C4%7D%2C+w%5E%7B2%2C3%7D%2C+w%5E%7B3%2C4%7D)组成，其中![[公式]](https://www.zhihu.com/equation?tex=w%5E%7Bi%2C+j%7D) 是节点 ![[公式]](https://www.zhihu.com/equation?tex=i) 和节点 ![[公式]](https://www.zhihu.com/equation?tex=j) 之间的参数。

![img](https://pic3.zhimg.com/80/v2-291190db9dc5fb98fa9eba2e31f38bda_1440w.jpg)

**在NAS中，** ![[公式]](https://www.zhihu.com/equation?tex=w%5E%7B1%2C2%7D%2C+w%5E%7B2%2C4%7D%2C+w%5E%7B2%2C3%7D%2C+w%5E%7B3%2C4%7D) **都是随机初始化，并在每个神经网络架构中从头开始训练的。在ENAS，这些参数是所有神经网络架构共享的。**如果下一次controller得到的神经网络架构如下，它参数由 ![[公式]](https://www.zhihu.com/equation?tex=w%5E%7B1%2C2%7D%2C+w%5E%7B2%2C4%7D%2C+w%5E%7B2%2C3%7D%2C+w%5E%7B3%2C4%7D) 组成，其中 ![[公式]](https://www.zhihu.com/equation?tex=w%5E%7B1%2C2%7D%2C++w%5E%7B2%2C3%7D%2C+w%5E%7B3%2C4%7D) 与上面神经网络架构相同的。

![img](https://pic4.zhimg.com/v2-932bfb13147fe238da3d474fe2e4671f_r.jpg)

通过参数共享，ENAS解决了NAS算力成本巨大的缺陷。

ENAS工作流程：

loop

loop
controller固定参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) ，采样一个神经网络架构，在训练集中训练该神经网络架构，并通过SGD调整神经网络架构的参数 ![[公式]](https://www.zhihu.com/equation?tex=w) 。
end loop

loop
controller采样一组神经网络架构，在验证集上计算ppl，并根据ppl和controller的交叉熵计算reward，通过Adam调整controller的参数![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta)。

end loop

end loop

### 3.One-Shot

参考：[认真学习的陆同学：2018.07.11- One-Shot -ICML 2018](https://zhuanlan.zhihu.com/p/73539339)

作者总结One-Shot模型的方法为4步：使用One-Shot模型设计搜索空间-训练-评估-选出有潜力的模型从头训练再评估。

1.设计搜索空间

![img](https://pic1.zhimg.com/v2-9b11731337cdf2e169a29d16099cdb14_r.jpg)

中间的三个Cell是相同的，每个Cell的构造如上图第二部分所示，但是在本文的实验中作者设定的Choice Block为4，图中少画了一个。每个Choice Block的构造如上图第三部分所示，设定每个Choice Block只能接受最多两个最近Cell 的输入，或者同一个Cell中其他Choice Block的输入，所以每个Choice Block可接受最少1个输入，最多7个输入。每个Choice Block中有7个可选操作，最多每次选2个操作，最少每次选1个操作。

2.使用SGD-M训练One-Shot模型：

（1）考虑模型的鲁棒性：因为评估的时候会去掉大量的分支，只留下所选择的路径，所以模型需要具备去掉分支后进行评估的鲁棒性，在训练的过程中使用Dropout来解决这一问题。

（2）稳定模型的训练：因为One-Shot模型的训练在早期是非常不稳定的，同时还在训练中引入了Dropout，所以训练很困难。作者引入了**“A variant of ghost batch normalization (Hoffer et al., 2017)”**来解决这一问题。

（3）防止过度正则化：让L2正则化只对当前模型的部分起作用。

### 4.DARTS

参考：

[谢震宇：NAS论文笔记（一）](https://zhuanlan.zhihu.com/p/62198487)

[Fisher Yu余梓彤：网络搜索之DARTS, GDAS, DenseNAS, P-DARTS, PC-DARTS](https://zhuanlan.zhihu.com/p/73740783)

公式推导可参考：[李斌：【论文笔记】DARTS公式推导](https://zhuanlan.zhihu.com/p/73037439)

DARTS 思想是直接把整个搜索空间看成supernet，学习如何sample出一个最优的subnet。这里存在的问题是子操作选择的过程是离散不可导，故DARTS 将单一操作子选择松弛软化为 softmax 的所有操作子权值叠加。

**1.搜索空间**

DARTS搜索示意图如下所示：

![img](https://pic2.zhimg.com/v2-74e28436f36eb92f459dc74793ed5e05_r.jpg)

(a) 定义的一个cell单元，可看成有向无环图，里面4个node，node之间的edge代表可能的操作（如：3x3 sep 卷积），初始化时unknown

(b) **把搜索空间连续松弛化，每个edge看成是所有子操作的混合（softmax权值叠加）**

(c) 联合优化，更新子操作混合概率上的edge超参（即架构搜索任务）和 架构无关的网络参数

(d) 优化完毕后，inference 直接取概率最大的子操作即可

cell中每个结点的值都是由其之前所有结点值计算而来的，具体计算公式为：

![[公式]](https://www.zhihu.com/equation?tex=x%5E%7B%28i%29%7D%3D%5Csum_%7Bj%3Ci%7D+o%5E%7B%28i%2C+j%29%7D%5Cleft%28x%5E%7Bj%7D%5Cright%29%5C%5C)

**2，搜索空间连续化**

搜索算法搜索的结果是某种特定结构的cell，比如cell中哪些结点之间有连线，连线上的操作是什么。但是到目前为止的搜索空间还是离散的。所以为了使搜索空间连续，作者为连线上的每一种操作赋上一个权重，所以对于连线上操作对于结点的操作可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=%5Coverline%7Bo%7D%5E%7B%28i%2C+j%29%7D%28x%29%3D%5Csum_%7Bo+%5Cin+O%7D+%5Cfrac%7B%5Cexp+%5Cleft%28%5Calpha_%7Bo%7D%5E%7B%28i%2C+j%29%7D%5Cright%29%7D%7B%5Csum_%7Bo%5E%7B%5Cprime%7D+%5Cin+O%7D+%5Cexp+%5Cleft%28%5Calpha_%7Bo%7D%5E%7B%28i%2C+j%29%7D%5Cright%29%7D+o%28x%29%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=O) 代表有向图中连线上操作的集合， ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7Bo%7D%5E%7B%28i%2C+j%29%7D) 代表连接结点 ![[公式]](https://www.zhihu.com/equation?tex=i%2C+j) 的连线上操作 ![[公式]](https://www.zhihu.com/equation?tex=o) 的权值。所以公式的含义就是使用连线上每个操作对结点进行操作，然后按照一定的权值将结果组合起来作为连线的输出。由于引入了参数权值 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) ，搜索空间变成连续空间的，而且 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 也可以通过反向传播算法学习更新。在结构搜索结束之后，也就是 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 学到之后，通过取每条连线上权值最大的操作作为连线的操作将结构离散化。数学表达式为：

![[公式]](https://www.zhihu.com/equation?tex=o%5E%7B%28i%2C+j%29%7D%3D%5Coperatorname%7Bargmax%7D_%7Bo+%5Cin+O%7D+%5Calpha_%7Bo%7D%5E%7B%28i%2C+j%29%7D%5C%5C)

**3，近似优化**

在优化过程中，除了需要优化更新结构参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 之外，还需要优化更新模型参数 ![[公式]](https://www.zhihu.com/equation?tex=w) 。如果同时进行优化的话，可能不太行。因为只要 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 改变了，就需要重新计算 ![[公式]](https://www.zhihu.com/equation?tex=w) 。所以作者采用了迭代近似优化的方法，即分开优化 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 和 ![[公式]](https://www.zhihu.com/equation?tex=w) 。优化算法如下图所示：

![img](https://pic2.zhimg.com/v2-3d761224fa76fdfb2ef010ae1b358e6d_r.jpg)

首先，在给定模型 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7Bk-1%7D) 时，通过最小化训练集损失 ![[公式]](https://www.zhihu.com/equation?tex=L_%7Bt+r+a+i+n%7D%5Cleft%28w_%7Bk-1%7D%2C+%5Calpha_%7Bk-1%7D%5Cright%29) 获得更新后参数 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bk%7D) 。之后固定住 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bk%7D)，通过最小化验证集损失

![[公式]](https://www.zhihu.com/equation?tex=L_%7Bv+a+l%7D%5Cleft%28w_%7Bk%7D-%5Cxi+%5Cpartial_%7Bw%7D+L_%7Bt+r+i+a+n%7D%5Cleft%28w_%7Bk%7D%2C+%5Calpha_%7Bk-1%7D%5Cright%29%2C+%5Calpha_%7Bk-1%7D%5Cright%29%5C%5C)

获得更新后的结构参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7Bk%7D) 。

**4，离散网络结构的获取**

对于cell中的每个中间结点，从它的所有前结点中选择k个连线权重最大的结点作为它的前结点。其中连线权重是连线上不同操作权重中最大的那个权重。通常对于CNN结构，k取2，对于RNN结构，k取1。得到每个中间结点的前结点之后，选区操作权重最大的操作作为两个结点间的操作。

### 5.ProxylessNAS

参考：[认真学习的陆同学：2018.12.02-ProxylessNAS-ICLR 2019](https://zhuanlan.zhihu.com/p/72604968)

1、针对Darts和One-Shot只能在小型数据集如CIFAR-10上进行搜索，而在大型数据集会出现显存爆炸的问题，提出了无代理的架构搜索，即可直接在大型数据集如ImageNet上完成搜索任务。

2、通过对不同硬件平台的时延进行建模，实现可搜索满足特定时延的神经网络，同时首次提出应针对不同的硬件平台搜索不同的网络架构。

**Method**

1、建立搜索空间：

 （1）主要借鉴Darts和One-Shot的思想，首先仍然建立超参数化网络，即每一条路径依然包含所有可能的操作：

![img](https://pic1.zhimg.com/v2-ccbc250bdb73de12bf2559d589f07398_r.jpg)

（2）二值化所有路径使得在训练时只有一条路径被激活，并且每条路径的激活概率设定为p，在训练中每个操作的概率p会被不断调整（第一个公式代表以概率p二值化所有操作，第二个公式代表以概率p二值化后一条边的输出）：

![img](https://pic1.zhimg.com/v2-80fa1bda778d1e7c4da48507ec85fab0_r.jpg)

![img](https://pic1.zhimg.com/v2-0c391e6f5782e5227e866f0469c44268_r.jpg)

2、训练二值化网络：

首先固定架构参数在训练集训练网络权重（**此时只有一条路被激活**，所以占用显存低），然后固定网络权重在验证集训练架构参数，二者交替进行，类似于Darts中的训练方法（同样是一个双优化问题）。

在对架构参数进行梯度反传时，作者使用了近似估计的方法，如下公式所示。由于最右边的第一项偏导和候选操作数N有关系，为了避免显存爆炸，本文提出在更新架构参数时，取样两条路径，然后使用梯度反传更新这两条路径的架构参数，之后将二者参数乘一个比例系数从而保持其他未被采样的架构参数所占比重不变。

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+%5Calpha_%7Bi%7D%7D%3D%5Csum_%7Bj%3D1%7D%5E%7BN%7D+%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+p_%7Bj%7D%7D+%5Cfrac%7B%5Cpartial+p_%7Bj%7D%7D%7B%5Cpartial+%5Calpha_%7Bi%7D%7D+%5Capprox+%5Csum_%7Bj%3D1%7D%5E%7BN%7D+%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+g_%7Bj%7D%7D+%5Cfrac%7B%5Cpartial+p_%7Bj%7D%7D%7B%5Cpartial+%5Calpha_%7Bi%7D%7D%3D%5Csum_%7Bj%3D1%7D%5E%7BN%7D+%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+g_%7Bj%7D%7D+%5Cfrac%7B%5Cpartial%5Cleft%28%5Csum_%7Bi%7D+%5Cexp+%5Cleft%28%5Calpha_%7Bj%7D%5Cright%29%5Cright.%7D%7B%5Cpartial+%5Calpha_%7Bi%7D%7D%3D%5Csum_%7Bj%3D1%7D%5E%7BN%7D+%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+g_%7Bj%7D%7D+p_%7Bj%7D%5Cleft%28%5Cdelta_%7Bi+j%7D-p_%7Bi%7D%5Cright%29%5C%5C)

3、提出两种方法解决不可微的硬件指标：

 （1）使时延可微：

对每个操作进行建模预测其时延，那么每条边的时延可表示为如下公式1，每条边时延的偏导为公式2，整个网络的时延为公式3，时延可微后直接放入损失函数如下公式4：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BE%7D%5Cleft%5B%5Ctext+%7B+latency+%7D_%7Bi%7D%5Cright%5D%3D%5Csum_%7Bj%7D+p_%7Bj%7D%5E%7Bi%7D+%5Ctimes+F%5Cleft%28o_%7Bj%7D%5E%7Bi%7D%5Cright%29%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial+%5Cmathbb%7BE%7D%5Cleft%5B%5Ctext+%7B+latency+%7D_%7Bi%7D%5Cright%5D+%2F+%5Cpartial+p_%7Bj%7D%5E%7Bi%7D%3DF%5Cleft%28o_%7Bj%7D%5E%7Bi%7D%5Cright%29%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BE%7D%5B%5Ctext+%7B+latency+%7D%5D%3D%5Csum_%7Bi%7D+%5Cmathbb%7BE%7D%5Cleft%5B%5Ctext+%7B+latency+%7D_%7Bi%7D%5Cright%5D%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Coperatorname%7BLoss%7D%3D%5Coperatorname%7BLoss%7D_%7BC+E%7D%2B%5Clambda_%7B1%7D%5C%7Cw%5C%7C_%7B2%7D%5E%7B2%7D%2B%5Clambda_%7B2%7D+%5Cmathbb%7BE%7D%5B%5Ctext+%7B+latency+%7D%5D%5C%5C)

（2）强化学习方法：

作为二进制连接的一种替代方法，我们也可以利用强化学习来训练二进制权重。考虑网络关键参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) ，更新关键参数的目标是找到最优二进制门 ![[公式]](https://www.zhihu.com/equation?tex=g) 来最大化奖励 ![[公式]](https://www.zhihu.com/equation?tex=R%28%5Ccdot%29) 。这里，为了便于说明，我们假设网络只有一个混合操作。我们对二值化参数的更新如下:

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+J%28%5Calpha%29+%26%3D%5Cmathbb%7BE%7D_%7Bg+%5Csim+%5Calpha%7D%5Cleft%5BR%5Cleft%28%5Cmathcal%7BN%7D_%7Bg%7D%5Cright%29%5Cright%5D%3D%5Csum_%7Bi%7D+p_%7Bi%7D+R%5Cleft%28%5Cmathcal%7BN%7D%5Cleft%28e%3Do_%7Bi%7D%5Cright%29%5Cright%29+%5C%5C+%5Cnabla_%7B%5Calpha%7D+J%28%5Calpha%29+%26%3D%5Csum_%7Bi%7D+R%5Cleft%28%5Cmathcal%7BN%7D%5Cleft%28e%3Do_%7Bi%7D%5Cright%29%5Cright%29+%5Cnabla_%7B%5Calpha%7D+p_%7Bi%7D%3D%5Csum_%7Bi%7D+R%5Cleft%28%5Cmathcal%7BN%7D%5Cleft%28e%3Do_%7Bi%7D%5Cright%29%5Cright%29+p_%7Bi%7D+%5Cnabla_%7B%5Calpha%7D+%5Clog+%5Cleft%28p_%7Bi%7D%5Cright%29+%5C%5C+%26%3D%5Cmathbb%7BE%7D_%7Bg+%5Csim+%5Calpha%7D%5Cleft%5BR%5Cleft%28%5Cmathcal%7BN%7D_%7Bg%7D%5Cright%29+%5Cnabla_%7B%5Calpha%7D+%5Clog+%28p%28g%29%29%5Cright%5D+%5Capprox+%5Cfrac%7B1%7D%7BM%7D+%5Csum_%7Bi%3D1%7D%5E%7BM%7D+R%5Cleft%28%5Cmathcal%7BN%7D_%7Bg%5E%7Bi%7D%7D%5Cright%29+%5Cnabla_%7B%5Calpha%7D+%5Clog+%5Cleft%28p%5Cleft%28g%5E%7Bi%7D%5Cright%29%5Cright%29+%5Cend%7Baligned%7D%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=g%5E%7Bi%7D) 表示第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个采样的二进制门， ![[公式]](https://www.zhihu.com/equation?tex=p%5Cleft%28g%5E%7Bi%7D%5Cright%29) 表示采样 ![[公式]](https://www.zhihu.com/equation?tex=g%5E%7Bi%7D) 的概率。

### 6.FBNet

参考：[雨宫夏一：CVPR2019 NAS 文章 (5)](https://zhuanlan.zhihu.com/p/70449149)

DARTS有一个很严重的问题，那就是内存消耗，内存消耗导致了一系列速度很慢的问题，所以作者打算直接从搜索空间中采样，然后进行梯度更新，但是“采样”这个过程必然导致不可导的问题，所以FBNE**采用了Gumbel function去代替求导的过程**。

DARTS直接对权重值求softmax 下一层的输入为之前一层所有操作空间的加权求和。

在超级网络的推理过程中，只对一个候选块进行采样并执行，采样概率为：

![[公式]](https://www.zhihu.com/equation?tex=P_%7B%5Ctheta_%7Bl%7D%7D%5Cleft%28b_%7Bl%7D%3Db_%7Bl%2C+i%7D%5Cright%29%3D%5Coperatorname%7Bsoftmax%7D%5Cleft%28%5Ctheta_%7Bl%2C+i%7D+%3B+%5Cboldsymbol%7B%5Ctheta%7D_%7Bl%7D%5Cright%29%3D%5Cfrac%7B%5Cexp+%5Cleft%28%5Ctheta_%7Bl%2C+i%7D%5Cright%29%7D%7B%5Csum_%7Bi%7D+%5Cexp+%5Cleft%28%5Ctheta_%7Bl%2C+i%7D%5Cright%29%7D%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7B%5Ctheta%7D_%7Bl%7D) 包含了决定每个块在第 ![[公式]](https://www.zhihu.com/equation?tex=l) 层采样概率的所有参数。同样，第 ![[公式]](https://www.zhihu.com/equation?tex=l) 层的输出可以表示为:

![[公式]](https://www.zhihu.com/equation?tex=x_%7Bl%2B1%7D%3D%5Csum_%7Bi%7D+m_%7Bl%2C+i%7D+%5Ccdot+b_%7Bl%2C+i%7D%5Cleft%28x_%7Bl%7D%5Cright%29%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=m_%7Bl%2C+i%7D) 为{0,1}中的一个随机变量，如果对块 ![[公式]](https://www.zhihu.com/equation?tex=b_%7Bl%2C+i%7D) 进行采样，则 ![[公式]](https://www.zhihu.com/equation?tex=i) 的值为1。

而FBNet直接利用Gumbel softmax 来进行一次采样，然后直接求导。具体的来说：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+m_%7Bl%2C+i%7D+%26%3D%5Ctext+%7B+GumbelSoftmax+%7D%5Cleft%28%5Ctheta_%7Bl%2C+i%7D+%7C+%5Cboldsymbol%7B%5Ctheta%7D_%7B%5Cboldsymbol%7Bl%7D%7D%5Cright%29+%5C%5C+%26%3D%5Cfrac%7B%5Cexp+%5Cleft%5B%5Cleft%28%5Ctheta_%7Bl%2C+i%7D%2Bg_%7Bl%2C+i%7D%5Cright%29+%2F+%5Ctau%5Cright%5D%7D%7B%5Csum_%7Bi%7D+%5Cexp+%5Cleft%5B%5Cleft%28%5Ctheta_%7Bl%2C+i%7D%2Bg_%7Bl%2C+i%7D%5Cright%29+%2F+%5Ctau%5Cright%5D%7D+%5Cend%7Baligned%7D%5C%5C)

Gumbel softmax 可以产生近似于one hot向量的输出。因此可以直接进行求导。

### 7.SPOS

参考：[认真学习的陆同学：2019.03.31-Single Path One-Shot-arXiv 2019](https://zhuanlan.zhihu.com/p/72736786)

1、本文的**主要思想是将权重优化和架构参数搜索这一双优化问题解耦分成两个步骤来进行**。首先对超网络不断进行均匀采样优化网络权重，然后通过进化算法可选择出满足不同约束的优秀网络架构。相对于之前的NAS方法，该文主要避免了在训练的过程中学习网络架构参数

，保证了在训练的过程中只有单一路径被激活（相比之前的ProxylessNAS，优化权重时单一路径激活，优化网络架构参数时需要两条路径被激活）。

2、由于在训练时不再同时优化网络权重和架构参数，所以在使用进化算法选择网络架构时可以精准满足不同的约束（如FLOPs和Latency等）。

3、基础的搜索块参照ShuffleNet v2设计，作者还提出可以搜索卷积层的输出通道数，同时还将本文方法应用于混合精度量化。

**Method**

1、单一路径的超网络和均匀采样：

之前的超网络训练是交替优化网络权重w和架构参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) ：

![[公式]](https://www.zhihu.com/equation?tex=w_%7Ba%7D%3D%5Cunderset%7Bw%7D%7B%5Coperatorname%7Bargmin%7D%7D+%5Cmathcal%7BL%7D_%7B%5Cmathrm%7Btrain%7D%7D%28%5Cmathcal%7BN%7D%28a%2C+w%29%29%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=a%5E%7B%2A%7D%3D%5Cunderset%7Ba+%5Cin+%5Cmathcal%7BA%7D%7D%7B%5Coperatorname%7Bargmax%7D%7D+%5Cmathrm%7BACC%7D_%7B%5Cmathrm%7Bval%7D%7D%5Cleft%28%5Cmathcal%7BN%7D%5Cleft%28a%2C+w_%7Ba%7D%5Cright%29%5Cright%29%5C%5C)

![img](https://pic2.zhimg.com/v2-62883b04a64a9cc20303b4f482794705_r.jpg)

本文提出的方法只优化权重，每次网络架构都是被均匀采样：

![[公式]](https://www.zhihu.com/equation?tex=W_%7B%5Cmathcal%7BA%7D%7D%3D%5Cunderset%7BW%7D%7B%5Coperatorname%7Bargmin%7D%7D+%5Cmathbb%7BE%7D_%7Ba+%5Csim+%5CGamma%28%5Cmathcal%7BA%7D%29%7D%5Cleft%5B%5Cmathcal%7BL%7D_%7B%5Ctext+%7B+train+%7D%7D%28%5Cmathcal%7BN%7D%28a%2C+W%28a%29%29%29%5Cright%5D%5C%5C)

2、通道搜索：

在训练时，让网络随机选择卷积的输出通道，权重训练好了以后使用进化算法选择表现优异的架构，此时选择出来的网络架构的通道就可理解为训练时网络随机搜索的。

3、进化算法搜索网络架构：

![img](https://pic4.zhimg.com/v2-f8623427a370ebe507b3b1db98b988cf_r.jpg)

### 8.Single-Path NAS

参考：[灵魂机器：论文解读：Single-Path NAS](https://zhuanlan.zhihu.com/p/63605721)

![img](https://pic2.zhimg.com/v2-524a92009dfeb4aa9a55037e1e57a355_r.jpg)

主要思想是**用一个7x7的大卷积，来代表3x3,5x5,7x7的三种卷积，把外边一圈mask清零掉就变成了3x3或5x5，这个大的卷积表示为 superkernel**，这样整个网络就只有一种卷积，看起来是一个直筒结构。这个操作相当于weight sharing，比如3x3的weights是和5x5,7x7的卷积共享的。

![img](https://pic3.zhimg.com/v2-c2368ee367b619a4265d63864f067bd2_r.jpg)

**搜索空间**

基于block的直筒结构，跟 ProxylessNAS, FBNet一样，都采用了Inverted Bottleneck 作为cell, 层数跟MobileNetV2都是22层。每层只有两个参数 expansion rate, kernel size 是需要搜索的。

**superkernel**

NAS中不同候选卷积操作可以看作一个参数过多的“超核”权值的子集。这样就可以把NAS看成一个寻找每个MBConv layer的核权值子集的组合问题，不同的MBConv结构选择共享核参数。

**Single-Path NAS formulation**

![img](https://pic2.zhimg.com/v2-bf320e31d6770eae747e9dd66fb95cf5_r.jpg)

如图3(左)所示，我们观察到，3×3核的权值可以看作是5×5核的权值的内核，而将“外层”的权值“归零”。我们将这个(外部)权值子集表示为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D_%7B%7B5+%5Ctimes+5%7D+%5Cbackslash+3+%5Ctimes+3%7D) 。因此，使用5×5卷积的NAS架构选择既对应于使用内部 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D_%7B3+%5Ctimes+3%7D) 权值，也对应于使用外壳，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bw%7D_%7B5+%5Ctimes+5%7D%3D%5Cmathbf%7Bw%7D_%7B3+%5Ctimes+3%7D%2B%5Cmathbf%7Bw%7D_%7B%7B5+%5Ctimes+5%7D+%5Cbackslash+3+%5Ctimes+3%7D) (图3，左)。

因此，我们可以将NAS决策直接编码到一个MBConv层作为内核权值的函数如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bw%7D_%7Bk%7D%3D%5Cmathbf%7Bw%7D_%7B3+%5Ctimes+3%7D%2B%5Cmathbb%7B1%7D%28%5Ctext+%7B+use+%7D+5+%5Ctimes+5%29+%5Ccdot+%5Cmathbf%7Bw%7D_%7B5+%5Ctimes+5+%5Cbackslash+3+%5Ctimes+3%7D%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7B1%7D%28%5Ccdot%29) 是对架构NAS选择进行编码的指示函数。

对于一个表示是否使用权值子集的指标函数，它的条件应该是权值子集的一个函数。因此，我们的目标是定义权值子集的“重要性”信号，该信号能够得到权值子集对总体损失的贡献。函数如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bw%7D_%7Bk%7D%3D%5Cmathbf%7Bw%7D_%7B3+%5Ctimes+3%7D%2B%5Cmathbb%7B1%7D%5Cleft%28%5Cleft%5C%7C%5Cmathbf%7Bw%7D_%7B5+%5Ctimes+5+%5Cbackslash+3+%5Ctimes+3%7D%5Cright%5C%7C%5E%7B2%7D%3Et_%7Bk%3D5%7D%5Cright%29+%5Ccdot+%5Cmathbf%7Bw%7D_%7B5+%5Ctimes+5+%5Cbackslash+3+%5Ctimes+3%7D%5C%5C)

其中tk=5是一个潜在的变量，它控制选择5×5核的决定(例如阈值)。阈值将与Lasso项进行比较，以确定是否将外部 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D_%7B%7B5+%5Ctimes+5%7D+%5Cbackslash+3+%5Ctimes+3%7D) 权值用于整个卷积。

由于基于核的NAS决策 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D_%7Bk%7D) 的结果本身就是卷积核，因此我们可以将我们的公式应用于编码 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D_%7Bk%7D) 的膨胀比的NAS决策。如图3所示(右),一个膨胀率为3的MBConv-k×k3层的通道数可以视为一个膨胀率为6的MBConv-k×k-6层通道数的一半,而置零的第二部分通道为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B%5Cmathbf%7Bw%7D_%7Bk%2C+6+%5Cbackslash+3%7D%5Cright%5C%7D) 。如果同时置零第一项输出过滤器，那么添加到MBConv layer的残差连接时，整个超核将没有贡献(相当于skip-op)。通过确定e是否等于3，我们可以编码NAS只使用或不使用“skip-op”路径的决策。对于两个 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D_%7Bk%7D) 核的决定，可以定义成:

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bw%7D%3D%5Cmathbb%7B1%7D%5Cleft%28%5Cleft%5C%7C%5Cmathbf%7Bw%7D_%7Bk%2C+3%7D%5Cright%5C%7C%5E%7B2%7D%3Et_%7Be%3D3%7D%5Cright%29+%5Ccdot%5Cleft%28%5Cmathbf%7Bw%7D_%7Bk%2C+3%7D%2B%5Cmathbb%7B1%7D%5Cleft%28%5Cleft%5C%7C%5Cmathbf%7Bw%7D_%7Bk%2C+6%7D+%7C+3%5Cright%5C%7C%5E%7B2%7D%3Et_%7Be%3D6%7D%5Cright%29+%5Ccdot+%5Cmathbf%7Bw%7D_%7Bk%2C+6+%5Cbackslash+3%7D%5Cright%29%5C%5C)

因此，对于输入 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D) ，网络第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个MBConv层的输出为:

![[公式]](https://www.zhihu.com/equation?tex=o%5E%7Bi%7D%28%5Cmathbf%7Bx%7D%29%3D%5Coperatorname%7Bconv%7D%5Cleft%28%5Cmathbf%7Bx%7D%2C+%5Cmathbf%7Bw%7D%5E%7Bi%7D+%7C+t_%7Bk%3D5%7D%5E%7Bi%7D%2C+t_%7Be%3D6%7D%5E%7Bi%7D%2C+t_%7Be%3D3%7D%5E%7Bi%7D%5Cright%29%5C%5C)

### 9.FairNAS

参考：[机器之心：超越MnasNet、Proxyless：小米开源全新神经架构搜索算法FairNAS](https://zhuanlan.zhihu.com/p/72392368)

[张俊：深度解读：小米AI实验室AutoML团队最新成果FairNAS](https://zhuanlan.zhihu.com/p/73093888)

根据模型真实能力进行排序的能力是神经架构搜索（NAS）的关键。传统方法采用不完整的训练来实现这一目的，但成本依然很高。而通过重复使用同一组权重，one-shot 方法可以降低成本。但是，我们无法确定共享权重是否真的有效。同样不明确的一点是，挑选出的模型具备更好的性能是因为其强大的表征能力，还是仅仅因为训练过度。

为了消除这种疑问，作者提出了一种全新方法——Fair Neural Architecture Search (FairNAS)，出于公平继承和训练的目的，该方法遵循严格的公平性约束。使用该方法，超网络训练收敛效果很好，且具备极高的训练准确率。与超网络共享权重的采样模型，在充分训练下的性能与独立模型（stand-alone model）的性能呈现出强烈的正相关。

![img](https://pic2.zhimg.com/v2-37dd6138b1def386c41dd54b7432bc55_r.jpg)

如上图所示，实验结果表明，在严格的公平性约束下，one-shot 模型在 ImageNet 训练集上的平均准确率稳步提升，没有出现振荡。与 EF相比，one-shot 模型的分层样本的准确率范围大大缩小。这是一个重大进展，研究者在快速评估模型的同时也能保证准确性。

**FairNAS 解决了两个基础问题：**

- 使用one-shot 超网络和采样技术得到不同子模型的方法真的公平吗？
- 如何根据模型性能进行快速排序，且排序结果具备较强的置信度？

具体而言，该研究具备以下贡献：

- 遵循严格公平性（strict fairness），强化 one-shot 方法；
- 在严格公平性条件下，实验结果表明平均准确率呈稳步上升，没有出现振荡（见图 1）；
- 尽管 one-shot 方法极大地加速了估计，但研究人员仍然面对多个现实约束以及广阔的搜索空间，于是研究人员选择多目标 NAS 方法来解决这个需求；
- 使用该研究提出的 pipeline，可在 ImageNet 数据集上生成一组新的 SOTA 架构。

**Strict Fairness**

在某种程度上，所有 one-shot 方法都是预定义搜索空间中任意单路径模型的不同性能预测器代理（proxies for performance predictor）。好的代理不能过度高估或低估模型得分。而目前还没有人对该主题进行深入的研究，并且以往多数研究仅仅侧重于搜索得分较好的几个模型。

为了减少超网络训练过程中的先验偏置（prior bias），研究人员定义了基本和直接的要求，如下所示：

![img](https://pic1.zhimg.com/v2-1413e35d66e62fe83685fabadea179cc_r.jpg)

不难看出，只有单路径 one-shot 方法符合上述定义。

在超网络训练的每个步骤中，只有相应激活选择块（choice block）的参数能够得到更新。笼统来说，参数更新的目的是减少模型在小批量数据上的损失，因此它虽然能够帮助激活选择块得到比未激活选择块更高的分数，但同时也产生了偏差。

研究人员将这种减少此类偏差的直接和基本要求称之为 Expectation Fairness，其定义如下：

![img](https://pic4.zhimg.com/v2-f1556bf70774bf136e85fd4ef333aef7_r.jpg)

研究人员提出了用于公平采样和训练的更严格要求，称之为 Strict Fairness，其定义如下：

![img](https://pic1.zhimg.com/v2-cce63d9b513886e1b1007cb18cc39134_r.jpg)

定义 3 施加了比定义 2 更严格的约束。定义 3 确保每个选择块的参数在任何阶段的更新次数相同，即 p(Y_l1 = Y_l2 = ... = Y_lm) = 1 在任何时候均成立。

**Method**

作者在严格遵循定义 3 的前提下，提出一种公平采样和训练算法（见 Algorithm 1）。**使用没有替换的均匀采样，在一步中采样 m 个模型，使得每个选择块在每次更新时都被激活**，参见下图 2：

![img](https://pic2.zhimg.com/v2-0d6b20f2509cf3a8279e7eced023dc25_r.jpg)

算法 1 如下图所示：

![img](https://pic3.zhimg.com/v2-3e539a9b50bb085ffb31f00592a54d9e_r.jpg)

**FairNAS 架构**

提出的 FairNAS 架构如下图 4 所示：

![img](https://pic2.zhimg.com/v2-38cfe7f4b2c2048765ba929d5bfb87ad_r.jpg)

## 总结

**SMASH**设计独立的hypernet来产生搜索空间中所有可能结构的权值，然而，hypernet的设计需要精细的专业知识，才能在采样模型的真实性能和生成的权值之间强相关。

**ENAS**其核心思想是让搜索中所有的子模型重用权重。它将NAS的过程看作是在一张大图中找子图，图中的边代表算子操作，基于LSTM的控制器产生候选网络结构(决定大图中的哪些边激活，以及使用什么样的操作)，这个LSTM控制器的参数和模型参数交替优化。

**One–Shot**进一步构建one-shot模型，覆盖搜索空间中的所有操作。在评估阶段，一个子网络通过drop-connect其它连接来进行模拟。不幸的是，它依赖于超参数(如drop-out率和每个block的操作可选数量)，这使得该方法的鲁棒性较差，必须采取特定的措施来稳定训练和防止过度正则化。另外，这样一个one-shot模型由于包含了所有的架构而存在内存爆炸的问题，当搜索空间增长时，内存变得太大而无法训练。

**DARTS**中最关键的是将候选操作使用softmax函数进行混合。这样就将搜索空间变成了连续空间，目标函数成为了可微函数。这样就可以用基于梯度的优化方法找寻最优结构了。搜索结束后，这些混合的操作会被权重最大的操作替代，形成最终的结果网络。

**ProxylessNAS**沿用了DARTS中连续松弛的方法和双优化的训练策略，将路径上的 arch parameter 二值化，在搜索时仅有一条路径处于激活状态。这样一来 GPU 显存就从 O(N) 降到了 O(1)，解决了显存占用和 search candidates 线性增长的问题。对于不同的硬件平台，通过 latency estimation model，将延迟建模为关于神经网络的连续函数，并提出了 Gradient 和 RL 两种方法在搜索过程对其优化。

**FBNet**通过Gumbel function来替代求导过程，解决了直接从搜索空间采样无法梯度更新的问题，减少了内存消耗。

**SPOS**将权重优化和架构参数搜索这一双优化问题解耦分成两个步骤来进行，避免了在训练的过程中学习网络架构参数，保证了在训练的过程中只有单一路径被激活。由于在训练时不再同时优化网络权重和架构参数，所以在使用进化算法选择网络架构时可以精准满足不同的约束（如FLOPs和Latency等）。

**Single-Path NAS**用superkenel统一3x3和5x5两种卷积，把网络结构变成了single path，使得在search method上可以选择更加快速的优化方法。

**FairNAS**提出了要满足 Strict Fairness，这个约束条件是超网的每单次迭代让每一层可选择运算模块的参数都要得到训练。FairNAS 与 SPOS 的均匀采样不同，采取了不放回采样方式和多步训练一次参数更新的方式，这带来了one-shot 分布和 supernet 训练的整体提升。

**总体趋势**：搜索空间越来越小，计算效率越来越高，训练速度越来越快，不同子模型的训练越来越公平。

## NAS应用

**目标检测**

[NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1904.07392)

[DetNAS: Backbone Search for Object Detection](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1903.10979)

**语义分割**

[Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1901.02985%3Fcontext%3Dcs.CV)

[Searching for Efficient Multi-Scale Architectures for Dense Image Prediction](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1809.04184)

**实例分割**

[InstaNAS: Instance-aware Neural Architecture Search](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1811.10201)

**ReID**

[Auto-ReID: Searching for a Part-aware ConvNet for Person Re-Identification](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1903.09776%3Fcontext%3Dcs)

**视频动作识别**

[Video Action Recognition Via Neural Architecture Searching](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1907.04632)



## References

[Neural Architecture Search: A Survey](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1808.05377)

[SMASH: One-Shot Model Architecture Search through HyperNetworks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1708.05344)

[Efficient Neural Architecture Search via Parameter Sharing](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1802.03268)

[The combination of electro-thermal stress, load cycling and thermal transients and its effects on the life of high voltage ac cables](https://link.zhihu.com/?target=https%3A//doi.org/10.1109/TDEI.2009.5211872)

[DARTS: Differentiable Architecture Search](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1806.09055)

[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1812.00332)

[FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1812.03443)

[Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1904.00420)

[Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1904.02877)

[FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1907.01845)