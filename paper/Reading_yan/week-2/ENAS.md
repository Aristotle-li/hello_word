> 题目：ENAS-Efficient Neural Architecture Search via Parameter Sharing
>
> 来源：plmr 2018
>
> 作者：HieuPham MelodyY.Guan BarretZoph  QuocV.Le  JeffDean

### Motivation：

NAS训练一个子模型直到收敛，只是为了度量其准确性，同时丢弃所有训练过得权值，计算量巨大。所以提出在子模型之间共享权重，提出有向无环图(DAG，directed acyclic graph)



### idea：

#### 思想：

the choice of a component of the architecture is regarded as an action. A sequence of actions defines an architecture of a neural network, whose dev set accuracy is used as the reward.

#### 细节：

在ENAS中，控制器通过在大型计算图中搜索最优子图来描述神经网络结构。使用policy gradient对控制器进行训练，以选择一个子图，使验证集上的期望回报最大化。同时对所选子图对应的模型进行训练，以使典型交叉熵损失最小化。在子模型之间共享参数使ENAS提供强大的经验性能。

用一个有向无环图（DAG）来表示NAS的搜索空间，ENAS的controller是一个RNN并决定：

1)哪些边被激活，2)在DAG的每个节点上进行了哪些计算（即决定RNN cell的拓扑结构和操作）

对于convolutional model 就是：1）连接之前的哪个节点2）使用什么计算操作。（注意其中还包括skip connection）







### 以后改进方向：

最近的一些工作着眼于解决共享权重带来的偏置问题和超级图的高显存占用问题，并将新的搜索目标如网络延时、结构稀疏性引入NAS中。





### 词句：

1、Meanwhile, using less resources tends to pro- duce less compelling results 

2、Central to the idea of ENAS is  ENAS的核心思想是

3、 ENAS’s DAG is the superposition of all possible child models in a search space of NAS  叠加

4、how to derive architectures from ENAS’s controller 如何从ENAS的控制器派生体系结构





## NAS论文阅读笔记（ENAS）

[![bfluss](https://pic1.zhimg.com/v2-e728bcfa0c5ade7b64db3a6a6a656026_xs.jpg?source=172ae18b)](https://www.zhihu.com/people/wu-rui-xia-1)

[bfluss](https://www.zhihu.com/people/wu-rui-xia-1)

东南大学研究生人工智能研究者

## Efficient Neural Architecture Search via Parameter Sharing

论文链接：[[Efficient Neural Architecture Search via Parameter Sharing](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1802.03268)]

use a single Nvidia GTX 1080Ti GPU, the search for architectures takes less than 16 hours.[Efficient Neural Architecture Search via Parameter Sharing](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1802.03268) use a single Nvidia GTX 1080Ti GPU, the search for architectures takes less than 16 hours.

**NAS的瓶颈**是训练一个子模型直到收敛，只是为了度量其准确性，同时丢弃所有训练过得权值。所以提出在子模型之间共享权重，提出有向无环图(DAG，directed acyclic graph)

**主要思路：**

在ENAS中，控制器通过在大型计算图中搜索最优子图来描述神经网络结构。使用policy gradient对控制器进行训练，以选择一个子图，使验证集上的期望回报最大化。同时对所选子图对应的模型进行训练，以使典型交叉熵损失最小化。在子模型之间共享参数使ENAS提供强大的经验性能。

**A DAG example**

![img](https://pic3.zhimg.com/v2-f91a9d085f3cb540c3eab9fed8a850e2_r.jpg)

节点表示本地计算，边表示信息流。每个节点上的计算量都有自己的参数，只有在特定的计算被激活时才会用到这些参数。

## 1、Designing Recurrent Cells

ENAS的controller是一个RNN并决定：

1)哪些边被激活，2)在DAG的每个节点上进行了哪些计算（即决定RNN cell的拓扑结构和操作）

controller对N个决策块采样

an example：

![img](https://pic3.zhimg.com/v2-79175b574dbc5aa905660a8784caa4ce_r.jpg)

h1 = tanh (xt· W(x)+ ht−1· W(h) 1)

h2 = ReLU(h1· W(h) 2,1)

h3 = ReLU(h2· W(h) 3,2)

h4 = tanh (h1· W(h) 4,1)

output ：ht = (h3+ h4)/2

search space：N notes 、4 activation functions(namely tanh、ReLU、identity、sigmoid) 4^N X N!

## 2、Training ENAS and Deriving Architectures

控制器网络是一个LSTM，有100个隐藏单元.这个LSTM通过softmax分类器以自回归的方式对决策进行采样:将上一步中的决策作为嵌入到下一步的输入。在第一步，控制器网络接收一个空嵌入作为输入。

在ENAS,有两套可学的参数:控制器LSTM的参数θ,和子模块共享参数ω。ENAS的培训过程包括两个交叉阶段。第一阶段通过整个训练数据集训练w。第二阶段以固定的步数训练θ。

## Training the shared parameters w of the child models

固定控制器策略π(m; θ) ，对w执行SGD来最小化期望损失函数Em∼π[L(m; ω)]，L(m; ω)是standard cross-entropy loss，对通过策略采样的模型在小批量数据集上计算得到。梯度是用蒙特卡罗估计来计算的。文中用根据策略采样的任意单一模型的梯度更新w。

![img](https://pic1.zhimg.com/v2-069b4456e8681c25e4a0192fc83fbdf4_r.jpg)

## Training the controller parameters θ

固定w更新策略参数θ，来使得其期望回报最大Em∼π(m;θ)[R(m, ω)]，使用Adam优化器，其中梯度是用REINFORCE计算的，使用moving average baseline来减小方差，回报R(m, ω)是在测试集上计算得到的。在图像分类中，回报函数是小批量测试图片的accuracy。

## Deriving Architectures

首先从训练策略采样几种模型，对每个模型，计算从验证集采样的单个小批量的回报，然后只对回报最高的模型从零开始进行在训练。

## 3、Designing Convolutional Networks

对于convolutional model的search space，控制器RNN在每个决策块采样两套决定：

1）连接之前的哪个节点，2）使用什么计算操作

注意其中还包括skip connection

（网络个数的计算公式详见原文）

> The 6 operations available for the con- troller are: convolutions with filter sizes 3 × 3 and 5 × 5, depthwise-separable convolutions with filter sizes 3×3 and 5×5 (Chollet, 2017), and max pooling and average pooling of kernel size 3 × 3.

![img](https://pic2.zhimg.com/v2-7bc3c5927238d2b0ab087001d810b0bd_r.jpg)

## 4、Designing Convolutional Cells

设计conv cell和reduction cell

![img](https://pic2.zhimg.com/v2-20ceb44683a06a024cb36fcc35bb9d21_r.jpg)

RNN controlller 做两个决定：

1)选择哪两个先前的节点作为当前节点的输入，2)应用于两个采样节点的两个操作。最后将两个结果相加。

> The 5 available operations are: identity, separable convolution with kernel size 3 × 3 and 5 × 5, and average pooling and max pooling with kernel size 3×3。

![img](https://pic4.zhimg.com/v2-39412a9a83b80e0fd148b3a79a869bf7_r.jpg)

如果输出节点不止一个的话，所有的输出节点将会在depth方向concatenate

搜索空间也可以实现一个reduction cell，只需:1)从搜索空间中采样一个计算图形，2)以stride = 2应用所有的操作。

（最后的search space的计算公式详见原文）

![img](https://pic2.zhimg.com/v2-7b1d2df3557d239941fc75ac725163a5_r.jpg)

![img](https://pic1.zhimg.com/v2-a63e8e761cd619d931d8359fdd333f10_r.jpg)

![img](https://pic3.zhimg.com/v2-c3259982e4eceb812346a4b3989f2a0a_r.jpg)

![img](https://pic3.zhimg.com/v2-397b5991c3de5692e290b414115de39e_r.jpg)