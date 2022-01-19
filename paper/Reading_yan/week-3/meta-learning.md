> 题目：Meta-Learning in Neural Networks: A Survey
>
> 作者：Timothy Hospedales, Antreas Antoniou, Paul Micaelli, Amos Storkey

## 现存的挑战

1. **泛化能力**。现有的meta-learning方法在一些任务上训练，在新的任务上测试。直接的假设是：训练和测试数据服从同一个分布。所以直接的问题是：如果meta-train和meta-test不是来自一个分布，如何做meta-learning？或者说，meta-train数据本身就包含了来自多个分布的数据，如何做meta-train？
2. **多模态的任务来源**。现有的meta-learning方法假设训练和测试任务来自单一模态的数据，如果是多模态如何进行元学习？
3. **任务家族**。特定任务下只有特定的任务家族可以用元学习，这大大限制了知识的表征和传播。
4. **计算复杂度**。通常来说meta-learning都涉及到一个bi-level的优化，这导致其计算非常复杂。如何提高计算效率是一个难点。
5. **跨模态迁移和异构的任务**。

### 解决的问题：

meta-learning aims to improve the learning algorithm itself, given the experience of multiple learning episodes.

元学习的目的是在经历多次学习的情况下，改进学习算法本身。

## 元学习的应用

元学习在机器学习的诸多场景下都有着广泛的应用。

- 计算机视觉中的**小样本学习（few-shot learning）**，包括分类、检测、分割、关键点定位、图像生成、密度估计等。可以说小样本的这个任务设定完美契合了meta-learning的学习过程。所以我们看到，绝大多数小样本研究都采用了meta-learning的学习方法。
- **强化学习**。一个很自然的扩展便是强化学习，在RL中的探索（exploration）和利用（exploitation），策略梯度等都有着广泛应用。
- **仿真（sym2real）**。由虚拟环境生成一些样本学习泛化的学习器，然后部属到真实环境中。
- **神经结构搜索（NAS）**。
- **贝叶斯元学习**。这个步骤非常有趣，它涉及到用贝叶斯的视角来对元学习进行重新表征，可以发现不一样的研究思维，对解决问题非常有帮助。
- **无监督元学习**。
- **终身学习、在线学习、自适应学习**。
- **领域自适应和领域泛化**。
- **超参数优化**。
- **自然语言处理**。
- **元学习系统**，等等。

思路：吃一句话，gan出，使用UE4快速生成大量低质量样本，通过随机采样少量低分辨率渲染成高质量实现领域泛化，NAS+泛化学习器，部属到真实环境中。

### detail：



#### 元学习：任务分配视角 

任务定义为数据集和loss函数T = {D,L}， 在任务分布p(T)上，评估w的性能，w指“如何学习”或“元知识”。L（D，w）是在数据集D上使用w衡量一个模型的性能

学习如何学习，因此成为：<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210512111842070.png" alt="image-20210512111842070" style="zoom:50%;" />

meta-training stage：

we denote the set of M source tasks used in the meta-training stage as：<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210512112606064.png" alt="image-20210512112606064" style="zoom:50%;" />  

其是从p(T)sample出来的子集（the source train and validation datasets are respectively called support and query sets.）

so，The meta-training step of ‘learning how to learn’ can be written as:<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210512112818126.png" alt="image-20210512112818126" style="zoom:50%;" />

meta-testing stage：

we denote the set of Q target tasks used in the meta-testing stage as <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210512113357018.png" alt="image-20210512113357018" style="zoom:50%;" />

在元测试阶段，我们使用所学的元知识w 训练基本模型on each previously unseen target task i ：<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210512113609003.png" alt="image-20210512113609003" style="zoom:50%;" />

#### 元学习：双层优化视角 

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210512114542091.png" alt="image-20210512114542091" style="zoom:50%;" />

内层优化公式6是以外层定义的学习策略w为条件的，但在训练过程中不能改变w

#### 元学习：前馈模型视角（amortized）

有许多元学习方法，它们以前馈方式而不是通过显式迭代优化来综合模型，定义元训练线性回归的toy示例：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210512115542070.png" alt="image-20210512115542070" style="zoom:50%;" />

训练集Dtr  is embedded  into a vector gw，gw定义为线性回归权重，以预测来自验证集的样本。

这一类方法，学习新任务的成本通过gw降低为前馈操作，迭代优化的时间用在了w元训练。

### 旧的分类：

元学习方法的分类：基于优化的方法、基于模型（或黑盒）的方法和基于度量（或非参数）的方法。

基于优化的方法：双层优化，如：MAML

基于模型的方法：前馈模型：该模型将当前数据集D嵌入到激活状态中，并根据该状态对测试数据进行预测。

外部优化和内部优化紧密耦合，因为w和D直接指定$ \theta$，缺点：泛化性能差且难以将大型训练集嵌入到模型中。

度量学习（Metric-Learning）： non-parametric algorithms

### 新的分类：

![image-20210512160254307](/Users/lishuo/Library/Application Support/typora-user-images/image-20210512160254307.png)



Meta-Representation：那些是可学习的元知识，那些是固定的。

应用：

在多任务场景中，任务不可知的知识是从一系列任务中提取出来的，并用于改进对该系列新任务的学习，

以及单个任务场景，其中单个问题被反复解决并在多个阶段得到改进



### 词句：

1、The previous discussion outlines the common flow of meta-learning in a multiple task scenario, but does not specify how to solve the meta-training step

前面的讨论概述了多任务场景中元学习的常见流程，但没有说明如何解决元训练步骤

2、the inner level optimization Eq. 6 is conditional on the learning strategy w defined by the outer level, but it cannot change w during its training.

内层优化公式6是以外层定义的学习策略w为条件的，但在训练过程中不能改变w

3、 because the cost of learning a new task is reduced to a feed-forward operation through gw, with iterative optimization already paid for during meta-training of w

因为学习一个新任务的成本通过gw降低为一个前馈操作，迭代优化的时间用在了w元训练。

4、elaborate alternatives



## 课程：

meta-learning  常常和few-shot learning一起搭配使用：

MAML：找到一个对他认为所有任务都最好的初始值

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518201719993.png" alt="image-20210518201719993" style="zoom:67%;" />

MAML：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518202849130.png" alt="image-20210518202849130" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518205147547.png" alt="image-20210518205147547" style="zoom:67%;" />

MAML和pre-training是有区别的：

pre-trainning是在task1上得到性能最好的模型参数，再应用到task2

MAML是在a set of tasks  supportset 上得到loss的和，但不更新模型参数，再到query set上，after training 后再更新参数

例如：update一次，因为few-shot时，update多了容易overfiting，也浪费时间

### MAML 数学推导：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518221557482.png" alt="image-20210518221557482" style="zoom:50%;" />

如何理解微分：一个量随另一个量变化的大小，也就是说一个量被另一个量影响的大小



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518222919447.png" alt="image-20210518222919447" style="zoom:50%;" />

计算细节：using a first-order approximation

 <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518224035247.png" alt="image-20210518224035247" style="zoom: 33%;" />

结果就是：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518224315214.png" alt="image-20210518224315214" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518224646508.png" alt="image-20210518224646508" style="zoom: 50%;" />

reptile：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210519121401568.png" alt="image-20210519121401568" style="zoom:50%;" />

Reptile 中，每更新一次 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) ，需要 sample 一个 batch 的 task（图中 batchsize=1），并在各个 task 上施加多次梯度下降，得到各个 task 对应的 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%5Ctheta) 。然后计算 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%5Ctheta) 和主任务的参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 的差向量，作为更新 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 的方向。这样反复迭代，最终得到全局的初始化参数。



### Learning to learn by gradient descent by gradient descent：



the learning algorithm look like RNN

RNN：优势是无论输入多长，模型的参数都不会增加

LSTM：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210519213846671.png" alt="image-20210519213846671" style="zoom: 67%;" />

改造LSTM，固定学习率，使其可以做gradient decent

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210519214739819.png" alt="image-20210519214739819" style="zoom: 67%;" />



更进一步：动态学习率

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210520182813616.png" alt="image-20210520182813616" style="zoom:67%;" />



结果就是：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210520183446854.png" alt="image-20210520183446854" style="zoom:67%;" />

类似于momentum，两层的LSTM可能会将过去的梯度考虑进去

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210520224551195.png" alt="image-20210520224551195" style="zoom:67%;" />





* 但是和真正的LSTM是不同的，c+h和输入x是independent，而theta和输入是有关系的，即虚线，但是实际上直接忽略了，就当没有关系硬train。

* 实际使用只train一个LSTM，然后应用到所有参数theta
* MAML中training和testing的model要一样，而用LSTM更新参数是可以不一样的

 OPTIMIZATION AS A MODEL FOR FEW-SHOT LEARNING

















# [Meta-Learning]对Reptile的深度解析

[![周威](https://pic4.zhimg.com/v2-e127be459ba552a5697994cd5a80ee2d_xs.jpg)](https://www.zhihu.com/people/zhou-wei-37-26)

[周威](https://www.zhihu.com/people/zhou-wei-37-26)[](https://www.zhihu.com/question/48510028)

东南大学 工学博士在读

23 人赞同了该文章

## 1 前言

接着上一篇对MAML的深度解析，链接如下：

[周威：[meta-learning\] 对MAML的深度解析zhuanlan.zhihu.com![图标](https://pic2.zhimg.com/v2-197832f5a287c3cff30dccbc4c6bc58d_180x120.jpg)](https://zhuanlan.zhihu.com/p/181709693)

我们介绍了一些meta-learning的基本概念，包括

- **N-way K-shot learning**：分类**类别为N**，**每个类别有K张图片**（或者其他数据）的分类问题，一般而言，一个task的数量为N-way K-shot；
- **support set**：用来进行**fast weight**的梯度下降（该过程又叫做inner loop update)的数据集，每次更新使用样本大小为N-way K-shot的数据集（为一个task）；
- **query set**：用来计算**一大轮**（包含m个tasks）总损失大小，一般大小为m（任务数）个N-way **K_test**-shot的数据集，K_test根据需求设定；

不过在上一讲中，我们并没有结合代码进行解析。因为当初概念较多，引入代码会使得内容用于冗长。不过我还是非常建议大家去看一下代码，加深理解。

代码链接如下：

[https://github.com/dragen1860/MAML-Pytorchgithub.com](https://link.zhihu.com/?target=https%3A//github.com/dragen1860/MAML-Pytorch)

有了一些对MAML的基础后，我们对其“升级版”Reptile进行解析。

大家发现了，meta-learning中取名都挺有意思，一会儿哺乳动物（MAML），一会儿又是爬行动物（Reptile），这个领域不会都**动物学家转过来**的吧？

有关论文和代码链接，下面给出

**Paper**：[On First-Order Meta-Learning Algorithms](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1803.02999)

**Code**：[gabrielhuang/reptile-pytorch](https://link.zhihu.com/?target=https%3A//github.com/gabrielhuang/reptile-pytorch)

## 2 Reptile论文中的一些创新点

作者在论文的introduction部分的最后，总结了该论文的一些贡献，具体如下：

![img](https://pic3.zhimg.com/80/v2-1e91573960830db20072caab6d1e3ec2_1440w.jpg)

也就是作者提出了一个叫做Reptile的Meta-Learning，该方法和First-Order MAML很像，但是实现起来更简单。

毕竟我们在之前的学习MAML的过程中，是要将训练数据集分为support set和query set的。support set和query set的作用各不相同。

而作者的Reptile并不需要将数据集分为support set和query set的。

注意一下，这里作者提到的fast weight 和slow weight是什么意思呢？

我们在MAML中提到，我们的目标是需要学习到**一个好的初始化权重**。

学习这个初始化权重需要**两个更新**（或者称为**两个梯度下降**）：

- **第一个更新/梯度下降（叫做inner loop update）**是使用一个个的task（也就是support set）不断的从初始化的权重 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 开始进行梯度下降，下降到 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bi%7D%5E%7B%27%7D%3D%5Cphi-g_%7B1%7D-g_%7B2%7D-g_%7B3%7D-...-g_%7Bn%7D) ，这里的i是task的索引，这里的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bi%7D) 就是fast weight ；
- **第二个更新/梯度下降（叫做outer loop update）**是使用一个个（索引为i）的query set去计算**总的损失函数**![[公式]](https://www.zhihu.com/equation?tex=L%3D%5Csum_%7Bi%3D0%7D%5E%7Btask%7D%7Bl%28qureyset_%7Bi%7D%3B%5Ctheta_%7Bi%7D%5E%7B%27%7D%29%7D) 。然后求解总损失函数**对初始化权重**的**导数**作为梯度即可实现**第二个更新**，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi%3D%5Cphi-%5Clambda%5Ccdot%5Cpartial%28L%29%2F%5Cpartial%28%5Cphi%29) ，这里的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 就是slow weight， ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) 是学习率。

Reptile中并没有进行support set和query set的划分，而是采用了一种更简单的方式进行slow weight更新，具体的算法流程如下。

## 3 Reptile的算法流程

这里我们引入Reptile的算法流程，如下

![img](https://pic1.zhimg.com/80/v2-58e3a968216888084267030f725988a0_1440w.jpg)

就这？如此简单？

对的，就是如此简单，没有support set和query set的纠缠，作者的意思非常明确。

给了**一堆tasks**，不断地从每个task中（采用**子集k个**）进行**inner loop update**，更新**fast weight**。

那么当该task的k个子集全部训练结束后，网络初始化的参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 就更新为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cphi%7D) ，其中

![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cphi%7D%3D%5Cphi-g_%7B1%7D-g%7B2%7D-...-g_%7Bk%7D)

简单来说，上面的更新其实就是**fast weight的更新**，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cphi%7D) 就是fast weight，这个和MAML中fast weight的更新**类似**。就是在**同一个任务中**随机采样的k个子集（每个子集包含N-way K-shot个数据）进行训练来更新参数。

在Reptile中，更新**slow weight的方向**并不是像MAML中由**总损失Loss对初始化参数的导数**决定的。因为Reptile中并**不需要query set**，所以无法计算总损失。

作者直接使用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi-%5Ctilde%7B%5Cphi%7D) 的方向，也就是 ![[公式]](https://www.zhihu.com/equation?tex=g_%7B1%7D%2Bg%7B2%7D%2B...%2Bg_%7Bk%7D) 来决定slow weight更新的方向，也就是上面流程图中**标黄的部分**，简单粗暴，简直amazing呀！

这里不妨使用图绘进行更清晰的表述，如下图所示

![img](https://pic1.zhimg.com/80/v2-94efe761b18d4827b44b76527e49de24_1440w.jpg)

这么一看，Reptile要比MAML简单很多，即便是简化后的First-Order MAML，也不如Reptile简单呀!

值得注意的是，上图中的Task1（1）、Task1（2）、Task1（3）、Task1（4）都是**同一个任务**（比如猫狗分类数据集）中**随机采样**的**4个子集**（每个子集N-way K-shot个数据）。

这里我们结合代码进行验证，代码如下：

```python
# Main loop
for meta_iteration in tqdm.trange(args.start_meta_iteration, args.meta_iterations):

    # Update learning rate
    meta_lr = args.meta_lr * (1. - meta_iteration/float(args.meta_iterations))
    set_learning_rate(meta_optimizer, meta_lr)

    # Clone model
    net = meta_net.clone()
    optimizer = get_optimizer(net, state)
    # load state of base optimizer?

    # Sample base task from Meta-Train
    train = meta_train.get_random_task(args.classes, args.train_shots or args.shots)
    train_iter = make_infinite(DataLoader(train, args.batch, shuffle=True))

    # Update fast net
    # do the first batch update steps
    # the grads of para are from grad to grad-p1-p2-...-pn
    loss = do_learning(net, optimizer, train_iter, args.iterations)
    state = optimizer.state_dict()  # save optimizer state

    # Update slow net
    # update the meta_net's grad parameter to meta_net.param - clone_net.param
    # correspond to p1+p2+p3+...+pn
    meta_net.point_grad_to(net)
    meta_optimizer.step()
```

核心代码非常简单、简短。值得注意的是，这里进行了一个**模型的clone**，意在进行fast weight的更新，同时**保留原模型的初始化参数**。

同样地，如果有n个任务（比如猫狗分类、香蕉苹果分类、男女分类等），那么更新的公式为

![img](https://pic1.zhimg.com/80/v2-d95f8d648ea9874c2635ce10e3808188_1440w.jpg)

这里的i是第i个任务的**索引**，一共有n个任务。上面图例中只展示了在1个任务中的slow weight更新。

至此，Reptile就讲解完毕，很简单的一个模型。但是具体背后的**数学原理**并不是很简单，即作者为什么选择![[公式]](https://www.zhihu.com/equation?tex=%5Cphi-%5Ctilde%7B%5Cphi%7D) 为slow weight的更新方向，我认为还是**有必要精读下原论文的数学推导的**，这里我就不细说了。

## 4 总结

最近看的东西比较多，也比较杂，而且还有很多idea进行验证与成果化，更新的速度比较慢，大家见谅！

看reptile原文的时候，我觉得这东西全是数学推导（看着就头大），等摸清楚了后，会发现Reptile是要比MAML简单的。具体的一些关于Reptile的应用，我也还在摸索，等有成果了会和大家进行额外分享的！







![《小王爱迁移》系列之二十四：元学习的前世今生](https://pic1.zhimg.com/v2-e5a4a17ab4fe4d2e51e8dbabdaadba1a_1440w.jpg?source=172ae18b)





# 《小王爱迁移》系列之二十四：元学习的前世今生

[![王晋东不在家](https://pic1.zhimg.com/d7bd4f986_xs.jpg?source=172ae18b)](https://www.zhihu.com/people/jindongwang)

[王晋东不在家](https://www.zhihu.com/people/jindongwang)[](https://www.zhihu.com/question/48510028)

中国科学院大学 计算机应用技术博士

166 人赞同了该文章

<img src="https://pic1.zhimg.com/v2-e5a4a17ab4fe4d2e51e8dbabdaadba1a_1440w.jpg?source=172ae18b" alt="《小王爱迁移》系列之二十四：元学习的前世今生" style="zoom:51%;" />

> 本文使用 [Zhihu On VSCode](https://zhuanlan.zhihu.com/p/106057556) 创作并发布

本文介绍由三星AI中心、爱丁堡大学Timothy Hospedales副教授团队刚刚在arXiv上放出的最新元学习综述：[《Meta-Learning in Neural Networks: A Survey》](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2004.05439)。

*请大家同时关注我的微信公众号：王晋东不在家。*

## 背景

一个传统的机器学习模型这样定义：在一个给定的数据集上![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BD%7D)学习一个目标函数![[公式]](https://www.zhihu.com/equation?tex=f)，使得其代价最小化。如果我们用![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta)来表示函数的参数，![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D)来表示代价函数（损失函数），则一个通用的机器学习目标可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta%5E%7B%5Cstar%7D%3D%5Carg+%5Cmin+_%7B%5Ctheta%7D+%5Cmathcal%7BL%7D%28%5Cmathcal%7BD%7D+%3B+%5Ctheta%29)

其中![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta%5E%5Cstar)表示最优的模型参数。

我们可以看到，给定一个任务（数据集），我们总是可以用上述过程学习一个最优的函数。此过程非常通用。

但是可以预见的是，如果任务数量非常庞大，或者学习过程非常缓慢，则我们的机器学习过程便捉襟见肘。

因此，很自然的一个想法是：如何最大限度地利用之前学习过的任务，来帮助新任务的学习？

迁移学习是其中一种有效的思维。简单来说，迁移学习强调我们有一个已学习好的源任务，然后将其直接应用于目标任务上，再通过在目标任务上的微调，达到学习目标。这已经被证明是一种有效的学习方式。

迁移学习过程可以被表示为：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta%5E%5Cstar+%3D+%5Carg+%5Cmin_%7B%5Ctheta%7D+%5Cmathcal%7BL%7D%28%5Ctheta%7C%5Ctheta_0%2C%5Cmathcal%7BD%7D%29)

其中的![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_0)是之前任务的超参数。这个过程也就是常用的微调（fine-tune）过程。

那么有没有别的学习模式？

## 元学习的概念

元学习（meta-learning, 也叫learning to learn）是一种非常有效的学习模式。与迁移学习的目标类似，元学习也强调从相关的任务上学习经验，以帮助新任务的学习。

不同点是，元学习是一种更为通用的模式，其核心在于“元知识”（meta-knowledge）的表征和获取。我们可以理解为，这种元知识是一大类任务上具有的通用知识，是通过某种学习方式可以获得的。其具有在这类任务上，非常强大的表征能力，因此可以被泛化于更多的任务上去。

为了获取元知识，通常，元学习假定我们可以获取一些任务，它们采样自任务分布![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cmathcal%7BT%7D%29)。我们假设可以从这个任务分布中采样出![[公式]](https://www.zhihu.com/equation?tex=M)个源任务，表示为![[公式]](https://www.zhihu.com/equation?tex=%5Cmathscr%7BD%7D_%7B%5Ctext+%7Bsource%7D%7D%3D%5Cleft%5C%7B%5Cleft%28%5Cmathcal%7BD%7D_%7B%5Ctext+%7Bsource%7D%7D%5E%7B%5Ctext+%7Btrain%7D%7D%2C+%5Cmathcal%7BD%7D_%7B%5Ctext+%7Bsource%7D%7D%5E%7B%5Ctext+%7Bval%7D%7D%5Cright%29%5E%7B%28i%29%7D%5Cright%5C%7D_%7Bi%3D1%7D%5E%7BM%7D)，其中两项分别表示在一个任务上的训练集和验证集。通常，在元学习中，它们又被称为**支持集（support set）** 和 **查询集（query set）**。

我们将学习元知识的过程叫做**meta-train**过程，它可以被表示为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cphi%5E%7B%5Cstar%7D%3D%5Carg+%5Cmax+_%7B%5Cphi%7D+%5Clog+p%5Cleft%28%5Cphi+%7C+%5Cmathcal%7BD%7D_%7B%5Ctext+%7Bsource%7D%7D%5Cright%29)

其中的![[公式]](https://www.zhihu.com/equation?tex=%5Cphi)表示元知识学习过程的参数。

为了验证元知识的效果，我们定义一个**meta-test**过程：从任务分布中采样![[公式]](https://www.zhihu.com/equation?tex=Q)个任务，构成meta-test数据，表示为![[公式]](https://www.zhihu.com/equation?tex=%5Cmathscr%7BD%7D_%7B%5Ctext+%7Btarget+%7D%7D%3D%5Cleft%5C%7B%5Cleft%28%5Cmathcal%7BD%7D_%7B%5Ctext+%7Btarget+%7D%7D%5E%7B%5Ctext+%7Btrain+%7D%7D%2C+%5Cmathcal%7BD%7D_%7B%5Ctext+%7Btarget+%7D%7D%5E%7B%5Ctext+%7Btest+%7D%7D%5Cright%29%5E%7B%28i%29%7D%5Cright%5C%7D_%7Bi%3D1%7D%5E%7BQ%7D)。于是，在meta-test过程时，我们便可以将学到的元知识应用于meta-test数据来训练我们真正的任务模型：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta%5E%7B%2A%28i%29%7D%3D%5Carg+%5Cmax+_%7B%5Ctheta%7D+%5Clog+p%5Cleft%28%5Ctheta+%7C+%5Cphi%5E%7B%2A%7D%2C+%5Cmathcal%7BD%7D_%7B%5Ctext+%7Btarget+%7D%7D%5E%7B%5Ctext+%7Btrain+%7D%28i%29%7D%5Cright%29)

值得注意的是，上式中我们是在针对每个任务，自适应地训练其参数，这也就完成了泛化的过程。

## 历史

Meta-learning的历史追溯到1987年。在这一年里，J. Schmidhuber和G. Hinton独立地在各自的研究中提出了类似的概念，被广泛认为是meta-learning的起源：

- J. Schmidhuber提出了元学习的整体形式化框架，提出了一种**self-referential** learning模式。在这个模式中，神经网络可以接收它们自己的权重作为输入，来输出目标权重。另外，模型可以自己通过进化算法来进行自我学习。
- G. Hinton提出了**fast weights**和**slow weights**的概念。在算法迭代过程中，slow weights获取知识较慢，而fast weights可以更快地获取知识。这一过程的核心是，fast weights可以回溯slow weights的值，从而进行全局性指导。

两篇工作都对meta-learning的产生有着直接推动作用。以今天的视角来看，J. Schmidhuber的版本更像是meta-learning的形式化定义，而G. Hinton的版本则更像是定义了meta-learning的bi-level优化过程。

后来，Bengio分别在1990和1995年提出了通过meta-learning的方式来学习生物学习规则，而J. Schmidhuber继续在他后续的工作中探索self-referential learning。S. Thrun在1998年的工作中首次介绍了**learning-to-learn**的概念，并将其表示为实现meta-learning的一种有效方法。

S. Hochreiter等人首次在2001年的研究中，用神经网络来进行元学习。后来，S. Thrun等人又在1998年的工作中，重新将现代深度网络引入meta-learning，这也奠定了如今用深度学习进行meta-learning的基础。

## 相关研究领域

- **迁移学习**。迁移学习（transfer learning）强调从已有任务中学习新任务的思维。与meta-learning相比，迁移学习更强调的是这种学习问题，而meta-learning则更侧重于学习方法。因此，二者不是完全没有联系，也并不是完全相等，取决于看待问题的角度。在很多情况下，二者的终极目标是一致的。
- **领域自适应和领域泛化**。Domain adaptation和domain generalization这两种学习模式是迁移学习的子集。与meta-learning显著区别的是，二者没有meta-objective，也就是说，没有bi-level优化的过程。当然最近有一些工作试图将meta-learning引入来解决这两个领域的问题。
- **终身学习和持续学习**。终身学习（lifelong learning）和持续学习（continual learning）强调在一个任务上进行连续不断地学习，而meta-learning则侧重于在多个任务上学习通用的知识，有着显著区别。
- **多任务学习**。多任务学习（multi-task learning）指从若干个相关的任务中联合学习出最终的优化目标。Meta-learning中的任务是不确定的，而多任务中的任务就是要学习的目标。
- **超参数优化**。严格来说，超参数优化（hyperparameter optimization）侧重学习率、网络架构等参数的设计，其是meta-learning的一个应用场景。

除此之外，meta-learning还与**贝叶斯层次优化**（Hierarchcal Bayesian model）和**AutoML**有着相近的联系，但这些模型的侧重和表征点，与meta-learning有所不同。特别地，贝叶斯模型从生成模型的角度，为meta-learning提供了一个有效的看问题角度。

## 元学习的基本问题

元学习的基本问题可以分成三大类：

- **元知识的表征**（meta-representation）。元知识应该如何进行表征，这回答了元学习的最重要问题，即学习什么的问题。
- **元学习器**（meta-optimizer）。即有了元知识的表征后，我们应该如何选择学习算法进行优化，也就回答了元学习中的如何学习的问题。
- **元目标**（meta-objective）。有了元知识的表征和学习方法后，我们应该朝着怎样的目标去进行学习？这回答了元学习中为什么要这样学习的问题。

围绕这三大基本问题，近年来元学习领域出现了诸多研究成果。按照此分类方法，meta-learning的研究框架可以由下图清晰地给出：



![img](https://pic2.zhimg.com/80/v2-7c0000260c89a33f1e34b7e4d9ddc52d_1440w.png)

元学习分类概览



### 元知识的表征

1. **初始化参数**。由meta-train任务给出模型的初始化参数，然后针对初始化参数进行二次学习，例如MAML等方法。此类方法存在的问题是：一组初始化参数是否具有足够的元知识表征能力，或者这些参数是否会受限于特定任务。
2. **优化器**。深度学习模型通常可以由SGD、Adam等常用的optimizer进行优化，这些优化器是固定不变的。那么，能否从相关的任务中，自动地学习适合于本任务的优化器？有一些工作，例如learning to optimize等，在这方面进行了尝试。
3. **黑盒模型**。这类工作的出发点非常直接：假定待学习模型的参数是由另一个网络所学习而来的，例如CNN、RNN等。比较著名的工作有Memory-augmented neural networks等。
4. **度量学习（嵌入函数）**。这类方法可以看作是上述黑盒模型的一个特例，即通过学习一个特征嵌入函数，使得在这个表征下，仅通过简单的相似度度量就可以进行分类。正因如此，这类方法目前只在few-shot的情景下适用。
5. **损失**。通过自适应地学习损失函数来进行meta-learning。
6. **网络结构**。通过设计额外的学习器来自动生成可用的网络结构，这一部分与AutoML很相关。
7. **注意力模块**。
8. **网络的特定层和模块**。
9. **超参数**。
10. **数据增强**。数据增强已成为深度学习领域的常用方法，如何用meta-learning来帮助进行数据增强，是一个非常有前景的领域。
11. **样本权重选择与课程学习**。如何选择mini-batch大小、如何设定样本权重，这设计到课程学习（curriculum learning）的研究，显然这些都可以通过meta-learning进行学习。
12. **数据集、标签、环境**。回归meta-learning本身的定义，为什么不自适应地学习数据集、标签和环境的切分和构造呢？
13. 除此之外，激活函数、卷积核、池化等，都可以用meta-learning进行学习。

纵观上述元知识的表征，我们可以说meta-learning其实设计到了一个机器学习过程的方方面面。因此，meta-learning也被视为非常有潜力的发展方向之一。

### 元学习器

有了meta-representation后，下一步便是meta-optimizer的学习，也就是说，如何进行优化的问题。

1. **梯度优化**。这是非常常用的手段，例如MAML等方法都采用了在bi-level优化过程中梯度回传的方式来进行优化。这里的问题包括：如何在一个较长的计算图中实现有效的梯度回传，以及如何避免计算过程中的梯度变差，并且如何base-learner含有不可微分的损失时如何计算。
2. **强化学习**。当base-learner含有不可微分的过程时，RL便是一个强有力的工具。此时涉及到如何将强化学习引入优化过程，使其变得更高效的问题。
3. **进化算法**。另一种非常有效的途径是进化算法，有别于深度学习，进化算法并没有BP的过程，同时也有自身的一些优点和缺点。

### 元目标

设定好元知识和元学习器后，最后一步便是整个meta-learning的学习目标。

1. **多样本 还是 小样本**。取决于目标任务的设定，meta-objective可以有针对性地被设计为多样本还是小样本的学习模式。
2. **快速的 还是 渐进式的结果**。许多meta-learning方法都要求在meta-test集上快速进行泛化。而一些RL的应用也需要渐进式的学习结果。
3. **多任务 还是 单任务**。
4. **在线 还是 离线**。
5. 另外的设计思路：有噪声样本的情况下如何学习，以及有domain-shift的情况下如何学习等。

## 元学习的应用

元学习在机器学习的诸多场景下都有着广泛的应用。

- 计算机视觉中的**小样本学习（few-shot learning）**，包括分类、检测、分割、关键点定位、图像生成、密度估计等。可以说小样本的这个任务设定完美契合了meta-learning的学习过程。所以我们看到，绝大多数小样本研究都采用了meta-learning的学习方法。
- **强化学习**。一个很自然的扩展便是强化学习，在RL中的探索（exploration）和利用（exploitation），策略梯度等都有着广泛应用。
- **仿真（sym2real）**。由虚拟环境生成一些样本学习泛化的学习器，然后部属到真实环境中。
- **神经结构搜索（NAS）**。
- **贝叶斯元学习**。这个步骤非常有趣，它涉及到用贝叶斯的视角来对元学习进行重新表征，可以发现不一样的研究思维，对解决问题非常有帮助。
- **无监督元学习**。
- **终身学习、在线学习、自适应学习**。
- **领域自适应和领域泛化**。
- **超参数优化**。
- **自然语言处理**。
- **元学习系统**，等等。

其实我们可以看到，因为元学习是机器学习的一种学习方法，因此，可以说，机器学习的绝大多数问题和研究领域，都可以与meta-learning进行结合，碰撞出不一样的火花。

## 现存的挑战

1. **泛化能力**。现有的meta-learning方法在一些任务上训练，在新的任务上测试。直接的假设是：训练和测试数据服从同一个分布。所以直接的问题是：如果meta-train和meta-test不是来自一个分布，如何做meta-learning？或者说，meta-train数据本身就包含了来自多个分布的数据，如何做meta-train？
2. **多模态的任务来源**。现有的meta-learning方法假设训练和测试任务来自单一模态的数据，如果是多模态如何进行元学习？
3. **任务家族**。特定任务下只有特定的任务家族可以用元学习，这大大限制了知识的表征和传播。
4. **计算复杂度**。通常来说meta-learning都涉及到一个bi-level的优化，这导致其计算非常复杂。如何提高计算效率是一个难点。
5. **跨模态迁移和异构的任务**。

## 结论

元学习是机器学习领域一种非常重要的方法，其贯穿整个机器学习过程的方方面面，并且在不同领域得到了长足的应用。其仍然存在一些问题有待解决，这将成为研究者们今后的方向。

=================