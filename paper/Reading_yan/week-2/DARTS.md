> 题目：DARTS: DIFFERENTIABLE ARCHITECTURE SEARCH
>
> 来源： ICLR 2019
>
> 作者：Hanxiao Liu   Karen Simonyan
>
> 缺点：high GPU memory consumption issue

在学习中学习学习

学习大进程纠正学习大路线

学习小行为学习局部小规律

### 解决的问题：

实现了一种可微的NAS，搜索时间比应用RL和进化算法快了好几个数量级

### idea：

直接把整个搜索空间看成supernet，学习如何sample出一个最优的subnet。这里存在的问题是子操作选择的过程是离散不可导，故DARTS 将单一操作子选择松弛软化为 softmax 的所有子操作权值叠加，将搜索空间弱化为一个**连续的**空间结构，并且将问题建模成了一个bilevel optimization problem，以此可以使用梯度下降进行性能优化。

【把搜索空间连续松弛化，每个edge看成是所有子操作的混合（softmax权值叠加）】

### 加速方法：

Several approaches for speeding up have been proposed, such as imposing a particular structure of the search space (Liu et al., 2018b;a), weights or performance prediction for each individual architecture (Brock et al., 2018; Baker et al., 2018) and weight sharing/inheritance across multiple architectures (Elsken et al., 2017; Pham et al., 2018b; Cai et al., 2018; Bender et al., 2018

### 传统方法为什么慢：

architecture search is treated as a black-box optimization problem over a discrete domain, which leads to a large number of architecture evaluations required.

### detail：

传统的连续搜索空间搜索方法：搜索一种特殊的结构(Filters Shape、Branch Pattern)，而DARTS是搜索一种完整的网络结构

架构（或其中的单元）的计算过程表示为有向无环图。然后，我们为搜索空间引入一个简单的连续松弛方案，这导致了针对体系结构及其权重的联合优化是可微分的学习目标。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210425143937902.png" alt="image-20210425143937902" style="zoom:67%;" />



* 结构：

  * 有向无环图的每一个节点$x(i)$：	对应一种潜在的表示（比如feature map）

  * each directed edge $(i,j)$： 	is associated with some operation $o(i,j)$，

  * Each intermediate node is computed based on all of its predecessors：$x(j) = 􏰄 \sum _{i<j}o(i,j)(x(i))$

  * $O$作为一个候选操作的集合（含有conv，maxpool等），每一个具体的操作可以表示为应用到$x$ 的$o()$函数，we relax the categorical choice of a particular operation to a softmax over all possible operations:

    <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210425161547386.png" alt="image-20210425161547386" style="zoom:50%;" />

    一对节点$（i，j）$的混合权重操作参数化为一组$O$维的$α(i、 j）$向量，架构搜索的任务简化为 learning a set of continuous variables $ \alpha=\left\{\alpha (i,j)\right\}$

    最后通过对每个edge 取softmax中最大概率(即向量$α(i、 j）$中最大的数)操作取代连续的混合操作，得到离散的体系结构。

  * The task of learning the cell therefore reduces to learning the operations on its edges.

* 优化：

  * 

  * This implies a bilevel optimization problem 

    <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210425165516921.png" alt="image-20210425165516921" style="zoom:50%;" />

* 算法：

  ![image-20210425165821450](/Users/lishuo/Library/Application Support/typora-user-images/image-20210425165821450.png)

例子：

![image-20210425174429150](/Users/lishuo/Library/Application Support/typora-user-images/image-20210425174429150.png)

### 改进方向：

1、 example, the current method may suffer from discrepancies between the continuous architecture encoding and the derived discrete architecture.可以通过对softmax温度进行退火（使用适当的时间表）to enforce one-hot selection  来缓解这种情况。

连续的最优解直接粗暴的对应到离散可能有问题，是否可以参考，线性规划和整数规划的关系，用优化整数规划的分支定界法，通过解一个个线性规划问题来逼近离散的最优解。

2、基于在搜索过程中学习到的共享参数来研究性能感知架构派生方案也将很有趣

3、Differentiable NAS can reduce the cost of GPU hours via a continuous representation of network architecture but 

suffers from the high GPU memory consumption issue.     (grow linearly w.r.t. candidate set size)

### 词句:

1、This paper addresses the scalability challenge of architecture search by formulating the task in a differentiable manner.

本文通过以可微的方式描述任务来解决架构搜索的可伸缩性挑战。

2、The best existing architecture search algorithms are computationally demanding despite their remark- able performance.

现有最佳的体系结构搜索算法尽管性能出色，但对计算的要求却很高。

3、approach the problem解决问题

4、using orders of magnitude less computation resources 使用较少数量级的计算资源

5、yet it is generic enough handle both convolutional and recurrent architectures.

6、as opposed to 与... 形成对照/相反 

we attribute to the use of gradient-based optimization as opposed to non-differentiable search techniques.

7、be comparable to 与...相当

8、based on bilevel optimization

9、is applicable to 适用于

10、 The learned cell could either be stacked to form a convolutional network or recursively connected to form a recurrent network.

11、which is related in a sense that the architecture α could be viewed as a special type of hyperparameter, although its dimension is substantially higher than scalar-valued hyperparameters such as the learning rate, and it is harder to optimize.

从某种意义上说α 可以看作是一种特殊类型的超参数，尽管它的维数远高于学习率等标量值超参数，而且更难优化。

12、where the computation procedure for an architecture (or a cell in it) is represented as a directed acyclic graph. We then introduce a simple continuous relaxation scheme for our search space which leads to a differentiable learning objective for the joint optimization of the architecture and its weight

然后，我们为我们的搜索空间引入一个简单的连续松弛方案，这导致了针对体系结构及其权重的联合优化的可微分学习目标（第2.2节）





- 