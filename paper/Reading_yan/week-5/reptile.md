> 题目：On First-Order Meta-Learning Algorithms
>
> 作者：Alex Nichol and Joshua Achiam and John Schulman OpenAI
>
> key：并不是找global minima对应的weight（pre-training），而是找对于task distribution里面random sample出来一个task，经过k步update，能得到global minima的initialization weight

### 解决的问题：

1、解决了MAML计算量太大的问题

2、first-order meta-learning algorithms perform well on some well-established benchmarks for few-shot classification

### idea：

通过忽略二阶导数，得到对于MAML的一阶近似，同时Reptile doesn’t need a training-test split for each task

### detail：

因为经典的pre-training方法并不能保证学习一个有利于fine-tuning的初始化参数，所以MAML被提出来，

FOMAML是对MAML的简化 ：

计算细节：using a first-order approximation

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518222919447.png" alt="image-20210518222919447" style="zoom:50%;" />

 <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518224035247.png" alt="image-20210518224035247" style="zoom: 33%;" />

结果就是：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518224315214.png" alt="image-20210518224315214" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518224646508.png" alt="image-20210518224646508" style="zoom: 50%;" />

reptile：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210519121345308.png" alt="image-20210519121345308" style="zoom:50%;" />

对比：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210519121723601.png" alt="image-20210519121723601" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210519122217343.png" alt="image-20210519122217343" style="zoom:50%;" />

reptile看起来和pre-training很相似，Instead, the update includes important terms coming from second-and-higher derivatives of Lτ ,

数学推导：

1、

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210519150626078.png" alt="image-20210519150626078" style="zoom:67%;" />

2、we argue that Reptile converges towards a solution φ that is close (in Euclidean distance) to each task τ’s manifold of optimal solutions. 



### 词句

1、be explained as  被解释为   emulate  模仿

2、may be computationally intractable  可能在计算上很难

3、A variety of different approaches to meta-learning have been proposed, each with its own pros and cons.

人们提出了多种不同的元学习方法，每种方法都有各自的优缺点。

4、this classic pre-training approach has no guarantee of learning an initialization that is good for fine-tuning, and ad-hoc tricks **the  are required for** good performance. 这种经典的预训练方法不能保证学习一个有利于微调的初始化，而要获得良好的性能需要一些特殊的技巧。

5、 proposed an algorithm called MAML, which directly optimizes performance with respect to this initialization—differentiating through the fine-tuning process. 提出了一种称为MAML的算法，该算法通过微调过程直接优化与此初始化微分有关的性能

6、In this approach, the learner   **falls back on** a sensible gradient-based learning algorithm even when it receives out-of-sample data, **thus allowing it to** generalize **better than** the RNN-based approaches

在这种方法中，学习者即使在接收到样本外的数据时也会**依赖**一种基于梯度的学习算法，从而使其比基于RNN的方法具有更好的泛化能力