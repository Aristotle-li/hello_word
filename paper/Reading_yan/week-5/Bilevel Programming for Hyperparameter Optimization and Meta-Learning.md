> 题目：Bilevel Programming for Hyperparameter Optimization and Meta-Learning
>
> 来源：PMLR 2018
>
> 作者：Luca Franceschi  Paolo Frasconi  Saverio Salzo Riccardo Grazzi  Massimiliano Pontil 



### 要解决的问题：

both HO and ML essentially **boil down to** nesting two search problems: at the inner level we seek a good hypothesis (as in standard supervised learning) while at the outer level we seek a good configuration (including a good hypothesis space) where the inner search takes place.

HO和ML基本上都归结为双层嵌套搜索问题：

在内部，我们寻找一个好的假设（如标准监督学习）

在外部，我们寻找一个好的配置（包括一个好的假设空间，在其中perform the inner search ）



HO：

在HO中，可用数据与单个任务相关联，并分成训练集（用于调整参数）和验证集（用于调整超参数）类似于NAS

在HO中，外部问题涉及超参数，而内部问题通常是经验损失的最小化。

ML：

在ML中，我们通常对所谓的“few-shot”学习设置感兴趣，其中数据以 short episodes的形式出现（小数据集，每个类只有很少的样本），是从监督任务的常见概率分布中采样的。

在ML中，外部问题可能涉及任务之间的共享表示，而内部问题可能涉及单个任务的分类器。





将神经网络最后一层的权值作为内变量，将表示映射参数化的剩余权值作为外变量。



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210519205005654.png" alt="image-20210519205005654" style="zoom:50%;" />





1、The search space in ML often incorporates choices associated with the hypothesis space and the features of the learning algorithm itself