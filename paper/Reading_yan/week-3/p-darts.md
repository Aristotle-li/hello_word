> 题目：Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation
>
> 来源： ICCV 2019
>
> 作者：Tongji University 、Huawei Noah’s Ark Lab
>
> 性能： 2.50% test set error rate on CIFAR-10.     top-1/5 errors of 24.4%/7.4%   on ImageNet

### 解决的问题

darts的不足：This is arguably due to the large gap between the architecture depths in search and evaluation scenarios.

浅层搜索，深层评估，制约是GPU内存。

### idea

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210507171805993.png" alt="image-20210507171805993" style="zoom:67%;" />

增加深度带来计算量的剧增：search space approximation 

将搜索过程分为多个阶段，并在每个阶段结束时逐步增加网络深度，同时随着深度的增加，根据上一个搜索过程中的得分减少下一过程候选（操作）的数量。

增加深度带来不稳定：search space regularization

算法可能严重偏向于 skip-connect ，因为它往往导致优化过程中最快速的错误衰减，但实际上，更好的选择往往存在于可学习的操作，如卷积。

(i) introduces operation-level Dropout  to alleviate the dominance of skip-connect during training

* 每个stage开始， train the (modified) architecture from scratch，即所有架构参数重新初始化，because several candidates have been dropped .

* skip-layer开始阻塞，在其他路径充分学习后，再同等对待： introduces operation-level Dropout after each skip-connect operation to partially ‘cut off’ the straightforward path through skip-connect, and facilitate the algorithm to explore other operations.但是，如果我们不断地阻塞通过skip-connect的路径，算法会通过给它们分配较低的权值来丢弃它们，这对最终的性能是有害的。To address this contradiction, we gradually decay the Dropout rate during the training process in each search stage

(ii) controls the appearance of skip-connect during evaluation.

* If the number of skip-connects is not exactly M, we search for the M skip-connect operations with the largest architecture weights in this cell topology and set the weights of others to 0, then redo cell construction with modified architecture parameters.

### 改进

分阶段优化，逐步丢弃表现不佳的操作，随着训练操作减少，深度增加使得GPU内存要求不大。但是skip-layer的处理比较简单粗暴，能否也将其作为一个可以学东西的量，既然skip-layer可以单独拎出来是否还有其他可以单独拎出来的，拎出来和不拎出来也可以当做一个优化的目标。

那么结果就是，先固定skip-layer及其类似操作，优化architecture weight，再固定architecture weight，优化skip-layer及其类似操作。

### 词句：

1、Such gap hinders these approaches in their application to more complex visual recognition tasks

2、While a deeper architec- ture requires heavier computational overhead

3、 a better option often resides in learnable operations such as convolution

4、 alleviate the dominance of skip-connect 

5、the emerging field of neural archi- tecture search (NAS)

6、 which contradicts the common sense that deeper networks tend to perform better

7、with respect to

8、First, the computational overhead increases linearly with the depth    首先，计算开销随深度线性增加

9、 The search process is split into multiple stages  搜索过程分为多个阶段

10、 which makes our approach easy to be deployed on regular GPUs  这使得我们的方法很容易部署在常规gpu上

11、 the number of ...  varies from 2 to 4.  ...的数量从2-4不等
