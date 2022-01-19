> 题目：SNAS: STOCHASTIC NEURAL ARCHITECTURE SEARCH
>
> 来源： ICLR 2019
>
> 作者：Sirui Xie, Hehui Zheng, Chunxiao Liu, Liang Lin
>
> 性能： 2.85±0.02%  test set error rate on CIFAR-10.    
>



### 解决的问题：

* 和ENAS相比，SNAS的搜索优化可微分，搜索效率更高，可以在更少的迭代次数下收敛到更高准确率。

* 与其他可微分的方法（DARTS）相比，SNAS直接优化NAS任务的目标函数，搜索结果偏差更小，可以直接通过一阶优化搜索。

- 此外，基于SNAS保持了概率建模的优势，作者提出同时优化网络损失函数的期望和网络正向时延的期望，扩大了有效的搜索空间，可以自动生成硬件友好的稀疏网络。

### idea：

NAS是一个确定环境中的完全延迟奖励的任务：

即ENAS只有在网络结构最终确定后，agent才能获得一个非零得分acc，而在一个网络被完全搭建好并训练及测试之前，agent的每一个动作都不能获得直接的得分奖励。agent只会在整一条动作序列（trajectory）结束之后，获得一个得分。

TD Learning与贡献分配：

强化学习的目标函数，是将来得分总和的期望。从每一个状态中动作的角度来说，agent应该尽量选择长期来说带来最大收益的动作。然而，如果没有辅助的预测机制，agent并不能在每一个状态预测每一个动作将来总得分的期望。TD Learning就是用来解决这个问题，预测每一个动作对将来总得分的贡献的。（1）这种动态规划的局部信息传递带来的风险就是，当将来某些状态的价值评估出现偏差时，它过去的状态的价值评估也会出现问题。而这个偏差只能通过更多次的动态规划来修复。



用损失函数（使输出与每个输入直接相关）替代准确率，就可以使NAS问题可微。具体来说，本文继承了DARTS的思想，但是并不是直接训练所有操作的权值，而是通过决策$Z$对子网络进行采样，并根据所有子网络的表现构建损失函数。具体来说，从母网络中产生子网络，可以通过在母网络的每一条边的所有可能神经变换的结果后乘上一个one-hot向量来实现。而对于子网络的采样，就因此自然转化为了对一系列one-hot随机变量的采样（包含0操作，即不连接这条边）。
<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210511160131527.png" alt="image-20210511160131527" style="zoom:67%;" />

### motivations：



One of the key motivations of SNAS is to replace the feedback mechanism triggered by constant rewards in reinforcement-learning-based NAS with more efficient gradient feedback from generic loss

SNAS的一个重要动机是用更有效的梯度反馈代替基于强化学习的NAS中由恒定奖励触发的反馈机制









### 词句：

1、NAS pipeline comprises architecture sampling, parameter learning, architecture validation, credit assignment and search direction update.

2、Due to the pervasive non-linearity in neural operations, it introduces untractable bias to the loss function.

3、 is reformulated as 

4、but assigns credits to structural de- cisions more efficiently. This credit assignment is further augmented with locally decomposable reward to enforce a resource-efficient constraint

5、pervasive

6、