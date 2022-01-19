> 题目：PROXYLESSNAS: DIRECT NEURAL ARCHITECTURE SEARCH ON TARGET TASK AND HARDWARE
>
> 来源： ICLR 2019
>
> 作者：Han Cai, Ligeng Zhu, Song Han
>
> 性能： 2.08% test set error rate on CIFAR-10.    3.1% better top-1 accuracy  on ImageNet
>
> 结构：get rid of the meta-controller (or hypernetwork) by modeling NAS as a single training process of an over-parameterized network that comprises all candidate paths.
>
> ​		   通过将NAS建模为包含所有候选路径的超参数化网络的单个训练过程，从而摆脱了元控制器（或超网络）。

### 解决的问题：

1、DARTS可以通过连续表示网络体系结构来减少GPU的时间开销，但存在GPU内存消耗高的问题，提出了

2、DARTS在代理任务上训练，进而迁移到目标任务，无法保证在目标数据集是最优的。

3、DARTS为了实现迁移学习，这种方法只搜索几个架构cell，然后重复地堆叠相同的模式，这限制了block 的多样性，从而损害了性能。

### idea：

1、它直接学习目标任务和硬件上的体系结构，而不是使用代理

2、we also remove the restriction of repeating blocks in previous NAS works

3、memory consumption grows linearly w.r.t. the number of choices：binarize the architecture  parameters(1 or 1)。and force only one path to be active at run-time.    propose a gradient-based approach to train these binarized parameters based on BinaryConnect

![image-20210505204351613](/Users/lishuo/Library/Application Support/typora-user-images/image-20210505204351613.png)

network pruning :

相同点：improve the efficiency of neural networks by removing insignificant neurons or channels .

不同点：can not change the topology of the network，focus on layer-level pruning that only modifies the filter (or units) number of a layer 

### detail：

1、超参数化网络的构建同darts

2、We propose a gradient-based algorithm to train these binarized architecture parameters.如图：

![image-20210505213246443](/Users/lishuo/Library/Application Support/typora-user-images/image-20210505213246443.png)

3、

![image-20210506111632852](/Users/lishuo/Library/Application Support/typora-user-images/image-20210506111632852.png)

在运行时内存中只有一条激活路径处于活动状态，因此训练超参数网络的内存需求降低到了训练紧凑模型的相同级别。

4、将从N候选路径选择一条转化成多个二选一的任务：直觉是若某条路径在所有候选路径最优，那么该路径和其他路径两两比较也是最最优的。

### 改进：

直觉是若某条路径在所有候选路径最优，那么该路径和其他路径两两比较也是最最优的

问题：最优的路径一开始可能不是表现最优秀的甚至可能比较差，还没等他发挥潜力就被优化掉了。



### 词句：

1、To address this issue, we consider factorizing the task of choosing one path out of N candidates into multiple binary selection tasks.

2、The intuition is that if a path is the best choice at a particular position, it should be the better choice when solely compared to any other path

3、 Finally, as path weights are computed by applying softmax to the architecture parameters, we need to rescale the value of these two updated architecture parameters by multiplying a ratio to keep the path weights of unsampled paths unchanged.

4、in each update step, one of the sampled paths is enhanced (path weight increases) and the other sampled path is attenuated (path weight decreases) while all other paths keep unchanged. 
