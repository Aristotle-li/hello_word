> 题目：Neural Architecture Optimization
>
> 来源： NeurIPS2018
>
> 作者：Renqian Luo  Fei Tian  Tao Qin Enhong Chen  Tie-Yan Liu
>
> 性能： 1.93% test set error rate on CIFAR-10.   56.0 perplexity on PTB dataset 

### motivation：

directly searching the best architecture within discrete space is inefficient given the exponentially growing search space with the number of choices increasing. 

### idea:

使用由离散字符串符号组成的序列来描述 CNN 或 RNN 架构。

* encode将神经网络结构嵌入/映射到一个连续的空间中：使用LSTM作为encode，将每个节点的每个分支的三个标记（包括它选择作为输入的节点索引，操作类型和操作大小）编码成长度T的序列，其中一个架构可表示为$x = \left\{x1 , · · · , xT \right\}$，将离散的架构编码成字符串序列作为输入，使用LSTM的隐状态作为架构的连续表示（embedding）
*  predictor将网络的连续表示作为输入并预测其精度：the optimization of f aims at minimizing the least-square regression loss 
*  decode将网络的连续表示映射回其体系结构：The decoder is an LSTM model equipped with an attention mechanism that makes the exact recovery easy

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210429092721469.png" alt="image-20210429092721469" style="zoom:67%;" />

采用联合训练，性能预测损失会充当一个regularizer，迫使未优化的编码器进入 trivial state ：![image-20210428144336012](/Users/lishuo/Library/Application Support/typora-user-images/image-20210428144336012.png)

### DARTS:

 continuous space  are different： in DARTS it is the mixture weights and in NAO it is the embedding of neural architectures.

### 改进方向：

Embedding 空间的构建只考虑了结构对应点对的对称性，没有考虑其他的关系。



### 词句：

1、with a significantly reduction of computational resources. 在大大减少了计算资源的前提下

2、 On one hand, similar to the distributed representation of natural language [36, 26], the continuous repre- sentation of an architecture is more compact and efficient in representing its topological information; On the other hand, optimizing in a continuous space is much easier than directly searching within discrete space due to better smoothness.

3、we can achieve improved efficiency in discovering powerful convolutional and recurrent architectures, 

4、 exploring the search space incrementally and sequentially

5、 is denoted as 表示为

6、 Empirically we found that acting in this way brings non-trivial gain 根据经验，我们发现以这种方式进行操作会带来不小的增益

7、 iterate such process for several rounds, 迭代这一过程几轮

8、Encoder：The encoder of NAO takes the string sequence describing an architecture as input, and maps it into a continuous space E. 

### 学习：

##### embedding：

将离散变量转为连续向量表示的一个方式。不仅减少离散变量的空间维数（区别于one-hot），同时还可以有意义的表示该变量（比如通过学习在embedding空间相似变量距离更近）。

1. 在 embedding 空间中查找最近邻，这可以很好的用于根据用户的兴趣来进行推荐。
2. 作为监督性学习任务的输入。
3. 用于可视化不同离散变量之间的关系。

##### Regularization：

拟合过程中通常都倾向于让权值尽可能小，最后构造一个所有参数都比较小的模型，最小二乘损失相当于L2正则化项。

##### 降低复杂度：

训练好的网络尽可能重用：

* 一种思路是利用网络态射从小网络开始，然后做加法
* 另一种思路是从大网络开始做减法，如One-Shot Architecture Search方法（ENAS、DARTS）