

> 题目：Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting
>
> 来源：IJCAI-18
>
> 作者：Peking University, Beijing

### motivation

交通预测是一种典型的时间序列预测问题，即利用数据之中所蕴含的时间、空间信息来对未来的区域内不同道路的交通流量进行预测。



要进行交通预测，首先要定义交通预测所需的时空图。本文使用39000个传感器来收集数据，每个传感器构成图中的一个节点。每30s进行一次数据采样，每5min的数据聚合成一张图，每 ![[公式]](https://www.zhihu.com/equation?tex=M) 张图构成一条数据（时空图)，即求得在已知t-M+1到t时间点内的交通流量，去求t+1到t+H时间点的交通流量。如图1所示：



**![img](https://pic2.zhimg.com/80/v2-db67c5cc6196353becaa27f50b9f5919_1440w.jpg)**

图1 时空图

用$G_t$表示时空图，则第 t 个时间步的图定义为 ![[公式]](https://www.zhihu.com/equation?tex=G_t%3D%28V_t%2C+E%2C+W%29) 。![[公式]](https://www.zhihu.com/equation?tex=V_t) 表示第 t 张图上的点集，不同时间步之间点的数量不变，但每个节点的特征改变。表示边集，通过从属关系、方向、出发点——目的地构成有向图。 ![[公式]](https://www.zhihu.com/equation?tex=W) 表示邻接矩阵，不同时间步的 ![[公式]](https://www.zhihu.com/equation?tex=W) 不变：

![[公式]](https://www.zhihu.com/equation?tex=w_%7Bij%7D+%3D%5Cleft%5C%7B+%5Cbegin%7Baligned%7D+exp%28%5Cfrac%7Bd_%7Bij%5E2%7D+%7D%7B%CF%83%5E2%7D%29%EF%BC%8Ci%E2%89%A0j%E4%B8%94exp%28%5Cfrac%7Bd_%7Bij%5E2%7D+%7D%7B%CF%83%5E2%7D%29%E2%89%A5%CF%B5%5C%5C+0%EF%BC%8C%E5%85%B6%E4%BB%96%E6%83%85%E5%86%B5+%5Cend%7Baligned%7D%5Cright.)

其中 ![[公式]](https://www.zhihu.com/equation?tex=d_%7Bij%7D) 表示节点 ![[公式]](https://www.zhihu.com/equation?tex=i) 和节点 ![[公式]](https://www.zhihu.com/equation?tex=j) 之间的距离， ![[公式]](https://www.zhihu.com/equation?tex=%CF%83) 和 ![[公式]](https://www.zhihu.com/equation?tex=%CF%B5) 用于调整 ![[公式]](https://www.zhihu.com/equation?tex=W) 的分布和稀疏度，本文所用数据集分别设置为10和0.5。

我们可以把交通流预测问题看作一个时空序列问题，

$\tilde v_{t+1},\cdots ,\tilde v_{t+H}=argmax_{ v_{t+1},\cdots,v_{t+h}}logP( v_{t+1},\cdots, v_{t+H}| v_{t-M+1},\cdots,v_t$









<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210615104803067.png" alt="image-20210615104803067" style="zoom:50%;" />










其中 ![[公式]](https://www.zhihu.com/equation?tex=v_t%E2%88%88R%5E%7Bn%C3%97C_i+%7D) 表示时间步 ![[公式]](https://www.zhihu.com/equation?tex=t) 时$n$  个观测点的观测向量（ ![[公式]](https://www.zhihu.com/equation?tex=C_i) 是数据的特征向量长度)。


数学和物理知识建模：由于使用算法复杂度太高、计算成本大、以及过多的对现实理想化的假设，基本上已经被抛弃。

传统统计学、机器学习方法：虽然可以在中短期交通流预测中实现较好的预测，但是无法再长期预测中达到较好的结果。

深度学习模型：传统CNN和LSTM（GRU）来分别学习空间和时间特征，然而问题就在于传统CNN只能处理标准的网格数据（例如图片），而LSTM因为是迭代训练容易发生误差积累，基于RNN的方法难以捕捉短时剧烈交通流的变化u，而且计算量大难训练。

通过时空图建模：时空图卷积网络，用于交通预测任务。该架构包含多个时空卷积块，是图卷积层和卷积序列学习层的组合，用于对空间和时间依赖性进行建模。纯粹采用卷积结构来设计神经网络进行时空预测，不但参数更少，训练的还更快了

### detail





STGCN网络整体架构如图2所示，网络输入是 ![[公式]](https://www.zhihu.com/equation?tex=M)个时间步的图的特征向量 ![[公式]](https://www.zhihu.com/equation?tex=X%E2%88%88R%5E%7BM%C3%97n%C3%97C_i+%7D) （论文中 ![[公式]](https://www.zhihu.com/equation?tex=C_i%3D1)）以及对应的邻接矩阵 ![[公式]](https://www.zhihu.com/equation?tex=W%E2%88%88R%5E%7Bn%C3%97n%7D)，经过两个时空卷积块和一个输出层，输出 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bv%7D%E2%88%88R%5En) 来预测第 ![[公式]](https://www.zhihu.com/equation?tex=t) 个时间步后某个时间步特征。

![img](https://pic3.zhimg.com/80/v2-030389b5592ad95cc19e3546ae70510e_1440w.jpg)

图2 图时空网络整体架构

### 1、时域卷积块

每个时空卷积块由两个时域卷积块和一个空域卷积块组成。其中时域卷积块如图2最右侧所示，每个节点处的输入 ![[公式]](https://www.zhihu.com/equation?tex=X%E2%88%88R%5E%7BM%C3%97C_i+%7D)，沿着时间维度进行一维卷积，卷积核 ![[公式]](https://www.zhihu.com/equation?tex=%CE%93%E2%88%88R%5E%7BK_t%C3%97C_i+%7D)，个数为 ![[公式]](https://www.zhihu.com/equation?tex=2C_o)，从而得到 ![[公式]](https://www.zhihu.com/equation?tex=%5BP+Q%5D%E2%88%88R%5E%7B%28M-K_t%2B1%29%C3%972C_o+%7D)(P, Q为通道数一半)。

然后进行 ![[公式]](https://www.zhihu.com/equation?tex=GLU)激活：

![[公式]](https://www.zhihu.com/equation?tex=%CE%93%2A_%CF%84+X%3DP%E2%8A%99%CF%83%28Q%29%E2%88%88R%5E%7B%28M-K_t%2B1%29%C3%97C_o+%7D)

其中，P, Q分别是GLU的门输入，$\odot$  表示哈达玛积（即元素对应相乘），sigmoid门$\sigma(Q)$ 控制当前状态的哪个输入P与时间序列中的组成结构和动态方差相关。这里GLU门以及作者论文中提到的时间卷积网络中的短路连接（residual connection）实际上都是为了防止梯度消失、而设计的，对于一张完整的时空图：输入 ![[公式]](https://www.zhihu.com/equation?tex=X%E2%88%88R%5E%7BM%C3%97n%C3%97C_i+%7D)，输出 ![[公式]](https://www.zhihu.com/equation?tex=Y%E2%88%88R%5E%7B%28M-K_t%2B1%29%C3%97n%C3%97C_o+%7D)。

### 2、空域卷积块

空域卷积在每个时间步的图上进行（不在时间步之间进行）。输入 ![[公式]](https://www.zhihu.com/equation?tex=X%E2%88%88R%5E%7Bn%C3%97C_i+%7D)，按照ChebGCN，输出

![[公式]](https://www.zhihu.com/equation?tex=Y%3D%5Csum_%7Bi%3D0%7D%5E%7BK-1%7D%CE%B8_i+T_i+%28%5Ctilde%7BL%7D%29X)

其中 ![[公式]](https://www.zhihu.com/equation?tex=T_i+%28x%29%3D2xT_%7Bi-1%7D+%28x%29-T_%7Bi-2%7D+%28x%29)， ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde+L%3D%5Cfrac%7B2L%7D%7B%CE%BB_%7Bmax%7D+%7D-I_n)， ![[公式]](https://www.zhihu.com/equation?tex=L%3DI_n-D%5E%7B-1%2F2%7D+AD%5E%7B-1%2F2%7D)， ![[公式]](https://www.zhihu.com/equation?tex=A) 是邻接矩阵，论文中 ![[公式]](https://www.zhihu.com/equation?tex=K%3D3) 。卷积核 ![[公式]](https://www.zhihu.com/equation?tex=%CE%98%E2%88%88R%5E%7BK%C3%97C_i+%7D)，个数为 ![[公式]](https://www.zhihu.com/equation?tex=C_o) 。

对于一张完整的时空图：输入 ![[公式]](https://www.zhihu.com/equation?tex=X%E2%88%88R%5E%7BM%C3%97n%C3%97C_i+%7D)，输出 ![[公式]](https://www.zhihu.com/equation?tex=Y%E2%88%88R%5E%7BM%C3%97n%C3%97C_o+%7D) 。

### 3、输出层

如图二所示，时空卷积块包括两个时间卷积层和一个空间卷积层。处于中间的空间卷积层，承接两个时间卷积层，可取得使空间状态能够从图卷积到时间卷积快速传播。这种网络结构也助于充分应用瓶颈策略（bottleneck strategy)来实现通过压缩通道C来进行规模缩放和特征压缩（目的是为了减少学习参数）。另外，每个时空块都使用层归一化（batchnormalization）抑制过拟合。
时空块的输入和输出都是三维张量。块的输入和输出通过下式计算：



其中，和是块的上下两个时间核，是图卷积的谱核，是激活函数。

根据时域卷积块的一维卷积，每经过一个时空卷积块，数据在时间维度的长度减小 ![[公式]](https://www.zhihu.com/equation?tex=2%28K_t-1%29) 。所以经过两个时空卷积块后，输出 ![[公式]](https://www.zhihu.com/equation?tex=Y%E2%88%88R%5E%7B%28M-4%28K_t-1%29%29%C3%97n%C3%97C_o+%7D) 。

输出层包括一个时域卷积层和一个全连接层，时域卷积层的卷积核大小 ![[公式]](https://www.zhihu.com/equation?tex=%CE%93%E2%88%88R%5E%7B%28M-4%28K_t-1%29%29%C3%97C_o%7D) ，个数为 ![[公式]](https://www.zhihu.com/equation?tex=C_o) ，将输出映射到 ![[公式]](https://www.zhihu.com/equation?tex=Z%E2%88%88R%5E%7Bn%C3%97C_o+%7D)。全连接层 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+v+%3DZw%2Bb)，其中 ![[公式]](https://www.zhihu.com/equation?tex=w%E2%88%88R%5E%7BC_o+%7D)，于是输出 ![[公式]](https://www.zhihu.com/equation?tex=v+%CC%82%E2%88%88R%5En) 。

损失函数是预测值和真实值的距离度量：

![[公式]](https://www.zhihu.com/equation?tex=L%28v+%CC%82%3B+W_%CE%B8+%29%3D%E2%80%96v+%CC%82%28v_%7Bt-M%2B1%7D%2C+%E2%8B%AF%2Cv_t%2CW_%CE%B8+%29-v_%7Bt%2B1%7D+%E2%80%96%5E2)

其中 ![[公式]](https://www.zhihu.com/equation?tex=W_%CE%B8) 是所有可训练参数， ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+v+) 是预测值， ![[公式]](https://www.zhihu.com/equation?tex=v_%7Bt%2B1%7D) 是真实值。

