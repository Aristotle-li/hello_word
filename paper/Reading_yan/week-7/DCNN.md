> 题目：Diffusion-Convolutional Neural Networks
>
> 来源：NIPS2016
>
> 作者：James Atwood and Don Towsley 
>
> College of Information and Computer Science University of Massachusetts

### 解决的问题：

NN4G是将所有邻居特征融合，不同的图对应不同的参数量，提出DCNN，只考虑 $H$ 阶邻居的特征，对应参数量$O(H*F)$，使diffusion-convolutional representation不受输入大小的限制。

### idea

假设有如下定义（截取自该文献的符号定义）：

* a set of T graphs $G = {\{Gt|t ∈ 1...T }\}$. Each graph $G_t = (Vt, Et)$is composed of vertices $V_t$ and edges$ E_t$ .

* ==$X_t$== ：The vertices are collectively described by an  $ N_t × F$ design matrix $X_t$ of ==features==, where $N_t$ is the number of nodes and $F$ is the dimension of features in $ G_t$ 

* ==$ A_t$==：the edges $E_t$ are encoded by an $N_t × N_t$ ==adjacency matrix==$ A_t$, 

* ==$P_t$==： we can compute a degree-normalized ==transition matrix== $P_t$that gives the probability of jumping from node $i$ to node$ j$ in one step. 

* ==$Y$==：Either the nodes, edges, or graphs have ==labels $Y$== associated with them, with the dimensionality of $Y$ differing in each case.

  

#### Task: predict Y

训练集：一些标记的entity（节点、图或边）。任务是预测剩余未标记entity的值。

DCNN takes $G$ as input and returns either a hard prediction for $Y$ or a conditional distribution$ P(Y |X)$

#### model：DCNN

该模型**对每一个节点(或边、或图)采用H个hop的矩阵**进行表示，每一个hop都表示该邻近范围的邻近信息，由此，对于局部信息的获取效果比较好，得到的节点的representation的表示能力很强。

> 所谓的hop，按照字面意思是“跳”，对于某一节点$n$ ，其**H-hop**的节点就是，从 $n$ 节点开始，跳跃 H次所到达的节点，比如对于$n$ 节点的1-hop的节点，就是 $n$ 节点的邻居节点。
> 这里对于节点representation并不是采用一个向量来表示，而是采用一个矩阵进行表示，**矩阵的第 ![[公式]](https://www.zhihu.com/equation?tex=i) 行就表示i-hop的邻接信息**。
>
> - 对于 Hidden state 0，计算出与所选节点距离为1的特征的和的平均，得到各个节点的特征![H^0](https://math.jianshu.com/math?formula=H%5E0)[![H^0](https://math.jianshu.com/math?formula=H%5E0)是由Hidden state 0所有节点组成的矩阵]
> - 对于 Hidden sate 1，计算出与所选节点距离为2的特征的和的平均，得到各个节点的特征![H^1](https://math.jianshu.com/math?formula=H%5E1)[![H^1](https://math.jianshu.com/math?formula=H%5E1)是由Hidden state 1所有节点组成的矩阵]
> - 以此类推，可以得到多个上述的矩阵将得到的矩阵按照如图形式表示
>
> <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210603164500161.png" alt="image-20210603164500161" style="zoom:50%;" />

#### 模型详述

DCNN模型输入图 $\mathcal G$  ，返回硬分类预测值 ![[公式]](https://www.zhihu.com/equation?tex=Y) 或者条件分布概率 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BP%7D%28Y%7CX%29) 。该模型将每一个预测的目标对象(节点、边或图)转化为一个diffusion-convolutional representation，大小为 ![[公式]](https://www.zhihu.com/equation?tex=H%5Ctimes%7B%7DF) ， ![[公式]](https://www.zhihu.com/equation?tex=H) 表示扩散的hops。因此，对于节点分类任务，图 ![[公式]](https://www.zhihu.com/equation?tex=t) 的confusion-convolutional representation为大小为 ![[公式]](https://www.zhihu.com/equation?tex=N_t%5Ctimes%7BH%7D%5Ctimes%7BF%7D) 的张量，表示为 ![[公式]](https://www.zhihu.com/equation?tex=Z_t) ，对于图分类任务，张量 ![[公式]](https://www.zhihu.com/equation?tex=Z_t) 为大小为 ![[公式]](https://www.zhihu.com/equation?tex=H%5Ctimes%7BF%7D) 的矩阵，对于边分类任务，张量 ![[公式]](https://www.zhihu.com/equation?tex=Z_t) 为大小为 ![[公式]](https://www.zhihu.com/equation?tex=M_t%5Ctimes%7BH%7D%5Ctimes%7BF%7D) 的矩阵。示意图如下

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210603164755802.png" alt="image-20210603164755802" style="zoom: 67%;" />

对于**节点分类任务**，假设 ![[公式]](https://www.zhihu.com/equation?tex=P_t%5E%2A) 为 ![[公式]](https://www.zhihu.com/equation?tex=P_t) 的power series，即 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7BP_t%2CP_t%5E2%2CP_t%5E3%2C...%5C%7D) ，大小为 ![[公式]](https://www.zhihu.com/equation?tex=N_t%5Ctimes%7BH%7D%5Ctimes%7BN_t%7D) ，那么对于图 ![[公式]](https://www.zhihu.com/equation?tex=t) 的节点 ![[公式]](https://www.zhihu.com/equation?tex=i) ，第 ![[公式]](https://www.zhihu.com/equation?tex=j) 个hop，第 ![[公式]](https://www.zhihu.com/equation?tex=k) 维特征值 ![[公式]](https://www.zhihu.com/equation?tex=Z_%7Btijk%7D) 计算公式为

![[公式]](https://www.zhihu.com/equation?tex=Z_%7Bt+i+j+k%7D%3Df%5Cleft%28W_%7Bj+k%7D%5E%7Bc%7D+%5Ccdot+%5Csum_%7Bl%3D1%7D%5E%7BN_%7Bt%7D%7D+P_%7Bt+i+j+l%7D%5E%7B%2A%7D+X_%7Bt+l+k%7D%5Cright%29+%5C%5C)

使用矩阵表示为

![[公式]](https://www.zhihu.com/equation?tex=Z_%7Bt%7D%3Df%5Cleft%28W%5E%7Bc%7D+%5Codot+P_%7Bt%7D%5E%7B%2A%7D+X_%7Bt%7D%5Cright%29+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Codot) 表示element-wise multiplication，由于模型只考虑 ![[公式]](https://www.zhihu.com/equation?tex=H) 跳的参数，即参数量为 ![[公式]](https://www.zhihu.com/equation?tex=O%28H%5Ctimes%7BF%7D%29) ，**使得diffusion-convolutional representation不受输入大小的限制**。

在计算出 ![[公式]](https://www.zhihu.com/equation?tex=Z) 之后，过一层全连接得到输出 ![[公式]](https://www.zhihu.com/equation?tex=Y) ，使用 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BY%7D) 表示硬分类预测结果，使用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BP%7D%28Y%7CX%29) 表示预测概率，计算方式如下

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BY%7D%3D%5Carg+%5Cmax+%5Cleft%28f%5Cleft%28W%5E%7Bd%7D+%5Codot+Z%5Cright%29%5Cright%29+%5C%5C+%5Cmathbb%7BP%7D%28Y+%7C+X%29%3D%5Coperatorname%7Bsoftmax%7D%5Cleft%28f%5Cleft%28W%5E%7Bd%7D+%5Codot+Z%5Cright%29%5Cright%29%5C%5C)

对于**图分类任务**，直接采用所有节点表示的均值作为graph的representation

![[公式]](https://www.zhihu.com/equation?tex=Z_%7Bt%7D%3Df%5Cleft%28W%5E%7Bc%7D+%5Codot+1_%7BN_%7Bt%7D%7D%5E%7BT%7D+P_%7Bt%7D%5E%7B%2A%7D+X_%7Bt%7D+%2F+N_%7Bt%7D%5Cright%29+%5C%5C)

其中，是全为1的的向量。

对于**边分类任务**，通过**将每一条边转化为一个节点来进行训练和预测**，这个节点与原来的边对应的首尾节点相连，转化后的图的邻接矩阵 ![[公式]](https://www.zhihu.com/equation?tex=A_t%27) 可以直接从原来的邻接矩阵增加一个**incidence matrix**得到

![[公式]](https://www.zhihu.com/equation?tex=A_%7Bt%7D%5E%7B%5Cprime%7D%3D%5Cleft%28%5Cbegin%7Barray%7D%7Bcc%7D%7BA_%7Bt%7D%7D+%26+%7BB_%7Bt%7D%5E%7BT%7D%7D+%5C%5C+%7BB_%7Bt%7D%7D+%26+%7B0%7D%5Cend%7Barray%7D%5Cright%29%5C%5C)

之后，使用 ![[公式]](https://www.zhihu.com/equation?tex=A_t%27) 来计算 ![[公式]](https://www.zhihu.com/equation?tex=P_t%27) ，并用来替换 ![[公式]](https://www.zhihu.com/equation?tex=P_t) 来进行分类。

对于**模型训练**，使用梯度下降法，并采用early-stop方式得到最终模型。

### 改进

- **内存占用大**：DCNN建立在密集的张量计算上，需要存储大量的张量，需要的空间复杂度。
- **长距离信息传播不足**：模型对于局部的信息获取较好，但是远距离的信息传播不足。















假设有如下定义

- 一个graph数据集 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BG%7D%3D%5Cleft%5C%7BG_%7Bt%7D+%7C+t+%5Cin+1+%5Cldots+T%5Cright%5C%7D) 。
- graph定义为 ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bt%7D%3D%5Cleft%28V_%7Bt%7D%2C+E_%7Bt%7D%5Cright%29) ，其中， ![[公式]](https://www.zhihu.com/equation?tex=V_t) 为节点集合， ![[公式]](https://www.zhihu.com/equation?tex=E_t) 为边集合。
- 所有节点的特征矩阵定义为 ![[公式]](https://www.zhihu.com/equation?tex=X_t) ，大小为 ![[公式]](https://www.zhihu.com/equation?tex=N_t%5Ctimes%7BF%7D) ，其中， ![[公式]](https://www.zhihu.com/equation?tex=N_t) 为图 ![[公式]](https://www.zhihu.com/equation?tex=G_t) 的节点个数， ![[公式]](https://www.zhihu.com/equation?tex=F) 为节点特征维度。
- 边信息 ![[公式]](https://www.zhihu.com/equation?tex=E_t) 定义为 ![[公式]](https://www.zhihu.com/equation?tex=N_t%5Ctimes%7B%7DN_t) 的邻接矩阵 ![[公式]](https://www.zhihu.com/equation?tex=A_t) ，由此可以计算出**节点度(degree)归一化**的转移概率矩阵 ![[公式]](https://www.zhihu.com/equation?tex=P_t) ，表示从 ![[公式]](https://www.zhihu.com/equation?tex=i) 节点转移到 ![[公式]](https://www.zhihu.com/equation?tex=j) 节点的概率。

对于graph来说没有任何限制，graph可以是带权重的或不带权重的，有向的或无向的。

模型的目标为预测 ![[公式]](https://www.zhihu.com/equation?tex=Y) ，也就是预测每一个图的节点标签，或者边的标签，或者每一个图的标签，**在每一种情况中，模型输入部分带有标签的数据集合，然后预测剩下的数据的标签**。