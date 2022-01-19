> 题目：node2vec: Scalable Feature Learning for Networks
>
> 来源：SIGKDD 2016
>
> 作者：Tsinghua University

### 解决的问题：

 1、eigendecomposition is expensive for large real-world networks.design an objective that seeks to preserve local neighborhoods of nodes.

2、node2vec是可并行的捕获局部关系，容易扩展到大型网络。

### idea：

node2vec生成了***“在一个d维特征空间中使得保留了节点的邻居节点的可能性最大化”***的特征表示



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210531220503624.png" alt="image-20210531220503624" style="zoom:50%;" />

考虑一种灵活的算法，该算法可以学习遵守两个原则的节点表示：

学习将来自 the same network community的节点（u和s1）紧密嵌入在一起的表示的能力

学习the different network community中起相似作用的节点（u和s6）具有相似嵌入的表示的能力

有偏随机游动算法能够通过可调参数控制搜索空间实现这一点，这些游走可以有效地探索给定节点的不同邻域。

#### 优化目标：最大似然估计

设$f(u)$是将顶点 $u $映射为embedding向量的映射函数,对于图中每个顶点 $u$，定义 $N_s(u)$  为通过采样策略 $S$  采样出的顶点 $u$   的近邻顶点集合。

node2vec优化的目标是给定每个顶点条件下，令其近邻顶点（**如何定义近邻顶点很重要**）出现的概率最大。

$max_f\sum_{u\in V}log\ Pr(N_s(u)|f(u))$

为了将上述最优化问题可解，文章提出两个假设：

- 条件独立性假设

假设给定源顶点下，其近邻顶点出现的概率与近邻集合中其余顶点无关。

$Pr(N_s(u)|f(u))=\prod_{n_i\in N_s(u)}Pr(n_i|f(u))$

- 特征空间对称性假设

这里是说一个顶点作为源顶点和作为近邻顶点的时候**共享同一套embedding向量**。(对比LINE中的2阶相似度，一个顶点作为源点和近邻点的时候是拥有不同的embedding向量的) 在这个假设下，上述条件概率公式可表示为$Pr(n_i|f(u))=\frac{exp\ f(n_i)\cdot f(u)}{\sum_{v\in V}expf(u)\cdot f(v)}$

根据以上两个假设条件，最终的目标函数表示为 ：

文章中可能问题：$max_f \ \sum_{u\in V}[-logZ_u+\sum_{n_i\in N_s(u)}f(n_i)*f(u)]$

==修改后的公式：$max_f \ \sum_{u\in V}[-|N_s(u)|logZ_u+\sum_{n_i\in N_s(u)}f(n_i)*f(u)]$==

由于归一化因子 $Z_u=\sum_{n_i\in N_s(u)}exp\ f(n_i)\cdot f(u)$ 的计算代价高，所以采用负采样技术优化。由于网络不是线性的，不能像文本那样可以在连续词上使用滑动窗口来定义邻居的概念，node2vec提出了一个随机过程来抽样一个给定源节点$u$的多个不同邻居节点。节点$u$通过顶点抽样策略S生成的网络邻居节点$N_S(u)$不限制于直接邻居，而是根据不同的抽样策略S有着许多不同的结构。

#### 顶点序列采样策略（如何定义近邻顶点很重要）：

将源节点的邻域采样问题视为局部搜索的一种形式，比较不同的采样策略：

- **宽度优先抽样(Breadth-first Sampling (BFS))**：$N_S(u)$中的节点必须是源节点的直接邻居

  BFS 采样的节点更准确地反映了邻域的局部结构，使得嵌入与结构等效性密切对应（例如，基于bridges and hubs等的结构等价性可以通过观察每个节点的邻域来推断）。通过对附近节点的搜索深度限制，BFS实现了对每个节点的邻居进行了一次微观扫描。此外，在BFS中，采样邻域中的节点往往会重复多次。这一点很重要，因为它减少了描述1-hop nodes相对于源节点的分布的方差。（This is also important as it reduces the variance in characterizing the distribution of 1-hop nodes with respect the source node. 然而，对于给定的k，图中只有非常小的一部分被探索。

- **深度优先抽样(Depth-first Sampling (DFS))**：邻居节点由到源节点的距离逐渐增加的连续抽样组成

  在DFS中，采样的节点更准确地反映了邻域的宏观结构，使得嵌入与同态性的推断密切对应。DFS不仅需要推断网络中点对点的相关性存在，同时需要描述出这些相关性的准确特征。当我们有一个**抽样规模的限制**和**很大邻域去探索**时，这很难做到。其次，采样节点可能远离源节点会导致复杂的相关性，也可能导致抽样缺乏代表性。

* the paper：网络中节点的预测任务经常在两种相似性之间变换：同质性和结构等价，即内容相似性和结构相似性。在同质性假设下，高度连通并且属于相似的网络聚类应该被embedded在一起（图1中的节点和节点属于相同的网络群体）。然而在结构等价的假设下，在网络中有相似结构的节点应该被embedded在一起（图1中的节点和节点是它们对应群体的中心节点）。与同质性不同的是，结构性不强调连接，网络中离得很远的节点仍然有相同的网络角色。真实世界的网络可能是一些节点表现出同质性而另一些节点表现出结构性。

node2vec依然采用随机游走的方式获取顶点的近邻序列，不同的是node2vec采用的是一种介于BFS和DFS之间的有偏的随机游走。

给定当前顶点 $v$ ，访问下一个顶点 $x$  的概率为

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210601122954998.png" alt="image-20210601122954998" style="zoom:50%;" />

 $\pi_{vx}$是顶点 $v$ 和顶点 $x$  之间的未归一化转移概率，$Z$   是归一化常数，$c_0=u$。



node2vec引入两个超参数 $p$ 和 $q$ 来控制随机游走的策略，假设当前随机游走经过边 $(t,v)$到达顶点 $v$ 设 $\pi_{vx}=\alpha_{pq}(t,x)\cdot w_{vx}$ ， $w_{vx}$ 是顶点  $v$ 和 $x$ 之间的边权，

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210601123019554.png" alt="image-20210601123019554" style="zoom:50%;" />

![[公式]](https://www.zhihu.com/equation?tex=d_%7Btx%7D) 为顶点 $t$ 和顶点 $x$ 之间的最短路径距离。

下面讨论超参数 $p$ 和 $q$ 对游走策略的影响 

- Return parameter,p 

参数 $p$ 控制重复访问刚刚访问过的顶点的概率。 注意到 $p $ 仅作用于 $d_{tx}=0$ 的情况，而  $d_{tx}=0$ 表示顶点 $x$ 就是访问当前顶点 $v$ 之前刚刚访问过的顶点。 那么若 $p$ 较高，则访问刚刚访问过的顶点的概率会变低，反之变高。 

- In-out papameter,q 

$q$  控制着游走是向外还是向内，若 $q>1$ ，随机游走倾向于访问和 $t$ 接近的顶点(偏向BFS)。若 $q<1$，倾向于访问远离 $t$  的顶点(偏向DFS)。

下面的图描述的是当从 $t$ 访问到 $v$ 时，决定下一个访问顶点时每个顶点对应的 $\alpha$ 。 

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210601123121031.png" alt="image-20210601123121031" style="zoom:50%;" />



#### 算法

采样完顶点序列后，剩下的步骤就和deepwalk一样了，用word2vec去学习顶点的embedding向量。 值得注意的是node2vecWalk中不再是随机抽取邻接点，而是按概率抽取，node2vec采用了Alias算法进行顶点采样。



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210601123248919.png" alt="image-20210601123248919" style="zoom:50%;" />

#### node2vecWalk

通过上面的伪代码可以看到，node2vec和deepwalk非常类似，主要区别在于顶点序列的采样策略不同，所以这里我们主要关注**node2vecWalk**的实现。

由于采样时需要考虑前面2步访问过的顶点，所以当访问序列中只有1个顶点时，直接使用当前顶点和邻居顶点之间的边权作为采样依据。 当序列多余2个顶点时，使用文章提到的有偏采样。

### 词句

1、**informative**, discriminating, and independent features 信息丰富的、区分性的和独立的特征

2、 The challenge in feature learning is defining an objective function, which **involves** a trade-off in balancing computational efficiency and predictive accuracy.

特征学习的挑战在于定义目标函数，这涉及平衡计算效率和预测准确性的权衡。

3、We also show how feature representations of individual nodes can **be extended to** pairs of nodes (i.e., edges). In order to generate feature representations of edges, we **compose** the learned feature representations of the individual nodes using simple binary operators. 

我们还展示了如何将单个节点的特征表示扩展到节点对（即边）。为了生成边的特征表示，我们使用简单的二元算子组合学习到的各个节点的特征表示。

4、Our experiments focus on two common prediction tasks in networks 我们的实验侧重于网络中的两个常见预测任务

5、We **experiment with** several real-world networks from diverse domains, such as social networks, information networks, as well as networks from systems biology.

我们对来自不同领域的几个真实世界的网络进行了实验，例如社交网络、信息网络以及系统生物学的网络。

6、node2vec **outperforms** state-of-the-art methods **by up to** 26.7% on multi-label classification and up to 12.6% on link prediction.

node2vec 在多标签分类上比最先进的方法高出 26.7%，在链接预测上高出 12.6%。

7、We show how node2vec **is in accordance with** **established principles** in network science, providing flexibility in discovering representations **conforming to** different equivalences.我们展示了 node2vec 如何符合网络科学中的既定原则，为发现符合不同等价的表示提供灵活性。

8、We **empirically evaluate** node2vec for multi-label classification and link prediction on several real-world datasets.我们凭经验评估 node2vec 在几个真实世界数据集上的多标签分类和链接预测。

9、conventional paradigm 常规范式  In terms of computational efficiency, 在计算效率方面，

10、eigendecomposition of a data matrix is expensive unless the solution quality **is significantly compromised with** approximations,

数据矩阵的特征分解是昂贵的，除非解的质量因近似值而受到显着影响

11、We **formulate** feature learning in networks **as** a maximum likelihood optimization problem.我们将网络中的特征学习表述为最大似然优化问题。



node2vec: Scalable Feature Learning for Networks
论文地址：https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf【KDD 2016】
Github实现：https://github.com/aditya-grover/node2vec（python/spark实现）
Github源码：https://github.com/snap-stanford/snap/tree/master/examples/node2vec （C++实现）
node2vec的软件实现(论文中给出的地址)：http://snap.stanford.edu/node2vec/

