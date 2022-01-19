



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210702200014193.png" alt="image-20210702200014193" style="zoom:50%;" />

![image-20210702200419495](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702200419495.png)

![image-20210702200809721](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702200809721.png)

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210702200934861.png" alt="image-20210702200934861" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210702201117976.png" alt="image-20210702201117976" style="zoom:50%;" />



按degree的3/4指数去采样，在全样本采样performance不好

![image-20210702201413165](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702201413165.png)

node2vec

![image-20210702201554124](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702201554124.png)

非对称的：

![image-20210702202059025](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702202059025.png)

PPR:每一步都有一定的概率停住。c:stopping probability，，启发式方法





![image-20210702202323930](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702202323930.png)

![image-20210702202510810](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702202510810.png)

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210702202603219.png" alt="image-20210702202603219" style="zoom:50%;" />

贡献：推导了这个loss function 是说明了个什么事

![image-20210702202627952](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702202627952.png)

![image-20210702202731912](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702202731912.png)

下面这篇文章证明了随即又走的方法其实都在做矩阵分解，，，被采样的概率和度成正比

![image-20210702203020849](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702203020849.png)



GCN和WL_1类似

![image-20210702204211520](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702204211520.png)

![image-20210702204344167](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702204344167.png)

更好的融入feature，

inductive，因为学的是一个卷积矩阵，和节点没关系的，如果节点改变，直接用train好的矩阵就好了



![image-20210702204640782](/Users/lishuo/Library/Application Support/typora-user-images/image-20210702204640782.png)







<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629172145517.png" alt="image-20210629172145517" style="zoom:50%;" />







一个有𝑁 个节点无向图中的最大边数 ：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629171853510.png" alt="image-20210629171853510" style="zoom:50%;" />

$𝐸_{𝑚𝑎𝑥} =(𝑁|2)=(𝑁(𝑁−1))/2 $



拥有𝐸=𝐸_𝑚𝑎𝑥 条边的无向图称为完全图，其平均度为𝑁−1
$$
average Degree=\frac{2\times |E|}{|V|}=\frac{2\times \frac{N(N-1)}{2}}{N}=N-1
$$

## **Bipartite** **Graph** 

二部图是一种图，其节点可以分为两个不相交的集合𝑈和𝑉，使得每个链接都将𝑈中的一个节点连接到𝑉中的一个节点；也就是说，𝑈 和 𝑉 是独立的集合

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629172604880.png" alt="image-20210629172604880" style="zoom:50%;" />

## **Representing** **Graph—

### adjacency matrix

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629172710396.png" alt="image-20210629172710396" style="zoom:50%;" />

无向图的邻接矩阵是对称矩阵，有向图不是

### **CSR** **(Compressed Sparse Representation)** 

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629173029732.png" alt="image-20210629173029732" style="zoom:50%;" />

现实中网络十分稀疏，<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629173210679.png" alt="image-20210629173210679" style="zoom: 33%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629173249067.png" alt="image-20210629173249067" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629173401792.png" alt="image-20210629173401792" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629173715679.png" alt="image-20210629173715679" style="zoom:50%;" />

### **Connectivity** **of** **directed** **graph**有向图的连通性

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629173823496.png" alt="image-20210629173823496" style="zoom:50%;" />



强连通组件SCCs：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629173956640.png" alt="image-20210629173956640" style="zoom:50%;" />

## How to build a graph:

Choice of the proper network representation of a given domain/problem determines our ability to use network successfully.

•In some cases, there is a unique, unambiguous representation

•In some cases, the representation is by no means of unique

•The way you assign links will determine the nature of the question you study.

例如：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629174255486.png" alt="image-20210629174255486" style="zoom:50%;" />





## part 2

**Schedule**

•**Graph** **Algorithms**

•**Knowledge** **Graphs**

•**Graph** **Learning**

•**Graph** **Systems**

### **Graph Algorithms：Subgraph** **Isomorphism**

子图同构：给定一个查询 𝑄 和一个数据图 𝐺， 𝑄 是 𝐺 的子图同构，当且仅当存在单射函数 𝑓:V(Q)→𝑉(𝐺)，使得 
∀𝑢∈𝑉(𝑄)、𝑓(𝑢)∈𝑉(𝐺)、𝐿_𝑉(𝑢)=𝐿_𝑉(𝑔(𝑢))，其中𝑉(𝑄) 和𝐺分别表示𝑉(𝑄) 和𝐺冰;并且𝐿_𝑉 (∙) 表示对应的顶点标签。 
∀(𝑢_1 𝑢_2 ) ̅∈𝐸(𝑄),(〖𝑔(𝑢〗_1)𝑔(𝑢_2)) ̅∈𝐸(𝐺), 𝐿_𝐸((𝑢_1)𝑐((𝑢_1)𝑔(𝑢_1)𝑔(𝑢_1)𝑔(𝑢_2) 𝑢_2))) ̅ )

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629174548898.png" alt="image-20210629174548898" style="zoom:50%;" />

### Graph Algorithms：子图搜索

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629174645556.png" alt="image-20210629174645556" style="zoom:50%;" />



### Graph Algorithms：**Reachability** **Query**: 

可达性查询：给定一个大的有向图 𝐺 和两个顶点 𝑢_1 和 𝑢_2，可达性查询验证是否存在从 𝑢_1 到 𝑢_2 的有向路径

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629174737987.png" alt="image-20210629174737987" style="zoom:50%;" />

## *Graph** **Database:

**Graph** **Database:
** 
 \1. A graph database (GDB) is a database that uses graph structures for semantic queries with nodes, edges, and properties to represent and store data.

 \2. Working with graph query language, such as SPARQL, Cypher, Gremlin

 \3. Two categories of graph databases: RDF and property graphs. 





## the small-world model

BA model：可以反应现实世界的模型

Step1：





## 社区检测：要知道整个图，耗费资源

团：完全图，极端情况，所有节点相互连接

就是子图聚类：使得子图内稠密，子图间稀疏

图上的层次聚类：how to do？

参考cluster：定义 distance，k-means只能聚类一个超球，条带状不可以。
$$
W=(\alpha A)^L 
$$
加入参数，意味着距离越远权重越弱。这样就可以使用层次聚类了。评价标准Q：正比于 [group内边的连接-group之间边的连接]
$$
Q \
$$
生成$\hat{G}$ 和$G$ 的度分布

 i 以多大概率连接到 j
$$
k_i \cdot \frac{k_j}{2m}
$$
Q 在0.3-0.7 是比较好的一个聚类，在不同的聚类层次Q不同。







### 另一个方法：

idea：找betweenness高的边，把他删除
$$
B(e)=\sum _{s,t\in V,s≠t}\frac{\delta_st(e)}{\delta_{st}}
$$


1、计算每条边的betweennesss，删除最高的

2、重新计算，删除，迭代计算 ， 直到全部分开

缺点：效率太低

如何计算：

BFS算法：正向计算节点的值，作为分母，回溯计算，上一层节点的值作为分子，得到概率edge score。



团（clique）：

找到一个给定大小的团是一个npc问题，

how to find 

P-R算法：

贪心算法，递归调用

每次增加一个和现有p里面都相连的节点，直到p中没有一个节点和极大团相连，结束

K-clique

K-core：

找到一个子图，所有节点度都大于K，可以应用其他算法对子图每个节点一个lable，然后在原图做预测，可以使用投票法。

### 生成模型做社区检测

membership

affiliation



现在的图数据是海量的，几乎所有的算法都在基于海量图来计算模型的参数，但是海量图由于噪声等原因，不容易发现规律，所以使用类似的K-core算法来对原始图进行计算，但是K-core算法对海量图操作依然是计算效率很低的，那么可以想到基于深度学习的方法，使用k-core的思想来讲子图抽取出来，怎么做？

如何更快的扒洋葱？基于node2vec获得子图，应用k-core的判别准则决定子图，最后得到最终的子图

AGM：可以产生不种类社区== 反过来 ==给定图找一个model对应的参数，使用极大似然估计的方法，找到出现样本可能性最大的参数。

$p_{(u,v)}$ 不直接计算，而是看u，v和A相连的权重越大，$p_{(u,v)}$ 越大
$$
p_A(u,v)=1-exp(-F_{u,A}\cdot F_{v,A})\\
矩阵形式:\\
p(u,v)=1-exp(-F_uF_v^T)
$$
<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629203314147.png" alt="image-20210629203314147" style="zoom:50%;" />



 本质上就是估计F矩阵：用learning的方式去学习



## 社区搜索

### 类似k-core   ：不需要知道全图，贪心策略





<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629203659038.png" alt="image-20210629203659038"  />



### k-truss

两个节点所有邻居的交集，就是共有边三角形的个数

1、每条边至少在k-2个三角形中：

2、所有边连通

3、极大子图



## graph partition

图割：两个子图边越少越好，minimum cut

最小割 最大流



### kl 算法 -类似k-means  1970



1、先随机cut，

2、交换两个点，计算看是不是好的交换

问题：效率很慢



### multi-level graph partition

1、压缩

2、切割

3、恢复

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210629212026798.png" alt="image-20210629212026798" style="zoom:50%;" />

## **Graph Isomorphism** ：Ullmann Algorithm [3]

![image-20210630184340411](/Users/lishuo/Library/Application Support/typora-user-images/image-20210630184340411.png)

思路：i如果能和j匹配，那么i的邻居和j的邻居也可以匹配：大大缩小了搜索空间



## VF2 Algorithm [**4**]  把子图匹配看作是状态的转移

idea： Finding the (sub)graph isomorphism between Q and G is **a sequence of state transition**. 

以上是DFS的方法，下面考虑BFS:

![image-20210630185645120](/Users/lishuo/Library/Application Support/typora-user-images/image-20210630185645120.png)

![image-20210630192428067](/Users/lishuo/Library/Application Support/typora-user-images/image-20210630192428067.png)







## **Graph** **Similarity**

1、相似的部分越多越好



![image-20210630195725264](/Users/lishuo/Library/Application Support/typora-user-images/image-20210630195725264.png)

2、用最小的步骤变换到另一个**Minimal Edit Distance**

![image-20210630195953422](/Users/lishuo/Library/Application Support/typora-user-images/image-20210630195953422.png)

### •Exact Algorithm (A*-algorithm )

  What’s A*-algorithm:

   A* uses a [best-first search](http://en.wikipedia.org/wiki/Best-first_search) and finds a least-cost path from a given initial [node](http://en.wikipedia.org/wiki/Node_(graph_theory)) to one [goal node](http://en.wikipedia.org/wiki/Goal_node) (out of one or more possible goals). As A* traverses the graph, it follows a path of the lowest *known* heuristic cost, keeping a sorted [priority queue](http://en.wikipedia.org/wiki/Priority_queue) of alternate path segments along the way.

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210630201424694.png" alt="image-20210630201424694" style="zoom:50%;" />

   where  (1) g(x) denotes the cost from the starting node to the current node; 

​          (2) h(x) denotes the  “heuristic estimate“ (lower bound) of the distance from to the goal. 