社交网络的时空性研究

行为分析-心理分析-贴标签



![preview](https://pic1.zhimg.com/v2-9308db5110b806f4334e910a418fe908_r.jpg)

我们将图神经网络划分为五大类别，分别是：图卷积网络（Graph Convolution Networks，GCN）、 图注意力网络（Graph  Attention Networks）、图自编码器（ Graph Autoencoders）、图生成网络（ Graph Generative  Networks） 和图时空网络（Graph Spatial-temporal Networks）

![img](https://pic2.zhimg.com/80/v2-c82e9a705cc6e1213f459fb89d3120fd_1440w.jpg)

### 理解二阶邻接矩阵:

一阶邻接矩阵：反映的是节点自己的邻居的链接情况

二阶邻接矩阵：反映的是节点的邻居的链接情况

实现原理：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210528191001106.png" alt="image-20210528191001106" style="zoom:50%;" />

根据矩阵乘法，$a*a_{:,j}$ 就是将和第$j$个节点相连的节点对应列累加（因为相连为1），不相连的置零（因为不相连为1），并将$j$列自己链接置零（对角线=0可以起到这个作用）

```
import numpy as np
a = np.array([[0,1,0,0],[1,0,0,1],[0,0,0,1],[0,1,1,0]])
b= np.dot(a,a)
print(b)
print(a)
结果：
b= [[1 0 0 1]
 [0 2 1 0]
 [0 1 1 0]
 [1 0 0 2]]
a= [[0 1 0 0]
 [1 0 0 1]
 [0 0 0 1]
 [0 1 1 0]]
```

![image-20210602142630413](/Users/lishuo/Library/Application Support/typora-user-images/image-20210602142630413.png)

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602142835222.png" alt="image-20210602142835222" style="zoom:67%;" />

spatial-based Convolution：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602143204102.png" alt="image-20210602143204102" style="zoom:67%;" />



NN4G:

https://github.com/EmanueleCosenza/NN4G

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602143819118.png" alt="image-20210602143819118" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602143909816.png" alt="image-20210602143909816" style="zoom:67%;" />

DCNN:

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602145513538.png" alt="image-20210602145513538" style="zoom:67%;" />

第一层:每个节点把和他距离为1的节点加起来

第二层:每个节点把和他距离为2的节点加起来



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602154733036.png" alt="image-20210602154733036" style="zoom: 67%;" />

MoNet：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602155850497.png" alt="image-20210602155850497" style="zoom:67%;" />

GraphSAGE:

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602160229077.png" alt="image-20210602160229077" style="zoom:67%;" />

GAT:

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602160349493.png" alt="image-20210602160349493" style="zoom:67%;" />

 <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602160523025.png" alt="image-20210602160523025" style="zoom:67%;" />

GIN:

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602160758174.png" alt="image-20210602160758174" style="zoom:67%;" />

 

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602161236016.png" alt="image-20210602161236016" style="zoom:67%;" />





<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602162055095.png" alt="image-20210602162055095" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602162249216.png" alt="image-20210602162249216" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602162550717.png" alt="image-20210602162550717" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602162855074.png" alt="image-20210602162855074" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602163346416.png" alt="image-20210602163346416" style="zoom:67%;" />





<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210602170605587.png" alt="image-20210602170605587" style="zoom:67%;" />
