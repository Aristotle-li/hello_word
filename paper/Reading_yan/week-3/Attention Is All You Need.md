> 题目：Attention Is All You Need
>
> 来源： NIPS 2017
>
> 作者：Google Brain
>



### 解决的问题：

1、在序列转换任务中，RNN系列学习长程依赖关系比较困难，因为前后 signals 必须跨越整个路径长度才能建立联系：

在这些模型中，将两个任意输入或输出位置的信号关联起来所需的操作数随着位置之间的距离而增长，这使得学习长程依赖关系变得更加困难。transformer中，此操作被减少为恒定的操作次数，尽管由于平均注意力加权位置而导致有效分辨率降低，使用Multi-Head Attention可以抵消。

2、并行计算，大大缩短了计算时间



### detail：



Self-attention：将单个序列的不同位置联系起来，以计算序列的表示形式。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210511202313397.png" alt="image-20210511202313397" style="zoom: 50%;" />

the output of each sub-layer ： $ LayerNorm(x + Sublayer(x)),$ 

Scaled Dot-Product Attention：The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$. 

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210511205233967.png" alt="image-20210511205233967" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210511205130560.png" alt="image-20210511205130560" style="zoom: 67%;" />



Multi-Head Attention：使用不同的线性投影将查询、键和值线性投影h次，分别投影到dk、dk和dv维，在查询、键和值的每个投影版本上，并行地执行注意函数，产生dv维的输出值。它们被连接起来并再次投影，从而得到最终值。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210511225634121.png" alt="image-20210511225634121" style="zoom:50%;" />



Multi-Head Attention使得模型能够在不同的位置共同关注来自不同表示子空间的信息

![image-20210511225955265](/Users/lishuo/Library/Application Support/typora-user-images/image-20210511225955265.png)



The Transformer 以三种不同的方式使用 multi-head attention：

* Encoder-Decnoder Attention：当前翻译和编码的特征向量之间的关系

  Q来自于前面的decode层，K,V来自于encode的输出，这允许decode中的每个位置都参与输入序列中的所有位置

* Self-Attention：

   Encoder：编码器中的每个位置都可以参与到编码器前一层中的所有位置

   Decode：需要防止decode中的信息流向左流动，以保持自回归特性，通过屏蔽softmax输入中与非法连接相对应的所有值来实现scaled dot-product attention



### 改进方向：

重要的位置信息被额外的编码，必然带来信息的丢失

### 词句：

1、dispensing with 免除    be superior in 在...方面胜过

2、 This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. 

这种固有的顺序性质阻止了训练示例内的并行化，这在较长的序列长度上变得至关重要，因为内存限制限制了示例之间的批处理。

3、Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences 

注意机制已成为各种任务中引人注目的序列建模和转导模型不可或缺的一部分，允许建模依赖关系，而不考虑它们在输入或输出序列中的距离。

4、albeit at 尽管在

5、The encoder is composed of a stack of N = 6 identical layers.编码器由N=6个相同层组成

6、 mimics 模仿

7、replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.