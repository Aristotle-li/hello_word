> 题目：Neural Network for Graphs: A Contextual Constructive Approach
>
> 来源：IEEE Transactions on Neural Networks 2009
>
> 作者：Alessio Micheli

### 解决的问题



RNN模型实现了递归（层次）数据结构的自适应处理（编码）。递归遍历算法用于处理（编码）所有图顶点，为每个访问的顶点生成状态变量值。

RNN方法的计算特点是过渡系统的因果关系[11]。因果关系假设强加在一个有向无环图上，即一个顶点的计算只依赖于当前顶点和从它下降的顶点

这种假设被RNN（和RCC）用来实现一种编码，该编码根据输入层次结构的拓扑顺序来匹配它们。然而，这种编码过程对RNN模型实现的信号传递（见[17]）和数据类构成了限制。实际上，序列、根树或有向（位置）无环图（DAG1和DPAG）都是拓扑排序存在的图的子类

为了克服因果关系的限制





<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210603100621201.png" alt="image-20210603100621201" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210603105537243.png" alt="image-20210603105537243" style="zoom: 50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210603112657897.png" alt="image-20210603112657897" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210603112711918.png" alt="image-20210603112711918" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210603112822173.png" alt="image-20210603112822173" style="zoom:50%;" />

