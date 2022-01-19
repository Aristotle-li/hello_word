

> 题目：Automatic Generation of Artistic Chinese Calligraphy
>
> 来源：2005
>
> 作者：



早期的生成模型基本都采用类比推理，based on an analogical reasoning process (ARP).

### motivation

抽取输入书法的笔画，组成笔画数据库，使用带有约束的类比推理部件生成准确的汉字、



特别是，我们采用形状分析技术来恢复训练示例的形状，并使用分层参数化表示它们。提出了一种类似的推理过程来对齐训练样本的形状表示，然后生成新颖的书法作品。该过程考虑了以几何约束形式表达的美学，以过滤出美学上不可接受的候选元素



人们可以将所采用的ARP理解为通过混合一些已知的模型来合成一个水平形状模型。为了支持混合，通常需要模型之间的形状原语映射，因此也构成所采用ARP的一部分。

![image-20211011150946790](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211011150946790.png)

### Calligraphic shape decomposition.



这是从书法作品的训练示例图像中提取层次化和参数表示的过程

It can be easily observed that manybasic features occur in different Chinese characters frequently.Totak eadv antage of this representation redundancy,we employahierarchical representation for Chinese characters. 可以很容易地观察到，许多基本特征经常出现在不同的汉字中。为了克服这种表示冗余，我们采用了汉字的分层表示。



书法作品的各个层次的参数化表示共同构成了参数空间E，用于随后的推理过程，

形状分解（或恢复）与识别P中的模型P和相应的参数集E的问题等价，以将模型P（Ei）实例化为最适合Ci.

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimage-20211011154814811.png" alt="image-20211011154814811" style="zoom:50%;" />

RP不仅可以应用于所有推理源的矩阵表示，还可以应用于所有推理源的拓扑构造函数。拓扑构造器的一些简单ARP模拟算子有：算术平均、几何平均和调和平均

constructive ellipse 由4*1矩阵表示，cat起来形成4*n的矩阵，形成的矩阵的每一行称为元素的稀疏表示域

### Calligraphy model generation from examples

给定一组n个模型Pi，每个模型对应于一个训练示例，可以通过混合n个模型实例Pi来定义一系列新的形状模型Pi。对于混合，需要根据形状信息计算最佳匹配。模型混合采用了类似的推理过程（直观上类似于插值/外推）。请注意，新导出的形状族是通过一组混合参数进行参数化的，w控制每个培训示例的贡献。



### Novelcalligraphy generation.

新的书法可以通过识别满足 $\theta$ 的w生成

