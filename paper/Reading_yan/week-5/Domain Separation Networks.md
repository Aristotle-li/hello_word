> 题目：Domain Separation Networks
>
> 来源：NIPS 2016
>
> 作者：Google Brain

### 解决的问题：

域自适应或域泛化：解决同类目标在不同域的表示不一致，避免重复标注大规模数据及，当前的方法：

1、mapping：mapping representations from one domain to the other

2、 shared representation： learning to extract features that are invariant to the domain from which they were extracted.

不足：

1、只关注不同域的差别，但是 ignore the individual characteristics of each domain，

2、shared representation：使得共享表示容易受到与底层共享分布相关的噪声的污染

解决方案：introduces the notion of a private subspace for each domain

### idea:

“low-level” differences： noise, resolution, illumination and color. 

“High-level” differences：  the number of classes, the types of objects, and geometric variations, such as 3D position and pose. 



A private subspace：模型为每个域引入了一个私有子空间的概念，它捕获特定于域的属性，例如背景和低级图像统计。

A shared subspace：一个共享的子空间，通过使用自动编码器和显式损失函数，捕获域共享的表示。

通过找到一个与私有子空间**正交**的共享子空间，模型能够分离出每个域所特有的信息，并在此过程中产生对手头任务更有意义的表示。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210520151854646.png" alt="image-20210520151854646" style="zoom:67%;" />

The private component of the representation is specific to a single domain and the shared component of the representation is shared by both domains.

为了使模型产生这样的分裂表示，增加了一个损失函数来鼓励这些部分的独立性。

在共享表示上训练的分类器能够更好地跨域泛化，因为其输入不受每个域所特有的表示方面的污染。

### loss：

**$L_{recon}$：**

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210520153424659.png" alt="image-20210520153424659" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210520153440022.png" alt="image-20210520153440022" style="zoom:50%;" />

scale–invariant mean squared error term ：2范数+规整项，假如 $\hat{x}$和$x$只在整体像素少一个常数，规整项会弥补这个差距。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210520154139513.png" alt="image-20210520154139513" style="zoom:50%;" />

理解尺度不变均方误差，通过论文Depth Map Prediction from a Single Image using a Multi-Scale Deep Network：<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210520155832250.png" alt="image-20210520155832250" style="zoom:50%;" />



这使模型能够学习重现要建模的对象的整体形状，而无需在输入的绝对颜色或亮度上花费建模能力

（This allows the model to learn to reproduce the overall shape of the objects being modeled without expending modeling power on the absolute color or intensity of the inputs. ）

**$L_{difference}$：**

差异损失鼓励了每个域的共享表示和私有表示之间的正交性：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210520161524577.png" alt="image-20210520161524577" style="zoom:50%;" />

使用余弦距离来度量 private and shared representation of each domain的距离，相关性越小损失越小，相互独立时为0

**$L_{similar}$：**

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210520170444294.png" alt="image-20210520170444294" style="zoom:50%;" />

### 改进方向：

1、we assume that they have high level parameters with similar distributions and the same label space.

而现实世界 label space 一般是不同的，而NN 改变了 label space 后需要重新训练，这是一个难题。

2、Gradient Reversal Layer (GRL) ：梯度方向是使得分类器更容易区分来自不同域样本，而梯度翻转确实让分类器更confuse，但是这个方向的选择是靠直觉选择的。

### 词句：

1、 circumventing this cost 规避这一成本

2、 vulnerable to contamination  易受污染

3、By partitioning the space in such a manner, the classifier trained on the shared representation is better able to generalize across domains as its inputs are uncontaminated with aspects of the representation that are unique to each domain.

通过以这种方式划分空间，在共享表示形式上训练的分类器可以更好地跨域进行泛化，因为其输入不受每个域唯一的表示形式的污染

4、 are partitioned into   .....分为....

5、Our novel architecture results in a model   我们新颖的架构形成了一个模型

6、 manipulate 

7、 Existing approaches focus either on mapping representations from one domain to the other, or on learning to extract features that **are invariant to** the domain from which they were extracted. 

现有方法侧重于将表示形式从一个域映射到另一域，或者着重于学习提取与提取它们的域不变的特征