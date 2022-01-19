> 题目：Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
>
> 来源： ICML 2017
>
> 作者：Chelsea Finn ， Pieter Abbeel ， Sergey Levine 



### 解决的问题:

提出了一种与模型无关的元学习算法，该算法可与与梯度下降训练的任何模型兼容，并适用于各种不同的学习问题，包括分类，回归和强化学习。



针对各种学习任务训练一个模型，使得它只需要少量的训练样本就可以解决新的学习任务，

the agent must integrate its prior experience with a small amount of new information，while avoiding overfitting to the new data.

所以元学习的机制需要对于任务和完成任务所需要的计算形式都应该是general的



### idea：

算法的基本思想是训练模型的初始参数，不会扩展学习参数的数量也不会对模型架构施加约束，可与CNN，FCN，RNN结合，可微和不可微的loss都可以用。

从特征学习的角度来看，训练模型参数的过程可以将其视为构建广泛适用于多任务的内部表征。



MAML：直觉是某些内部表示形式比其他内部表示形式更具可移植性。

因此我们将以基于梯度的学习规则来学习模型，可以在从p（T）中提取的新任务上取得快速进展，而不会过度拟合。

目标是找到对任务的变化敏感的模型参数，即当沿p（T）的梯度方向改变参数时，参数的细微变化将极大改善从p（T）提取的任何任务的损失函数

### detail：



### 算法：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210512193929247.png" alt="image-20210512193929247" style="zoom:50%;" />

#### N-way K-shot：

N-way指训练数据中有N个类别，K-shot指每个类别下有K个被标记数据。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210512212907053.png" alt="image-20210512212907053" style="zoom:67%;" />



$f(\theta) $是参数为$\theta$的参数化的模型

meta-train classes: C1~C10，训练元模型![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D)，包含的共计300个样本，即 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-train%7D) 

meta-test classes：P1~P5，精调（fine-tune）得到最终的模型 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bfine-tune%7D) ，共计100个样本，即 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-test%7D) 



#### 根据5-way 5-shot的实验设置:

在训练 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) 阶段：

* 从 ![[公式]](https://www.zhihu.com/equation?tex=C_1%EF%BD%9EC_%7B10%7D) 中随机取5个类别，每个类别再随机取20个已标注样本，组成一个**task** ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+T%7D) 。  双随机：随机类别，随机样本，有点随机森林的意思，通过这种方式获得良好的泛化能力。
* 设置5个已标注样本称为 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+T%7D) 的**support set**，另外15个样本称为 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+T%7D) 的**query set**。 
* 这个**task** ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+T%7D) ， 就相当于普通深度学习模型训练过程中的一条训练数据。那我们肯定要组成一个batch，才能做随机梯度下降SGD
* 所以我们反复在训练数据分布中抽取若干个这样的**task** ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+T%7D) ，组成一个batch。
* 在训练 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bfine-tune%7D) 阶段，**task**、**support set**、**query set**的含义与训练 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) 阶段均相同。



#### gradient by gradient：

第一种梯度更新：support set

可以理解为copy了一个原模型，计算出新的参数，用在第二轮梯度的计算过程中。

用了batch size个task对 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) 进行了训练，然后我们使用上述batch个task中地**query set**去测试参数为![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D%5E%7Bi%7D%2Ci%5Cin%5B1%2Cbatch+size%5D) 的 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) 模型效果，获得总损失函数 ![[公式]](https://www.zhihu.com/equation?tex=L%28%5Cphi%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bbs%7D%7Bl%5E%7Bi%7D%28%5Chat%7B%5Ctheta%7D%5E%7Bi%7D%29%7D) ，这个损失函数就是一个**batch task**中**每个task**的**query set**在各自参数为 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D%5E%7Bi%7D%2Ci%5Cin%5B1%2Cbatch+size%5D) 的 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) 中的损失 ![[公式]](https://www.zhihu.com/equation?tex=l%5E%7Bi%7D%28%5Chat%7B%5Ctheta%7D%5E%7Bi%7D%29) 之和。

第二种梯度更新：query set，更新模型参数

即更新初始化参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) ，也就是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi%5CLeftarrow%5Cphi+-%5Ceta.%5Cpartial+L%28%5Cphi%29%2F%5Cpartial+%5Cphi) 来更新初始化参数。

对第一种里面batch size个task的loss融合后的loss的进行梯度下降，为了更好地泛化性，不对单一任务的loss更新模型参数，综合了多个task后再更新

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210513113344270.png" alt="image-20210513113344270" style="zoom:67%;" />









### 词句:

1、it is compatible with any model trained with gradient de- scent and applicable to a variety of different learning problems, including classification

它与任何用梯度下降法训练的模型兼容，适用于各种不同的学习问题，包括分类问题。

2、 In our approach, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task.

在我们的方法中，模型的参数被显式地训练，使得少量的梯度步长和少量的训练数据能够在新任务上产生良好的泛化性能。

3、As such, for the greatest applicability, the mechanism for learning to learn (or meta-learning) should be general to the task and the form of computation required to complete the task.        required to 

因此，为了最大化适用性，元学习的机制需要对于任务和完成任务所需要的都应该是general的

4、The key idea underlying our method is to train the model’s initial parameters such that the model has maximal perfor- mance on a new task after the parameters have been up- dated through one or more gradient steps computed with a small amount of data from that new task. 

我们方法的基本思想是训练模型的初始参数，以便在通过一个或多个梯度步骤更新参数后，该模型在新任务上具有最佳性能，而该梯度步骤是使用来自该新任务的少量数据计算出的

5、The process of training a model’s parameters can be viewed from a fea- ture learning standpoint as building an internal representa- tion that is broadly suitable for many tasks.

从特征学习的角度来看，训练模型参数的过程可以将其视为构建广泛适用于许多任务的内部表示。

6、with respect to 关于   The intuition behind this approach is  这种方法背后的直觉是











**步骤1**，随机初始化模型的参数，没什么好说的，任何模型训练前都有这一步。

**步骤2，**是一个循环，可以理解为一轮迭代过程或一个epoch，当然啦预训练的过程是可以有多个epoch的。

**步骤3，**相当于更新模型参数中的DataLoader，即随机对若干个（e.g., 4个）**task**进行采样，形成一个batch。

**步骤4～步骤7，**是第一次梯度更新的过程。注意这里我们可以理解为copy了一个原模型，计算出新的参数，用在第二轮梯度的计算过程中。我们说过，MAML是gradient by gradient的，有两次梯度更新的过程。步骤4～7中，利用batch中的每一个task，我们分别对模型的参数进行更新（4个task即更新4次）。注意这一个过程**在算法中是可以反复执行多次的**，伪代码没有体现这一层循环，但是作者再分析的部分明确提到" using multiple gradient updates is a straightforward extension"。

**步骤5，**即对利用batch中的某一个task中的support set，计算每个参数的梯度。在N-way K-shot的设置下，这里的**support set**应该有**NK**个。作者在算法中写with respect to K examples，默认对每一个class下的K个样本做计算。实际上参与计算的总计有NK个样本。这里的loss计算方法，在回归问题中，就是MSE；在分类问题中，就是**cross-entropy**。

**步骤6，**即第一次梯度的更新。

**步骤4～步骤7，**结束后，MAML完成了**第一次**梯度更新。接下来我们要做的，是根据第一次梯度更新得到的参数，通过gradient by gradient，计算第二次梯度更新。第二次梯度更新时计算出的梯度，直接通过SGD作用于原模型上，也就是我们的模型真正用于更新其参数的梯度。

**步骤8**即对应第二次梯度更新的过程。这里的loss计算方法，大致与步骤5相同，但是不同点有两处。一处是我们不再是分别利用每个task的loss更新梯度，而是像常见的模型训练过程一样，计算一个batch的loss总和，对梯度进行随机梯度下降SGD。另一处是这里参与计算的样本，是task中的**query set**，在我们的例子中，即5-way*15=75个样本，目的是增强模型在task上的泛化能力，避免过拟合**support set**。步骤8结束后，模型结束在该batch中的训练，开始回到步骤3，继续采样下一个batch。

以上即时MAML预训练得到 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) 的全部过程，是不是很简单呢？事实上，MAML正是因为其简单的思想与惊人的表现，在元学习领域迅速流行了起来。接下来，应该是面对新的**task**，在 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) 的基础上，精调得到 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bfine-tune%7D) 的方法。原文中没有介绍**fine-tune**的过程，这里我向小伙伴们简单介绍一下。

**fine-tune**的过程与预训练的过程大致相同，不同的地方主要在于以下几点：

- 步骤1中，fine-tune不用再随机初始化参数，而是利用训练好的 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) 初始化参数。
- 步骤3中，fine-tune只需要抽取一个task进行学习，自然也不用形成batch。fine-tune利用这个task的support set训练模型，利用query set测试模型。实际操作中，我们会在 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-test%7D) 上随机抽取许多个task（e.g., 500个），分别微调模型 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) ，并对最后的测试结果进行平均，从而避免极端情况。
- fine-tune没有步骤8，因为task的query set是用来**测试**模型的，标签对模型是未知的。因此**fine-tune过程没有第二次梯度更新，而是直接利用第一次梯度计算的结果更新参数。**

以上就是MAML的全部算法思路啦。我也是在摸索学习中，如有不足之处，敬请指正。

