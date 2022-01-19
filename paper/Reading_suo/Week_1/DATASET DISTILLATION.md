> 题目：Dataset Distillation
>
> 来源：ICLR 2019
>
> 作者 Facebook AI Research, MIT CSAIL



### motivation：

we keep the model fixed and instead attempt to distill the knowledge from a large training dataset into a small one. The idea is to synthesize a small number of data points that do not need to come from the correct data distribution, but will, when given to the learning algorithm as training data, approximate the model trained on the original data.

我们保持模型固定，而不是尝试从一个大的训练数据集提取知识到一个小的。其思想是合成少量的数据点，这些数据点不需要来自于正确的数据分布，但当给学习算法作为训练数据时，会近似于在原始数据上训练的模型。





### 效果：

如，在图1a中，在给定固定网络初始化的情况下，将MNIST数字数据集的60000个训练图像合成仅10个图像（每个类别一个）。在这10幅图像上训练标准的LeNet，使得测试性能达到94％，而原始数据集的这一性能达到99％。对于具有未知随机权重的网络，100个合成图像通过几个梯度下降步骤训练到80％。我们将方法命名为数据集蒸馏，这些图像是蒸馏图像。

传统的常常使用数万个梯度下降步骤的训练相比，给定一些蒸馏图像，我们可以更加有效地给网络“加载”整个数据集的知识。

我的理解：

和few-shot learning区别，data distillation有大数据集

迁移学习：使用pre-train权重文件初始化权重

数据集蒸馏：学到一组远小于训练集的合成数据，和对应的学习率，梯度下降后得到在原始数据集中好的权重。

方式：在合成数据集中训练，在原始数据集中验证。



### 一个问题：

是否有可能将数据集压缩为一小组合成数据样本。例如，是否有可能在合成图像上训练分类模型？传统观点认为答案是否定的，因为合成训练数据可能不遵循真实测试数据的分布。然而在这项工作中，我们表明这确实是可能的。我们提出了一种新的优化算法用于合成少量的合成数据，不仅可以捕获大部分原始训练数据，而且还可以在几个梯度步骤中快速地训练模型。

### 方法：



我们提出了一种新的优化算法用于合成少量的合成数据，不仅可以捕获大部分原始训练数据，而且还可以在几个梯度步骤中快速地训练模型

**建模：**我们首先将网络权重推导为合成数据的可微函数。鉴于这种联系，我们优化蒸馏图像的像素值，而不是优化特定训练目标的网络权重。

**初始化：**但是，这种方法需要获取网络的初始权重。为了放宽这个假设，我们开发了一种为随机初始化网络生成蒸馏图像的方法。为了进一步提高性能，我们提出了一个迭代版本，可以获得了一系列蒸馏图像，这些蒸馏图像可以用在多个阶段进行训练。

**下限：**最后，我们研究了一个简单的线性模型，得出了获得完整数据集性能的蒸馏数据大小下限。



我们证明，在固定初始化的模型下，可以使用少量蒸馏图像训练出实现令人惊讶的性能。对于在其他任务上预训练的网络，我们的方法可以找到用于快速模型微调的蒸馏图像。我们在几个初始化设置上测试我们的方法：固定初始化，随机初始化，固定预训练权重和随机预训练权重，以及两个训练目标：图像分类和恶意数据集毒性攻击。对四个公开可用的数据集MNIST，CIFAR10，PASCAL-VOC和CUB-200进行了大量实验，结果表明我们的方法通常优于现有方法。



### 目标

给定一个模型和一个数据集，我们的目标是获得一个新的，大大减少的合成数据集，其性能几乎与原始数据集一样好。



训练集：

![image-20211220214940932](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220214940932.png)

神经网络优化目标：

![image-20211220214725325](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220214725325.png)



优化方式：梯度下降

![image-20211220214712286](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220214712286.png)

η是学习速率。这样的训练过程往往需要数万甚至数百万个更新步骤才能收敛。相反，我们的目标是学习一个小集合的合成蒸馏训练数据̃<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220215013161.png" alt="image-20211220215013161" style="zoom:50%;" />

和相应的学习速率̃η，以便单个GD步骤，如

![image-20211220215053752](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220215053752.png)



使用这些学习到的合成数据̃x可以极大地提高真实测试集上的性能。给定初始θ0，通过最小化L以下的目标，我们得到这些合成数据̃x和学习速率̃η:





![image-20211220215526786](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220215526786.png)





where we derive the new weights θ1 as a function of distilled data  ̃ x and learning rate  ̃ η using Equation 2 and then evaluate the new weights over all the training data x.

其中，我们使用公式2推导新的权值θ1作为蒸馏数据̃x和学习速率̃η的函数，然后对所有训练数据x计算新的权值。





不足和改进：



不幸的是，上面为给定初始化而优化的数据不能很好地泛化到其他初始化。被提取的数据通常看起来像随机噪声(例如，在图2a中)，因为它编码了训练数据集x和特定网络初始化θ0的信息。为了解决这个问题，我们转而计算一小部分经过提炼的数据，这些数据可以用于特定分布的随机初始化的网络。我们将优化问题表述如下

![image-20211220220425132](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220220425132.png)



其中网络初始化θ0随机从分布p(θ0)中采样。在我们的优化过程中，蒸馏后的数据被优化到适合随机初始化的网络。算法1说明了我们的主要方法。在实践中，我们观察到最终提炼的数据很好地概括为不可见的初始化。此外，这些经过提炼的图像通常看起来信息量很大，它们对每个类别的区别特征进行了编码(如图3所示)。





3.3简单线性情况分析

本节研究具有二次损失的简单线性回归问题的公式。通过一个GD步骤，我们得到了获得与在完整数据集上进行任意初始化训练相同的性能所需的经过提炼的数据大小的下界。考虑一个数据集x包含N数据目标对{(di, ti)} N i = 1, di∈RD和ti∈R,我们代表两个矩阵:一个N×维数据矩阵D和一个N×1目标矩阵t。鉴于均方误差和θD×1权重矩阵,

![image-20211220231249818](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220231249818.png)

对于二次损失，对于任意初始化θ0，**总存在经过学习的蒸馏数据̃x，可以在全数据集x上达到与训练相同的性能(**即达到全局最小值)。例如

![image-20211220231431118](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220231431118.png)

任意最优的θ∗ 满足dT dθ∗= dT t，将式(6)代入上述条件，得到



![image-20211220231623902](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220231623902.png)



假设即数据矩阵d的特征列是独立的(即dT d是满秩的)。对于̃x =(̃d，̃t)对于任意θ0满足上述方程，我们必须

![image-20211220231810396](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220231810396.png)

which implies that  ̃ dT  ̃ d has full rank and M ≥ D.





证明出来了：当M大于D，即样本数大于特征维度可以优化到最小值可以。π







鸡生蛋蛋生鸡

激发图片，通过这些合成图片一步激发出在测试集上最优的权重

把合成图像作为网络参数的一部分

https://zhuanlan.zhihu.com/p/67907418

https://zhuanlan.zhihu.com/p/56328042