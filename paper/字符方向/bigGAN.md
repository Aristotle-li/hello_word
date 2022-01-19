

> 题目：LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS
>
> 来源： ICLR 2018
>
> 作者：Yoshua Bengio Petar Velicˇkovic ́∗Guillem Cucurull∗

### motivation

启发于Spectral normalization for generative adversarial networks，谱归一化

采用正交正则化

我们发现，将正交正则化应用于生成器可使其服从一个简单的“截断技巧”，通过减少生成器输入的方差，允许对样本保真度和多样性之间的权衡进行精细控制

截断技巧”，这是一种简单的采样技术，允许显式、细粒度地控制样本种类和保真度之间的权衡。



GAN的发展：

**One line** of work is focused on changing the **objective function** (Arjovsky et al., 2017; Mao et al., 2016; Lim & Ye, 2017; Bellemare et al., 2017; Salimans et al., 2018) to encourage convergence. 

**Another line** is focused on c**onstraining D** **through gradient penalties** (Gulrajani et al., 2017; Kodali et al., 2017; Mescheder et al., 2018) or **normalization** (Miyato et al., 2018), both to counteract（抵消） the use of unbounded loss functions（无界的损失函数） and **ensure D provides gradients everywhere to G.**



### ieda：

谱归一化：

与我们的工作特别相关的是频谱归一化（Miyato 等人，2018 年），它通过使用其第一个奇异值的运行估计对其参数进行归一化，从而引入自适应正则化顶部奇异方向的向后动态，从而在 D 上强制 Lipschitz 连续性。与 Odena 等人相关。 (2018) 分析 G 的 Jacobian 的条件数，发现性能取决于 G 的条件。张等人。 (2018) 发现在 G 中使用谱归一化提高了稳定性，允许每次迭代更少的 D 步。我们扩展了这些分析，以进一步了解 GAN 训练的病理学。

借鉴：

SAGAN：

1、hinge loss

2、BN

3、SN

SNGAN和SAGAN：共享嵌入，而不是为每一个嵌入分别设置一个层，这个嵌入线性投影到每个层的bias和weight

本文：

1、大规模：64-2048batch，channel变大，参数量变大，16亿参数

2、采用先验分布z的截断技巧，允许对样本和保真度进行精细控制

3、使用技巧减小大规模训练的不稳定

4、一般的GAN是将z作为输入直接嵌入生成网络，而bigGAN将噪声z送到G的多个层而不仅仅是初始层，文章认为潜空间z可以直接影响不同分辨率和层次结构级别的特征，bigGAN将z分成每个分辨率的一个块儿，并将每个块连接到条件向量c来实现，4%的性能提升，速度提升18%







其他工作：

侧重于架构的选择，例如 SA-GAN (Zhang et al., 2018)，它添加了 (Wang et al., 2018) 中的自注意力块，以提高 G 和 D 对全局建模的能力结构体。 ProGAN（Karras 等人，2018 年）通过在一系列分辨率不断增加的情况下训练单个模型，在单类设置中训练高分辨率 GAN。

类信息可以通过多种方式输入模型：

在 (Odena et al., 2017) 中，它通过将 1-hot 类向量连接到噪声向量来提供给 G，并且修改目标以鼓励条件样本最大化由辅助分类器预测的相应类概率。（G by concatenating a 1-hot class vector to the noise vector），objective被修改为鼓励条件样本最大化由辅助分类器预测的相应类别概率。

德弗里斯等人。 (2017) 和 Dumoulin 等人。 (2017) 通过在 BatchNorm (Ioffe & Szegedy, 2015) 层中为 G 提供类条件增益和偏差来修改类条件传递给 G 的方式。在 Miyato & Koyama (2018) 中，D 通过使用其特征与一组学习类嵌入之间的余弦相似性作为区分真实样本和生成样本的额外证据来进行条件化，有效地鼓励生成特征与学习类原型相匹配的样本。





我们采用 Zhang 等人的 SA-GAN 架构，hinge loss 

用截断技巧权衡多样性和保真度