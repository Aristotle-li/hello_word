

> 题目：Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering
>
> 来源：NIPS 2016
>
> 作者：Michaël Defferrard Xavier Bresson Pierre Vandergheynst

### motivation

第一代GCN：SCNN存在计算复杂度高和无法保证局部链接的缺点，为了解决这一缺陷，设计了ChebNet。

- $\boldsymbol{f}$的Graph傅里叶变换：$\hat{\boldsymbol{f}} = \boldsymbol{\Phi}^T \boldsymbol{f}$
- 卷积核$\boldsymbol{h}$的Graph傅里叶变换为：$\hat{\boldsymbol{h}}=(\hat{h}_1,…,\hat{h}_n)$，其中，$\hat{h}_k=\langle h, \phi_k \rangle, k=1,2…,n$。实际上，$\hat{\boldsymbol{h}}=\boldsymbol{\Phi}^T \boldsymbol{h}$.
- 求图傅里叶变换向量$\hat{\boldsymbol{f}} \in \mathbb{R}^{N \times 1}$和$\hat{\boldsymbol{h}} \in \mathbb{R}^{N \times 1}$ 的element-wise乘积，等价于将$\hat{\boldsymbol{h}}$组织成对角矩阵的形式，即$diag[\hat{h}(\lambda_k)] \in \mathbb{R}^{N \times N}$，再求$diag[\hat{h}(\lambda_k)] $和$\hat{\boldsymbol{f}}$矩阵乘法。
- 求上述结果的逆傅里叶变换，即左乘$\boldsymbol{\Phi}$。

则：图上的卷积定义为：
$$
(\boldsymbol{f} * \boldsymbol{h})_\mathcal{G}=\boldsymbol{\Phi} \text{diag}[\hat{h}(\lambda_1),…,\hat{h}(\lambda_n)] \boldsymbol{\Phi}^T \boldsymbol{f} \tag{1}
\\为什么\hat h是\lambda 的函数？
\\因为\hat h 就是"频率域"上的滤波器，就是自变量"频率"的函数，假如h=1,那么h就是全通滤波器，f经过傅里叶变换和反傅里叶变换后没有改变
$$



### idea

为了解决上述问题(1、在谱域中定义的滤波器不是自然就是局部的，2、与图傅立叶基的 $O(n^2)$  乘法，平移计算成本很高)，首先回顾一下，**图傅里叶变换**是关于特征值(频率)的函数$F(\lambda_1), …,F(\lambda_n)$, 即，$F(\boldsymbol{\Lambda})$，因此可以将上述卷积核$\boldsymbol{g}_{\theta}$写作$\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{\Lambda})$。接着，将$\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{\Lambda})$定义成如下**k阶多项式**形式：
$$
\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}}(\boldsymbol{\Lambda}) \approx \sum_{k=0}^K \theta_k^{\prime} \boldsymbol{\Lambda}^k \\
\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}}(\boldsymbol{\Lambda})类比成频域滤波器，假如=e^x，那么级数就是1+x+\frac{x^2}{2}+\frac{x^3}{3}+\cdots,对应高通滤波器
$$
代入可以得到：
$$
\begin{aligned}
\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}} * \boldsymbol{x} &\approx \boldsymbol{\Phi} \sum_{k=0}^K \theta_k^{\prime} \boldsymbol{\Lambda}^k \boldsymbol{\Phi}^T \boldsymbol{x} \\
&= \sum_{k=0}^K \theta_k^{\prime} (\boldsymbol{\Phi} \boldsymbol{\Lambda}^k \boldsymbol{\Phi}^T) \boldsymbol{x} \\
&= \sum_{k=0}^K \theta_k^{\prime} (\boldsymbol{\Phi} \boldsymbol{\Lambda}\boldsymbol{\Phi}^T)^k \boldsymbol{x} \\
& = \sum_{k=0}^K \theta_k^{\prime} \boldsymbol{L}^k \boldsymbol{x}
\end{aligned}
$$
上述推导第三步应用了特征分解的性质。

上式为第二代的GCN。不需要做特征分解了，直接对拉普拉斯矩阵进行变换。可以事先把$\boldsymbol{L}^{k}$计算出来，这样前向传播的时候，就只需要计算矩阵和相邻的乘法。复杂度为$O(Kn^2)$。如果使用稀疏矩阵（$L$比较稀疏）算法，复杂度为$O(k|E|)$.

那么上式是如何体现localization呢？我们知道，矩阵的$k$次方可以用于求连通性，即1个节点经过$k$步能否到达另一个顶点，矩阵$k$次方结果中对应元素非0的话可达，为0不可达。因此$L$矩阵的$k$次方的含义就是代表$\text{k-hop}$之内的节点。进一步，根据拉普拉斯算子的性质。可以证明，如果两个节点的最短路径大于$K$的话，那么$L^{K}$在相应位置的元素值为0。因此，实际上只利用到了节点的K-Localized信息。

另外，作者提到，可以引入切比雪夫展开式来近似$\boldsymbol{L}^k$，因为**任何k次多项式都可以使用切比雪夫展开式来近似**。(类比泰勒展开式对函数进行近似）。

引入切比雪夫多项式（Chebyshev polynomial) $T_k(x)$的$K$阶截断获得对$\boldsymbol{L}^k$的近似，进而获得对$\boldsymbol{g}_{\theta}(\boldsymbol{\Lambda})$的近似，来降低时间复杂度。
$$
\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}}(\boldsymbol{\Lambda}) \approx \sum_{k=0}^K \theta_k^{\prime} T_k(\tilde{\boldsymbol{\Lambda}})
$$
其中，$\tilde{\boldsymbol{\Lambda}}=\frac{2}{\lambda_{max}}\boldsymbol{\Lambda}-\boldsymbol{I}_n$为经图拉普拉斯矩阵$L$的最大特征值（即谱半径）缩放后的特征向量矩阵（防止连乘爆炸）。$\boldsymbol{\theta}^{\prime} \in \mathbb{R}^{K}$表示一个**切比雪夫向量**，$\theta_k^{\prime}$是第$k$维分量。切比雪夫多项式$T_k(x)$使用递归的方式进行定义：$T_k(x)=2xT_{k-1}(x)-T_{k-2}(x)$，其中$T_0(x)=1,T_1(x)=x$。

此时，可以使用近似的$\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}}$替换原来的$\boldsymbol{g}_{\theta}$，可以得到：
$$
\begin{aligned}
\boldsymbol{g}_{\boldsymbol{\theta^{\prime}}} * \boldsymbol{x} &\approx \boldsymbol{\Phi} \sum_{k=0}^K \theta_k^{\prime} T_k(\tilde{\boldsymbol{\Lambda}}) \boldsymbol{\Phi}^T \boldsymbol{x} \
&\approx \sum_{k=0}^K \theta_k^{\prime} (\boldsymbol{\Phi} T_k(\tilde{\boldsymbol{\Lambda}}) \boldsymbol{\Phi}^T) \boldsymbol{x} \\
&=\sum_{k=0}^K \theta_k^{\prime} T_k(\tilde{\boldsymbol{L}}) \boldsymbol{x}
\end{aligned}
$$
其中，$\tilde{\boldsymbol{L}}=\frac{2}{\lambda_{max}} \boldsymbol{L}- \boldsymbol{I}_n$。

因此有递归式，
$$
\bar x_k =T_k(\tilde L)x = 2\tilde L\bar x_{k-1}-\bar x_{k-2} \in \mathbb R^n  \\
\boldsymbol{y}_{output} = \sigma(\sum_{k=0}^K \theta_k^{\prime} T_k(\tilde{\boldsymbol{L}}) \boldsymbol{x})= \sigma(\sum_{k=0}^K \theta_k^{\prime} \bar x_k)= \sigma([\bar x_0 ,...,\bar x_{K-1} ]\theta)
$$
参数向量$\boldsymbol{\theta}^{\prime} \in \mathbb{R}^{k}$，需要通过反向传播学习。时间复杂度也是$O(K|E|)$。



