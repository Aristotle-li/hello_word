作者：言煜秋
链接：https://zhuanlan.zhihu.com/p/368076507
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



## BN

源自机器学习中的独立同分布假设，细化到网络每一层，每轮训练数据分布不一致就不太好（层间的covariate shift ，协变量偏移）。也是一个控制梯度的手段。还可加学习率提速，可抛弃一些其他正则化手段如dropout和L2正则化。让网络更“泛”了（防止过拟合）。

为啥加在relu前？首先归一化对想高斯分布做的，假设让relu在前，那么0右边好不容易激活的值又被“归”了。

论文上看到的步骤：

- mean：E
- variance：Var
- 对x归一化（均值0方差1）：x^（z-score标准化：给定数据，距离其均值多少个标准差）
- 对y做scale&shift（y=γx^+β）：都归一化了还学啥？gamma&beta。

合起来就是y = ax +b。（a包含**γ**,Var，b包含**γ,β**,Var,E）

实现：类似维纳滤波的累计统计方法。设置momentum（如0.99），每轮batch训练不断统计moving_mean和moving_variance （当前batch的mean/var*0.01+历史值*0.99），测试时再固定。

PS：下了个新数据集finetune，结果因为老数据导致一些网络结点为0（weight-decay等正则化技术带来的dead kernel），训不动了。解决方法是**把Var固定为1**（weightdecay的问题作用在scale上，也就是a上），mxnet可以直接fix_gamma，pytorch没有，所以把网络conv(bias=false)+bn（scale）+relu设计为bn（scale的两个参数不学）+conv(bias=true)+relu。前者为什么bias是False？因为偏执经过bn其实不起作用（就算有bias，也要被归一化平移了）,后者为什么是True？因为我会将bn中的affine参数设为false（不加γ,β），然后为了补上这个β，在后面的conv里补个bias（平移）。（简单说就是conv bn relu变成bn conv relu，反正relu在最后就行了，且前两个针对平移的设置都改了）实际上

## SE注意力机制

chennel维度，提特征的深层（也许channel=512），这个512里有用没用的都有，需要一个权重来筛选。于是先全局池化，FC把这个512压小了之后，relu（nonlinear），再FC回去，然后sigmoid。这里压小了是为了融合全局信息，sigmoid可以当成一个0/1开关（毕竟很容易让梯度离谱）。

## MobileNetV2

1x1 bn relu        3x3 bn relu         1x1 bn        shortcut

最后为啥不加relu？低维信息怕丢失。

## 如何判断过拟合？怎么解决？

- training的指标特好，testing不行。
- trainloss降，valloss不变。

解决：

- 降模型复杂度
- 增加数据集，数据增强
- 正则化：对w。L1：稀疏解，L2平滑解。[我要鼓励娜扎：常见的过拟合解决方法](https://zhuanlan.zhihu.com/p/89590793)
- dropout（其实也算正则）
- 早停训练
- 决策树剪枝
- BN
- 集成学习

## 分割iou怎么代码实现

## mAP

- Precision：找的多少对 
- Recall：对的多少找到了

PR曲线，锯齿状，负相关。锯齿的正相关部分是因为在某个情况下多检测出一个（PR都涨）。所以计算取max precision to the right计算AUC（最后是阶梯状）。（调节置信度阈值，后续多找到一个目标，同时提升了pr，说明右侧最大值等更能体现左侧能力）。

m：所有种类平均

## 定点化量化怎么做的，精度下降多少

## anchor free检测方法

## 压缩模型

不断减少深度宽度，最后做效果-性能曲线，取性价比最高的斜率拐点（trade-off）

蒸馏

量化

剪枝

## 人脸误检

通用：数据增强，注意力，过拟合方法。

人脸和耳朵同时被检测：IOU加一个系数r=Big box area/Small box area，也就是大小box差距大时，按照这个比例扩大IOU结果，视情况把r映射到指数上平移后使用。同时除以小box的score的线性映射（当小box置信度很高，则保留）。

圆形小物体的误检：二分类网络二级检测



## 检测极端情况

通用：数据增强，注意力，过拟合方法。

逆光（脸黑背景亮）：使用分割结果压暗脸训练。

暗光：

- 全图暗，inf前处理抬亮。
- 二级检测：高阈值（0.5-1）是本来就检测到的，现在算中阈值（0.2-0.5）框平均亮度，如果较暗则进入二级检测，抬亮重新检测，置信度提升超过一定比例的，和进入高阈值的，保留。

小人脸：

- mosaic
- 先训练仅含m,l分支的网络，再迁移到包含s分支的网络针对s训练。

大人脸：

- 人脸超出屏幕，直接裁剪人脸做全图数据集。
- 视频上直接拼凑4帧到一张图检测。

大小通用：

- 改网络，yolo输出fm的size，sml分支的增减。
- 数据集。

眼镜墨镜：

- 同遮挡问题。
- 在yolo输出的分类加入有无眼镜的类别。
- 在关键点中，眼附近轮廓点出现偏差，可用下面的轮廓点拟合。C++中使用ceres实现。

头发：

分割和关键点都有这个问题。

马尔科夫随机场分割。

## 框的稳定性

- loss：C-IOU
- 维纳滤波
- anchor的值kmeans重新标定

## 遮挡问题

数据增强：随机擦除。使用关键点数据，人眼以下部分擦除（补白，模拟口罩），需要同时在背景处随机补白防止过拟合。

注意力机制：在中间block加注意力模块

dropblock



## 关键点偏移恢复

基于分割恢复

基于边缘恢复



## 测距

实际距离-pixel距离函数：人脸边缘的warping

pixel距离-yaw微调函数









Initialization参考[深度学习调参有哪些技巧？9645 关注 · 52 回答问题](https://www.zhihu.com/question/25097993)random_uniform_initializerxavieglorot_uniform
Learning rate参考：[学习率 Learning Rate](https://www.cnblogs.com/keguo/p/6244253.html)[www.cnblogs.com/keguo/p/6244253.html![img](https://pic2.zhimg.com/v2-b78698b31a42ab3d9eca6278ce4512f5_180x120.jpg)](https://www.cnblogs.com/keguo/p/6244253.html)[Momentum and Learning Rate Adaptation](https://willamette.edu/~gorr/classes/cs449/momrate.html)[willamette.edu/~gorr/classes/cs449/momrate.html![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-03849f4b36ffe0d8bdf51b058c316ec4_ipico.jpg)](https://willamette.edu/~gorr/classes/cs449/momrate.html)[陈志远：[译\]如何找到一个好的学习率(learning rate)57 赞同 · 4 评论文章![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-b72dcc7461d242c254814a2195b2a106_180x120.jpg)](https://zhuanlan.zhihu.com/p/50499794)batch大，可以支持更高的学习率工具：fast.aiBatch Normalization早期解释主要是说概率分布的，将每一层输入分布都归一化到N（0,1）上，减少了Internal Covariate Shift（让输入落在敏感区域，让梯度变大-->收敛快）。作用是解决反传梯度消失的问题，提升训练稳定性，且加速训练。在去年的论文[《How Does Batch Normalization Help Optimization?》](https://arxiv.org/abs/1805.11604)里边，作者否定了原来的一些观点，提出了关于BN的新理解：**BN主要作用是使得整个损失函数的landscape更为平滑，从而使得可以更平稳地进行训练。**参考：[BN究竟起了什么作用？一个闭门造车的分析 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/6992/comment-page-1)[spaces.ac.cn/archives/6992/comment-page-1](https://spaces.ac.cn/archives/6992/comment-page-1)
**ReLU or others** 为什么需要激活函数？因为如果没有，再深的网络也是线性，学不了复杂的映射关系。**sigmoid和tanh** sigmoid：值域（0,1） tanh：      值域（-1,1）比上者稍微好点二者缺点是都容易梯度消失（除非特殊情况，如二分类输出层用sigmoid，多分类softmax）**ReLU** ReLU常用，缺点是x为负导数为0，可能导致某些神经元死亡。Xception已经实验证明了   Depthwise卷积后再加ReLU效果会变差，作者认为Depthwise输出太浅了用ReLU会带来信息丢失。LeakyReLU & PReLUx为负时加个小斜率，防止神经元失活。前者的斜率固定，后者随数据变化 。ReLU6x>6后为横线，防止大网络爆炸。在移动端设备float16的低精度的时候，也能有很好的数值分辨率。

LOSS交叉熵：[飞鱼Talk：损失函数 - 交叉熵损失函数1673 赞同 · 118 评论文章![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-8f3a8a11696f62e720bf6b860a1a6a76_180x120.jpg)](https://zhuanlan.zhihu.com/p/35709485)Focal loss[中国移不动：5分钟理解Focal Loss与GHM——解决样本不平衡利器1672 赞同 · 131 评论文章![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-24d633d8da3ada9c9e020dbd19017521_180x120.jpg)](https://zhuanlan.zhihu.com/p/80594704)[机器学习大牛最常用的5个回归损失函数，你知道几个？](https://baijiahao.baidu.com/s?id=1603857666277651546&wfr=spider&for=pc)[中国移不动：5分钟理解Focal Loss与GHM——解决样本不平衡利器1672 赞同 · 131 评论文章![img](https://pic2.zhimg.com/v2-24d633d8da3ada9c9e020dbd19017521_180x120.jpg)](https://zhuanlan.zhihu.com/p/80594704)常用loss和区别**[机器学习大牛最常用的5个回归损失函数，你知道几个？](https://baijiahao.baidu.com/s?id=1603857666277651546&wfr=spider&for=pc)[baijiahao.baidu.com/s?id=1603857666277651546&wfr=spider&for=pc![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-a777867a65349b137950290deb2f6525_180x120.jpg)](https://baijiahao.baidu.com/s?id=1603857666277651546&wfr=spider&for=pc)