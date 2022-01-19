Masked Feature Prediction for Self-Supervised Visual Pre-Training



BEiT ：

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222140734689.png" alt="image-20211222140734689" style="zoom:50%;" />



One immediate solution is to imitate the language vocabulary by building a **visual vocabulary** that discretizes frame patches **into tokens**, as explored in BEiT [2, 73].

However, this requires an **external tokenizer** which can be limited in compute-intensive **video understanding** scenario.

not only an extra pre-training stage on 250M images, but also non-negligible training overhead in masked modeling.

一个直接的解决方案是通过构建**视觉词汇**来模拟语言词汇，将框架补丁**离散化为标记**，如BEiT[2,73]所述。"“但是，这需要一个**外部标记器**，它可以限制在计算密集型**视频理解**场景中。”“这不仅是一个额外的250米图像预训练阶段，而且在蒙面建模方面也有不可忽略的训练开销。



MAE：

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222141054727.png" alt="image-20211222141054727" style="zoom:50%;" />

像素作为预测目标，有一个潜在的缺点，那就是会让模型过度拟合局部统计数据（例如光照和对比度变化）和高频细节，而这些对于视觉内容的解释来说很可能并不是特别重要。



MaskFeat：

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222141422499.png" alt="image-20211222141422499" style="zoom:50%;" />



一个原图经过masked后进入encoder，然后linear一下，预测这个原图的HOG，具体做法是，首先获得原图的HOG特征图，然后分块，把要mask的部分平坦化，最小化预测的HOG和原始HOG的L2损失。

使用最小的数据扩充：随机调整大小的裁剪和水平翻转。mask 40%的图像块





随机mask：序列中的一些tokens被替换为[MASK]tokens（这是一个可学习的嵌入，指mask patch。）



像素作为预测目标，有一个潜在的缺点，那就是会让模型过度拟合局部统计数据（例如光照和对比度变化）和高频细节，而这些对于视觉内容的解释来说很可能并不是特别重要。





erms of both performance and efficiency. (ii) The discretization (tokenization) of visual signals is not necessary for masked visual prediction, and continuous feature regression (i.e. MaskFeat) can work well.

与BEiT相比，MaskFeat只需要计算HOG特征，摆脱了dVAE的tokenizer，而后者在250M DALL-E数据集上引入了额外的预训练阶段，并在mask预测期间引入了不可忽视的推理开销。



不同：没有decode，直接回归mask 区域的HOG特征















考虑五种不同类型的目标特征

Pixel colors：使用RGB值，这些值通过数据集的平均值和标准偏差标准化。我们最小化了模型预测和地面真实RGB值之间的l2距离。

不足：过度拟合局部统计数据（例如照明和对比度变化）和高频细节，这可能对视觉内容的解释无关紧要

HOG：是描述局部子区域内梯度方向或边缘方向分布的特征描述符。

好处：方向梯度直方图（HOG）是描述局部子区域内梯度方向或边缘方向分布的特征描述符，通过简单的梯度滤波（即减去相邻像素）来计算每个像素的梯度大小和方向来实现的。

 HOG的特点是善于捕捉局部形状和外观，同时对几何变化不敏感，对光的变化也有不变性，计算引入的开销还很小，可以忽略不计。它可以实现为两通道卷积，以在x轴和y轴上生成梯度（或通过减去相邻的水平和垂直像素）。

局部对比度归一化对于MaskFeat的预训练也是必不可少的

dVAE：BEiT

不足：额外计算

Deep features：预测有监督的CNN或ViT的特征，CNN：the last layers’ features corresponding to the masked patches  ViT： patch tokens

不足：额外计算，过拟合：

Pseudo-label：consider predicting class labels of masked patches. 每个patch被分配一个特定于位置的IN-1K伪标签。

不足：额外计算

有监督特征表现不好原因：同一类标签对于同一对象的局部形状和纹理是不变的，这使得MaskFeat无法对对象的内部结构建模。有监督特征不好，可能这些特征对于局部的mask来说过于全局



![image-20211222162952840](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222162952840.png)



![image-20211222163011353](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222163011353.png)





![image-20211222163544399](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222163544399.png)

MAE：

![image-20211222180231563](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222180231563.png)

MoCo v3[18]和DINO[9]是需要多视图训练和精心设计的增强的对比方法，而MaskFeat只使用单视图和最小增强

![image-20211222171705036](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222171705036.png)



Multi-tasking：

![image-20211222173342568](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222173342568.png)

型模型从更长的时间表中获益更多

![image-20211222173618259](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222173618259.png)

Masking ratio.

![image-20211222173750442](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222173750442.png)





Data augmentation.

![image-20211222173951799](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222173951799.png)

Linear probing：MoCo v3在ViT-L的最后一个区块中占77.6%，这表明基于对比的视觉预测方法和掩蔽视觉预测方法具有非常不同的特征。

MaskFeat通过fine-tuning学习良好的视觉知识，但不是线性可分离的特征。我们在这里的假设是，对比学习中的实例辨别损失为不同的图像创建了不同的嵌入（类），这些嵌入（类）可以在很大程度上简化为具有线性层的类级信息（类的子集）。 Our hypothesis here is that instance discrimination losses in contrastive learning create different embeddings (classes) for different images which can be largely reduced to classlevel information (a subset of classes) with a linear layer

![image-20211222174126079](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222174126079.png)



效果图

![image-20211222172738248](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211222172738248.png)

问题？

beit是预测被mask的visual token，mae是预测被mask的pixel，maskfeat去预测被mask的部分的HOG

HOG就能代表视觉内容的意义了么？ 是否有HOG手工特征就是适用于识别分类等任务设计的原因

文本是人类制造的本就有意义的东西，视觉图片自然界本就有，到底用什么才能拟合出来人类理解到的那个意义？



HOG是从原图提取出来的一种特征，目前证明预测HOG图比预测原图性能好，是不是可以理解为目前的结构并未从原图中挖掘出十分有效的“意义”信息。







其他人的思路：

像素级特征

各种手工特征

tokenizer：BEIT

离线预训练网络得到的特征



，在作为判断生成图像质量方面，没有本质区别。

- MAE、SimMIM：直接用像素评判；
- BEIT、PeCo：使用一个离线预训练的tokenizer：这个tokenizer和VQ-VAE挂钩，而VQ-VAE的目标是恢复像素——因此几乎可以认为，这种tokenizer的作用和像素级恢复是相当的；
- iBOT：将上述tokenizer改为在线训练，利用类似于teacher-student的方式做监督——我很喜欢它无需引入离线预训练的性质，虽然它的训练效率要低一些；
- SaGe：使用一个离线BYOL预训练的网络来抽特征；
- MaskFeat：使用手工的HOG特征——这是2005年的CVPR paper，新人们有多少能第一时间反应出HOG是啥玩意儿的？





- BEiT: 使用dVAE tokenizer构造bottleneck，将pixel-level details学在tokenzier参数中 ("BEiT  overcomes the above issue by predicting discrete visual tokens, which  summarizes the details to high-level abstractions.")
- MAE: 1) 增加了decoder部分用来记忆pixel-level details；2) encoder部分去除了[M]，把masked  patch信息推到decoder中；3) per-patch-norm 归一化掉细节信息，鼓励学习semantic content
- PeCo: 在BEiT tokenizer中加入perceptual loss (在style transfer里面充当content loss)，鼓励visual tokens保留semantic content，抑制具体的纹理、style等信息
- iBOT: 框架上类似BEiT+DINO，其中DINO部分得到的online tokenizer，通过data augmentation抑制细节信息的学习
- MaskFeat: 利用人工构造的HOG features作为学习目标，消除细节信息



基于BEiT中提出的masked image modeling  (MIM)预训练任务，可以发现目前的绝大多数工作都是从上面说的这个insight去提升自监督效果。问题中的提到的MaskFeat验证了人工构造的HOG特征，也可以起到很好的效果。希望未来有更形式化的工作，去指引大家创新。