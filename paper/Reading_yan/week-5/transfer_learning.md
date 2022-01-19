## transfer learning

### 1、

* 当target data很少时，为了防止overfitting：在source data 上train一个model，冻结大部分layer权重，在target上，只train剩下的layer

* which layer can be transferred？

  speech：copy the last few layer

  image：copy the first few layer



### 2、multitask learning

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518095005727.png" alt="image-20210518095005727" style="zoom:67%;" />

### 3、source data 和target data 的domain相差比较大，且target data 没有label 

domain-adversarial training

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518100139751.png" alt="image-20210518100139751" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518100727100.png" alt="image-20210518100727100" style="zoom:67%;" />

像GAN一样，不好train，domain classifier一定要努力分辨不同的domain label，奋力挣扎最后失败。

通常神经网络的参数都是朝着最小化损失的目标共同前进的，但在这个神经网络里，三个组成部分的参数各怀鬼胎：

- 对标签预测变量，要把不同数字的分类准确率做的超越越好
- 对域分类器，要正确地区分某张图片是属于哪个域
- 对特征提取器，要提高标签预测器的准确率，但要降低域分类器的准确率

这里，特征提取器和域分类器的目标是相反的，要做到这一点，只需要在两者之间加一层梯度反转的层即可，当NN做向后的时候，其中的参数更新往相反的方向走

### Domain adaptation：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518162106708.png" alt="image-20210518162106708" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518162553408.png" alt="image-20210518162553408" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518164156845.png" alt="image-20210518164156845" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518165217289.png" alt="image-20210518165217289" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518165527878.png" alt="image-20210518165527878" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518165543536.png" alt="image-20210518165543536" style="zoom:67%;" />

soft label： 

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518170841368.png" alt="image-20210518170841368" style="zoom:50%;" />

### zero-shot learning

source中没有target的类别：

语音到蜂鸣再辨识

image到attribute再分类：字典学习，attribute embedding，在embedding space 中，可以聚类

有时候属性的维数也很大，以至于我们对属性要做embedding的降维映射，同样的，还要把训练集中的每张图像都通过某种转换投影到embeddding空间上一个地方点，并且要保证属性投影的$ g（y ^ i）$和对应图像投影的$ f（x ^ i）$越接近越好，这里的$ f（x ^ n）$和$ g（y ^ n） $可以是两个神经网络

当遇到新的图像时，只需要将其投影到相同的空间，即可判断它与哪些属性对应的类别更接近

但如果我们根本就无法发现每个动物的属性$ y ^ i $是什么，那该怎么办？可以使用word vector，直接从维基百科上爬取图像对应的文字描述，再用word vector降级维提取特征，映射到同样的空间即可

以下这个损失函数存在一些问题，它会使模型把所有不同的x和y都投影到同一个点上：$$ f ^ *，g ^* = \ arg \ min \ limits_ {f，g} \ sum \ limits_n || f（x ^ n）-g（y ^ n）|| *2 $$类似用t-SNE的思想，我们既要考虑同一对$ x ^ n $和$ y ^ n $距离要接近，又要考虑不属于同一对的$ x ^ n $与$ y ^ m $距离要拉大（这是前面的式子没有考虑到的），于是有：$$ f ^ *g ^* = \ arg \ min \ limits {f，g} \ sum \ limits_n \ max（0，kf （x ^ n）\ cdot g（y ^ n）+ \ max \ limits_ {m \ ne n} f（x ^ n）\ cdot g（y ^ m））$$

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518111841520.png" alt="image-20210518111841520" style="zoom:67%;" />

类内点积（类内最大距离）-类间最大点积（类间最小距离）> k

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518113021691.png" alt="image-20210518113021691" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518113038186.png" alt="image-20210518113038186" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518113500507.png" alt="image-20210518113500507" style="zoom:50%;" />



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210518194631270.png" alt="image-20210518194631270" style="zoom:50%;" />

unit

munit

