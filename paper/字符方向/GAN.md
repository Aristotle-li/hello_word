## Generative Adversarial Network

### 思想：轮替优化

## 公式推导

* 造假水平要高   ==：目标==
* 鉴定水平要高   ==：手段==                    `所以盗墓的造假的一般都有很高的鉴赏水平`

  <img src="../../Library/Application Support/typora-user-images/image-20210328222120190.png" alt="image-20210328222120190" style="zoom:50%;" />

造假造美了，鉴赏水平该提高了，假的都识别不出来；

造假造碎了，造假水平该提高了，这个水平的鉴赏专家都欺骗不过去。

不直接对pg 建模，而是用神经网络逼近pg的分布，

假定有一个z~pz（z）

  KL散度：信息熵-交叉熵

​			  ：在机器学习领域的物理意义则是用来度量两个函数的相似程度或者相近程度



![image-20210330152654864](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330152654864.png)

![image-20210330152707709](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330152707709.png)





#### 注：

$p_g$太复杂，其他模型会想一些办法去直接面对他，而GAN 没有选择直接面对$p_g$，而是绕过了$p_g$，从采样角度去逼近$p_g$，所以GAN 是implicit density model，并非likelihood based model。



常规的生成模型，就是直接对$p_g$建模，例如概率图模型，概率图模型中的势函数就是 gan中的discriminator

MDSL

machine、deep、structured learning

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330095131558.png" alt="image-20210330095131558" style="zoom:67%;" />



## 算法实现：





<img src="../../Library/Application Support/typora-user-images/image-20210329162248441.png" alt="image-20210329162248441" style="zoom:67%;" />



### GAN 的实现可以类似于Auto-encode的思路

<img src="../../Library/Application Support/typora-user-images/image-20210329165945120.png" alt="image-20210329165945120" style="zoom:67%;" />

<img src="../../Library/Application Support/typora-user-images/image-20210329170309477.png" alt="image-20210329170309477" style="zoom:67%;" />

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330104718378.png" alt="image-20210330104718378" style="zoom:67%;" />



变分自编码机：encode和decode都相当于乘一个矩阵，encode是降维，decode是升维。

理论上用之前讲的自动编码器也可以生成图片，但是自动编码器在判断生成的图片与真实图片的距离的时候，不好定义loss，因为一个像素差别可能还不如6个像素的差别。一般来说使用自动编码器生成图片往往需要更大的网络才能生成更GAN接近的图片。

同样效果，和auto encode 相比，gan只需要更少的网络层数

Q：discrimination is easier to catch the relation between the components by top-down evaluation.

A：生成的时候每个component是独立生成的，不容易考虑他们之间的关系。等整张图片生成后，再去检查每个component之间的关系对不对，是比较容易的。

### 使用了GAN 的好处和decode相比

对于discriminator：generator为他生成了负样本

对于generator：虽然object是逐个component生成的，但是通过discriminator获得了全局视野（Hard to learn the correlation between components）

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330095922874.png" alt="image-20210330095922874" style="zoom:67%;" />

## 训练



<img src="../../Library/Application Support/typora-user-images/image-20210329190154847.png" alt="image-20210329190154847" style="zoom:67%;" />

> **`理论上可以这样做，在真实区域分布的数据给高分，其他区域给低分，但是实际上并不是这样做的，因为，实际的真实样本维度很高，在如此高维的空间里面样本量是十分稀疏的，所以discriminator并不知道真实样本在如此高维空间里面的真实分布,也就是说，在高维空间中，并不能让discriminator 没有出现real example的地方都判低分。`**

* **实际上是怎么train的：**

<img src="../../Library/Application Support/typora-user-images/image-20210329193027947.png" alt="image-20210329193027947" style="zoom:67%;" />



在这个过程中，$D(x)$也在学习，开始只会给fake example低分，而real example高分和其他区域可能高分，随着不断学习，$D(x)$也不断趋近于 real example的分布。





<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330095744278.png" alt="image-20210330095744278" style="zoom: 67%;" />



<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330102932524.png" alt="image-20210330102932524" style="zoom:67%;" />

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330103025573.png" alt="image-20210330103025573" style="zoom:67%;" />

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330103542783.png" alt="image-20210330103542783" style="zoom:67%;" />

### 小demo：

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330120549167.png" alt="image-20210330120549167" style="zoom:67%;" />



<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330122822124.png" alt="image-20210330122822124" style="zoom:67%;" />

### 如何反向传播：

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330172619058.png" alt="image-20210330172619058" style="zoom:67%;" />

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330172853926.png" alt="image-20210330172853926" style="zoom:67%;" />

p（1）=0.9

p(2)=0.1

1 * 0.9+2 * 0.1=1.1

9+2/10=1.1

训练discriminator的过程就是量JS散度，update参数很多次，找到JS散度的局部最高点。

g不能update太多，不然可能不能使得js散度下降



 

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330194105855.png" alt="image-20210330194105855" style="zoom:67%;" />

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330214716473.png" alt="image-20210330214716473" style="zoom:67%;" />

实际是优化这个函数，好处是update $ p_g$时 等同于是让$ p_g$产生的data当做是discriminator的positive example

## 一些问题：



无论generater的data质量怎么样，discriminator的loss都几乎为0。只要他能分开真的和假的，那么loss都是0，那么能识别的假钞和白纸对于discriminator来说就没有区别了，他只能给出真假，却对白纸该如何进步闭口不语，对于假钞应该如何改善也沉默寡言。 

也就是说无法进行阶段性进步啊，如何解决呢？

why：1、因为我们是用sample的方法，discriminator如果非常powerful的话，那么就可以overfitting，在低维空间找了不是规律的规律硬生生把他们分开了，

但是呢，JS散度又要求discriminator非常powerful，才可以量

2、p-data和p-g都是高维空间的low-dim manifold，所以这两个分布在manifold里面的overlap就没有！这就造成演化很困难，没有overlap，0-99步的的loss都是0，天敌是恐龙生物是小白兔，根本不给演化的机会，没有动力从小白兔进化成大白兔。理想的应该是猎豹和羚羊，他们的速度能力有overlap，可以在对抗中进步。

WGAN解决这个问题。

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330224259931.png" alt="image-20210330224259931" style="zoom:67%;" />



## 一些小技巧：加噪声

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330224923833.png" alt="image-20210330224923833" style="zoom:67%;" />

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330234248781.png" alt="image-20210330234248781" style="zoom:67%;" />

这种问题不会容易发现，比如，gan只会画红狗，画的很好，那他没画出其他颜色的狗，谁知道他画的怎么样，他也不会，没学到啊

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210330234832949.png" alt="image-20210330234832949" style="zoom:67%;" />

why：kl比较不容易产生，因为p-g在分母上，所以他不敢随便有0，因为一旦他有0 而p-data有值，loss就会趋向于无穷

reverse kl 容易产生，因为p-g不敢随便非0，一旦他非0，而p-data为0 loss就会无穷，所以这个时候gan比较保守，拟合了一个峰值分布后，不再前进。



## conditional gan

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210423115913114.png" alt="image-20210423115913114" style="zoom:67%;" />



若discriminator只看generator的输出，generator会学到生成几乎真实的图片，但是忽视所给的条件，比如给火车生成几乎真实的猫。

所以discriminator不仅判断 x is realistic or not 而且判断 c and x are matched or not

注意这个时候需要输入图片和文字的pair，作为真实数据，另外生成虚假的数据。

现在的辨别器优化的目标包括，本身的好坏，配对的好坏

但是，分数合在一起可能会不知道是什么原因犯了错，所以另一个思路是把这两个分开。

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210423180029469.png" alt="image-20210423180029469" style="zoom:67%;" />



### 算法：



<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210423162813479.png" alt="image-20210423162813479" style="zoom:67%;" />

### stackGAN：分阶段产生高分别率的图：

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210423180550471.png" alt="image-20210423180550471" style="zoom: 67%;" />

### Patch GAN：discriminator吃整张图参数太多，效果不好：

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210423181319052.png" alt="image-20210423181319052" style="zoom:50%;" />







f散度



<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210331195445623.png" alt="image-20210331195445623" style="zoom:67%;" />