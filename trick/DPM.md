# DPM(Deformable Parts Model)

**目标检测方法**

（1）基于cascade的目标检测

cascade的级联思想可以快速抛弃没有目标的平滑窗（sliding  window）,因而大大提高了检测效率，但也不是没缺点，缺点就是它仅仅使用了很弱的特征，用它做分类的检测器也是弱分类器，仅仅比随机猜的要好一些，它的精度靠的是多个弱分类器来实行一票否决式推举（就是大家都检测是对的）来提高命中率，确定分类器的个数也是经验问题。这节就来说说改进的特征，尽量使得改进的特征可以检测任何物体，当然Deep Learning学习特征很有效，但今天还是按论文发表顺序来说下其他方法，（服务器还没配置好，现在还不能大批跑Deep Learning ）,。
 



（2）基于形变部件的目标检测

形变部件模型检测方法是现在除了深度学习之外的还相对不错的目标检测方法，先来看下为什么要使用形变部件，在下图1中，同一个人的不同姿态，试问用前面几节中的什么方法可以检测到这些不同姿态的人？阈值不行，广义霍夫变换行吗？人的姿态是变换无穷的，需要太多的模板。霍夫森林投票？貌似可以，但是霍夫森立的特征是图像块，只适用于一些形变不大的物体，当图像块内的形变很大时同样不太适用。那么ASM可以吗？想想也是和广义霍夫变换一样，需要太多的均值模板。归根结底就是我们没有很好的形状描述方法，没有好的特征。而Pedro几乎每发表一篇论文就改进一下形状描述的方法，最终由简单的表示方法到语法形式的表示方法，其演化过程可以在参考文献[4]中看出，参考文献[4]是Pedro的博士论文。



![img](https://img-blog.csdn.net/20130704160223000?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3VvcXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

图1



**1 思路**



 DPM是一个非常成功的目标检测算法，连续获得VOC（Visual Object  Class）07,08,09年的检测冠军。目前已成为众多分类器、分割、人体姿态和行为分类的重要部分。DPM可以看做是HOG（Histogrrams of Oriented Gradients）的扩展，大体思路与HOG一致。先计算梯度方向直方图，然后用SVM（Surpport Vector  Machine  ）训练得到物体的梯度模型（Model）。有了这样的模板就可以直接用来分类了，简单理解就是模型和目标匹配。DPM只是在模型上做了很多改进工作。

 ![img](http://images.cnitblog.com/blog/460184/201310/23084751-224aec7a393b4fd7ba3d8769b77826c0.png)

 上图是HOG论文中训练出来的人形模型。它是单模型，对直立的正面和背面人检测效果很好，较以前取得了重大的突破。也是目前为止最好的的特征（最近被CVPR20 13年的一篇论文 《Histograms of Sparse Codes for Object Detection》 超过了）。但是，  如果是侧面呢？所以自然我们会想到用多模型来做。DPM就使用了2个模型，主页上最新版本Versio5的程序使用了12个模型。

 
 

​             ![img](http://images.cnitblog.com/blog/460184/201310/23084752-0e98ddb40be74be0a8b28fe86bdb5007.png)![img](http://images.cnitblog.com/blog/460184/201310/23084752-cba1c1687ac24355b7a715f048b2c08d.png)

 

 

 上图就是自行车的模型，左图为侧面看，右图为从正前方看。训练的时候只是给了一堆自行车的照片，没有标注是属于component 1，还是component  2.直接按照边界的长宽比，分为2半训练。这样肯定会有很多很多分错了的情况，训练出来的自然就失真了。不过没关系，论文里面只是把这两个Model当做初始值。重点就是作者用了多模型。

 ![img](http://images.cnitblog.com/blog/460184/201310/23084752-6a7b573e1e8c42bf946eb03ac1d9e668.png)

  

 

 上图右边的两个模型各使用了6个子模型，白色矩形框出来的区域就是一个子模型。基本上见过自行车的人都知道这是自行车。之所以会比左边好辨识，是因为分错component类别的问题基本上解决了，还有就是图像分辨率是左边的两倍，这个就不细说，看论文。

 有了多模型就能解决视角的问题了，还有个严重的问题，动物是动的，就算是没有生命的车也有很多款式，单单用一个Model，如果动物动一下，比如美女搔首弄姿，那模型和这个美女的匹配程度就低了很多。也就是说，我们的模型太死板了，不能适应物体的运动,特别是非刚性物体的运动。自然我们又能想到添加子模型，比如给手一个子模型，当手移动时，子模型能够检测到手的位置。把子模型和主模型的匹配程度综合起来，最简单的就是相加，那模型匹配程度不就提高了吗？还有个小细节，子模型肯定不能离主模型太远了，试想下假如手到身体的位置有两倍身高那么远，那这还是人吗？也许这是个检测是不是鬼的好主意。所以我们加入子模型与主模型的位置偏移作为Cost,也就是说综合得分要减去偏移Cost.本质上就是使用子模型和主模型的空间先验知识。

 ![img](http://images.cnitblog.com/blog/460184/201310/23084753-d2050789027843fbb42051d4d951e3fa.png)

  

 

 来一张合影。最右边就是我们的偏移Cost,圆圈中心自然就是子模型的理性位置，如果检测出来的子模型的位置恰好在此，那Cost就为0，在周边那就要减掉一定的值，偏离的越远减掉的值越大。


 



参考文献[1]、[2]、[3]分别讲述了如何利用形变模型描述物体（特征阶段）、如何利用形变部件来做检测（特征处理+分类阶段）、如何加速检测。文献[1]的形变部件。在Deformable Part Model中，通过描述每一部分和部分间的位置关系来表示物体（part+deformable  configuration）。其实早在1973年，Part Model就已经在 “Therepresentation and matching  of pictorial structures” 这篇文章中被提出了。



 ![img](https://img-blog.csdn.net/20130704160310671?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3VvcXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



图2

​      Part Model中，我们通过描述a collection of parts以及connection between  parts来表示物体。图2表示经典的弹簧模型，物体的每一部分通过弹簧连接。我们定义一个energy  function，该函数度量了两部分之和：每一部分的匹配程度，部分间连接的变化程度（可以想象为弹簧的形变量），与模型匹配最好的图像就是能使这个energy function最小的图片。

形式化表示中，我们可以用一无向图 G=(V,E) 来表示物体的模型，V={v1,…,vn} 代表n个部分，边 (vi,vj)∈E 代表两部分间的连接。物体的某个实例的configuration可以表示为 L=(l1,…,ln)，li 表示为 vi 的位置（可以简单的将图片的configuration理解为各部分的位置布局，实际configuration可以包含part的其他属性）。

给定一幅图像，用 mi(li) 来度量vi 被放置图片中的 li 位置时，与模板的不匹配程度；用 dij(li,lj) 度量 vi,vj 被分别放置在图片中的 li,lj位置时，模型的变化程度。因此，一副图像相对于模型的最优configuration，就是既能使每一部分匹配的好，又能使部分间的相对关系与模型尽可能的相符的那一个。同样的，模型也自然要描述这两部分。可以通过下面的（公式一）描述最优configuration：



 ![img](https://img-blog.csdn.net/20130704160407828?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3VvcXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



 公式1 

​    优化公式1其实就是马尔科夫随机场中的经典问题求解，可以用上面说的BP算法求解。说的理论些就是最大化后验概率（MAP），因为从随机场中很容易转换到概率测度中（gibbs measure）,相关理论可以学习概率图模型（probabilistic graphical  model）。识别的时候采用就是采用部件匹配，并且使得能量最小，这有点类似于ASM,但是ASM没有使用部件之间的关系，只是单纯的让各匹配点之间的代价和最小。匹配结果如图3所示：



 ![img](https://img-blog.csdn.net/20130704160436609?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3VvcXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



 图3

​    上面的方法没有用到机器学习，另外部件的寻找也不是一件容易的事情，因为首先要大概预估部件的位置，因此这个方法也有缺点，但这个形变部件的思想可以用来作为特征，接着就来看下Pedro的第二篇文献[2]如何用它来做目标检测。

Pedro在文献[2]中基于形变模型的目标检测用到了三方面的知识：1.Hog Features 2.Part Model 3. Latent SVM。

\1.   作者通过Hog特征模板来刻画每一部分，然后进行匹配。并且采用了金字塔，即在不同的分辨率上提取Hog特征。

\2.   利用上段提出的Part Model。在进行object detection时，detect window的得分等于part的匹配得分减去模型变化的花费。

\3.   在训练模型时，需要训练得到每一个part的Hog模板，以及衡量part位置分布cost的参数。文章中提出了Latent SVM方法，将deformable part model的学习问题转换为一个分类问题。利用SVM学习，将part的位置分布作为latent  values，模型的参数转化为SVM的分割超平面。具体实现中，作者采用了迭代计算的方法，不断地更新模型。

针对上面三条，发现难题如下：

1）部件从何而来？

2）如何用部件做检测？

在基于部件做目标检测之前，PASCAL VOC 2006年Dalal-Triggs的方法是直接用HOG作为特征，然后直接基于不同尺度的滑动窗口做判别，像一个滤波器，靠这个滤波器赢得短时的荣誉，但不能抗大形变的目标。

Pedro改进了Dalal-Triggs的方法，他计算作为一个得分，其中beta是滤波器，phi(x)是特征向量。通过滤波器找到一个根（root）部件p0，根部件有专门的滤波器，另外还有一系列非根部件（parts）p1…pn,然后把他们组成一个星形结构（此时回顾图1的形变模型思想）。

每个部件用![img](https://img-blog.csdn.net/20130704160617000?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3VvcXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)来表示，其中X,Y是坐标，L表示金字塔级别。当这个星形结构的匹配得分减去模型变化的代价得到最终分最高时，就完成了匹配，如公式2所示：



 ![img](https://img-blog.csdn.net/20130704160642703?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3VvcXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



 公式2

​    其中F’表示滤波器的向量化表示,b是偏移项，H表示特征金字塔。

现在假设滤波器解决了部件，完成了匹配，解答了第二个疑问，但是滤波器从何而来，简单的说就是这个滤波器的权重beta是多少？

现在不知道部件，也不知道滤波器，没有滤波器就没有部件，没有部件也求不出滤波器的参数，这就是典型的EM算法要解决的事情，但是作者没有使用EM算法，而是使用隐SVM（Latent SVM）的方法，隐变量其实就是类似统计中的因子分析，在这里就是找到潜在部件。

在训练的时候对部分部件进行打标签，用他们求beta,然后用beta再来找潜在部件，因此使用coordinatedescent迭代求解，再一次遇到这个求解方法。有了部件和打分，就是寻找根部件和其他部件的结合匹配最优问题，可以使用动态规划，但很慢，具体请参考文献[2]。



# 2.检测



 检测过程比较简单：

 综合得分：

 ![img](http://images.cnitblog.com/blog/460184/201310/23084753-5d001d1fa16e4f0f978f8aeea37bf3ec.png)

 ![img](http://images.cnitblog.com/blog/460184/201310/23084753-cbf162f80f694132a2d57d81d296d197.png)是rootfilter (我前面称之为主模型)的得分，或者说是匹配程度，本质就是![img](http://images.cnitblog.com/blog/460184/201310/23084754-5ca73b32ee8b41339bd1d5736abd33bd.png)和![img](http://images.cnitblog.com/blog/460184/201310/23084754-e7e5aa0805c34b0f818c2d4eab016fb7.png)的卷积，后面的partfilter也是如此。中间是n个partfilter（前面称之为子模型）的得分。![img](http://images.cnitblog.com/blog/460184/201310/23084754-43ff9657a8014a44b997d05712caa794.png)是为了component之间对齐而设的rootoffset.![img](http://images.cnitblog.com/blog/460184/201310/23084754-e3acf36523ac45d69dd7a1fc59539596.png)![img](http://images.cnitblog.com/blog/460184/201310/23084754-f65c36ecbd374ddebcafc7b9b37bfa25.png) 为rootfilter的left-top位置在root feature map中的坐标，![img](http://images.cnitblog.com/blog/460184/201310/23084754-2433dd97b9c94e1e8ab34d76e9a6edd1.png)为第![img](http://images.cnitblog.com/blog/460184/201310/23084755-90f46651ef7d4186999aad851e407e73.png)个partfilter映射到part feature map中的坐标。![img](http://images.cnitblog.com/blog/460184/201310/23084755-63b8b5256bbd4df0952cc3509a4ffdd3.png)是因为part feature map的分辨率是root feature map的两倍，![img](http://images.cnitblog.com/blog/460184/201310/23084755-5d3b2f25b97746cda6449c3185fa376e.png)为相对于rootfilter left-top 的偏移。

 ![img](http://images.cnitblog.com/blog/460184/201310/23084755-3f75f247d4c340a68d8d8bc57d49cb82.png) 的得分如下：

 ![img](http://images.cnitblog.com/blog/460184/201310/23084755-e2e0d3d96cfe48c8be4a874246c4f73c.png)

 上式是在patfilter理想位置![img](http://images.cnitblog.com/blog/460184/201310/23084756-a49b7c08e1cb47679ffa491b56ce4bc0.png),即anchor position的一定范围内，寻找一个综合匹配和形变最优的位置。![img](http://images.cnitblog.com/blog/460184/201310/23084756-334ba3734f0f47a099fcda6bb8461083.png)为偏移向量，![img](http://images.cnitblog.com/blog/460184/201310/23084756-35a67d8f59d54e0fb485dd425af0a5ad.png)为偏移向量![img](http://images.cnitblog.com/blog/460184/201310/23084756-67b88c6ed0b44e51af0b70365a1869d1.png)，![img](http://images.cnitblog.com/blog/460184/201310/23084757-8be6610b23b04639b307f956abab9bf9.png)![img](http://images.cnitblog.com/blog/460184/201310/23084757-d7430d79f6d048f3a04ae8f42e06e814.png)为偏移的Cost权值。比如![img](http://images.cnitblog.com/blog/460184/201310/23084757-a233368f148a479ebcb6ce1e3a7f678d.png)则![img](http://images.cnitblog.com/blog/460184/201310/23084757-ad9baa34103a4f71b19fcf12e9db54fd.png)![img](http://images.cnitblog.com/blog/460184/201310/23084757-5b7da003973b4f228ec5caf8147f4bd4.png)即为最普遍的欧氏距离。这一步称为距离变换，即下图中的transformed response。这部分的主要程序有**train.m、featpyramid.m、dt.cc.**

![img]()
 

 ![img](http://images.cnitblog.com/blog/460184/201310/23084757-d60c6d0ceb8d419385a2c8ad1017ee55.jpg)

​    在文献[2]中虽然使用了金字塔来加速搜寻速度，但是对星形结构组合的搜索匹配计算量也很大，检测速度稍慢。

因此接着来看第三篇文献[3],文献[3]就是加速检测过程，对于星形结构模型采用cascade来判断，来快速抛去没有有效信息的part，其实实际中根部件的位置对匹配起着很大作用，然后依次对其他部件（n+1）,有了这种关系，取一些部件子集后我们可以采用cascade来修剪、抛去一些不是好配置的部件组合（官方用语叫**配置**），这样一些在弱分类器中评分高的组合进入更一步的判断，类似**于cascade的级联思想**，但是要注意形变模型的每个部件应该是相关的，而不应该像上节那样harr-like特征之间独立,依次判断在这里行不通，这里其实是个子序列匹配问题,文献[7]提出过一种解决方法，pedro又改进了此方法，在原来n+1个部件的基础上增加n+1可以快速计算的简单部件，这样打乱之后，子序列匹配的代价就小了一些。

下面正式进入检测流程，看看怎么来加速的，大概流程如图4所示：



 ![img](https://img-blog.csdn.net/20130704160712656?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3VvcXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



图4

  


 

# 3.训练

## 3.1多示例学习（Multiple-instance learning）

### 3.1.1 MI-SVM

 一般机器学习算法，每一个训练样本都需要类别标号（对于二分类：1/-1）。实际上那样的数据其实已经经过了抽象，实际的数据要获得这样的标号还是很难，图像就是个典型。还有就是数据标记的工作量太大，我们想偷懒了，所以多只是给了正负样本集。负样本集里面的样本都是负的，但是正样本里面的样本不一定都是正的，但是至少有一个样本是正的。比如检测人的问题，一张天空的照片就可以是一个负样本集；一张某某自拍照就是一个正样本集（你可以在N个区域取N个样本，但是只有部分是有人的正样本）。这样正样本的类别就很不明确，传统的方法就没法训练。

 疑问来了，图像的不是有标注吗？有标注就应该有类别标号啊?这是因为图片是人标的，数据量特大，难免会有些标的不够好,这就是所谓的弱监督集（weakly supervised set）。所以如果算法能够自动找出最优的位置，那分类器不就更精确吗？  标注位置不是很准确，这个例子不是很明显，还记得前面讲过的子模型的位置吗？比如自行车的车轮的位置，是完全没有位置标注的，只知道在bounding  box区域附件有一个车轮。不知道精确位置，就没法提取样本。这种情况下，车轮会有很多个可能的位置，也就会形成一个正样本集，但里面只有部分是包含轮子的。

 针对上述问题《Support Vector Machines for Multiple-Instance  Learning》提出了MI-SVM。本质思想是将标准SVM的最大化样本间距扩展为最大化样本集间距。具体来说是选取正样本集中最像正样本的样本用作训练，正样本集内其它的样本就等候发落。同样取负样本中离分界面最近的负样本作为负样本。因为我们的目的是要保证正样本中有正，负样本不能为正。就基本上化为了标准SVM。取最大正样本（离分界面最远），最小负样本（离分界面最近）：

 ![img](http://images.cnitblog.com/blog/460184/201310/23084758-a3f04fd2c01246c08a0081b35aac4061.png)

 对于正样本：![img](http://images.cnitblog.com/blog/460184/201310/23084758-e2444955eaab4ae893757c1b7514dc4c.png) 为正样本集中选中的最像大正样本的样本。

 对于负样本：可以将max展开，因为最小的负样本满足的话，其余负样本就都能满足，所以任意负样本有：

 ![img](http://images.cnitblog.com/blog/460184/201310/23084758-abd1efa7cb354f7fb9917520619426be.png)

 目标函数：

 ![img](http://images.cnitblog.com/blog/460184/201310/23084758-34e48bbe7bdd4a7e9fed85defd0ff8f7.png)

 也就是说选取正样本集中最大的正样本，负样本集中的所有样本。与标准SVM的唯一不同之处在于拉格朗日系数的界限。

 ![img](http://images.cnitblog.com/blog/460184/201310/23084758-244a445e2504494cbb852e56be6ff18b.png)

 而标准SVM的约束是：

 ![img](http://images.cnitblog.com/blog/460184/201310/23084759-c780a833274d429b8cf7d60b35c4a432.png)

 最终化为一个迭代优化问题:

 ![img](http://images.cnitblog.com/blog/460184/201310/23084759-a9bbb5649e794654b405a1cc61970a89.png)

 思想很简单:第一步是在正样本集中优化；第二步是优化SVM模型。与K-Means这类聚类算法一样都只是简单的两步，却爆发了无穷的力量。

 这里可以参考一篇博客[Multiple-instance learning](http://blog.csdn.net/pkueecser/article/details/8274713)。

### 3.1.2 Latent SVM

 1）我觉得MI-SVM可以看成 Latent-SVM的一种特殊情况。首先解释下Latent变量，MI-SVM决定正样本集中哪一个样本作为正样本的![img](http://images.cnitblog.com/blog/460184/201310/23084759-73e980f6b17a44eda22396e66b067cfd.png)就是一个latent变量。不过这个变量是单一的，比较简单，取值只是正样本集中的序号而已。而LSVM 的latent变量就特别多，比如bounding box的实际位置x,y，在HOG特征金字塔中的某level中，样本component  ID。也就是说我们有了一张正样本的图片，标注了bounding box，我们要在某一位置，某一尺度，提取出一个区域作为某一component  的正样本。

 直接看Latent-SVM的训练过程：

 ![img](http://images.cnitblog.com/blog/460184/201310/23084759-30a312f229c0413388ae09299c55bbaa.png)

 这一部分还牵扯到了Data-minig。先不管，先只看循环中的3-6,12.

 3-6就对于MI-SVM的第一步。12就对应了MI-SVM的第二步。作者这里直接用了梯度下降法，求解最优模型β。

 2）现在说下Data-minig。作者为什么不直接优化，还搞个Data-minig干嘛呢？因为，负样本数目巨大，Version3中用到的总样本数为2^28，其中Pos样本数目占的比例特别低，负样本太多，直接导致优化过程很慢，因为很多负样本远离分界面对于优化几乎没有帮助。Data-minig的作用就是去掉那些对优化作用很小的Easy-examples保留靠近分界面的Hard-examples。分别对应13和10。这样做的的理论支撑证明如下：

 ![img](http://images.cnitblog.com/blog/460184/201310/23084800-2e55cda5c2cb416697dd469ab2b692e4.png)

 3）再简单说下随机梯度下降法（Stochastic Gradient Decent）：

 首先梯度表达式：

 ![img](http://images.cnitblog.com/blog/460184/201310/23084800-1bb6e2d62d004dde89e49ff99760d0d5.png)

 梯度近似：

 ![img](http://images.cnitblog.com/blog/460184/201310/23084800-d646a9deafc244d1ae5a879eac04377a.png)

 优化流程：

 ![img](http://images.cnitblog.com/blog/460184/201310/23084801-6d05fc3435d048aa931b6a6bc17b15e5.png)

 这部分的主要程序：**pascal_train.m->train.m->detect.m->learn.cc**

## 3.2 训练初始化

 LSVM对初始值很敏感，因此初始化也是个重头戏。分为三个阶段。英语方面我就不班门弄斧了，直接上截图。

 ![img](http://images.cnitblog.com/blog/460184/201310/23084801-29c222f6873c4a75868d47f9ce8b8152.png)

 
 

 下面稍稍提下各阶段的工作，主要是论文中没有的Latent 变量分析：

 Phase1:是传统的SVM训练过程，与HOG算法一致。作者是随机将正样本按照aspect  ration（长宽比）排序，然后很粗糙的均分为两半训练两个component的rootfilte。这两个rootfilter的size也就直接由分到的pos examples决定了。后续取正样本时，直接将正样本缩放成rootfilter的大小。

 Phase2:是LSVM训练。Latent variables  有图像中正样本的实际位置包括空间位置（x,y）,尺度位置level，以及component的类别c，即属于component1 还是属于  component 2。要训练的参数为两个 rootfilter，offset（b）

 Phase3:也是LSVM过程。

 先提下子模型的添加。作者固定了每个component有6个partfilter，但实际上还会根据实际情况减少。为了减少参数，partfilter都是对称的。partfilter在rootfilter中的锚点（anchor location）在按最大energy选取partfilter的时候就已经固定下来了。

 这阶段的Latent variables是最多的有：rootfilter（x,y,scale）,partfilters(x,y,scale)。要训练的参数为 rootfilters, rootoffset, partfilters, defs(![img](http://images.cnitblog.com/blog/460184/201310/23084803-cc736ddcc38849cf9973d27ea1408349.png)的偏移Cost)。

 这部分的主要程序：**pascal_train.m**

 **4.细节**

## 4.1轮廓预测（Bounding Box Prediction）

 ![img](http://images.cnitblog.com/blog/460184/201310/23084803-97f9fb0a6f354664ab1d973a7aa9cbf9.png)![img](http://images.cnitblog.com/blog/460184/201310/23084803-827cc92da46b4571ae898698e47fa55e.png)

 仔细看下自行车的左轮，如果我们只用rootfilter检测出来的区域，即红色区域，那么前轮会被切掉一部分，但是如果能综合partfilter检测出来的bounding box就能得到更加准确的bounding box如右图。

 这部分很简单就是用最小二乘（Least Squres）回归，程序中**trainbox.m**中直接左除搞定。

## 4.2 HOG

 作者对HOG进行了很大的改动。作者没有用4*9=36维向量，而是对每个8x8的cell提取18+9+4=31维特征向量。作者还讨论了依据PCA（Principle Component Analysis）可视化的结果选9+4维特征，能达到HOG 4*9维特征的效果。

 这里很多就不细说了。开题一个字都还没写，要赶着开题……主要是**features.cc。**

 源码分析：

 [DPM(Defomable Parts Model) 源码分析-检测](http://blog.csdn.net/soidnhp/article/details/12954195)

 [DPM(Defomable Parts Model) 源码分析-训练](http://blog.csdn.net/soidnhp/article/details/12954631)

 
 

 

参考文献：

[1] Pictorial Structures for Object Recognition. Pedro F.Felzenszwalb

[2]Object Detection with Discriminatively Trained Part Based Models.Pedro F. Felzenszwalb

[3]Cascade Object Detection with Deformable Part Models. Pedro F.Felzenszwalb

[4]From RigidTemplates To Grammars: Object Detection With Structured Models. Pedro F.Felzenszwalb

[5]Histogramsof oriented gradients for human detection. N. Dalal and B. Triggs

[6] http://bubblexc.com/y2011/422/

[7]A computational model for visual selection.Y. Amit and D.Geman


 

  

参考：http://blog.csdn.net/ttransposition/article/details/12966521


 



##              [     DPM（Deformable Part Model）原理详解（汇总）        ](https://www.cnblogs.com/aerkate/p/7648771.html)         

**写在前面：**

**DPM**（**Deformable Part Model**），正如其名称所述，可变形的组件模型，是一种基于组件的检测算法，其所见即其意。该模型由大神**Felzenszwalb**在2008年提出，并发表了一系列的cvpr，NIPS。并且还拿下了2010年，**PASCAL VOC**的“终身成就奖”。

 

由于DPM用到了HOG的东西，可以参考本人http://blog.csdn.net/qq_14845119/article/details/52187774

 

**算法思想：**

（1）Root filter+ Part filter：

该模型包含了一个8*8分辨率的根滤波器（Root filter）（左）和4*4分辨率的组件滤波器（Part  filter）（中）。其中，中图的分辨率为左图的2倍，并且Part filter的大小是Root  filter的2倍，因此，看的梯度会更加精细。右图为其高斯滤波后的2倍空间模型。

![img](http://img.blog.csdn.net/20160922175322951)

 

(左)Rootfilter(中) Part filter (右)高斯滤波后模型

（2）响应值（score）的计算：

响应值得分公式如下：

![img](http://img.blog.csdn.net/20160922175412140)

 

其中，

x 0, y 0, l 0分别为锚点的横坐标，纵坐标，尺度。

R 0,l 0 (x 0, y 0)为根模型的响应分数

Di,l 0−λ(2(x 0, y 0) + vi)为部件模型的响应分数

b为不同模型组件之间的偏移系数，加上这个偏移量使其与跟模型进行对齐

2(x 0, y 0)表示组件模型的像素为原始的2倍，所以，锚点*2

vi为锚点和理想检测点之间的偏移系数，如下图中红框和黄框

 

其部件模型的详细响应得分公式如下：

![img](http://img.blog.csdn.net/20160922175451906)

 

其中，

x, y为训练的理想模型的位置

Ri,l(x + dx, y + dy)为组件模型的匹配得分

di · φd(dx, dy))为组件的偏移损失得分

di ·为偏移损失系数

φd(dx, dy))为组件模型的锚点和组件模型的检测点之间的距离

简单的说，这个公式表明，组件模型的响应越高，各个组件和其相应的锚点距离越小，则响应分数越高，越有可能是待检测的物体。

 

（3）DPM特征定义：

![img](http://img.blog.csdn.net/20160922175541292)

 

DPM首先采用的是HOG进行特征的提取，但是又有别于HOG，DPM中，只保留了HOG中的Cell。如上图所示，假设，一个8*8的Cell，将该细胞单元与其对角线临域的4个细胞单元做归一化操作。

​     提取有符号的HOG梯度，0-360度将产生18个梯度向量，提取无符号的HOG梯度，0-180度将产生9个梯度向量。因此，一个8*8的细胞单元将会产生，（18+9）*4=108，维度有点高，**Felzenszwalb**大神给出了其优化思想。

​      首先，只提取无符号的HOG梯度，将会产生4*9=36维特征，将其看成一个4*9的矩阵，分别将行和列分别相加，最终将生成4+9=13个特征向量，为了进一步提高精度，将提取的18维有符号的梯度特征也加进来，这样，一共产生13+18=31维梯度特征。实现了很好的目标检测。

（4）DPM检测流程：

![img](http://img.blog.csdn.net/20160922175853692)

 

如上图所示，对于任意一张输入图像，提取其DPM特征图，然后将原始图像进行高斯金字塔上采样，然后提取其DPM特征图。对于原始图像的DPM特征图和训练好的Root filter做卷积操作，从而得到Root filter的响应图。对于2倍图像的DPM特征图，和训练好的Part  filter做卷积操作，从而得到Part filter的响应图。然后对其精细高斯金字塔的下采样操作。这样Root filter的响应图和Part filter的响应图就具有相同的分辨率了。然后将其进行加权平均，得到最终的响应图。亮度越大表示响应值越大。

（5）Latent SVM:

![img](http://img.blog.csdn.net/20160922180002716)

 

传统的Hog+SVM和DPM+LatentSVM的区别如上面公式所示。

​      由于，训练的样本中，负样本集肯定是100%的准确的，而正样本集中就可能有噪声。因为，正样本的标注是人工进行的，人是会犯错的，标注的也肯定会有不精确的。因此，需要首先去除里面的噪声数据。而对于剩下的数据，里面由于各种角度，姿势的不一样，导致训练的模型的梯度图也比较发散，无规则。因此需要选择其中的具有相同的姿势的数据，即离正负样本的分界线最近的那些样本，将离分界线很近的样本称为Hard-examples，相反，那些距离较远的称为Easy-examples。

​     实际效果图如下图所示：

![img](http://img.blog.csdn.net/20160922180050935)

 

**实验效果：**

如下图所示，左面为检测自行车的检测效果，右面为Root filter，Part filter，2维高斯滤波下的偏离损失图

![img](http://img.blog.csdn.net/20160922180205388)

 

**References:**

[1]: https://people.eecs.berkeley.edu/~rbg/latent/index.html

[2]: P. Felzenszwalb, D. McAllester, D.Ramanan A Discriminatively  Trained, Multiscale, Deformable Part Model IEEEConference on Computer  Vision and Pattern Recognition (CVPR), 2008

[3]: P. Felzenszwalb, R. Girshick, D.McAllester, D. Ramanan Object  Detection with Discriminatively TrainedPart Based Models IEEE  Transactions on Pattern Analysis and MachineIntelligence, Vol. 32, No.  9, Sep. 2010 

[4]: P. Felzenszwalb, R. Girshick, D.McAllester Cascade Object  Detection with Deformable Part Models IEEEConference on Computer Vision  and Pattern Recognition (CVPR), 2010

[5]: P. Felzenszwalb, D. McAllester ObjectDetection Grammars University of Chicago, Computer Science TR-2010-02, February2010

[6]: R. Girshick, P. Felzenszwalb, D.McAllester Object Detection with Grammar Models Neural InformationProcessing Systems (NIPS), 2011

[7]: R. Girshick From RigidTemplates to Grammars: Object Detection with Structured Models
Ph.D. dissertation, The University of Chicago, Apr. 2012