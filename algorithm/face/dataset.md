首发于[AI on chip](https://www.zhihu.com/column/c_1178097838814363648)

# 常用公开人脸数据集汇总

[![言煜秋](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-fbb66a4ba07a08732d3cf8b417569f22_xs-20211222215013193.jpg)](https://www.zhihu.com/people/waitsop)

[言煜秋](https://www.zhihu.com/people/waitsop)

## 目录

- 人脸数据汇总

- - 人脸检测
  - 人脸关键点检测
  - 人脸表情
  - 年轻与性别
  - 人脸姿态
  - 人脸识别

- 常用人脸数据详细介绍

- - 人脸检测
  - 人脸关键点检测
  - 人脸表情
  - 年轻与性别
  - 人脸姿态
  - 人脸识别

人脸数据集续更新完善中~~~

## 一. 人脸数据汇总

## 1.人脸检测

人脸检测是要定位出图像中人脸的位置。

**Caltech 10000 (2007)**

这是一个灰度人脸数据集，使用Google图片搜索引擎用关键词爬取所得，包含了7,092张图，10,524个人脸，平均分辨率在304x312，除此之外还提供双眼，鼻子和嘴巴共4个坐标位置。在早期被使用的较多，现在的方法已经很少用灰度数据集做评测。[链接](http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/)

**FDDB (2010)**

这是被广泛用于人脸检测方法评测的一个数据集，FDDB（Face Detection Data Set and  Benchmark），它的提出是用于研究无约束人脸检测。所谓无约束指的是人脸表情、尺度、姿态、外观等具有较大的可变性。FDDB的图片都来自于  Faces in the Wild  数据集，图片来源于美联社和路透社的新闻报道图片，所以大部分都是名人，而且是自然环境下拍摄的。共2845张图片，里面有5171张人脸图像。在FDDB当中采用了椭圆标记法，它可以适应人脸的轮廓。具体来说，每个标注的椭圆形人脸由六个元素组成。(ra, rb, Θ, cx, cy, s)，其中ra，rb是椭圆的半长轴、半短轴，cx,  cy是椭圆的中心点坐标，Θ是长轴与水平轴夹角（头往左偏Θ为正，头往右偏Θ为负），s则是置信度得分。标注的结果是通过多人独立完成标注之后取标注的平均值，而且排除了以下的样本：长或宽小于20个像素的人脸区域；设定一个阈值，将像素低于阈值的区域标记为非人脸；远离相机的人脸区域被标记为非人脸；人脸被遮挡，2个眼睛都不在区域内的标记为非人脸。[链接](http://vis-www.cs.umass.edu/fddb/index.html)

**AFW (2013)**

FW数据集是人脸关键点检测非常早期使用的数据集，共包含205个图像，其中有473个标记的人脸。每一个人脸提供了方形边界框，6个关键点和3个姿势角度的标注。

**WIDER Face （2015）**

FDDB评测标准由于只有几千张图像，这样的数据集在人脸的姿态、尺度、表情、遮挡和背景等多样性上非常有限，训练出来的模型难以被很好的评判，算法很快就达到饱和。在这样的背景下香港中文大学提出了Wider-face数据集，在很长一段时间里，大型互联网公司和科研机构都在Wider-face上做人脸检测算法竞赛。Wider-face总共有32203张图片，共有393703张人脸,比FDDB数据集大10倍，而且在面部的尺寸、姿势、遮挡、表情、妆容、光照上都有很大的变化，算法不仅标注了框，还提供了遮挡和姿态的信息，自发布后广泛应用于评估性能比传统方法更强大的卷积神经网络。[链接](http://shuoyang1213.me/WIDERFACE/)

**MALF（2015）**

Multi-Attribute Labelled Faces  ，MALF是为了更加细粒度地评估野外环境中人脸检测模型而设计的数据库。数据主要来源于Internet，包含5250个图像，11931个人脸。每一幅图像包含正方形边界框，头部姿态的俯仰程度，包括小中大三个等级的标注。该数据集忽略了小于20*20或者非常难以检测的人脸，共包含大约838个人脸，占该数据集的7%。同时该数据集还提供了性别，是否带眼镜，是否遮挡，是否是夸张的表情等辅助信息。  [链接](http://www.cbsr.ia.ac.cn/faceevaluation/)

## 2.人脸关键点检测

**AR Face Database（1998） 标注：22**

包括126个人，超过4000张图。[链接](http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html)

**XM2VTS （1999）标注：68**

包含295个人，2360张正面图，大部分的图像是无表情，而且在同样的光照环境下。[链接](http://www.ee.surrey.ac.uk/CVSSP/xm2vtsdb/)

**BioID（2001）**

约1000幅图像,每个人脸标定20个关键点。[链接](https://www.bioid.com/About/BioID-Face-Database)

**FRGC-V2（2002）标注：5**

共466个人的4950张图，包括均匀的光照条件下的高质量图和不均匀的光照条件下的低质量图，标注了5个关键点。[链接](https://www.nist.gov/programs-projects/face-recognition-grand-challenge-frgc)

**CMU Multi-PIE（2010） 标注：39~68**

包含6152张图像[链接](https://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html)

**LFPW（CVPR2011）标注：29**

[Localizing Parts of Faces Using a Consensus of Exemplars](https://neerajkumar.org/projects/face-parts/facepartlocalization_cvpr_2011.pdf)

1432张图片，每张图片上有29个点。[链接](https://neerajkumar.org/projects/face-parts/)

**AFLW（ECCV 2011）标注：21**

[Annotated Facial Landmarks in the Wild: A Large-scale, Real-world Database](https://files.icg.tugraz.at/seafhttp/files/30f0dede-3e33-4d23-81d7-ef11a9d0fda4/koestinger_befit_11.pdf)

包括多姿态、多视角的大规模人脸数据库，一般用于评估面部关键点检测效果，图片来自于flickr的爬取。总共有21,997张图，25,993张面孔，由于是肉眼标记，不可见的关键点不进行标注。除了关键点之外，还提供了矩形框和椭圆框的脸部位还提供了矩形框和椭圆框的脸部位置标注，其中椭圆框的标注方法与FDDB相同。另外还有从平均3D人脸重建提供的3D的人脸姿态角标注。大部分图像是彩色图，也有少部分是灰度图，59%为女性，41%为男性，这个数据集非常适合做多角度多人脸检测，关键点定位和头部姿态估计，是关键点检测领域里非常重要的一个数据集。[链接](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)

**Helen（ECCV2012）标注：68**

[Interactive Facial Feature Localization](http://www.ifp.illinois.edu/~vuongle2/helen/eccv2012_helen_final.pdf)

包括训练集和测试集，测试集包含了330张人脸图片，训练集包括了2000张人脸图片[链接](http://www.ifp.illinois.edu/~vuongle2/helen/)

**IBUG（2013）标注：68**

随着300W一起发布的数据集，包含了135张人脸图片，每张人脸图片被标注了68个特征点。[链接](https://ibug.doc.ic.ac.uk/resources/300-W/)

**AFW（Annotated Faces in the Wild）（CVPR2012）标注：6**

[Face detection, pose estimation and landmark localization in the wild](https://www.cs.cmu.edu/~deva/papers/face/face-cvpr12.pdf)

AFW数据集是人脸关键点检测非常早期使用的数据集，共包含205个图像，其中有473个标记的人脸。每一个人脸提供了方形边界框，6个关键点和3个姿势角度的标注，数据库虽然不大，额外的好处是作者给出了其2012 CVPR的论文和程序以及训练好的模型。[链接](https://www.cs.cmu.edu/~deva/papers/face/index.html)

**COFW（ICCV2013）标注：29**

[Robust face landmark estimation under occlusion](http://www.vision.caltech.edu/xpburgos/ICCV13/Data/Poster.pdf)

遮挡图像较多，包括1852张图像,其中训练姐1345张图像，测试集507张图像[链接](http://www.vision.caltech.edu/xpburgos/ICCV13/)

**300W（ICCV2013）标注：68**

[300 Faces in-the-Wild Challenge: The first facial landmark localization Challenge](https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_iccv_2013_300_w.pdf)

包含了300张室内图和300张室外图，其中数据集内部的表情，光照条件，姿态，遮挡，脸部大小变化非常大，因为是通过Google搜索“party”,  “conference”等较难等场景搜集而来。该数据集标注了68个关键点，一定程度上在这个数据集能取得好结果的，在其他数据集也能取得好结果。该数据集每个图像上包含不止一张人脸，但是对于每张图像只标注一张人脸。其中：AFW(337)，Helen(train 2000+test 330)，IBUG(135)，LFPW(train 811+test  224)。共计3148张图像，测试集有554+135=689张图像。[链接](https://ibug.doc.ic.ac.uk/resources/300-W/)

**300-W challenge（ICCV2013）标注：68**

[300 Faces in-the-Wild Challenge: The first facial landmark localization Challenge](https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_iccv_2013_300_w.pdf)

300-W challenge所使用的训练数据集实际上并不是一个全新的数据集，它是采用了半监督的标注工具，将AFLW，AFW，Helen，IBUG，LFPW，FRGC-V2，XM2VTS等数据集进行了统一标注然后得到的，关键信息是68个点[链接](https://ibug.doc.ic.ac.uk/resources/300-W/)

**300-VW（ICCV2015）标注：68**

在ICCV2015年拓展成了视频标注，即300 Videos in the Wild (300-VW)[链接](https://ibug.doc.ic.ac.uk/resources/300-VW/)

**MTFL/MAFL（2014）标注：68**

这里包含了两个数据集。Multi-Task Facial Landmark (MTFL) 数据集包含了12,995  张脸，5个关键点标注，另外也提供了性别，是否微笑，是否佩戴眼镜以及头部姿态的信息。Multi-Attribute Facial Landmark (MAFL)  数据集则包含了20,000张脸，5个关键点标注与40个面部属性，实际上后面被包含在了Celeba数据集中，该数据集我们后面会进行介绍。这两个数据集都使用TCDCN方法将其拓展到了68个关键点的标注。[链接](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html)

**OCFW（2014）：标注：68**

3837幅图像,每个人脸标定68个关键点。[链接](https://sites.google.com/site/junliangxing/codes)

**CelebA（2015）标注：5**

10177个人,共202599幅人脸图像。[链接](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

**SCUT-FBP（2017）标注：86**

数据集共5500个正面人脸，年龄分布为15-60，全部都是自然表情。包含不同的性别分布和种族分布（2000亚洲女性，2000亚洲男性，750高加索男性，750高加索女性），数据分别来自于数据堂，US Adult  database等。每一张图由60个人进行评分，共评为5个等级，这60个人的年龄分布为18～27岁，均为年轻人。适用于基于apperance/shape等的模型研究。[链接](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)

**WFLW（ECCV2018）标注：98**

[Look at Boundary: A Boundary-Aware Face Alignment Algorithm](https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_iccv_2013_300_w.pdf)

图像，其中7500训练集，2500测试集，标注信息除了98个关键点之外，还有遮挡，姿态，妆容，光照， 模糊和表情等信息的标注。[数据集](https://wywu.github.io/projects/LAB/WFLW.html) [论文开源code](https://github.com/wywu/LAB)

## 3.人脸表情

人脸表情识别(facial expression recognition, FER)是人脸属性识别技术中的一个重要组成部分，在人机交互、安全控制、直播娱乐、自动驾驶等领域都非常具有应用价值

**JAFFE (1998)**

这是比较小和老的数据库。该数据库是由10位日本女性在实验环境下根据指示做出各种表情，再由照相机拍摄获取的人脸表情图像。整个数据库一共有213张图像，10个人，全部都是女性，每个人做出7种表情，这7种表情分别是：sad, happy, angry, disgust, surprise, fear, neutral，每组大概20张样图。[链接](https://zenodo.org/record/3451524#.XrZwXkszaUk)

**KDEF与AKDEF（1998）**

这个数据集最初是被开发用于心理和医学研究目的。它主要用于知觉，注意，情绪，记忆等实验。在创建数据集的过程中，特意使用比较均匀，柔和的光照，被采集者身穿统一的T恤颜色。这个数据集，包含70个人，35个男性，35个女性，年龄在20至30岁之间。没有胡须，耳环或眼镜，且没有明显的化妆。7种不同的表情，每个表情有5个角度。总共4900张彩色图，尺寸为562*762像素。[链接](https://www.emotionlab.se/kdef/)

**GENKI（2009）**

GENKI数据集是由加利福尼亚大学的机器概念实验室收集。该数据集包含GENKI-R2009a，GENKI-4K，GENKI-SZSL三个部分。GENKI-R2009a包含11159个图像，GENKI-4K包含4000个图像，分为“笑”和“不笑”两种，每个图片拥有不同的尺度大小，姿势，光照变化，头部姿态，可专门用于做笑脸识别。这些图像包括广泛的背景，光照条件，地理位置，个人身份和种族等。[链接](https://inc.ucsd.edu/mplab/index.php)

**RaFD（2010）**

该数据集是Radboud大学Nijmegen行为科学研究所整理的，这是一个高质量的脸部数据库，总共包含67个模特，其中20名白人男性成年人，19名白人女性成年人，4个白人男孩，6个白人女孩，18名摩洛哥男性成年人。总共8040张图，包含8种表情，即愤怒，厌恶，恐惧，快乐，悲伤，惊奇，蔑视和中立。每一个表情，包含3个不同的注视方向，且使用5个相机从不同的角度同时拍摄的。[链接](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main)

**CK（2010）**

这个数据库是在Cohn-Kanade Dataset的基础上扩展来的，它包含137个人的不同人脸表情视频帧。这个数据库比起JAFFE要大的多。而且也可以免费获取，包含表情的标注和基本动作单元的标注。[链接](https://www.pitt.edu/~emotion/ck-spread.htm)

**Fer2013（2013）**

该数据集包含共26190张48*48灰度图，图片的分辨率比较低，共6种表情。分别为0 anger生气、1 disgust 厌恶、2 fear 恐惧、3 happy 开心、4 sad 伤心、5 surprised 惊讶、6 normal 中性。[链接](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

**RAF（2017）**

包含总共29672 张图片，其中7个基本表情和12 个复合表情，而且每张图还提供了5个精确的人脸关键点，年龄范围和性别标注。[链接](http://www.whdeng.cn/RAF/model1.html)

**EmotionNet（2017）**

共950,000张图，其中包含基本表情，复合表情，以及表情单元的标注。[链接](http://cbcsl.ece.ohio-state.edu/EmotionNetChallenge/)

表情识别目前的关注点已经从实验室环境下转移到具有挑战性的真实场景条件下，研究者们开始利用深度学习技术来解决如光照变化、遮挡、非正面头部姿势等问题，仍然有很多的问题需要解决。

另一方面，尽管目前表情识别技术被广泛研究，但是我们所定义的表情只涵盖了特定种类的一小部分，尤其是面部表情，而实际上人类还有很多其他的表情。表情的研究相对于颜值年龄等要难得多，应用也要广泛的多，相信这几年会不断出现有意思的应用。

## 4.年龄与性别

人脸的年龄和性别识别在安全控制，人机交互领域有着非常广泛的使用，而且由于人脸差异性，人脸的年龄估计仍然是一个难点。

**FGNet（2000）**

第一个意义重大的年龄数据集，包含了82个人的1002张图，年龄范围是0到69岁。[链接](http://www-prima.inrialpes.fr/FGnet/html/benchmarks.html)

**CACD2000（2013）**

这是一个名人数据集，包含了2,000个人的163446张名人图片，其范围是16到62岁。[链接](https://bcsiriuschen.github.io/CARC/)

**Adience（2014）**

采用iPhone5或更新的智能手机拍摄的数据，共2284个人26580张图像。它的标注采用的是年龄段的形式而不是具体的年龄，其中年龄段为（0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60+）[链接](https://talhassner.github.io/home/projects/Adience/Adience-data.html#frontalized)

**IMDB-wiki（2015）**

IMDB-WIKI人脸数据库是由IMDB数据库和Wikipedia数据库组成，其中IMDB人脸数据库包含了460,723张人脸图片，而Wikipedia人脸数据库包含了62,328张人脸数据库，总共523,051张人脸数据。都是从IMDb和维基百科上爬取的名人图片，根据照片拍摄时间戳和出生日期计算得到的年龄信息，以及性别信息，对于年龄识别和性别识别的研究有着重要的意义，这是目前年龄和性别识别最大的数据集。[链接](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

**MORPH（2017）**

包括13,000多个人的55,000张图，年龄范围是16到77。[链接](https://uncw.edu/oic/tech/morph.html)

## 5.人脸姿态

人脸的姿态估计在考勤，支付以及各类社交应用中有非常广泛的应用。

## 6.人脸识别

## 二. 常用人脸数据详细介绍

## 1.人脸检测

## 2.人脸关键点检测

### 2.1 CMU Multi-PIE

为了系统地捕捉具有不同姿势和照明的图像，我们使用了一个由15个摄像头和18个闪光灯连接到一组Linux pc上的系统。13个摄像头位于头部高度，间隔15°，另外两个摄像头位于受试者上方，模拟典型的监控场景。下图显示了摄像机的位置。

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='300' height='211'></svg>)


下图显示了所有15个带有正面闪光灯照明的相机视图。

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='609' height='276'></svg>)


在一个记录过程中，每个相机捕获20张照片:一张没有任何闪光灯照明，18张照片每个闪光灯单独发射，然后另一张没有任何闪光灯。所有相机在0.7秒内总共拍摄了300张照片。下面我们展示了正面拍摄的全部20张照明图片。

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='594' height='358'></svg>)


**高分辨率的图像:**
我们使用佳能EOS 10D(630万像素CMOS相机)和Macro Ring Lite MR-14X Ring flash拍摄正面图像。受试者坐在离摄像机很近的蓝色背景前。得到的图像大小为3072 x 2048，受试者的瞳孔间距通常超过400像素。

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='618' height='216'></svg>)


**面部表情**
在这四段录音的每一段中，受试者都被要求展示不同的面部表情。下图显示了在每个会话中捕获的表达式的图像。

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='622' height='737'></svg>)



### 2.2 LFPW

LFPW (Labeled Face Parts in the Wild) 展示了从互联网上收集的新数据集.共1432张图片，每张图片上有29个点。

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='1000' height='633'></svg>)


人脸关键点标注顺序如下图所示：

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='205'></svg>)



### 2.3 AFLW

Annotated Facial Landmarks in the Wild (AFLW)提供了大量从网上收集的带注释的面部图像，展示了各种各样的外观(如姿势、表情、种族、年龄、性别)以及一般的成像和环境条件。总共有大约25k张脸被标注上了多达21个地标。

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='600' height='324'></svg>)



### 2.4 Helen

数据集包括2000张训练图像和330张测试图像

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='900'></svg>)



### 2.5 COFW

所有图片都是手注释的29地标,标注了地标位置以及它们的occlusion /no occlusion 状态。COFW的平均 occlusion 率超过23%。

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='532' height='361'></svg>)



### 2.6 300W

该数据集每个图像上包含不止一张人脸，但是对于每张图像只标注一张人脸。其中：AFW(337)，Helen(train 2000+test 330)，IBUG(135)，LFPW(train 811+test  224)。共计3148张图像，测试集有554+135=689张图像。



![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='785' height='512'></svg>)



![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='1856' height='1496'></svg>)



### 2.7 WFLW

Wider Facial Landmarks in-the-wild  (WFLW)包含10000个面部(7500个用于训练，2500个用于测试)和98个全手工标注的地标。除了地标标注外，新数据集还包含了丰富的属性标注，即，遮挡，姿势，化妆，照明，模糊和现有的综合分析算法的表达。与以前的数据集相比，新数据集中的人脸在表情、姿态和遮挡方面存在较大的变化。我们可以简单地评估提出的数据集上的位姿、遮挡和表达式的鲁棒性，而不是在不同数据集中的多个评估协议之间切换。

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='716' height='373'></svg>)


**Landmark Definition**

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='828' height='822'></svg>)


**Multi-View Illustration**

![img](data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='1844' height='523'></svg>)



## 3.人脸表情

## 4.年龄与性别

## 5.人脸姿态

## 6.人脸识别



参考：[常用公开人脸数据集汇总，持续更新中~~ - 程序员大本营](https://www.pianshen.com/article/98811386642/)

编辑于 2020-11-20 15:42

[人脸识别](https://www.zhihu.com/topic/19559196)

[图像识别](https://www.zhihu.com/topic/19588774)

[数据集](https://www.zhihu.com/topic/19612270)

### 文章被以下专栏收录

- [![AI on chip](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-4092259e0fc66d3a7d2fc134848cf24e_xs-20211222215013238.jpg)](https://www.zhihu.com/column/c_1178097838814363648)

- ## [AI on chip](https://www.zhihu.com/column/c_1178097838814363648)

- 深度学习算法部署嵌入端

### 推荐阅读



[![重磅！GroupFace 人脸识别，刷新 9 个数据集SOTA](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-3247921621a608387a1754d53d49871b_250x0.jpg)重磅！GroupFace 人脸识别，刷新 9 个数据集SOTA我爱计算机...发表于我爱计算机...](https://zhuanlan.zhihu.com/p/143845931)[![人脸识别常用数据集大全（6/11更新）](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-d351b2e7aaa41718a48bfd58709d0d71_250x0.jpg)人脸识别常用数据集大全（6/11更新）极市平台](https://zhuanlan.zhihu.com/p/31378836)[![分享几个业界新出人脸识别相关数据集](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-476acaa7d2592e86d1835406ef0d2c8e_250x0.jpg)分享几个业界新出人脸识别相关数据集我爱计算机...发表于CV Da...](https://zhuanlan.zhihu.com/p/343907168)[人脸识别数据集精粹（下）人脸识别数据集精粹（下） 5. 人脸检测数据集所谓人脸检测任务，就是要定位出图像中人脸的大概位置。通常检测完之后根据得到的框再进行特征的提取，包括关键点等信息，然后做一系列后续的分…吴建明wujianming](https://zhuanlan.zhihu.com/p/145984551)

## 还没有评论

评论区功能升级中