

效果：

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211013104921817.png" alt="image-20211013104921817" style="zoom:50%;" />

基于机械方式的合成，使得汉字看起来十分不自然。



### motivation

使用从用户手写体中提取的成分合成汉字

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211012201750672.png" alt="image-20211012201750672" style="zoom:50%;" />

三个汉字合成30个

1） 所需的组件是什么？
2） 用户必须手写多少个汉字？
3） 如何从用户的手写体中提取所需的组件？
4） 给定目标汉字，将这些组件放置在何处？这些组件有多大

### idea

we tested preliminarily the approach: to synthesize Chinese characters using components extracted from the user’s handwritings,

用从用户手写体中提取的成分合成汉字



### structure

A. Component Analysis

我们利用了[21]的结果，其中《汉语大字典》中记录的所有汉字被分解成大约1000个词根，如表二所示。我们对[21]进行了扩展，并将Unicode 3.1编码的其他汉字分解为部首[24-25]。

在预处理阶段，我们分析[24-25]中报告的结构，以找到一小部分汉字，其中包含合成目标汉字字符集编码中其他汉字所需的所有成分

在本研究中，当汉字的所有成分都可用时，即从其他汉字中提取的成分，汉字是可合成的。我们采用了一个简单的贪婪算法，如图3所示。最初，所有目标汉字都被放入候选库中。接着是一个循环。在每次迭代中，一个目标汉字被选择到输出子集中，并从候选池中删除。然后，我们检查候选库中的其他汉字是否可以合成。如果是，则将其从候选人才库中删除。当候选池为空时，循环停止，这意味着所有目标汉字要么被选入输出子集，要么可合成。



B.Positons and Sizes of Components in a Chinese Character

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211013095642499.png" alt="image-20211013095642499" style="zoom:50%;" />

在预处理阶段，我们还创建了一个Web界面，用于标记目标字符集中每个汉字组件的位置和大小

用户首先单击指定汉字的一个组件，然后绘制一个覆盖该组件的最小矩形。两个矩形可能有重叠

C.Component Extraction

由于应用程序知道字符的所有组成部分，因此它可以根据笔划的起点、终点和顺序轻松地跟踪和分类每个笔划，并使用不同颜色的组件向用户显示结果。



D. Glyph Synthesis

如[23]所述，采用了以下原则。

a） 处于相似相对位置的组件优于处于其他位置的组件。

b） 手写组件比合成组件更好。

c） 当一个以上的组件满足（a）和（b）时，我们随机选择一个。

在预处理阶段，我们根据从标准Kai字体提取的大小和位置信息放置组件



<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211012173643333.png" alt="image-20211012173643333" style="zoom:50%;" />

g. 1. Concept of glyph synthesis of Chinese characters from components extracted from a user’s handwritings.



首先，汉字容易分解：在本研究中，如果汉字的结构是左右或上下两部分的组合，则认为汉字容易分解。因此，建议的应用程序可以更好地从中提取组件。前五个选定的字符是

Characters easy to decompose first: in this study, if the structure of a Chinese character is left-and-right or topand-bottom combination of two components, it was considered easy to decompose. Thus, the proposed APP can perform better to extract components from it. The first five selected characters were
