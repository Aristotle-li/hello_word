str2text: Handwritten Chinese Text Generation via Inexact Supervision



```
备用标题：
with component collaborator
via modeling global order and local details
via weakly supervised learning
with spatial perception module

which demands for a full exploration. 
```



Abstract—In recent years, researchers have investigated a variety of approaches to the generation of Chinese fonts, but the generation of handwritten Chinese sequence text is usually achieved by stacking single characters. Due to the large number of complex glyphs in Chinese characters, it is impossible to generate readable content by directly migrating the alphabetic handwritten sequence text model to the Chinese task, as shown in Figure 1.

For real handwritten Chinese text, text-level features and differential output of the same character in the single handwriting style are indispensable. Inspired by inexact supervision, we propose a str2text framework to directly generate handwritten Chinese text, which integrates sequence recognition module(SRM) and spatial perception module(SPM) into standard GAN to generate variable size Chinese text images with correct contents. Different from the common Chinese font generation paradigm, only Chinese strings are fed into the network, and a large number of handwritten texts with various styles are obtained by combining the prior domain knowledge of Chinese strings. And we show that integrating the generated Chinese text data into the existing training data of the text recognition system can improve performance.





，迁移到手写中文文本后是不可读的，如图1所示



我们提出了一种直接生成手写中文文本的框架。与常用的字母形文本相比，我们进一步编码了汉字字符串的先验知识，并将序列识别模块（SRM）和空间感知模块（SPM）集成到标准GAN中，生成内容正确的可变大小中文文本图像。我们还表明，将生成的中文文本数据集成到文本识别系统的现有训练数据中可以提高性能。



摘要

近年来，研究人员研究了多种汉字字体生成方法，但手写汉字序列文本的生成通常是通过单个字符的叠加来实现的。由于汉字存在海量复杂字形的原因，直接将字母形手写序列文本工作迁移到中文任务中无法生成可读的内容，如图1所示。受不精确监督的启发，我们提出了一个有效的模型，名为str2text，直接从文本标签上的潜在先验（如高斯分布）生成手写中文文本。具体而言，str2text被设计为一种基于CGAN的架构，集成了序列识别模块（SRM）和空间感知模块（SPM）。SRM提供了序列级约束，以确保生成文本的可识别性。SPM可以在无监督的情况下自适应地学习字符内部组件之间的空间相关性，这将有助于复杂拓扑结构字符的建模。得益于这种巧妙的建模，我们的模型足以生成任意长度的序列汉字图像。



1.Introduction+图片（可控序列文本的生成）



历史+未解决的问题+前人方案的不足+基于以上观察我们提出了什么模型+有什么结构+好处+贡献

generating Chinese characters with diverse styles is considered as a difficult task



With the development of deep learning, automatic generation of Chinese fonts with complicated structures is not a difficult task, but the Handwritten Chinese text generation remains unexplored. Compared with single Chinese font generation, Handwritten Chinese text generation is a much more challenging task. In addition to the inherent difficulties of Chinese character font generation, such as Chinese characters sharing an extremely large glyph with complicated content, handwritten Chinese text generation has the following challenges.  (i) handwritten Chinese text contain text-level features, such as the dependencies between adjacent characters, text line offset, etc. (ii) differential output of the same character in the single handwriting style.

随着深度学习的发展，汉字字体生成的工作取得了令人满意的效果，但是对于手写中文文本的生成工作还没有人探索。与单一汉字字体生成相比，手写汉字文本生成更具挑战性。除了汉字字体生成的固有困难（例如汉字共享超大字形且内容复杂）之外，手写汉字文本生成还面临以下挑战：(i)手写中文文本包含文本级特征，例如相邻字符之间的依赖关系、文本行偏移量等。(ii)在同一书写风格下同一汉字的差异化输出的问题



The study of character generation has experienced two stages of development. The first stage is the component assembly method based on computer graphics. This kind of method regards glyphs as a combination of strokes or radicals, which first extract strokes from Chinese character samples, and then some strokes are selected and assembled into unseen characters by reusing parts of glyphs. The above-mentioned methods are largely dependent on the effect of strokes extraction. The stroke extraction module does not typically perform well when the structure is written in a cursive style.
In the second stage, the utilization of U-Net as encoder-decoder architecture enables the font generation problem to be treated as an image-to-image translation problem via learning to map the source style to a target style. Considering blur and ghosting problems, it is Insufficient only to treat a single Chinese character as a glyph category. 
Recent works such as “XX” and advanced version “XX” generated Chinese characters by reintroducing prior knowledge of structures and components to alleviate this problem, and helps to generate more diverse fonts with fewer stylized samples.  



字符生成的研究工作经历了两个阶段的发展：第一个阶段是基于计算机图形学的部件装配法。这种方法将字形视为笔画或部首的组合，首先从汉字样本中提取笔画，然后选择一些笔画并组合成看不见的汉字依通过重用字形的部分内容。上述方法在很大程度上取决于笔画提取的效果。当结构以草书风格书写时，笔画提取模块通常不能很好地执行。在第二阶段，利用基于编码器-解码器的方法，通过学习将源样式映射到目标样式，可以将字体生成问题视为图像到图像的翻译问题。考虑到模糊和虚影的问题，仅仅把一个汉字作为一个字形类别是不够的。

最近的工作，如**[SVAE]**通过重新引入结构和组件的先验信息编码来生成汉字，以缓解此问题，并有助于以较少的样式化样本生成更多样化的字体。

However, all previous single Chinese font generation works ignore the basic fact that only standard Chinese characters can be separable. For real handwritten Chinese text, text-level features and differential output of the same character in the single handwriting style are indispensable, which is unrealizable by the common Chinese font generation paradigm.

但是，以前所有的单汉字字体生成工作都忽略了一个基本事实，即只有标准汉字才能被分离。对于真实的手写中文文本，文本级特征和同一字符在单个手写样式中的差异输出是必不可少的，这是普通中文字体生成范式无法实现的。

IInspired by inexact supervision**[A brief introduction to weakly supervised learning]**, we propose a str2text framework to directly generate handwritten Chinese text, which integrates sequence recognition module(SRM) and spatial perception module(SPM) into standard GAN to generate variable size Chinese text images with correct contents.

受弱监督学习启发，我们提出str2text框架直接生成手写中文文本，该框架将序列识别模块(SRM)和空间感知模块(SPM)集成到标准GAN中，生成具有正确内容的可变尺寸中文文本图像。

Due to the complex spatial structure of Chinese characters, the traditional text generation using only a character sequence recognition module(SRM) is not enough to generate readable handwritten Chinese text. Thus, we introduce a component-level spatial perception module(SPM) based on inexact supervised learning to enrich the details of the text. 

由于汉字丰富的组成细节和复杂的空间结构，传统的仅使用字符序列识别模块(SRM)的文本生成不足以生成可读的手写中文文本。因此，我们引入了基于非精确监督学习的空间感知模块（SPM）丰富文本的细节。



Unlike the past Chinese font generation paradigm, we only use the prior domain knowledge of Chinese characters(i.e., component and structure) to encode Chinese strings as the network input. The coding method not only enables the network to generate Chinese text containing arbitrary Chinese characters, but also some non-sense glyphs. As we all know, data augmentation is conducive to the improvement of accuracy in recognition tasks. In this paper, we find that non-sense glyphs can play the same effect by making the glyph classification boundary of feature space more accurate. The concrete impact of this discovery is discussed in Section III-B.

与以往的汉字字体生成模式不同，我们只使用汉字的先验领域知识（即成分和结构）来编码汉字字符串作为网络的输入。这种编码方法不仅可以使网络生成包含任意汉字的中文文本，还可以生成一些无意义的字形。众所周知，数据扩充有助于提高识别任务的准确性。在本文中，我们发现无意义字形通过使特征空间的字形分类边界更加准确可以起到同样的效果。第III-B节讨论了这种转变的具体影响。

我们解决了差异化输出的问题

To sum up, the advantages of the proposed str2text framework can be summarized as follows: 

Exploiting inexact supervision, we directly generate handwritten Chinese text with text-level features.

We solved the problem of differentiated output of the same Chinese character in the single handwriting style

we improve text recognition performance on the HWDB dataset, using a dataset extended with generated handwritten Chinese text.

利用非精确监督，直接生成具有文本级特征的可读手写中文文本。