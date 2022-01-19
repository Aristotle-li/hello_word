str2text: Handwritten Chinese Text Generation via Inexact Supervision



```
备用标题：
with component collaborator
via modeling global order and local details
via weakly supervised learning
with spatial perception module

which demands for a full exploration. 
```



Abstract—In recent years, researchers have investigated a variety of approaches to the generation of Chinese single character, but the obtainment of handwritten Chinese sequence text images is usually achieved by stacking single character image. Due to the large number of complex glyphs in Chinese characters, it is impossible to generate readable handwritten Chinese sequence text images by directly exploiting the alphabetic handwritten text model, as shown in Figure 1. Inspired by the inexact supervision, we propose an effective model named str2text to generate handwritten Chinese sequence text directly from latent content representation (e.g., learnable embedding vector) conditioned on text labels. Specifically, str2text is designed as a CGAN-based architecture that additionally integrates a Chinese string encoder (CSE), a sequence recognition module(SRM) and a spatial perception module (SPM). Compared with the one-hot embedding, CSE can obtains the latent priori conditioned on text labels by reusing the structure and component information shared among all the Chinese characters. SRM provides sequence-level constraints to ensure the recognizability of the generated text. SPM can adaptively learn the spatial correlation between the internal components of a character in an inexact supervised manner, which will facilitate the modeling of characters with complicated topological structures. Benefiting from such artful modeling, our model suffices to generate images of sequence Chinese characters with arbitrary length. Extensive experimental results demonstrate that our model achieves state-of-the-art performance in handwritten Chinese text generation.



摘要

近年来，研究者们研究了各种生成汉字的方法，但手写体中文序列文本图像的获取通常是通过对汉字图像进行叠加来实现的。由于汉字中有大量复杂的字形，直接利用字母手写文本模型无法生成可读的手写汉字序列文本图像，如图1所示。受不精确监督的启发，我们提出了一个有效的模型，名为str2text，直接从文本标签上的潜在先验（如高斯分布）生成手写中文序列文本。具体而言，str2text被设计为基于CGAN的体系结构，另外集成了中文字符串编码器（CSE）、序列识别模块（SRM）和空间感知模块（SPM）。与热嵌入相比，CSE通过重用所有汉字之间共享的结构和成分信息，可以获得以文本标签为条件的潜在先验信息。SRM提供了序列级约束，以确保生成文本的可识别性。SPM能够以一种不精确的监督方式自适应地学习字符内部组件之间的空间相关性，这将有助于复杂拓扑结构字符的建模。得益于这种巧妙的建模，我们的模型足以生成任意长度的序列汉字图像。大量的实验结果表明，我们的模型在手写体中文文本生成中达到了最先进的性能。



1.Introduction+图片（可控序列文本的生成）



```
写清楚挑战，要明确是单字字符挑战？还是序列字符挑战？

有点空，太泛。建议写清楚模型主体架构是什么样的，提出的SRM和SPM分别有什么作用，互补性，为什么能提高（单）序列字符生成质量等等。最终，该方法能够干一件什么事（z+text ->image）。此外，还能以生成式的方式用于增广数据，实现多大提升等等
```



历史+未解决的问题+前人方案的不足+基于以上观察我们提出了什么模型+有什么结构+好处+贡献

With the development of deep learning, automatic generation of Chinese fonts with complicated structures is not a difficult task, but the Handwritten sequence text generation remains underexplored. Compared with single Chinese font generation(Hereinafter referred to as Chinese font generation), Handwritten Chinese sequence text generation(Hereinafter referred to as Chinese text generation) is a much more challenging task. In addition to the inherent difficulties of Chinese font generation, such as Chinese characters sharing an extremely large glyph with complicated content, handwritten Chinese text generation has the following challenges.  (i) handwritten Chinese text contain text-level features, such as the dependencies between adjacent characters, text line offset, etc. (ii) differential output of the same character in the single handwriting style. For the current Chinese font generation model, although the readable and accurate single font can be obtained, it can not be applied to the sequence text generation task. On the contrary, the current alphabetic text generation model is enough to generate sequence text images, due to a large number of complex glyphs in Chinese characters, it is impossible to generate readable content by directly exploiting the alphabetic handwritten text model.

随着深度学习的发展，复杂结构汉字字体的自动生成已不再是一项艰巨的任务，但手写体序列文本的自动生成仍处于探索阶段。与单个汉字字体生成（以下简称汉字字体生成）相比，手写汉字序列文本生成（以下简称汉字文本生成）是一项更具挑战性的任务。除了汉字字体生成的固有困难（例如汉字共享一个超大字形，内容复杂）之外，手写汉字文本生成还有以下挑战。（i） 手写中文文本包含文本级特征，如相邻字符之间的依赖关系、文本行偏移量等。（ii）单个手写样式中相同字符的差异输出。对于目前的中文字体生成模型，虽然可以得到可读、准确的单个字体，但不能应用于序列文本生成任务。相反，当前的字母文本生成模型足以生成序列文本图像，由于汉字中有大量复杂的字形，直接利用字母手写文本模型无法生成可读内容。



The study of character generation has experienced two stages of development. The first stage is the component assembly method based on computer graphics. This kind of method regards glyphs as a combination of strokes or radicals, which first extract strokes from Chinese character samples, and then some strokes are selected and assembled into unseen characters by reusing parts of glyphs. The above-mentioned methods are largely dependent on the effect of strokes extraction. The stroke extraction module does not typically perform well when the structure is written in a cursive style.
In the second stage, the utilization of U-Net as encoder-decoder architecture enables the font generation problem to be treated as an image-to-image translation problem via learning to map the source style to a target style. Considering blur and ghosting problems, it is Insufficient only to treat a single Chinese character as a glyph category. 
Recent works such as “XX” and advanced version “XX” generated Chinese characters by reintroducing prior knowledge( structures and components )to alleviate this problem, and helps to generate more diverse fonts with fewer stylized samples.  However, all previous Chinese font generation works ignore the basic fact that real Handwritten Chinese text image can not be obtain by stacking single character image.

字符生成的研究工作经历了两个阶段的发展：第一个阶段是基于计算机图形学的部件装配法。这种方法将字形视为笔画或部首的组合，首先从汉字样本中提取笔画，然后选择一些笔画并组合成看不见的汉字依通过重用字形的部分内容。上述方法在很大程度上取决于笔画提取的效果。当结构以草书风格书写时，笔画提取模块通常不能很好地执行。在第二阶段，利用基于编码器-解码器的方法，通过学习将源样式映射到目标样式，可以将字体生成问题视为图像到图像的翻译问题。考虑到模糊和虚影的问题，仅仅把一个汉字作为一个字形类别是不够的。最近的工作，如**[SVAE]**通过重新引入结构和组件的先验信息编码来生成汉字，以缓解此问题，并有助于以较少的样式化样本生成更多样化的字体。然而，以往的中文字体生成工作都忽略了一个基本事实，即通过堆垛汉字图像无法获得真正的手写中文文本图像。



Handwriting text generation (HTG)\cite{graves2013generating} is originally proposed for the generation of alphabetic text images, which basic structure can naturally model the dependencies between adjacent characters. The most advanced model ScrabbleGAN\cite{fogel2020scrabblegan} uses a one-hot vector to encode a letter/symbol as the conditional information of CGan and adds a sequence recognition module(SRM) to constrain the text content. As shown in Figure 1, after applying the modified ScrabbleGan model(one-hot embedding will lead to an explosive growth of model parameters) to the handwritten Chinese text generation (HCTG) task, we can not obtain the readable handwritten Chinese text image. The reason is not only the huge number of Chinese characters (more than 100,000 Chinese characters, but only dozens of letters in English), but also the reuse mechanism of Chinese components brings the problems of near glyph interference and information redundancy, so it is difficult to accurately generate more 100,000 glyphs.

手写文本生成（HTG）\cite{graves 2013generating}最初用于生成字母文本图像，其基本结构可以自然地模拟相邻字符之间的依赖关系。最先进的ScrabbleGAN\cite{fogel2020scrabblegan}模型使用一个热向量编码字母/符号作为CGan的条件信息，并添加序列识别模块（SRM）来约束文本内容。如图1所示，将改进的ScrabbleGan模型（一次热嵌入将导致模型参数的爆炸性增长）应用于手写中文文本生成（HCTG）任务后，我们无法获得可读的手写中文文本图像。究其原因，不仅是汉字数量庞大（超过10万个汉字，但英文只有几十个字母），而且汉字组件的重用机制带来了近字形干扰和信息冗余的问题，因此很难准确生成10万个以上的字形。







Inspired by the inexact supervision, we propose an effective model named str2text to generate handwritten Chinese sequence text directly from latent content representation conditioned on text labels. Specifically, str2text is designed as a CGAN-based architecture that additionally integrates a Chinese string encoder (CSE), a sequence recognition module(SRM) and a spatial perception module (SPM).

受到不精确监管的启发，我们提出了一个有效的模型，名为str2text，直接从文本标签上的潜在先验条件生成手写中文序列文本。具体而言，str2text被设计为基于CGAN的体系结构，另外集成了中文字符串编码器（CSE）、序列识别模块（SRM）和空间感知模块（SPM）

With such a large amount of Chinese character classes, if the one-hot embedding method is simply adopted, not only will it lead to an explosive growth of model parameters, but also it is difficult to obtain the content representation conducive to generating tasks. To address this issue, we propose a learnable character encoding method(CSE) by introducing the structure and component information of Chinese characters into our model. Compared with the one-hot embedding, CSE can reuse the structure and component information shared among all the Chinese characters. Every character in Chinese strings can be transformed to a unique structure embedding vector and four components sequence embedding vector. 

在汉字类数量如此庞大的情况下，如果简单地采用one-hot嵌入方法，不仅会导致模型参数的爆炸性增长，而且很难获得有利于生成任务的潜在内容表示。为了解决这个问题，我们通过在模型中引入汉字的结构和成分信息，提出了一种可学习字符编码方法（CSE）。与热嵌入相比，CSE可以重用所有汉字之间共享的结构和部件信息。中文字符串中的每个字符都可以转化为一个唯一的结构嵌入向量和四分量序列嵌入向量。

Alphabetic text generation model generally only use a sequence recognition module(SRM) with a CTC-based recognition loss function, which provides sequence-level constraints to ensure the recognizability of generated text.

字母文本生成模型一般只使用序列识别模块(SRM)，该模块具有基于ctc的识别损失函数，提供序列级约束以保证生成文本的可识别性。

Due to the complex spatial structure of Chinese characters, the traditional alphabetic text generation model using only a character sequence recognition module(SRM) is not enough to generate readable handwritten Chinese text. Thus, we further introduce a spatial perception module(SPM) to refine the details of the handwritten Chinese text images to make the content readable.

由于汉字复杂的空间结构，传统的仅使用字符序列识别模块（SRM）的字母文本生成模型不足以生成可读的手写汉字文本。因此，我们进一步引入空间感知模块（SPM）来细化手写中文文本图像的细节，使其内容更具可读性。

SPM can adaptively learn the spatial correlation between the internal components of a character in an inexact supervised manner, which will facilitate the modeling of characters with complicated topological structures. Benefiting from such artful modeling, our model suffices to generate images of handwritten Chinese text with arbitrary length. Extensive experimental results demonstrate that our model achieves state-of-the-art performance in handwritten Chinese text generation.

SPM能够以一种不精确的监督方式自适应地学习字符内部组件之间的空间相关性，这将有助于复杂拓扑结构字符的建模。得益于这种巧妙的建模，我们的模型足以生成任意长度的手写中文文本图像。大量的实验结果表明，我们的模型在手写体中文文本生成中达到了最先进的性能。





Unlike the past Chinese font generation paradigm, we only take the latent content representation of Chinese characters(i.e., component and structure learnable embedding vector)  as the network input. Since any appropriate combination of component and structure will correspond to a glyph, our model not only generate Chinese text containing arbitrary Chinese characters, but also some non-sense glyphs. As we all know, data augmentation is conducive to the improvement of accuracy in recognition tasks. In this paper, we find that non-sense glyphs can play the same effect. The concrete impact of this discovery is discussed in Section III-B.



与以往的汉字字体生成范式不同，我们只将汉字的潜在内容表示（即部件和结构可学习的嵌入向量）作为网络输入。鉴于组件和结构的任何适当组合都对应于一个字形，我们的模型不仅生成包含任意汉字的中文文本，还生成一些无意义字形。众所周知，数据扩充有助于提高识别任务的准确性。在本文中，我们发现无意义的字形也可以发挥同样的效果。第III-B节讨论了这一发现的具体影响。



To sum up, the advantages of the proposed str2text model can be summarized as follows: 

Exploiting inexact supervision, our proposed model directly generate readable handwritten Chinese text image.

We proposed a Chinese string encoder(CSE) that can facilitate handwritten Chinese text generation.

we improve text recognition performance on the HWDB dataset, using a dataset extended with generated handwritten Chinese text.

利用非精确监督，直接生成具有文本级特征的可读手写中文文本。









Overview of the proposed method at training. Given Chinese character string(i.e.,呗员...),it was queried in the dictionary L to obtain corresponding components and structures sequence(i.e.,$s_1s_2,c_1c_2$). Then, The sequences pass through E to obtain content representation e. Later, the e concatenated by noise $z_1$ was fed into G to generate handwritten Chinese text image(i.e., fake). Multi-scale nature of the fake is controlled by additional noise vectors,$z_2,z_3 and z_4$ fed to G each layers through Conditional Batch Normalization (CBN) ayers. The output generated images and the real images are transmitted to Discriminator, SRM and SPM, which respectively correspond to adversarial loss(),CTC loss(), and ACE loss().





建议的培训方法概述。对于中文字符串(即，…)，在字典L中查询得到对应的成分和结构序列(即，s1s2…，c1c2…)。然后序列经过E得到内容表示E，再将噪声z1连接的E输入G生成手写中文文本图像(即。假)。伪图像的多尺度特性由额外的噪声矢量控制，z2、z3和z4通过条件批处理归一化(CBN)层馈给每一层G。输出生成的图像和真实图像分别传输给鉴别器、识别器和增强器，分别对应对抗损失()、CTC损失()和ACE损失()。



编码方式：
前四个方格为component sequence，第五个是结构信息嵌入，颜色相同的部分嵌入向量一致，重用汉字之间共享的结构和部件信息。



生成器的输入方式





