

**2.related work**



**Chinese font generation(CFG).** 

到目前为止，已经提出了大量的字体合成方法。一般来说，现有的方法可以分为两类：早期基于笔画合成的方法和基于深度学习的方法。

Up to now, large numbers of methods for font generation have been proposed. Generally speaking, those existing methods can be classified into two categories: Computer Graphics based methods and Deep Learning-based methods.

基于传统的部件组合：

Traditional methods are typically based on the basic idea of stroke or radical extraction and reassembly. **(Automatic Generation of Artistic Chinese Calligraphy)** proposed a small structural stroke database to represent each character in a multi-level way, and calligraphy is generated via a reasoning-based approach.

传统方法通常基于笔画或部首提取和重组的基本思想。(Automatic Generation of Artistic Chinese Calligraphy) 提出了一个小型结构化笔画数据库，以多层次的方式表示每个字符，并通过基于推理的方法生成书法。



**(Handwritten Chinese Character Font Generation Based on Stroke Correspondence)** consider each stroke to be a vector and proposed individual’s handwritten Chinese character font generation method by vector quantization.

In **the (Automatic Generation of Chinese Calligraphic Writings with Style Imitation)** method, he derive a parametric representation of stroke shapes and generate a new character topology via weighted averaging of a few given character topologies derived from the individual’s previous handwriting. 

(Handwritten Chinese Character Font Generation Based on Stroke Correspondence) ，将每一笔画作为向量，提出了矢量量化的手写体汉字字形生成方法。

 在 (Automatic Generation of Chinese Calligraphic Writings with Style Imitation)方法中，他推导了笔画形状的参数表示，并通过加权平均从个人以前的笔迹中导出的几个给定字符拓扑生成了新的字符拓扑。



**(Automatic Shape Morphing for Chinese Characters)** and **(Automatic Generation of Large-scale Handwriting Fonts via Style Learning)** by applying the Coherent Point Drift (CPD) algorithm to achieve non-rigid point set registration between each character to extract stroke/radical.

**(Font Generation of Personal Handwritten Chinese Characters)**exploit a radical placement database of Chinese characters, vector glyphs of some other Chinese characters are automatically created,and these vector glyphs are merged into the userspecific font.(Font Generation of Personal Handwritten Chinese Characters)

**(Automatic Shape Morphing for Chinese Characters)** and **(Automatic Generation of Large-scale Handwriting Fonts via Style Learning)** 通过应用相干点漂移（CPD）算法实现每个字符之间的非刚性点集注册，以提取笔划/部首。

(Font Generation of Personal Handwritten Chinese Characters)利用汉字的部首位置数据库，自动创建一些其他汉字的矢量字形，并将这些矢量图示符被合并到特定于用户的字体中。



Later, the Radical composition Model **(Easy generation of personal Chinese handwritten fonts)** and StrokeBank **(StrokeBank: Automating Personalized Chinese Handwriting Generation)** were proposed by mapping the standard font component to their handwritten counterparts to synthesize characters. 

后来，提出了汉字部首组合模型（Easy generation of personal Chinese handwritten fonts）和StrokeBank（StrokeBank: Automating Personalized Chinese Handwriting Generation），通过将标准字体组件映射到手写体来合成字符。



In the era of deep learning, generative adversarial networks (GANs) **[ Generative adversarial nets,Conditional generative adversarial nets,styleGAN,SAGAN]** are widely adopted for style transfe.**[Stylebank: An explicit representation for neural image style transfer,Perceptual losses for real-time style transfer and super-resolution,Image-to-image translation with conditional adversarial networks,Unpaired image-to-image translation using cycleconsistent adversarial networks.StarGAN: Unified generative adversarial networks for multi-domain image-to-image translation.StarGAN v2: Diverse image synthesis for multiple domains.]**

Several attempts have been recently made to model font synthesis as an image-to-image translation problem**[rewrite,zi2zi,Multi-content GAN for Few-Shot Font Style Transfer,Generating Handwritten Chinese Characters using CycleGAN,W-Net: One-Shot Arbitrary-Style Chinese Character Generation with Deep Neural Networks: 25th International Conference, ICONIP 2018, Siem Reap, Cambodia, December 13–16, 2018, Proceedings, Part V,DCFont: an end-to-end deep chinese font generation system,Auto-Encoder Guided GAN for Chinese Calligraphy Synthesis,Pyramid Embedded Generative Adversarial Network for Automated Font Generation,DG-Font: Deformable Generative Networks for Unsupervised Font Generation,TET-GAN: Text Effects Transfer via Stylization and Destylization,Separating Style and Content for Generalized Style Transfer,Coconditional Autoencoding Adversarial Networks for Chinese Font Feature Learning,SCFont: Structure-Guided Chinese Font Generation via Deep Stacked Networks]**, which transforms the image style while preserving the content consistency.



生成性对抗网络（GAN）被提出后，其衍生版本被广泛用于神经类型转换。此外，字体生成可以被视为图像到图像翻译问题的一个实例，它将图像从一个域转换到另一个域，同时保持内容的一致性。





**Handwritten text generation(HTG).** 

Since the high inter-class variability of text styles from writer to writer and intra-class variability of same writer styles\cite{krishnan2021textstylebrush}, the handwritten text generation is challenging.

At present, handwritten text generation mainly focuses on alphabetic text. Alonso et al. [34] proposed an offline handwritten text generation model for fixed-size word images. ScrabbleGAN [13] used a fully-convolutional handwritten text generation model.

For handwritten Chinese text generation(HCTG) tasks, the existing text generation model cannot generate readable content. In contrast, our method is applicable for images of handwritten Chinese text with arbitrary length. 

由于不同作者的文本样式的类间可变性很高，以及相同作者样式的类内可变性，手写文本生成具有挑战性。目前，手写文本生成工作主要集中在字母型文本

Alonso等人[34]提出了一种用于固定尺寸文字图像的脱机手写文本生成模型。ScrabbleGAN[13]使用了一个完全卷积的手写文本生成模型。

对于手写书写中文文本生成任务，现有的文本生成模型无法生成可读的内容。

相比之下，本方法适用于任意长度的手写中文图像。



