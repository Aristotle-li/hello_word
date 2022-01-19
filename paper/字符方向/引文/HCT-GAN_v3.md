On the contrary, the current alphabetic text generation model is enough to generate sequence text images, due to a large number of complex glyphs in Chinese characters, it is impossible to generate readable content by directly exploiting the alphabetic handwritten text model





CSE：每个汉字结构复杂度差别很大，但手写中文文本中每个字符占据的空间都是基本相同的。这种编码方式很符合人类的直觉，复杂字和简单字虽然占据空间位置相同，但是不同的位置承载的信息量差别很大。

显示构建了每个汉字字符的部件和结构信息，

SPM：内容识别网络的作用是引导生成器生成能很好识别的内容，同时D保持其真实的风格。受启发于scrabbleGAN中，采用基于CTCloss的CRNN内容识别器，当移除rnn时，生成质量得到提升。那么可以想到，内容识别网络需要满足一个基本的条件，即能够引导生成正确的结构上下文关系（顺序，拓扑，相邻交叠等），但是不可以让语言上下文关系促进识别准确率提升。和1D识别crnn相比，带有attention机制的rnn识别网络可以构建2D的上下文关系，但是其固有的特性使其必须使用上下文关系来促进识别。我们选择采用空间attention机制的卷积网络得到2D特征图，使用抛弃序列概念的ACEloss引导G在正确的位置生成部件，这正确构建了每个汉字的拓扑结构，但是并没有让拓扑关系反过来促进识别，这有利于生成高质量的文本图像。

While state-of-the-art handwriting recognition methods [9, 31, 34] may predict characters based on linguistic context instead of visual character shapes. In contrast, R only uses local visual features for character predictions and therefore provides better feedback for generating characters.

当最先进的手写识别方法[9,31,34]使用CNN RNN时，我们使用R作为基于[34]的完全卷积网络获得了更好的结果。RNN具有任意大的上下文窗口，可以基于语言上下文而不是视觉字符形状预测字符。



相反，R仅使用局部视觉特征进行角色预测，因此为生成角色提供更好的反馈。



HCT-GAN: Handwritten Chinese Text Generation with Inexact Supervision

## \begin{abstract}

In recent years, researchers have investigated a variety of approaches to the generation of Chinese single character, but the obtainment of handwritten Chinese sequence text images is usually achieved by stacking single character image. Due to the large number of complex glyphs in Chinese characters, it is impossible to generate readable handwritten Chinese sequence text images by directly exploiting the alphabetic handwritten text model. Inspired by the inexact supervision, we propose an effective model named HCT-GAN to generate handwritten Chinese sequence text directly from latent content representation (e.g., learnable embedding vector) conditioned on text labels. Specifically, HCT-GAN is designed as a CGAN-based architecture that additionally integrates a Chinese string encoder (CSE), a sequence recognition module(SRM) and a spatial perception module (SPM). Compared with the one-hot embedding, CSE can obtains the latent priori conditioned on text labels by reusing the structure and component information shared among all the Chinese characters. SRM provides sequence-level constraints to ensure the recognizability of the generated text. SPM can adaptively learn the spatial correlation between the internal components of a character in an inexact supervised manner, which will facilitate the modeling of characters with complicated topological structures. Benefiting from such artful modeling, our model suffices to generate images of sequence Chinese characters with arbitrary length. Extensive experimental results demonstrate that our model achieves state-of-the-art performance in handwritten Chinese text generation.

近年来，研究人员研究了多种汉字单字生成方法，但手写汉字序列文本图像的获取通常是通过叠加单字图像来实现的。由于汉字中含有大量复杂的字形，直接利用字母手写文本模型无法生成可读的手写序列文本图像。在不精确监督的启发下，我们提出了一种有效的模型HCT-GAN，该模型可以从文本标签条件下的潜在内容表示(如可学习嵌入向量)直接生成手写中文序列文本。具体来说，HCT-GAN是一种基于cgan的架构，它集成了中文字符串编码器(CSE)、序列识别模块(SRM)和空间感知模块(SPM)。与一次性热嵌入相比，CSE通过重用所有汉字共享的结构和构件信息，获得文本标签的潜在先验。SRM提供序列级约束以确保生成的文本的可识别性。SPM能够以不精确的监督方式自适应地学习字符内部构件之间的空间相关性，为复杂拓扑结构字符的建模提供了方便。得益于这种巧妙的建模，我们的模型足以生成任意长度的序列汉字图像。大量的实验结果表明，我们的模型在手写中文文本生成方面达到了最先进的性能。

\end{abstract}



## \section{Introduction}



With the development of deep learning, automatic generation of Chinese single font with complicated structure is not a difficult task, but the handwritten Chinese sequence text generation remains underexplored. Compared with single Chinese font generation(hereinafter referred to as Chinese font generation), Handwritten Chinese sequence text generation(hereinafter referred to as Chinese text generation) is a much more challenging task. In addition to the inherent difficulties of Chinese font generation, such as Chinese characters sharing an extremely large glyph with complicated content, handwritten Chinese text generation has the following challengings. (i) handwritten Chinese text contain text-level features, such as the adhesions between adjacent characters. (ii) There are subtle differences in the same Chinese characters of a single writer. For the current Chinese font generation model, handwritten text images are obtained by splicing isolated character images. The method typically has the problem of lack of authenticity for failing to solve the above two critical challengings. Our work alleviates the problem by directly generating variable-sized handwritten Chinese text images.(e.g., see Figure \ref{fig:introduction_show}).

随着深度学习的发展，自动生成结构复杂的汉字单字体已不再是一项艰巨的任务，但手写体中文序列文本的生成还没有得到充分的研究。与单个汉字字体生成（以下简称汉字字体生成）相比，手写汉字序列文本生成（以下简称汉字文本生成）是一项更具挑战性的任务。除了汉字字体生成固有的困难，例如汉字共享一个超大的字形和复杂的内容外，手写汉字文本生成还具有以下特点。（i） 手写中文文本包含文本级特征，例如相邻字符之间的粘连。（二）同一作者的汉字存在细微差异。在现有的汉字字体生成模型中，手写文本图像是通过对孤立字符图像进行拼接得到的。由于未能解决上述两个关键挑战，该方法通常存在缺乏真实性的问题。我们的工作通过直接生成可变长度的手写中文文本图像来缓解这个问题（例如，见图1）。

The study of character generation has experienced two stages of development. The first stage is the component assembly method based on computer graphic\cite{mybib:Automatic_Generation_of_Artistic_Chinese_Calligraphy,mybib:Handwritten_Chinese_Character_Font_Generation_Based_on_Stroke_Correspondence,lian2012automatic,lin2014font,lian2016automatic,xu2009automatic,zhou2011easy,zong2014strokebank}. This kind of method regards glyphs as a combination of strokes or radicals, which first extract strokes from Chinese character samples, and then some strokes are selected and assembled into unseen characters by reusing parts of glyphs. The above-mentioned methods are largely dependent on the effect of strokes extraction. The stroke extraction module does not typically perform well when the structure is written in a cursive style.In the second stage, the utilization of U-Net as encoder-decoder architecture enables the font generation problem to be treated as an image-to-image translation problem via learning to map the source style to a target style \cite{azadi2018multi, chang2018generating,jiang2018w,jiang2017dcfont,jiang2019scfont,liu2019fontgan,lyu2017auto,sun2018pyramid,xie2021dg,Yang2019TETGANTE,zhang2018separating,zheng2018coconditional,wen2019handwritten}. Considering blur and ghosting problems, it is Insufficient only to treat a single Chinese character as a glyph category. Recent works such as \cite{wu2020calligan,2020RD,2017Learning,cha2020fewshot,park2020fewshot,2021Multiple} generated Chinese characters by reintroducing prior knowledge of structures and components to alleviate this problem, and helps to generate more diverse fonts with fewer stylized samples. However, all previous Chinese font generation works ignore the basic fact that real Handwritten Chinese text image can not be obtain by stacking single character image.

Handwriting text generation (HTG)\cite{graves2013generating} is originally proposed for the generation of alphabetic text images, which basic structure can naturally model the dependencies between adjacent characters. The most advanced model ScrabbleGAN\cite{fogel2020scrabblegan} uses a one-hot vector to encode a letter/symbol as the conditional information of CGan\cite{mirza2014conditional} and adds a sequence character recognizer to constrain the text content. As shown in Figure\ref{fig:arch}, after applying the ScrabbleGan to the handwritten Chinese text generation task, we can not obtain the readable handwritten Chinese text image. The reason is not only the huge number of Chinese characters (more than 100,000 Chinese characters, but only dozens of letters in English), but also the reuse mechanism of Chinese components brings the problems of near glyph interference and information redundancy, so it is difficult to accurately generate more 100,000 glyphs.

字符生成的研究经历了两个发展阶段。第一阶段是基于计算机图形的组件组装方法，引用{mybib：艺术汉字自动生成，mybib：手写汉字字体自动生成，基于笔划对应，Lian2012自动，lin2014font，Lian2016自动，Xu2009自动，zhou2011easy，zong2014strokebank}。这种方法将字形视为笔画或部首的组合，首先从汉字样本中提取笔画，然后通过重用字形中的部分笔画，选择一些笔画并组合成看不见的字符。上述方法在很大程度上取决于笔划提取的效果。当结构以草书风格书写时，笔划提取模块通常表现不佳。在第二阶段，使用U-Net作为编码器-解码器体系结构，通过学习将源样式映射到目标样式，使字体生成问题被视为图像到图像的翻译问题\cite{azadi2018multi，chang2018generating，jiang2018w，jiang2017dcfont，jiang2019scfont，Liu2019Fonggan，lyu2017auto，sun2018pyramid，xie2021dg，Yang2019TETGANTE，zhang2018separating，zheng2018coconditional，Wen2019Handwrite}}。考虑到模糊和重影问题，仅将单个汉字视为字形类别是不够的。最近的作品如\cite{Wu2020caligan，2020RD，2017Learning，cha2020fewshot，park2020fewshot，2021Multiple}通过重新引入结构和组件的先验知识来生成汉字，以缓解这一问题，并有助于以较少的样式化样本生成更多样化的字体。然而，以前所有的汉字字体生成工作都忽略了一个基本事实，即单字符叠加无法获得真实的手写中文文本图像r图像。

手写文本生成（HTG）\cite{graves 2013generating}最初用于生成字母文本图像，其基本结构可以自然地模拟相邻字符之间的依赖关系。最先进的ScrabbleGAN\cite{fogel2020scrabblegan}模型使用一个热向量编码字母/符号作为CGan\cite{mirza2014conditional}的条件信息，并添加序列字符识别器来约束文本内容。如图\ref{fig:arch}所示，将ScrabbleGan应用于手写中文文本生成任务后，我们无法获得可读的手写中文文本图像。究其原因，不仅是汉字数量庞大（超过10万个汉字，但英文只有几十个字母），而且汉字组件的重用机制带来了近字形干扰和信息冗余的问题，因此很难准确生成10万个以上的字形。

Inspired by the inexact supervision\cite{10.1093/nsr/nwx106}, we propose an effective model named HCT-GAN to generate handwritten Chinese sequence text directly from latent content representation conditioned on text labels. Specifically, HCT-GAN is designed as a CGAN-based architecture that additionally integrates a Chinese string encoder (CSE), a sequence recognition module(SRM) and a spatial perception module (SPM).

受不精确监督{10.1093/nsr/nwx106}的启发，我们提出了一个有效的模型HCT-GAN，该模型直接从文本标签条件下的潜在内容表示生成手写中文序列文本。具体而言，HCT-GAN设计为基于CGAN的架构，另外集成了中文字符串编码器（CSE）、序列识别模块（SRM）和空间感知模块（SPM）。

With such a large amount of Chinese character classes, if the one-hot embedding method is simply adopted, not only will it lead to an explosive growth of model parameters, but also it is difficult to obtain the content representation conducive to generating tasks. To address this issue, we propose a learnable character encoding method(CSE). By reusing the structure and component information shared among all the Chinese characters, CSE enables every character to be transformed to structure embedding vector and component sequence embedding vector. Compared with the one-hot embedding, CSE is amore informative character encoding method. The alphabetic text generation model generally only uses a sequence character recognizer with a CTC-based recognition loss function, which provides sequence-level constraints to ensure the recognizability of generated text.
Due to the complex spatial structure of Chinese characters, the traditional alphabetic text generation model using only a character sequence recognizer is not enough to generate readable handwritten Chinese text. Thus, we modify the character recognizer to obtain SRM. The former predicts character sequences and the latter predicts component sequences. we further introduce a spatial perception module(SPM) to refine the details of the handwritten Chinese text images to make the content readable.

在汉字类数量如此庞大的情况下，如果简单地采用单热嵌入方法，不仅会导致模型参数的爆炸性增长，而且很难获得有利于生成任务的内容表示。为了解决这个问题，我们提出了一种可学习的字符编码方法（CSE）。通过重用所有汉字之间共享的结构和部件信息，CSE可以将每个汉字转换为结构嵌入向量和部件序列嵌入向量。和热嵌入相比，CSE是一种更具信息性的字符编码方法。字母文本生成模型通常仅使用带有基于CTC的识别丢失功能的序列字符识别器，该功能提供序列级约束以确保生成文本的可识别性。由于汉字复杂的空间结构，传统的仅使用字符序列识别器的字母文本生成模型不足以生成可读的手写中文文本。因此，我们修改字符识别器以获得SRM。前者预测字符序列，后者预测成分序列。我们进一步引入空间感知模块（SPM）来细化手写中文文本图像的细节，使其内容更具可读性。

SPM can guide generator to adaptively learn the spatial correlation between the internal components of a character in an inexact supervised manner, which will facilitate the modeling of characters with complicated topological structures. Benefiting from such artful modeling, our model suffices to generate images of handwritten Chinese text with arbitrary lengths. Extensive experimental results demonstrate that our model achieves state-of-the-art performance in handwritten Chinese text generation.

Unlike the past Chinese font generation paradigm, we only take the latent content representation of Chinese characters(i.e., component and structure learnable embedding vector) as the network input. Since any appropriate combination of component and structure will correspond to a glyph, our model not only generate Chinese text containing arbitrary Chinese characters, but also some non-sense glyphs. As we all know, data augmentation is conducive to the improvement of accuracy in recognition tasks. In this paper, we find that non-sense glyphs can play the same effect. The concrete impact of this discovery is discussed in Section IV-B.

SPM可以引导生成器以不精确的监督方式自适应地学习字符内部组件之间的空间相关性，这将有助于复杂拓扑结构字符的建模。得益于这种巧妙的建模，我们的模型足以生成任意长度的手写中文文本图像。大量的实验结果表明，我们的模型在手写中文文本生成方面达到了最先进的性能。”“与以往的汉字字体生成范式不同，我们只将汉字的潜在内容表示（即组件和结构可学习的嵌入向量）作为网络输入。由于组件和结构的任何适当组合都将对应于一个字形，因此我们的模型不仅生成包含任意汉字的中文文本，还生成一些无意义字形。众所周知，数据扩充有助于提高识别任务的准确性。在本文中，我们发现无意义的字形也可以发挥同样的效果。第IV-B节讨论了这一发现的具体影响。

To sum up, the advantages of the proposed HCT-GAN framework can be summarized as follows: 
\begin{itemize}
\item To the best of our knowledge, the proposed method is the first to directly generate readable and more realistic-looking handwritten Chinese text image. 
\end{itemize}
\begin{itemize}
\item We propose a novel Chinese string encoder(CSE) and spatial perception module (SPM) that can facilitate handwritten Chinese text generation.
\end{itemize}
\begin{itemize}
\item We improve text recognition performance, using a dataset extended with generated handwritten Chinese text.
\end{itemize}

综上所述，提出的HCT-GAN框架的优点可以总结如下:
开始\{逐条列记}
据我们所知，该方法是第一个直接生成可读性强、外观更逼真的手写体中文序列文本图像的方法。
结束\{逐条列记}
开始\{逐条列记}
我们提出了一种新的中文字符串编码器（CSE）和空间感知模块（SPM），可以方便地生成手写中文文本。
结束\{逐条列记}
开始\{逐条列记}
我们利用生成的手写中文文本扩展数据集，提高了HWDB数据集的文本识别性能。
结束\{逐条列记}





% figue:introduction_show
\begin{figure}
\centering
\includegraphics[width=\linewidth]{introduction.pdf}
\caption{Examples of our model. Top:  Handwritten text images are obtained by splicing independent character images. Bottom:  Generating handwritten text image directly. When we handwrite Chinese text, the same words will be slightly different under the influence of neighbors, and there may be adhesion between characters. The Chinese characters in the box are exactly the same, but there are some differences in the circle. The feature marked in green is unique to the bottom. 
}
\label{fig:introduction_show}
\end{figure}





## \section{Related Work}

\textbf{Chinese font generation(CFG).} Up to now, large numbers of methods for font generation have been proposed. Generally speaking, those existing methods can be classified into two categories: Computer Graphics based methods and Deep Learning-based methods.Traditional methods are typically based on the basic idea of stroke or radical extraction and reassembly.

\cite{mybib:Automatic_Generation_of_Artistic_Chinese_Calligraphy} proposed a small structural stroke database to represent each character in a multi-level way, and calligraphy is generated via a reasoning-based approach. \cite{mybib:Handwritten_Chinese_Character_Font_Generation_Based_on_Stroke_Correspondence}consider each stroke to be a vector and proposed individual’s handwritten Chinese character font generation method by vector quantization.
In the \cite{xu2009automatic} method, he derive a parametric representation of stroke shapes and generate a new character topology via weighted averaging of a few given character topologies derived from the individual’s previous handwriting. 
\cite{lian2012automatic} and \cite{lian2016automatic} by applying the Coherent Point Drift (CPD) algorithm to achieve non-rigid point set registration between each character to extract stroke/radical.
\cite{lin2014font} exploit a radical placement database of Chinese characters, vector glyphs of some other Chinese characters are automatically created, and these vector glyphs are merged into the user-specific font. 
Later, the Radical composition Model \cite{zhou2011easy} and \cite{zong2014strokebank} were proposed by mapping the standard font component to their handwritten counterparts to synthesize characters. 

After the generative adversarial networks(GANs)\cite{goodfellow2014generative} was proposed, its derivative version\cite{mirza2014conditional,zhang2019selfattention,karras2019stylebased} was widely adopted for style transfe\cite{chen2017stylebank,johnson2016perceptual,isola2017image,zhu2017unpaired,choi2018stargan,choi2020stargan}. Several attempts have been recently made to model font synthesis as an image-to-image translation problem \cite{Rewrite,Zi-to-zi,azadi2018multi, chang2018generating,jiang2018w,jiang2017dcfont,jiang2019scfont,liu2019fontgan,lyu2017auto,sun2018pyramid,xie2021dg,Yang2019TETGANTE,zhang2018separating,zheng2018coconditional,wen2019handwritten}, which transforms the image style while preserving the content consistency. All of these methods generate a single font from target style image and so are not directly comparable with our work.



\textbf{Handwritten text generation(HTG).} Since the high inter-class variability of text styles from writer to writer and intra-class variability of same writer styles\cite{krishnan2021textstylebrush}, the handwritten text generation is challenging.
At present, handwritten text generation mainly focuses on alphabetic text. Alonso et al.\cite{alonso2019adversarial} proposed an offline handwritten text generation model for fixed-size word images. ScrabbleGAN\cite{fogel2020scrabblegan} used a fully-convolutional handwritten text generation model.
For handwritten Chinese text generation(HCTG) tasks, the existing text generation model cannot generate readable content. In contrast, our method is applicable for images of handwritten Chinese text with arbitrary length. 

\ textbf{中国字体一代(CFG)。到目前为止，已经提出了大量的字体生成方法。一般来说，现有的方法可以分为两类:基于计算机图形学的方法和基于深度学习的方法。传统的方法通常基于脑卒中或自由基提取和重组的基本思想。

cite{mybib:Automatic_Generation_of_Artistic_Chinese_Calligraphy}提出了一个小的结构笔画数据库，以多层次的方式表示每个汉字，而书法是通过基于推理的方法生成的。引用{mybib:Handwritten_Chinese_Character_Font_Generation_Based_on_Stroke_Correspondence}将每一笔画视为一个矢量，并提出了基于矢量量化的个人手写汉字字体生成方法。在cite{xu2009automatic}方法中，他推导出笔画形状的参数表示，并通过从个人以前的笔迹中得到的几个给定的字符拓扑的加权平均，生成一个新的字符拓扑。
利用相干点漂移(CPD)算法实现字符间的非刚性点集配准，提取笔画/笔根。
利用汉字的基位数据库，自动创建一些其他汉字的矢量字形，并将这些矢量字形合并到用户特定的字体中。随后，通过将标准字体组件映射到它们的手写组件来合成字符，提出了Radical composition Model \cite{zhou2011easy}和\cite{zong2014strokebank}。
后生成对抗网络(甘斯)\引用{goodfellow2014generative},提出了它的导数版本\引用{mirza2014conditional、zhang2019selfattention karras2019stylebased}被广泛采用风格怪罪别人\引用{chen2017stylebank、johnson2016perceptual isola2017image, zhu2017unpaired, choi2018stargan, choi2020stargan}。几个最近尝试合成模型字体作为image-to-image翻译问题\引用{重写,Zi-to-zi azadi2018multi, chang2018generating, jiang2018w, jiang2017dcfont, jiang2019scfont, liu2019fontgan, lyu2017auto, sun2018pyramid, xie2021dg, Yang2019TETGANTE, zhang2018separating, zheng2018coconditional, wen2019handwritten},它在保持内容一致性的同时转换图像样式。 所有这些方法从目标样式图像生成单一字体，因此不能直接与我们的工作相比较。



\ textbf{手写文本生成(高温凝胶)。由于文本样式从一个作者到另一个作者之间的类间高可变性以及同一作者样式的类内可变性，手写文本生成具有挑战性。目前，手写文本的生成主要集中在字母文本上。Alonso et al. cite{alonso2019adversarial}提出了一种用于固定大小文字图像的离线手写文本生成模型。ScrabbleGAN\cite{fogel2020scrabblegan}使用一个完全卷积手写文本生成模型。对于手写中文文本生成任务，现有的文本生成模型无法生成可读内容。相比之下，本方法适用于任意长度的手写中文图像。







### \section{Our method}

Any appropriate combination of component and structure will correspond to a glyph. More than 100,000 Chinese glyphs are a subset of the combination. Since the real handwritten Chinese text can not be obtained by stacking a single character, our proposed model aims to learn a way to generate real handwritten Chinese text, which can contain non-sense glyph.

### \subsection{architecture}

The framework mainly consists of five modules of a Chinese string encoder(CSE), a generator(G), a discriminator(D), a sequence recognition module(SRM), and a spatial perception module(SPM). CSE is mainly responsible for encoding the components and structures of Chinese string into latent content representation $e$. G is constructed based on a full convolution network(FCN)\cite{long2015fully} and can generate variable-sized output. D promotes realistic-looking handwriting styles. SRM is a 1-D sequence recognizer with connectionist temporal classification (CTC) loss\cite{graves2006connectionist} to ensure that the generated handwritten Chinese text is in the correct character order. SPM is a 2-D prediction model that guides the network to refine each type of component in correct position. Due to the high cost of components spatial relationships annotation, SPM adopts an inexact supervised approach.

组件和结构的任何适当组合都将对应于一个符号。超过10万个汉字是这个组合的一个子集。由于不能通过叠加单个字符来获得真实的手写汉字，我们提出的模型旨在学习一种生成真实手写汉字的方法，该方法可以包含无意义的字形。

\分段{架构}
该框架主要由字符串编码模块(CSE)、生成器模块(G)、识别器模块(D)、序列识别模块(SRM)和空间感知模块(SPM)五个模块组成。CSE主要负责将中文字符串的组成部分和结构编码为内容表示形式$e$。G是基于全卷积网络(FCN)构造的，可以生成可变大小的输出。促进逼真的笔迹风格。SRM是一个具有连接时态分类(CTC)的一维序列识别器，以确保生成的手写中文文本的字符顺序正确。SPM是一种二维预测模型，它指导网络在正确的位置细化每种类型的组件。由于组件空间关系标注的成本很高，SPM采用了一种不精确的监督方法。





#### \subsection{Chinese string encoder(CSE)}



Our input is a Chinese character string, e.g., “\begin{CJK}{UTF8}{gbsn}呗员...\end{CJK}”. \cite{wu2020calligan} embed component sequences into fixed-length vectors, which is not suitable for the generation of text image. We propose a intuitive method that decompose each character of Chinese character string into components sequence and structure index, and padding to equal length to fit the generation of text image. 

我们的输入是一个中文字符串，例如“\begin{CJK}{UTF8}{gbsn}…\end{CJK}”。\引用{wu2020caligan}将分量序列嵌入到固定长度的向量中，这不适合生成文本图像。我们提出了一种直观的方法，将汉字字符串中的每个字符分解为组件序列和结构索引，并进行等长填充以适应文本图像的生成。

Since the same components and structure appear repeatedly in various glyphs, the number of indexe is much smaller than the number of character classes in the Dictionary. Figure \ref{fig:arch}(left) shows a few examples of the Dictionary. CSE directly learns the embedding of each index, then which is reshaped into latent content representation tensor $e\in\mathbb{R}^{480\times 4\times 4W}$ , instead of mapping the embedding to a fixed-length vector in\cite{wu2020calligan}, which helps to mimic the writing process that the components of Chinese characters are only connected with their local spatial components, and avoid using a recurrent network to learn the coupling embedding of the whole text. Compared with the one-hot embedding, CSE is a more informative character encoding method. Table \ref{table:encoder}  shows the details of CSE.

由于相同的组件和结构在不同的glyph中重复出现，索引的数量远小于字典中字符类的数量。图\ref{fig:arch}（左）显示了字典的一些示例。CSE直接学习每个索引的嵌入，然后将其重塑为潜在内容表示张量$e\in\mathbb{R}{480\times 4\times 4W}$，而不是将嵌入映射到\cite{wu2020caligan}中的固定长度向量，这有助于模拟汉字成分仅与其局部空间成分相连的书写过程，避免使用递归网络来学习整个文本的耦合嵌入。和热嵌入相比，CSE是一种信息量更大的字符编码方法。表\ref{Table:encoder}显示了CSE的详细信息。





% CSE_encoder 
\begin{table}[]
\renewcommand{\arraystretch}{2.0}
    \centering
\begin{tabular}{c}
\hline \hline
Embed_s $(s_1,s_2,...,s_W) \in \mathbb{R}^{1\times128\times W$ \\
Embed_c $(c_1,c_2,...,c_W) \in \mathbb{R}^{4\times 256\times W$\\
\hline Linear $((128+4\times 256) \times W)\\ \rightarrow 
$ \it{e} \in\mathbb{R}^{480\times 4\times 4W}$\\
\hline \hline\\
\end{tabular}
\caption{ learning the embedding of each index. Here, 480 and W is the number of channels and characters respectively.}
    \label{tab:my_label}
\end{table}







#### \subsection{Generator(G) and Discriminator(D)}

G and D are inspired by SAGAN\cite{zhang2019selfattention}, and some common techniques in the training, such as self-attention mechanism\cite{vaswani2017attention} to refine local area image quality, spectral normalization\cite{miyato2018spectral} and hinge loss function\cite{lim2017geometric} to stabilizes the training, was used. FCN is a common technique to deal with variable size input and output problems\cite{krishnan2021textstylebrush,fogel2020scrabblegan}. In this paper, we inherit this paradigm in SAGAN-based generators. At the same time, FCN allows the network to learn the dependencies between adjacent components, which solves the problem of the differentiated output of the same characters in a single handwriting style. 

As shown in Tabel \ref{Tabel:encoder}, CSE module outputs the latent content representation tensor $e\in\mathbb{R}^{480\times 4\times 4W}$.  Later, $e$ is concatenated with noise vector $z_1 \in \mathbb{R}^{32\times 4\times4W} $ in the channel layer and fed into G, and the additional noise vectors $z_2, z_3,z_4$ and $ z_5$ are fed into each layer of G through Conditional Batch Normalization (CBN)\cite{shell2015bare} layers. This allow G to control the low-resolution and high-resolution details of the appearance of Chinese text to match the required handwritten style. Compared with the common Chinese font generation network, the optimization objective of G is to generate appropriate components in the appropriate position. To cope with varying size image generation, D is also fully convolutional structure to promote the generator output realistic handwritten text. 

G和D是灵感来自萨根\ {zhang2019selfattention},和一些常见的技术培训,如self-attention机制\引用{vaswani2017attention}来完善当地的图像质量,光谱归一化\引用{miyato2018spectral}和铰链损失函数\引用{lim2017geometric}来稳定的训练,是使用。FCN是处理可变大小输入和输出问题的常见技术，引用{krishnan2021textstylebrush,fogel2020scrabblegan}。在本文中，我们在基于sagan的生成器中继承了这一范例。同时，FCN允许网络学习相邻组件之间的依赖关系，解决了相同字符在单一手写方式下的差异化输出问题。
如图Fig\ref{Fig:encoder}(下)所示，CSE模块输出内容表示为$e$。之后，e在通道层中与噪声向量$z_1$连接并送入G，另外的噪声向量$z_2, z_3$和$ z_4$通过条件批处理归一化(CBN)层送入G的每一层。这允许G控制中文文本外观的低分辨率和高分辨率细节，以匹配所需的手写风格。与普通的汉字字体生成网络相比，G的优化目标是在合适的位置生成合适的组件。为了应对不同大小的图像生成，D还采用了全卷积结构，以促进生成器输出逼真的手写文本。



#### \subsection{spatial perception module(SPM) and sequence recognition modul(SRM)}

In addition to our proposed coding method, The synergy between sequence recognition module(SRM) and spatial perception module(SPM) is our main improvement.
The generation of alphabetic text usually adopts 1-D sequence prediction architecture to evaluate the content of the generated images. However, due to the huge number of Chinese characters (more than 100,000 Chinese characters compared with dozens of letters in English), it is very difficult to accurately generate 100,000 types of glyphs. The reuse mechanism of Chinese components also brings the problems of near glyph interference and information redundancy. Therefore, we treat a Chinese text as a string of components with complex spatial relationships rather than as a single character category.

We consider starting from two aspects, introducing a sequence recognition module(SRM) for promoting the correct character order of handwritten text, and designing a spatial perception module(SPM) to refine each type of component in 2-D space. In section IV. Ablation experiments prove the necessity. 
SRM consist of a feature extractor based on a convolutional neural network, followed by a max-pooling on the vertical dimension and a full connection layer. CTC loss is used to train the system. Using SRM alone is not enough to generate detailed handwritten Chinese characters. The reason is that the SRM is essentially a 1-D sequence prediction system, which is difficult to capture the complex spatial structure of Chinese characters. The same situation is that the performance of CRNN based system is insufficient in the irregular text recognition task.

SPM is inspired by inexact supervision where the training data are given with only coarse-grained labels. We treat it as a non-sequence-based 2D prediction problem with image-level annotations. 
SPM is composed of a full convolution network with a spatial attention module, followed by a full connection layer. SPM constraint the component category at each location in the 2-D feature map, and guide the network to generate the corresponding component at the correct location. 

Note tha most of recognition model use a recurrent network, typically bidirectional LSTM \cite{shi2016end,he2016reading}, which may predict characters based on linguistic context instead of clear character shapes. In contrast, SRM and SPM only uses local visual features for character recognition and therefore provides better optimization direction for generating characters.

 We present the detail of our proposed model in Table\ref{table:GDRP}.

除了我们提出的编码方法外，序列识别模块(SRM)和空间感知模块(SPM)之间的协同是我们的主要改进。字母文本的生成通常采用一维序列预测架构来评估生成图像的内容。然而，由于汉字数量庞大(超过10万个汉字，而英语只有几十个字母)，要准确生成10万个字形是非常困难的。中文构件的重用机制也带来了近字形干扰和信息冗余的问题。因此，我们将中文文本视为一串具有复杂空间关系的组成部分，而不是单一的字符类别。
我们考虑从两个方面着手，引入序列识别模块(SRM)来提高手写文本的正确字符顺序，设计空间感知模块(SPM)来细化二维空间中每种类型的组件。在第四节。烧蚀实验证明这是必要的。SRM由基于卷积神经网络的特征提取器、垂直维度上的最大池和全连接层组成。利用CTC损耗对系统进行训练。仅使用SRM不足以生成详细的手写汉字。究其原因，SRM本质上是一个一维预测系统，难以捕捉汉字复杂的空间结构。同样的情况是，基于CRNN的系统在不规则文本识别任务中的性能不足。
SPM的灵感来源于不精确的监督，即只使用粗粒度标签给出训练数据。我们将其视为具有图像级注释的基于非序列的二维预测问题。SPM是由一个带有空间注意模块的全卷积网络和一个重塑操作组成的。SPM约束二维特征图中每个位置的组件类别，引导网络在正确的位置生成对应的组件。

请注意，大多数识别模型使用循环网络，通常是双向LSTM，它可以根据语言上下文而不是清晰的字符形状预测字符。相比之下，SRM和SPM仅使用局部视觉特征进行字符识别，因此为生成字符提供了更好的优化方向。

我们在表\ref{Table:GDRP}中展示了我们提出的模型的细节。





## \subsection{Losses and optimization settings}

We implement the hinge version of the adversarial loss from Geometric GAN\cite{lim2017geometric}

我们实现了Geometric GAN\cite{lim2017geometric}的敌对损失的铰链版本
$$
\begin{equation}
\begin{aligned}
L_{G}=&-\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[D(G(\boldsymbol{z}, \boldsymbol{e}))] 
\\
L_{D}=&+\mathbb{E}_{(\boldsymbol{x}) \sim p_{\text {data }}}[\max (0,1-D(\boldsymbol{x}))] \\
&+\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[\max (0,1+D(G(\boldsymbol{z}, \boldsymbol{e})))] \\
\end{aligned}
\end{equation}
$$
to promote our generated images look realistic. To encourage the correct character order in the generated handwritten Chinese text image, we use the CTC loss.

为了提升我们生成的图像的真实感。为了在生成的手写中文文本图像中鼓励正确的字符顺序，我们使用CTC丢失。
$$
\begin{equation}
\begin{aligned}
L_{SRM} = 
&+\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[\operatorname{CTC}(\boldsymbol{e}, SRM(G(\boldsymbol{z}, \boldsymbol{e}))]\\
\end{aligned}
\end{equation}
$$
The generated images should retain the fine strokes, we use the ACE loss.

生成的图像应该保留精细的笔划，我们使用ACE损失。
$$
\begin{equation}
    \begin{aligned}
L_{SPM_g} = 
&+\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[\operatorname{ACE}(\boldsymbol{e}, SPM(G(\boldsymbol{z}, \boldsymbol{e})))]\\
\end{aligned}
\end{equation}
$$


Here,  $p_{data}$ denotes the distribution of real Handwritten Chinese text image, $p_z$ is a prior distribution on input noise $z$ and $p_{text}$  refers to a prior distribution of  the text.

这里，$p{data}$表示真实手写中文文本图像的分布，$p{z$是输入噪声的先验分布，$p{text}$表示文本的先验分布。

We adopt the loss terms balance rule\cite{alonso2019adversarial} to obtain blance coefficient $\alpha$ and $\lambda$ , where the parameter $\beta$ and $\gamma$ respectively control the relative importance of $\boldsymbol L_{SRM_g}$ and $\boldsymbol L_{SPM_g}$  in the updating generator network. $\nabla L_{G},\nabla L_{SRM}$ and $\nabla L_{SPM}$ are respectively the gradients of $L_G,L_{SRM}$ and $L_{SPM}$ w.r.t. the image.
$$
\begin{equation}
\begin{aligned}
\alpha \leftarrow & \beta \frac{\sigma\left(\nabla L_{G}\right)}{\sigma\left(\nabla L_{S R M_{}}\right) \nabla L_{S R M}} \cdot\left[\nabla L_{S R M}-\mu\left(\nabla L_{S R M}\right)\right]+ \\
& \mu\left(\nabla L_{G}\right)
\end{aligned}
\end{equation}
$$

$$
\begin{equation}
\begin{aligned}
\lambda  \leftarrow & \gamma \frac{\sigma\left(\nabla L_G\right)}{\sigma\left(\nabla L_{SPM}\right)\nabla L_{SPM}} \cdot\left[\nabla  L_{SPM}-\mu\left(\nabla L_{SPM}\right)\right]+\\
&\mu\left(\nabla L_{G}\right)
\end{aligned}
\end{equation}
$$

here $\sigma$ and $\mu$ are respectively the empirical standard deviation and mean.

In D training step, For SRM and SPM, we only use real images to avoid learning the generated content representation. In G training step, we freeze the weight of D, SRM and SPM, let it processes the generated image, and update the weight of G. The total loss is

在D训练步骤中，对于SRM和SPM，我们只使用真实图像来避免学习生成的内容表示。在G训练步骤中，我们冻结D、SRM和SPM的权重，让其处理生成的图像，并更新G的权重。总损失为
$$
\begin{equation}
    L = L_G+\alpha L_{SRM}+\lambda L_{SPM}
\end{equation}
$$




## \section{Experiments}

In this  section, we first introduce Implementation details and evaluation metrics. Then we evaluate our proposed method from both qualitative and quantitative aspects on publicly available datasets.

在本节中，我们首先介绍我们的操作细节和评价准则。然后在公开数据集上，我们分别从定性和定量两个方面评估我们提出的方法，将我们的结果与以前的工作进行了比较。



### Dataset

The offline handwritten Chinese database, CASIA-HWDB \cite{liu2011casia} is a widely used database for handwritten Chinese character recognition. It contains samples of isolated characters and handwritten texts that were produced by 1020 writers on dot matrix paper with Anoto pen. Offline handwritten Chinese character sample are divided into three databases: HWDB1.0~1.2(Including 7,185 classes Chinese characters and 171 classes English letters, numbers and symbols). Handwritten texts is also divided into three databases: HWDB2.0~2.2(Its character classes is contained in HWDB1.0~1.2). 

The datasets HWDB1.0-Train and HWDB2.0~2.1-Train is added to HWDB1.1-Train and HWDB2.2-Train respectively for enlarging the training set size to promote generation of characters/texts. The datasets HWDB1.0~1.1-Test and HWDB2.0~2.2-Test are used for inspecting performance. Table \ref{tab:hwdb} shows the dataset we selected, while Figure\ref{fig:hwdb} shows some images from these two datasets.

脱机手写汉字数据库CASIA-HWDB\cite{liu2011casia}是一个广泛使用的手写汉字识别库。它包含1020名作家用Anoto笔在点阵纸上创作的孤立字符和手写文本样本。脱机手写汉字样本分为三个数据库：HWDB1.0-1.2（包括7185类汉字和171类英文字母、数字和符号）。手写文本也分为三个数据库：HWDB2.0-2.2（其字符类包含在HWDB1.0~1.2中。

在HWDB1.1-Train和HWDB2.2-Train中分别增加数据集HWDB1.0-Train和HWDB2.0-2.1-Train，扩大训练集大小，促进字符/文本生成。性能检测使用数据集HWDB1.0-1.1-Test和HWDB2.0~2.2-Test。表ref{tab:hwdb}显示了我们选择的数据集，而图ref{fig:hwdb}显示了来自这两个数据集的一些图像。



% figue: dataset_show
\begin{figure}
\centering
\includegraphics[width=\linewidth]{4A_show_dataset.pdf}
\caption{Examples of the CASIA-HWDB dataset. First line: HWDB2.0(124-P17). Second line: HWDB1.0(156-t).}
\label{fig:hwdb}
\end{figure}



% table：HWDB_dataset
\begin{table}[]
\renewcommand{\arraystretch}{1.3}
    \centering
        \caption{Details of the CASIA-HWDB dataset}
    \label{tab:hwdb}
    \begin{tabular}{l c|cccc}
    \hline \hline \multirow{2}{*}{\text { Dataset }} & \multirow{2}{*}{\text { class }} & \multicolumn{2}{c}{\text { writer }} & \multicolumn{2}{c}{\text { sample }} \\
\cline { 3 - 6 } & & \text { Train } & \text { Test } & \text { Train } & \text { Test } \\
\hline \text { HWDB1.0 } & 7,356 & 336 & 84 & 1,246,991 & 309,684 \\
\hline \text { HWDB 1.1 } & 7,356 & 240 & 60 & 897,758 & 223,991 \\
\hline \text { HWDB2.0 } & 2,486 & 336 & 84 & 16，358 & 4，137 \\
\hline \text { HWDB2.1 } & 2,486 & 240 & 60 & 13,766 & 3，524 \\
\hline \text { HWDB2.2 } & 2,486 & 240 & 60 & 11，655 & 2，790 \\
\hline\hline
    \end{tabular}
\end{table}



![image-20211202192616792](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimage-20211202192616792.png)





### Implementation details and evaluation metrics



we decompose 7,318 classes of Chinese characters in CICAS-HWDB into 1,179 classes components and 31 classes structure. Some rare character classes almost no longer appear in modern Chinese, e.g.,\begin{CJK}{UTF8}{gbsn}”嚚,橐,畿,曩,爨,黉,夔,龠,etc“\end{CJK}, we remain unchanged for rare-complex character. We use a publicly available Chinese character decomposition document\utl{https://github.com/CoSeCant-csc/chaizi} and modify it to fit the proposed model.

我们将数据集CICAS-HWDB中7185类汉字拆解得到1179 类component和 31 类 structure index（HDWD2.0文本中的汉字是HWDB1.0的一个子集），一些类在现代常用汉语中几乎已经不再出现，这部分难以拆解的生僻字我们选择不拆分。例如：嚚、橐、畿、曩、爨 、黉、夔、韱、龠 、巤、彘，等等。拆字从开源的chaizi文档修改而来，使其符合我们的生成任务。

In our experiments, 32 × 32W generated images of handwritten Chinese texts are obtained with the following processing:  Each Chinese character is decomposed into four component indexes and one structure index. Given Chinese indexs sequence encoded by the CSE, each Chinese character can be represented by a tensor of size $480\times 4\times4$, and the latent content representation $e$  is a stack of the tensor in width. Later, the $e$ concatenated by noise $z_1$ is fed into the first residual blocks in generator which upsample the spatial resolution, and lead to the final image size of 32 × 32W. Four additional 32 dimensional noise vectors,$ z_2, z_3,z_4$ and $z_5$ through conditional Instance Normalization layersonditional Instance Normalization layers modulate the other residual blocks.

We resize the real images to a height of 32 pixels, width of 32W pixels, and use 32 × 32W images of handwritten Chinese text/font to better deceive the discriminator, where W is the number of characters contained in labels. We use the Adam optimizer with a fixed learning rate of 0.0002 and batch size of 16. Finally, our method is implemented using the PyTorch framework. Training was performed on 3GPUS with 12GB of RAM each.

在我们的实验中，通过以下处理获得了32×32W的手写中文文本假图像：每个汉字被分解成四个成分索引和一个结构索引。给定CSE编码的中文索引序列，每个汉字可以用大小为$480\times 4\times4$的张量表示，潜在内容表示法$e$是张量宽度的堆栈。随后，由噪声$z_1$连接的$e$被馈送到生成器中的第一个剩余块中，这将提高空间分辨率的采样，并导致最终图像大小为32×32W。四个额外的32维噪声向量，$z_2、z_3、z_4$和$z_5$通过条件实例规范化层。条件实例规范化层调制其他剩余块。

我们将图像调整为32像素的高度，32W像素的宽度，并使用32×32W的手写中文文本/字体图像以更好的欺骗判别器，其中W是标签中包含的字符数。我们使用Adam优化器，其固定学习率为0.0002，批量大小为16。最后，我们的方法是使用PyTorch框架实现的。训练在3gpu上进行，每个gpu有12GB RAM。



We follow the same quantitative evaluation measures as previous handwritten text Images generation methods\cite{alonso2019adversarial,fogel2020scrabblegan}. We compare real handwritten images with generated results using these measures: (1)Fréchet Inception Distance (FID)  is widely used and calculates the distance between the real and generated images; (2)the Geometric Score (GS), which compares the topology between the real and generated manifolds. For the above two indicators, lower is better. We evaluate FID/GS on a sampled set(HWDB1.0~1.1: with HWDB1.0 test set for FID and 7k samples for GS, HWDB2.0~2.2: with 20k samples for FID and 5k samples for GS), due to its computational costs. FID/GS was computed on sampled real and generated images using 32×32W images. 

我们遵循与以前的手写文本图像生成方法相同的定量评估方法\cite{alonso2019adversarial，fogel2020scrabblegan}。我们使用这些度量将真实的手写图像与生成的结果进行比较：（1）Fréchet起始距离（FID）被广泛使用，并计算真实图像与生成图像之间的距离；（2） 几何分数（GS），用于比较真实流形和生成流形之间的拓扑。由于FID/GS的计算成本，我们在采样集（HWDB1.0-1.1：使用HWDB1.0测试集，HWDB2.0~2.2：所有样本）上对其进行评估。使用32×32W图像对采样的真实图像和生成的图像计算FID/GS。



For promoting handwritten Chinese text/character recognition, we evaluate the performance with recognition accuracy,  character error rate(CER) and edit-distance(ED). we use accuracy to measure handwritten Chinese isolated character recognition accuracy.  we use CER and ED to measure handwritten Chinese texts recognition performance.  CER is the number of misread  out of the number of characters in the test set. The ED is derived from the Levenshtein Distance algorithm, calculated as the minimum edit distance between the predicted and true text. Most of Chinese character generation work focusing on the transfer of font style use different evaluation metrics and so are not directly comparable with our work.

```
It is computed as:
\begin{equation}
\text { Accuracy }=\frac{T P+T N}{T P+T N+F P+F N}   =\frac{1}{\# \text {HWDB1.0Test}}\left(\sum_{i} \mathbb{I}\left(R\left(\mathcal{O}_{s_{i}, c_{i}}\right)==c_{i}\right)\right)
\end{equation}
```



为了促进手写体中文文本/字符识别，我们从识别精度、单词错误率（WER）和编辑距离（ED）三个方面对其性能进行了评估。我们使用准确度来衡量手写汉字孤立字符识别的准确度。我们使用CER和ED来衡量手写中文文本的识别性能。CER是测试集中字符数的误读数。ED源自Levenshtein距离算法，计算为预测文本和真实文本之间的最小编辑距离。大多数侧重于字体风格转换的汉字生成工作使用不同的评估指标，因此与我们的工作没有直接的可比性。





### Handwritten Chinese generation rsults.



We report the experimental results on HWDB1.1 and HWDB2.2 respectively. 

我们分别在HWDB1和HWDB2上分别报告实验结果



\textbf{Ablation study. } We conducted an ablation study by removing key Dmodules of our model: the Chinese string encoder(CSE), the sequence recognition module(SRM), and the spatial perception module(SPM). Without the CSE,  the model produces blurry results in both isolated characters images and handwritten text images. Appling the SRM or SPM, generated samples leads to a improvement in readability, but the character strokes are still not clear compared with HCT-GAN. Compared with SRM, SPM is more critical for generating text images, while it is not obvious in isolated character images. Without the SPM, generated text images can not reflect the true perceived visual quality, which shows that SPM is indispensable. We attempte a model with Replace SRM,  which are not able to obtain better the quality of the images generated.

\textbf{消融研究}我们通过移除模型中的关键模块：中文字符串编码器（CSE）、序列识别模块（SRM）和空间感知模块（SPM），进行了消融研究。如果没有CSE，该模型在孤立字符图像和手写文本图像中都会产生模糊结果。应用SRM或SPM，生成的样本可提高可读性，但与HCT-GAN相比，字符笔划仍然不清晰。与SRM相比，SPM在生成文本图像方面更为关键，而在孤立字符图像中则不明显。没有SPM，生成的文本图像无法反映真实的视觉感知质量，这说明SPM是必不可少的。我们尝试使用替换SRM的模型，该模型无法获得更好的图像质量。

Compared with the character recognizer, SRM effectively improves the image quality and does not increase the workload (the ground true components sequence comes from the dictionary L in CSE). Meanwhile SRM saves the memory overhead(Component class: 1,179. Character class: 7,356). The character recognizer predicts characters or character sequences, and the SRM predicts component sequences, both using the same network. FID and GS are reported in Table \ref{tab:ablation_study}. Figure \ref{fig:ablation_HWDB1} and Figue \ref{fig:ablation_HWDB2} shows some examples of all versions.

与字符识别器相比，SRM有效地提高了图像质量，并且不会增加工作量（地面真实分量序列来自CSE中的字典L）。同时，SRM节省了内存开销（组件类：1179。字符类：7356）。字符识别器预测字符或字符序列，SRM预测组件序列，两者都使用相同的网络。FID和GS报告在表\ref{tab:ablation_study}中。图\ref{fig:ablation_HWDB1}和Figue\ref{fig:ablation_HWDB1}显示了所有版本的一些示例。







![image-20211207194834485](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimage-20211207194834485.png)

% table:ablation_study
\begin{table}[]
\renewcommand{\arraystretch}{1.3}
    \centering
        \caption{Caption}
    \label{tab:my_label}
    \begin{tabular}{c|c|ll}
     \text { Dataset } & \text {  } & \text { FID} & \text { GS } \\
     
\hline \hline   & HCT-GAN(without CSE) &   - & - \\
\text {HWDB1}   & HCT-GAN(without CSE and SPM)	  & 21.14  & $1.07\times 10^{-3}$ \\
                & HCT-GAN(without SRM and SPM) &  18.65 & $1.25\times 10^{-3}$   \\
                & HCT-GAN &  -  & -  \\

\hline          & HCT-GAN(without CSE)	 & 22.76 & $1.50\times 10^{-1}$  \\
\text {HWDB2}   & HCT-GAN(without SPM)	 & 20.42 & $4.86\times 10^{-3}$ \\
                & HCT-GAN(replace SRM)   & 20.69 & $4.10\times 10^{-3}$  \\
                & HCT-GAN                & 17.82 & $3.33\times 10^{-3}$ \\
\hline
    \end{tabular}
\end{table}





<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211214230452546.png" alt="image-20211214230452546" style="zoom:67%;" />



![image-20211214230516465](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211214230516465.png)

% figue:ablation_HWDB1
\begin{figure*}
\centering
\includegraphics[width=\linewidth]{ablation_HWDB2.pdf}
\caption{ Ablation study on HWDB2.2. Replace SRM means that replace SRM with character recognizer.
}
\label{fig:ablation_HWDB1}
\end{figure*}

% figue:ablation_HWDB2
\begin{figure}
\centering
\includegraphics[width=\linewidth]{ablation_HWDB1.pdf}
\caption{  Ablation study on HWDB1.1. Replace SRM means that replace SRM with character recognizer.
}
\label{fig:ablation_HWDB2}
\end{figure}



### Showing generation rsults





\textbf{Gradient balancing study.} Gradient balancing is an important trick in multi-task learning. To compare with \cite{fogel2020scrabblegan}, we apply the gradient balancing scheme suggested in \cite{alonso2019adversarial}. Table \ref{tab:gradient_balancing} reports FID, GS for different gradient balancing settings. 

```
Figure 6 further illustrates generated images for the different settings. 
图6进一步说明了不同设置下生成的图像


```



\textbf{梯度平衡研究}梯度平衡是多任务学习中的一个重要技巧。为了与\cite{fogel2020scrabblegan}进行比较，我们采用了\cite{alonso2019adversarial}中建议的梯度平衡方案。正如在以前的工作中所报告的，适当地平衡$L_G、L_{SRM}$和$L_{SPM}$之间的重要性可以提高图像质量。"“表II报告了不同梯度平衡设置的FID、GS。





\textbf{ Comparison to scrabbleGAN.} We trained the HCT-GAN on the two datasets described in Section IV-A, HWDB1.1 and HWDB2.2. Figure \ref{fig:comparison} represent results trained by ScrabbleGAN\cite{fogel2020scrabblegan}  alongside results of our method on the same characters/texts images. It is obvious from the figure that our network produces images that are much clearer, whether for isolated characters or variable-size text images. 

我们根据第IV-A节HWDB1中描述的两个数据集对HCT-GAN进行了培训HWDB1.1 and HWDB2.2. 2.图\ref{fig:comparison}表示ScrabbleGAN\cite{fogel2020scrabblegan}训练的结果以及我们方法在相同字符/文本图像上的结果。从图中可以明显看出，我们的网络生成的图像更清晰，无论是孤立字符还是可变大小的文本图像。

When the character strokes becomes complex, the details of the top-left character images begin to blur. On the contrary, our network produces images that are still clear.Directly generating variable-size handwritten text image is more challenging. The bottom-left text have lost readability, and the bottom-right text still has realistic-looking handwriting and diverse writing styles. Later, we will further represent generated arbitrarily text images with various styles, not only in the training set.

当字符笔划变得复杂时，左上角字符图像的细节开始模糊。相反，我们的网络生成的图像仍然清晰。直接生成可变大小的手写文本图像更具挑战性。左下角的文字已失去可读性，右下角的文字仍具有逼真的笔迹和多样化的书写风格。稍后，我们将进一步表示生成的具有各种样式的任意文本图像，而不仅仅是在训练集中。

 



% figue:comparison
\begin{figure*}
\centering
\includegraphics[width=\linewidth]{comparison.pdf}
\caption{ Comparing to our results(right side) to those from ScrabbleGAN \cite{fogel2020scrabblegan}(left side) on the CASIA-HWDB \cite{liu2011casia} dataset. Top: training on HWDB1.1, down: training on HWDB2.2. *

}
\label{fig:comparison}
\end{figure*}



% table:comparison
\begin{table}[]
\renewcommand{\arraystretch}{1.3}
    \centering
        \caption{FID and GS scores in comparison to ScrabbleGAN.}
    \label{tab:comparison}
    \begin{tabular}{c|c|ll}
     \text { Dataset } & \text {  } & \text { FID} & \text { GS } \\
     
\hline \hline\text {HWDB1.1}   
& ScrabbleGAN &   22.83 & $3.49\times 10^{-2}$ \\
& HCT-GAN	  & 15.14  & $5.8\times 10^{4}$ \\

\hline\text {HWDB2.2}          
& ScrabbleGAN	 & 32.04 & $7.40\times10^{-3}$  \\
& HCT-GAN	 & 17.81 & $3.33\times 10^{-3}$ \\

\hline
    \end{tabular}
\end{table}





<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimage-20211209220156478.png" alt="image-20211209220156478" style="zoom:67%;" />

![image-20211209210848560](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimgimage-20211209210848560.png)





\textbf{Showing diverse handwriting.} We are able to generate different handwriting styles by changing the noise vector z. Figure \ref{fig:comparison} shows examples of randomly selected character/text generated in different handwriting styles.

\textbf{显示不同的笔迹。}我们可以通过改变噪声向量z来生成不同的手写样式。图\ref{fig:comparison}显示了以不同手写样式生成的随机选择字符/文本的示例。

\textbf{Interpolation between different style.} We are able to capture variations of the different handwriting styles by interpolating between the two noise vector. Figure \ref{fig:style_interpolation} shows the interpolation between two styles obtained from the randomly noises. We note that interpolation hardly changes the shape of the character and focuses more on the overall global style elements(e.g., tilt, ink thickness).

通过在两个噪声矢量之间插入，我们能够捕获不同笔迹风格的变化。图\ref{fig:style_interpolation}显示了由随机噪声得到的两种风格之间的插值。我们注意到插值几乎不会改变字符的形状，而是更多地关注整体的全局样式元素(例如。，倾斜，油墨厚度)。

\textbf{Generating unsee text.} Each the character in the Figure \ref{fig:show_test_unsee}(top) text is not in the training set, which indicates that HCT-GAN can generate handwritten text containing unsee characters in through CSE. Figure \ref{fig:show_test_unsee}(down) shows generated text in the HWDB2.2-Test.

\ textbf{生成unsee文本。}图\ref{fig:show_test_unsee}(top)文本中的每个字符都不在训练集中，说明HCT-GAN可以通过CSE生成包含不可见字符的手写文本。图\ref{fig:show_test_unsee}(下)显示hwdb22.2 - test中生成的文本。

\textbf{Generating non-sense glyph.} Figure \ref{fig:show_nonsense} shows glyphs formed by the random combination of components and structures. These glyphs look like Chinese characters, but they are not. 

\textbf{生成无意义标志符号。}图\ref{fig:show_}显示了由组件和结构的随机组合形成的图示符。这些字形看起来像汉字，但它们不是汉字。



% figue:different_style_show
\begin{figure*}
\centering
\includegraphics[width=\linewidth]{different_style_show.pdf}
\caption{ Showing diverse handwriting. Left: The characters generated are:\begin{CJK}{UTF8}{gbsn} 跋袄肮扒\end{CJK}. Right: The text generated are:\begin{CJK}{UTF8}{gbsn}泛树织做只迁挺椒淅枳森哒达沃送玳代边代认做\end{CJK}. All characters are selected randomly, and each row represents one style.

}
\label{fig:different_style_show}
\end{figure*}





% figue:style_interpolation
\begin{figure*}
\centering
\includegraphics[width=\linewidth]{style_interpolation.pdf}
\caption{ An interpolation between two different styles of handwriting generated by HCT-GAN. 
}
\label{fig:style_interpolation}
\end{figure*







% figue:show_test_unsee
\begin{figure}
\centering
\includegraphics[width=\linewidth]{show_test_unsee.pdf}
\caption{ Generating unsee text. Top: The text generated are \begin{CJK}{UTF8}{gbsn}"犰犴圪扪忉岫岣圩圬圪圹圮坜圻圮拊卟叱叩呒"\end{CJK}. Each character in the text is not in the training set. Down: The text( \begin{CJK}{UTF8}{gbsn}"而其搭建的“金字塔”策略' 在最短的时间里扩大了品牌知名度和市场占有份额"\end{CJK}) not in the training set.
}
\label{fig:show_test_unsee}
\end{figure}





% figue:show_nonsense
\begin{figure}
\centering
\includegraphics[width=\linewidth]{show_nonsense.pdf}
\caption{ Examples of non-sense glyphs.
}
\label{fig:show_nonsense}
\end{figure}





![image-20211213212417184](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimage-20211213212417184.png)



![image-20211213212346766](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimage-20211213212346766.png)

![image-20211213212504330](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimage-20211213212504330.png)

![image-20211213212514745](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimage-20211213212514745.png)





### Improving recognition performance



Our proposed HCT-GAN not only promotes generating more realistic handwritten Chinese text images but also improves the handwritten Chinese text recognition(HCTR) performance. For all experiments in this section, we use the code provided by \cite{shi2016end} as our HCTR framework. The purpose of using the basic CRNN model is to prove that HCTR performance can be improved by simply appending the generated image to the training set. Note that like the random affine transformations, we only use the training set. 

我们提出的HCT-GAN不仅可以生成更真实的手写中文文本图像，而且可以提高手写中文文本识别（HCTR）的性能。对于本节中的所有实验，我们使用\cite{shi2016end}提供的代码作为我们的HCTR框架。使用基本CRNN模型的目的是证明，只需将生成的图像附加到训练集，即可提高HCTR性能。请注意，与随机仿射变换一样，我们只使用训练集。

CER, ED and ACC are reported in Table \ref{tab:recognition_performance}. As can be seen in the table, using the HCT-GAN generated samples further improve the recognition performance compared to only using affine augmentation. Moreover, we find that appling non-sense glyphs as negative samples is more conductive to improvement in performance than equivalent character augmentation.

CER、ED和ACC报告在表\ref{tab:recognition\u performance}中。从表中可以看出，与仅使用仿射增强相比，使用HCT-GAN生成的样本进一步提高了识别性能。此外，我们发现将无意义的字形作为负样本比等效字符增强更有利于性能的提高。







\textbf{Appling non-sense glyph.} 

Non-sense combination of components and structure will lead to a non-sense glyph. As can be seen in the table, using the HCT-GAN generated non-sense glyphs during training leads to a significant improvement in performance compared to using generated character. 

组件和结构的无意义组合将产生无意义图示符。从表中可以看出，与使用生成的字符相比，在训练期间使用HCT-GAN生成的无意义图示符可以显著提高性能。





图X证明负样本的引入使得模型对于形近字在特征空间的分类边界更加清晰，提升了模型的识别准确率。













### conclusion

We have presented a model capable of directly generating handwriting Chinese text image of arbitrary length. In our model, we design a Chinese sequence encoder(CSE) suitable for Chinese text image generation, and introduce the spatial perception module(SPM) into the text image generation model. Experimental results show that the proposed method generates high-quality images of handwritten Chinese text.

Our research is still ongoing and there are some deficiencies. Generated characters with many strokes and close coupling are still blurred. However, these Chinese characters almost no longer appear in modern commonly used Chinese, which has a limited impact on our methods. In the future, we additionally plan to address handwritten text image generation with controllable style.





、

