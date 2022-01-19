![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimage-20220106153744813.png)



there is a picture corresponding to the target character.
由于字母形文字的最小组成是字母，通过截取两个单词，拼接Create a Made Up Word 是很简单
例如，sleeping and boring is sloring，虽然sloring是错误单词，但是年轻人借此表达个性，激发创新。

。但是对于汉字而言，无法从输入法直接获得想要的部件组合。互联网是一个彰显个性的平台，各种部件组合的新含义层出不穷，但是有限的字库限制了创造力。我们的模型只需要输入通用的结构的对应的部件就可以得到对应的虚构字符，这对于促进网络文化交流十分有意义。

左面生成的是最近几年的大火的新造汉字字符，而右侧的无意义字符是我们大规模自动生成的，使用者可以用我们的模型尽情发挥创造力。





HCT-GAN: Handwritten Chinese Text Generation Conditioned on Text-lines

Our model achieves state-of-the-art visual quality for handwriting generation and, unlike prior methods [2, 19], generates entire lines of handwriting conditioned on arbitrarily long text. 

### abstract

In recent years, researchers have investigated a variety of approaches to generating isolated Chinese character images, but the handwritten Chinese lines image conditioned on arbitrary quantity characters is typically achieved by stacking single-character images. Due to complicated topological structures in Chinese, it is hard to generate readable images of handwritten Chinese lines by directly exploiting the text generation model. To address the problem, we propose an effective model named HCT-GAN to produce entire lines of Chinese handwriting from text-line labels. Specifically, HCT-GAN is designed as a CGAN-based\cite{mirza2014conditional} architecture that additionally integrates a Chinese text-line encoder (CTE), a sequence recognition module(SRM), and a spatial perception module (SPM). Compared with the one-hot embedding, CTE learns the latent content representation conditioned on text-line labels by reusing the structure and component embedding shared among the Chinese characters. SRM provides component sequence-level constraints to the generated images. SPM can adaptively constrain the spatial correlation between the generated components in an inexact supervised manner, which facilitates the modeling of characters with complicated topological structures. Benefiting from such artful modeling, our model suffices to generate images of handwritten Chinese lines with arbitrary length. Extensive experimental results demonstrate that our model achieves state-of-the-art performance in handwritten Chinese lines generation.

近年来，研究人员研究了各种生成孤立汉字图像的方法，但任意数量汉字的手写体线条图像通常是通过叠加单个汉字图像来实现的。由于中文中复杂的拓扑结构，直接利用文本生成模型很难生成可读的手写中文行图像。为了解决这个问题，我们提出了一个名为HCT-GAN的有效模型，用于从文本行标签生成完整的中文手写行。具体而言，HCT-GAN设计为基于CGAN的架构，另外集成了中文文本行编码器（CTE）、序列识别模块（SRM）和空间感知模块（SPM）。与热嵌入相比，CTE通过重用汉字之间共享的结构和组件嵌入来学习以文本行标签为条件的潜在内容表示。SRM为生成的图像提供组件序列级约束。SPM能够以一种不精确的监督方式自适应地约束生成部件之间的空间相关性，这有利于复杂拓扑结构字符的建模。得益于这种巧妙的建模，我们的模型足以生成任意长度的手写中文线条图像。大量的实验结果表明，我们的模型在手写体中文行生成中达到了最先进的性能。

### introduction



With the development of deep learning, the automatic generation of the Chinese character with complicated structures is not a challenge, but the handwritten Chinese lines generation remains underexplored. Compared to generating the Chinese character, the Handwritten Chinese lines generation is much more challenging. In addition to the inherent difficulties of the Chinese character generation, such as Chinese characters sharing an extremely large glyph with complicated structure, the handwritten Chinese lines generation has the following challenges. (i) Handwritten Chinese lines contain line-level features, such as the adhesions between adjacent characters. (ii) There are subtle differences in the same characters of a single style. Currently, handwritten lines images are acquired by splicing isolated character images. The method typically lacks authenticity for failing to solve the above two critical challenges. Our work alleviates the problem by directly generating images of handwritten Chinese variable-length lines.(e.g., see Figure \ref{fig:introduction_show}). Due to nefarious uses of forgery Handwriting technology, our proposed model does not aim to the specific writing styles. Text images with various styles can also provide additional data to improve handwriting recognition performance.



随着深度学习的发展，复杂结构汉字的自动生成已不再是一个挑战，但手写体汉字线条的自动生成仍处于探索阶段。与汉字生成相比，手写体汉字行的生成更具挑战性。除了汉字生成固有的困难，例如汉字共享一个非常大的字形和复杂的结构，手写汉字行生成还有以下挑战。（i） 手写汉字行包含行级特征，例如相邻字符之间的粘连。（ii）同一风格的相同人物之间存在细微差异。目前，手写线条图像是通过对孤立的字符图像进行拼接来获取的。由于未能解决上述两个关键挑战，该方法通常缺乏真实性。我们的工作通过直接生成手写中文可变长度行的图像来缓解这个问题。（例如，参见图\ref{fig:introduction\u show}）。由于伪造笔迹技术的恶意使用，我们提出的模型并不针对特定的书写风格。具有各种样式的文本图像还可以提供额外的数据，以提高手写识别性能。





Recent Chinese character generation are treated as an image-to-image translation problem via learning to map the source style to a target-style \cite{azadi2018multi, chang2018generating,jiang2018w,jiang2017dcfont,jiang2019scfont,liu2019fontgan,lyu2017auto,sun2018pyramid,xie2021dg,Yang2019TETGANTE,zhang2018separating,zheng2018coconditional,wen2019handwritten,wu2020calligan,2020RD,2017Learning,cha2020fewshot,park2020fewshot,2021Multiple}. However, previous Chinese character generation works ignore the basic fact that real Handwritten Chinese lines can not be obtain by stacking single character image.

通过学习将源样式映射到目标样式，最近的汉字生成被视为图像到图像的翻译问题\cite{azadi2018multi，chang2018generating，jiang2018w，jiang2017dcfont，jiang2019scfont，Liu2019Fonggan，lyu2017auto，sun2018pyramid，xie2021dg，Yang2019TETGANTE，Zhang2018Coconditional，Zhang2018W，jiang2017dcfont，Jing2019SCFONT，Liu2019Fongan，lyu2017auto，sun2018pyramid，XIE2022DG，Yang2019TETGANTE，Zhang2018Separation，Zhang2018Cocon。然而，以往的中文字体生成工作忽略了一个基本事实，即单字符图像叠加不能得到真实的手写中文文本图像。

Handwriting text generation (HTG)\cite{graves2013generating} is originally proposed for the generation of alphabetic text images, which basic structure has the nature of modeling the relationship between adjacent characters. The latest model ScrabbleGAN\cite{fogel2020scrabblegan} produce word images conditioned on one-hot embedding of letter and applies a character recognizer to constrain the text content. ScrabbleGan can not obtain the readable handwritten Chinese text image. The reason is not only large scale vocabulary in Chinese (say, as many as 70244 in GB180102005 standard), but also the reuse mechanism of Chinese components brings the problems of near-glyph interference. To acquire realistic-looking handwriting, we propose an effective model named HCT-GAN to generate handwritten Chinese lines directly from latent content representation conditioned on text-lines labels. Specifically, HCT-GAN is designed as a CGAN-based architecture that additionally integrates a Chinese text-line encoder (CTE), a sequence recognition module(SRM), and a spatial perception module (SPM).

手写文本生成（HTG）\cite{graves 2013generating}最初用于生成字母文本图像，其基本结构具有建模相邻字符之间关系的性质。最新型号的ScrabbleGAN\cite{fogel2020scrabblegan}使用一个热嵌入作为CGan\cite{mirza2014conditional}的条件信息，并应用识别器来约束文本内容。ScrabbleGan无法获得可读的手写中文文本图像。究其原因，不仅是汉语词汇量大（GB180102005标准中多达70244个），而且汉语组件的重用机制带来了近字形干扰的问题。为了获得逼真的笔迹，我们提出了一个有效的模型HCT-GAN，该模型可以直接从基于文本行标签的潜在内容表示生成手写体中文行。具体而言，HCT-GAN设计为基于CGAN的架构，另外集成了中文文本行编码器（CTE）、序列识别模块（SRM）和空间感知模块（SPM）。

Appling one-hot embedding to Chinese characters will lead to an explosive growth of model parameters, and it is difficult to learn the content representation conducive to generation. To address this issue, we propose a learnable Chinese text-line encoding method. By reusing the structure and component embedding shared among the Chinese characters, CTE enables every character to be transformed to an embedding vectors combination. Compared with the one-hot embedding, CTE is a more informative text-line encoding method. The alphabetic text generation model generally uses a 1-D recognition network to induce legibility. Due to complicated spatial structures in Chinese, only using a 1-D recognition network is not enough. Thus, we modify the 1-D recognition network to SRM. The former predicts character sequences and the latter predicts component sequences. 

We further introduce a spatial perception module(SPM) to promote higher-quality images. SPM performs 2-D predictions to guide the generator to adaptively learn the spatial correlation between the internal components of lines in an inexact supervised manner, which facilitates the modeling of characters with complicated topological structures. Benefiting from such artful modeling, our model suffices to generate images of handwritten Chinese text with arbitrary lengths. 

将一个热嵌入应用于汉字将导致模型参数的爆炸性增长，并且很难学习有利于生成的内容表示。为了解决这个问题，我们提出了一种可学习的中文文本行编码方法。通过重用汉字之间共享的结构和组件嵌入，CTE可以将每个汉字转换为嵌入向量组合。和热嵌入相比，CTE是一种信息量更大的文本行编码方法。字母文本生成模型通常使用一维识别网络来诱导易读性。由于汉语空间结构复杂，仅使用一维识别网络是不够的。因此，我们将一维识别网络修改为SRM。前者预测字符序列，后者预测成分序列。

我们进一步引入了空间感知模块（SPM）推广更高质量的图像。SPM是一个二维预测模块，它可以引导生成器以不精确的监督方式自适应地学习线的内部组件之间的空间相关性，这有助于对具有复杂拓扑结构的字符进行建模。得益于这种巧妙的建模，我们的模型足以生成任意长度的手写中文文本图像。



Unlike prior work, we only take the latent content representation conditioned on text-line labels(i.e., component and structure learnable embedding vector) as the network input. Since any appropriate combination of component and structure will correspond to a glyph, our model not only generate Chinese lines, but also some non-sense glyphs. We know that data augmentation is conducive to recognition performance. Experimental results demonstrate that non-sense glyphs work better.

与以前的工作不同，我们只将以文本行标签（即组件和结构可学习的嵌入向量）为条件的潜在内容表示作为网络输入。由于组件和结构的任何适当组合都将对应于一个字形，因此我们的模型不仅生成中文线条，还生成一些无意义的字形。我们知道数据增强有助于提高识别性能。实验结果表明，非感觉字形效果更好。



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



### related work

\textbf{Chinese character generation.} Generally speaking, existing Chinese character generation methods can be classified into two categories: component assembling-based methods and deep learning-based methods. The former regards a character as a combination of strokes or components, which first extract strokes from character samples, and then some strokes are selected and assembled into unseen characters by reusing parts of a character. The latter frame character generation as conditional image generation, directly learning from and predicting pixels.

 \cite{mybib:Automatic_Generation_of_Artistic_Chinese_Calligraphy} proposed a small structural stroke database to represent each character in a multi-level way, and calligraphy is generated via a reasoning-based approach. \cite{mybib:Handwritten_Chinese_Character_Font_Generation_Based_on_Stroke_Correspondence}consider each stroke to be a vector and proposed individual’s handwritten Chinese character font generation method by vector quantization.
In the \cite{xu2009automatic} method, he derive a parametric representation of stroke shapes and generate a new character topology via weighted averaging of a few given character topologies derived from the individual’s previous handwriting. 
\cite{lian2012automatic} and \cite{lian2016automatic} by applying the Coherent Point Drift (CPD) algorithm to achieve non-rigid point set registration between each character to extract stroke/radical.
\cite{lin2014font} exploit a radical placement database of Chinese characters, vector glyphs of some other Chinese characters are automatically created, and these vector glyphs are merged into the user-specific font. 
Later, the Radical composition Model \cite{zhou2011easy} and \cite{zong2014strokebank} were proposed by mapping the standard font component to their handwritten counterparts to synthesize characters. 

After the generative adversarial networks(GANs)\cite{goodfellow2014generative} was proposed, its derivative version\cite{mirza2014conditional,zhang2019selfattention,karras2019stylebased} was widely adopted for style transfe\cite{chen2017stylebank,johnson2016perceptual,isola2017image,zhu2017unpaired,choi2018stargan,choi2020stargan}. Several attempts have been recently made to model font synthesis as an image-to-image translation problem \cite{Rewrite,Zi-to-zi,azadi2018multi, chang2018generating,jiang2018w,jiang2017dcfont,jiang2019scfont,liu2019fontgan,lyu2017auto,sun2018pyramid,xie2021dg,Yang2019TETGANTE,zhang2018separating,zheng2018coconditional,wen2019handwritten}, which transforms the image style while preserving the content consistency.

\textbf{汉字生成}到目前为止，已经提出了大量的字体生成方法。一般来说，现有的方法可以分为两类：基于计算机图形学的方法和基于深度学习的方法。传统的方法通常基于笔划或根提取和重组的基本思想。"""“\cite{mybib:Automatic_Generation_of_art_Chinese_horthip}提出了一个小型的结构化笔划数据库，以多级方式表示每个字符，并通过基于推理的方法生成书法。\cite{mybib:Handwrited_Chinese_character_Font_Generation_on_stroke_correlation}以笔画为载体，提出了矢量量化的手写体汉字字形生成方法。在{xu2009automatic}方法中，他推导出笔划形状的参数表示，并通过对从个人先前笔迹中导出的几个给定字符拓扑进行加权平均，生成新的字符拓扑。"“\cite{lian2012automatic}和\cite{lian2016automatic}通过应用相干点漂移（CPD）算法实现每个字符之间的非刚性点集注册，以提取笔划/部首。”“\cite{lin2014font}利用汉字的部首位置数据库，自动创建一些其他汉字的矢量字形，并将这些矢量字形合并到用户特定的字体中。随后，部首组合模型\cite{zhou2011easy}和\cite{zong2014strokebank}”通过将标准字体组件映射到手写体组件来合成字符。"“在生成性对抗网络（GANs）被提出后，其衍生版本{mirza2014conditional，zhang2019selfattention，karras2019stylebased}被广泛用于风格转换{chen2017stylebank，Johnson 2016Perceptive，isola2017image，zhu2017unpaired，choi2018stargan，choi2020stargan}”. 最近有人尝试将字体合成建模为一个图像到图像的翻译问题，例如{重写，Zi to Zi，azadi2018multi，chang2018generating，Jiang 2018W，Jiang 2017DCFONT，Jiang 2019SCFONT，Liu2019Fonggan，lyu2017auto，sun2018pyramid，xie2021dg，Yang2019TETGANTE，zhang2018separating，zheng2018coconditional，Wen2019Handwrited}，它在保持内容一致性的同时变换图像样式。





\textbf{Handwritten text generation(HTG).} Since the high inter-class variability of text styles from writer to writer and intra-class variability of same writer styles\cite{krishnan2021textstylebrush}, the handwritten text generation is challenging.
At present, handwritten text generation mainly focuses on alphabetic text. Alonso et al.\cite{alonso2019adversarial} proposed an offline handwritten text generation model for fixed-size word images. ScrabbleGAN\cite{fogel2020scrabblegan} used a fully-convolutional handwritten text generation model.
For handwritten Chinese text generation(HCTG) tasks, the existing text generation model cannot generate readable content. In contrast, our method is applicable for images of handwritten Chinese text with arbitrary length.



### Our method

Our model only have two inputs: text-line labels, style noise. Figue \ref{fig:arch} shows an overview of the proposed model, mainly consisting of five modules of a Chinese text encoder(CTE), a generator(G), a discriminator(D), a sequence recognition module(SRM), and a spatial perception module(SPM). CSE is mainly responsible for obtaining latent content representation($e$) conducive to generation, and providing component labels for SPM/SRM(not pictured in Figue \ref{fig:arch} ).

G is constructed based on a full convolution network(FCN)\cite{long2015fully} and can generate variable-length output. D promotes realistic-looking handwriting styles. SRM is a 1-D sequence recognizition network with connectionist temporal classification (CTC) loss\cite{graves2006connectionist} to induce basic legibility. SPM is a 2-D prediction module that guides the generator to capture the spatial offset of the writing process and refine each type of component in correct position. 

我们的模型只有两个输入：文本行标签、样式噪声。Figue\ref{fig:arch}展示了所提出模型的概述，主要由中文文本编码器（CTE）、生成器（G）、鉴别器（D）、序列识别模块（SRM）和空间感知模块（SPM）五个模块组成。CSE主要负责获取有助于生成的潜在内容表示（$e$），并为SPM/SRM提供组件标签（Figue\ref{fig:arch}中未显示）。"“G基于全卷积网络（FCN）构建，可以生成可变长度的输出。D促进逼真的笔迹风格。SRM是一个一维序列识别网络，具有连接主义时间分类（CTC）丢失\cite{Graves 2006Connectionist}”使基本易读。SPM是一个二维预测模块，用于引导生成器捕获写入过程的空间偏移，并在正确位置优化每种类型的组件。



#### Chinese text encoder(CTE)

Our input is a Chinese character string or character, e.g., “\begin{CJK}{UTF8}{gbsn}呗员...\end{CJK}”. \cite{wu2020calligan} embed component sequences into fixed-length vectors, which is not suitable for the generation of the text-line image. We propose an intuitive method that decomposes each character of the Chinese text-line into component sequence index and structure index, and padding to equal length to fit the generation of lines image. Since the same components and structure appear repeatedly in various characters, the number of indexes is much smaller than character classes in the Dictionary. Figure \ref{fig:arch}(left) shows a few examples of the Dictionary. 

CTE directly learns the shared embedding of indexes between characters. Separate embedding combinations is reshaped into latent content representation tensor e\in\mathbb{R}^{480\times 4\times 4W} according to different text labels, instead of mapping the embedding to a fixed-length vector in\cite{wu2020calligan}, which helps to mimic the writing process that the components of Chinese characters are only connected with their local spatial components, and avoid using a recurrent network to learn the coupling embedding of the whole text. An additional benefit of CTE is generating non-sense glyphs with a random combination of components. Table \ref{table:encoder} shows the details of CTE.



我们的输入是一个中文字符串或字符，例如“\begin{CJK}{UTF8}{gbsn}..\end{CJK}”。\引用{wu2020caligan}将组件序列嵌入到固定长度向量中，这不适合生成文本行图像。我们提出了一种直观的方法，将中文文本行中的每个字符分解为组件序列索引和结构索引，并填充等长以适应行图像的生成。由于相同的组件和结构在不同的字符中重复出现，索引的数量比字典中的字符类少得多。图\ref{fig:arch}（左）显示了字典的一些示例。"“



CTE直接学习字符之间索引的共享嵌入。根据不同的文本标签，单独的嵌入组合被重塑为潜在内容表示张量e\in\mathbb{R}{480\times 4\times 4W}，而不是将嵌入映射到\cite{wu2020caligan}中的固定长度向量，这有助于模拟汉字成分仅与其局部空间成分相连的书写过程，避免使用递归网络来学习整个文本的耦合嵌入。和热嵌入相比，CTE是一种信息量更大的字符编码方法。CTE的另一个好处是通过组件的随机组合生成无意义图示符。表\ref{Table:encoder}显示了CTE的详细信息。





#### Generator(G) and Discriminator(D)

G is inspired by SAGAN\cite{zhang2019selfattention}, but differs in architecture and receives the variable-length latent content representation tensor e as input with the style vector concatenated at each spatial position. As shown in Tabel \ref{table:encoder}, CSE module outputs the latent content representation tensor $e\in\mathbb{R}^{480\times 4\times 4W}$. Later, e is concatenated with noise vector $z_1 \in \mathbb{R}^{32\times 4\times4W}$  and fed into G, and the additional noise vectors $z_2, z_3$, and  $z_4$ are fed into each layer of G through Conditional Batch Normalization (CBN)\cite{shell2015bare} layers. This allows G to control the low-resolution and high-resolution details of the appearance of Chinese text to match the required handwritten style. Some common techniques in the training, such as self-attention mechanism\cite{vaswani2017attention} to refine local area image quality, spectral normalization\cite{miyato2018spectral} and hinge loss function\cite{lim2017geometric} to stabilizes the training, FCN to deal with variable-length images\cite{krishnan2021textstylebrush,fogel2020scrabblegan}, was used. 

To cope with variable-length images, D is also fully convolutional structure and performs global average pooling. The pooling layer aggregates logis from the variable-length feature map into the final discriminator output.

#### spatial perception module(SPM) and sequence recognition modul(SRM)



In addition to our proposed coding method, The synergy between sequence recognition module(SRM) and spatial perception module(SPM) is our main improvement. The alphabetic text generation model generally uses a 1-D sequence character recognition network to evaluate the content of the generated images. Due to the huge number of Chinese characters with complicated topological structures, only using a 1-D sequence character recognition network is not enough to obtain high-quality readable text images. The reuse mechanism of Chinese components also brings the problems of near glyph interference and information redundancy. Therefore, we treat a Chinese text-line as a irregular component sequences with complex spatial relationships rather than as 1-D character sequences.



We consider starting from two aspects, introducing a sequence recognition module(SRM) for promoting the basic legibility of handwritten text, and designing a spatial perception module(SPM) to performs 2-D predictions to guide the generator capture spatial correlation between the internal components of lines. In section IV. Ablation study prove the necessity. SRM modified from character sequence prediction predict component sequence trained on CTC loss. It consist of a feature extractor based on a convolutional neural network, followed by a max-pooling on the vertical dimension and a full connection layer. Using SRM alone is not enough to generate realistic-looking handwritten Chinese lines. The reason is that the SRM is essentially a 1-D sequence prediction system, which is difficult to capture the complex spatial correlation of Chinese characters/components. A similar situation is that the performance of CRNN\cite{shi2016end} based system is insufficient in the irregular text recognition.



SPM is inspired by inexact supervision where the training data are given with only coarse-grained labels. We treat it as a non-sequence 2-D prediction problem with image-level annotations. 
SPM is composed of a full convolution network with a spatial attention module, followed by a full connection layer trained on ACE\cite{xie2019aggregation} loss. SPM constraint the component category at each location in the 2-D feature map, and guide the network to generate the corresponding component at the correct location. It is conceivable that the 2-D prediction method can guide the generator to learn the up-down offset of characters in the writing process and the more  realistic-looking details of components.



Note tha most of recognition model use a recurrent network, typically bidirectional LSTM \cite{shi2016end,he2016reading}, which may predict characters based on linguistic context instead of clear character shapes. In contrast, SRM and SPM only uses local visual features for character recognition and therefore provides better optimization direction for generating characters.



### Experiments

The offline handwritten Chinese database, CASIA-HWDB \cite{liu2011casia} is a widely used database for handwritten Chinese character recognition. It contains samples of isolated characters and handwritten lines that were produced by 1020 writers on dot-matrix paper with Anoto pen. Offline handwritten Chinese character samples are divided into three databases: HWDB1.0~1.2(Including 7,185 classes Chinese characters and 171 classes English letters, numbers, and symbols). Handwritten texts are also divided into three databases: HWDB2.0~2.2(Its character classes are contained in HWDB1.0~1.2). 







formed





Chinese characters share an extremely large vocabulary(say, as many as 70244 in GB180102005 standard), one-hot embedding may lead to an explosive growth of model parameters, and it is difficult to learn the content representation conducive to generation. To address this issue, we propose a learnable Chinese textline encoding method. By reusing the structure and component embedding shared among the Chinese characters, CTE enables every character to be transformed to a combination of embedding vectors. Compared with the one-hot embedding, CTE is a more informative encoding method. The alphabetic text generation model generally uses a 1-D character recognition network to induce legibility. Due to large vocabulary, only using a character recognition network is not enough. 
To guide clearer content, we no longer regard images of Chinese lines as a 1-D character sequence, but as irregular text of components.
Thus, we modify the character recognition network to SRM to predict 1-D component sequences. Considering the spatial structure between components, we further introduce a spatial perception module(SPM) to promote higher-quality images. SPM performs 2-D predictions to guide the generator to adaptively learn the spatial correlation between the internal components of lines in an inexact supervised manner, which facilitates the modeling of characters with complicated topological structures. Benefiting from such artful modeling, our model suffices to generate images of handwritten Chinese text with arbitrary lengths. 

汉字的词汇量非常大（GB180102005标准中多达70244个），一次热嵌入将导致模型参数的爆炸性增长，并且很难学习有利于生成的内容表示。为了解决这个问题，我们提出了一种可学习的中文文本行编码方法。通过重用汉字之间共享的结构和组件嵌入，CTE可以将每个汉字转换为嵌入向量的组合。和热嵌入相比，CTE是一种信息量更大的编码方法。字母文本生成模型通常使用一维字符识别网络来诱导易读性。由于词汇量大，仅使用字符识别网络是不够的。为了引导更清晰的内容，我们不再将中文线条图像视为一维字符序列，而是视为组件的不规则文本。因此，我们将字符识别网络修改为SRM来预测一维分量序列。考虑到组件之间的空间结构，我们进一步引入了空间感知模块（SPM）来提升图像质量。SPM执行二维预测以引导生成器以不精确的监督方式自适应地学习线条内部组件之间的空间相关性，这有助于对具有复杂拓扑结构的字符进行建模。得益于这种巧妙的建模，我们的模型足以生成任意长度的手写中文文本图像。



