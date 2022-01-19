





```

Chinese characters consist of an extremely large glyphs,  more than 100,000 characters can be composed by approximately 1000 components. 

汉字是由大量的字形组成的，大约有1000个组成部分可以组成超过10万个汉字。

we note that each Chinese character can be basically decomposed to the combination of components string and a structure

我们注意到，每个汉字基本上可以分解为部件字符串和结构的组合，如图X所示。

which makes our model have the ability to generate nonsense glyphs, and then these glyphs can promote Handwritten Chinese recognition tasks.
 这使得我们的模型能够生成无意义字形，然后这些字形可以促进手写体中文识别任务。
our framework is designed to learn a way to generate real handwritten Chinese text ,which can contain non-sense glyphs.

我们的框架旨在学习一种生成真实手写中文文本的方法，该文本可以包含无意义的字形。
实验证明负样本的引入使得模型对于形近字在特征空间的分类边界更加清晰，提升了模型的识别准确率。

由于汉字字符具有丰富的部件细节和复杂的空间结构，传统以character为基本生成单位只使用全局序列识别模块的文本生成工作对于生成可读的手写中文文本是不足的。目标检测是解决目标在哪里是什么的问题，受目标检测任务启发，我们提出引入空间感知模块，以component 为生成任务的基本单位，在2D特征图上预测每一类component的空间分布，并要求生成的图像服从这一分布。由于component的位置标注成本太高，我们采取了弱监督的方式。


Some previous works based on encoder-decoder have also exploit the prior domain knowledge of Chinese characters to 

以往基于encoder-decode的有些文章也利用汉字的先验信息通过整合到进生成器的方式实现了性能提升，但是都是作为辅助先验信息和汉字的图像信息搭配编码使用，从而实现少样本学习的效果，目的是使用更少的风格化或手写汉字创造出一套风格化的或手写的字体库，


 In this way,这样，

To be specific, given a small number of characters written by a user, 

具体地说，给定用户书写的少量字符，
DCFont system that contains two major components, 该系统包含两个主要组件，



```









3.Our method



```
In this section, we first formally introduce our tasks and the detailed architecture of the proposed zi2text system. Then, we will present all details of each module and the process of optimization. Finally, we show how to use our model to generate handwritten Chinese text.

在本节中，我们首先正式介绍我们的任务和整个框架的体系结构。然后，我们详细介绍了各个模块所有详细信息以及优化过程。最后，我们展示了如何使用我们的模型进行手写汉字文本的生成。
```



First of all, the appropriate combination of component and structure will correspond to a glyph. More than 100000 Chinese character are a subset of the combination. Secondly, The handwritten text of any writer should have text-level features and differentiated output of the same Chinese characters. which is different from the text obtained by stacking single glyph. Our framework is designed to learn a way to generate real handwritten Chinese text ,which can contain non-sense glyphs. 

```
。
```

首先，任意component与structure合适组合都会对应一个字形，10万多个汉字字形是组合的一个子集。任何一个作者的手写文本要具备文本级特征和相同汉字的差异化输出，不同于单字堆叠得到的文本。我们提出的模型旨在学习一种方式去生成真实的手写汉字文本，该文本可以包含无意义的字形。

**architecture.** Fig. 2 illustrates the architecture proposed str2text model. 

图2示出了str2text系统提出的体系结构。

Here, we introduce proposed str2text framework. The framework mainly consists of five modules of a Chinese string encoder, a Generator, a Discriminator,a sequence recognition modul(SRM) and a spatial perception module(SPM). The Chinese character encoder is mainly responsible for encoding the components and structures information of Chinese string into content representation $e$ . the Generator is constructed based on full convolution network and has the ability to generate variable sized output. Discriminator promotes realistic looking handwriting styles. The SRM is a 1-D sequence recognizer with connectionist temporal classification (CTC) loss**[]** to  ensures that the generated handwritten Chinese text are in accurate character order. The SPM is a 2-D prediction model that guide the network to refines each type of component in spatial position. Due to the high cost of component coordinate annotation, SPM adopts an inexact supervised approach.

这里，我们将介绍我们模型的细节。该模型主要由中文字符串编码器、生成器、鉴别器、序列识别模块(SRM)和空间感知模块(SPM)五个模块组成。其中汉字字符编码器主要负责将汉字字符序列的部件和结构信息编码到内容表示e，生成器基于全卷积网络构建，具备不定长文本的生成能力。鉴别器促进逼真的笔迹风格。SRM是一种具有CTC损失的一维序列识别器[]，以确保生成的手写中文文本符合准确的字符顺序。SPM是一种二维预测模型，用于在空间位置上细化各种类型的部件。SRM确保生成的手写汉字文本顺序准确，内容基本一致。SPM是一个二维预测模型，指导生成器在空间位置上细化每种类型的组件。由于组件坐标标注成本高，SPM采用了一种不精确的监督方法。

**encoder：**  

组件和结构编码器：

Our input is a Chinese string, e.g.,  Fig. 2 (“呗员...”).  Each Chinese character can be decomposed into a string of components and a unique structure. Meanwhile, the same component may present in various glyphs at various locations. Based on the appeal characteristic, we design an encoding method.  Figure 3 shows a few examples of how to encode the information.

许多汉字是由一套基本成分和结构组成的。

我们的输入是一个中文字符串，如图2（“呗员…”）。每个汉字可以分解成一系列的组成部分和一个独特的结构。同时，同一组件可能出现在不同位置的不同图示符中，基于上诉特征，我们设计了一种编码方式，图3显示了一些如何编码信息的示例。

To generate variable sized output, the encoder needs to send a variable size content representation conforming to the full convolution network into the generator. For this, The encoder directly learns the embedding of each component and structure, and then reshapes it into content representation $e\in\mathbb{R}^{512\times 4\times W}$ as shown in Figure 4, instead of mapping the embedding to a fixed length vector in **[CalliGAN]**, which helps to simulate the writing process by convolution, that is, the components of Chinese characters are only connected with their local spatial components, and avoid using a recurrent network to learn the embedding of the whole text. Here, 512 is the number of channels, and W corresponds to the width of the generated handwritten Chinese text.

为了生成可变大小的输出，编码器需要向生成器发送符合全卷积网络的可变大小内容表示。为此编码器直接学习每个组件和结构的嵌入，如图5，整合后得到 $e \in\mathbb{R}^{512\times 4\times W}$，而不是将嵌入映射到***[CalliGAN]***中的固定长度向量，这有助于通过卷积模拟书写过程，即汉字的组件仅与其局部空间组件发生联系，避免使用循环网络学习整个句子文本的嵌入。这里512是通道数，W对应生成手写汉字文本的宽度。

```
我们为我们的任务定制了编码器，如图X。如图X所示，我们将汉字拆解为四个位置的部件序列(不够四个部件的为blank)，和一个位置的结构信息分别嵌入128维向量，如图X，按顺序排列，以此来表示唯一的一个汉字，这样做的好处是相比于将部件序列信息耦合进一个向量的方式，这种将字与字之间部件和结构分离


**Others:** ，我们并未采用SVAE中附加ID给每个汉字的方式，也未采用CalliGAN中将部件序列经过LSTM融合为一个固定维度向量的方式，

且放弃ID的嵌入使得我们可以可控的生成不存在的汉字，例如图X，将通过生成大量形近字的衍生负样本，使得形近字的识别率得以提高，

CalliGAN中采用LSTM将部件序列整合到固定维度，这对于单字的生成是有利的，但是我们汉字文本生成的任务中只希望生成的每个汉字只和前后字符产生关联，所以我们对汉字重新编码并引入结构信息。

SVAE中使用101bits表示101个高频部首信息、12bits表示12种高频结构信息，但是文章并未详细说明编码方式，代码未开源。calliGAN中编码未引入结构信息。

**ours：**我们为了任务需要去掉 calliGAN中的LSTM，只将component序列信息映射到固定维度，同时我们加入结构序列信息，如图XX

Given Chinese character string(i.e. 呗员...), we consult the dictionary L to obtain corresponding components and structures sequence $s,c$ to generate  content representation $ e\in \mathbb{R}^{512\times 4 \times  W}$ containing components and structures  embedding through a encoder E. Here, 512 is the number of channels in last convolutional feature block.

给定中文字符串（即…），我们查阅字典L以获得相应的组件和结构序列$s$，从而生成包含通过编码器e嵌入的组件和结构的特征向量$c\in\mathbb{R}^{512\times 4\times W}$。这里，512是最后一个卷积特征块中的信道数。



汉字是部件和空间结构的组合，我们利用开源代码文档得到汉字拆字的部件序列，修改至适合我们任务后，加入我们设计的结构类别补充信息，得到每个汉字独一无二的部件+结构序列，示例如图X所示。

```





**生成器和判别器：**

The discriminator network D and The generater network G are inspired by SAGAN **[Self Attention Gan(SAGAN)]**，The structure is shown in table X, and some common techniques in the training Gan are used, such as self-attention mechanism to refins local area image quality**[[Attention is All You Need]**, spectral normalization**[Spectral normalization for generative adversarial networks]** and Hinge loss function to stabilizes the training **[Generative Adversarial Nets (GANs)]** .



鉴别器网络D和生成器网络G受SAGAN[ ]的启发，其结构如表X所示，并使用训练GAN中的一些常用技术，如注意机制改善局部图像质量、hinge loss和光谱归一化以稳定训练。





```
SAGAN将self attention layer 加入到GAN中，使得编码器和解码器能够更好地对图像不同区域的关系进行建模，使用hinge loss目标函数。通过class-conditional BatchNorm为G提供类信息，通过projection为D提供类信息。对G使用了谱归一化


and conditional instance normalization**[A Learned Representation For Artistic Style]**. 

和条件实例归一化。
```

As shown in Fig. 2, the Chinese string encoder module outputs the content representation $e$, Later, the content representation is concatenated with noise vector $z_1$  in the channel layer and fed into G, and the additional noise vectors $z_2, z_3\text{and} z_4$  are fed into each layer of the generator through Conditional Batch Normalization (CBN)**[Modulating early visual processing by language]** layers. Thus we allow the generator to control the low resolution and high resolution details of the appearance of Chinese text to match the required handwritten style.

Full convolution networks(FCN) is  a commonly technique to deal with variable size input and output problems **[TextStyleBrush: Transfer of Text Aesthetics from a Single Example , ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation]**. In this paper, we inherit this paradigm in SAGAN based generators.

如图2所示，汉字字符串编码器模块输出内容表示 e，随后，在通道层中的噪声z1 cat的内容表示后馈入G，同时将附加噪声矢量z2、z3和z4通过条件批量归一化（CBN）馈送至生成器的每一层。因此，我们允许生成器控制中文文本外观的低分辨率和高分辨率细节，以匹配所需的手写样式。

全卷积网络是处理可变大小输入和输出问题的通用技术**[TextStyleBrush：从单个示例中传递文本美学，ScrabbleGAN：半监督变长手写文本生成]**。在本文中，我们在基于SAGAN的生成器中继承了这一范式。

At the same time, the FCN allows the network to learn the dependencies between adjacent components, which solves the problem of different output of the same characters in a single handwriting style.

同时，全卷积的结构允许网络学习相邻部件之间的依赖关系，这解决了在单一手写样式中相同字符的差异输出的问题

 To cope with varying width image generation, D is also fully convolutional. In the Gan paradigm, the discriminator is used to distinguish the authenticity of the generated image. In our task, the role of D is to promote the generator output realistic handwritten text.

为了处理不同宽度的图像生成，D也是完全卷积的。在GAN的范式中，判别器用于区分生成的图像的真假，在我们的任务中，D的作用是促进生成器输出真实的手写文本。

Different from the common Chinese font generation network, the optimization objectives of the network are different. Taking advantage of the repetitiveness of the same component in different Chinese characters, we introduce component-level spatial perception module(SPM) based on inexact supervised learning to  promote the network to generate components at a variety of locations and sizes. 

与普通的中文字型生成网络不同，该网络的优化目标也有所不同。利用相同构件在不同汉字中的重复性，引入构件级空间感知模块(SPM)，促进网络生成不同位置和大小的部件。











```
条件实例归一化层[9]用于使用三个额外的32维噪声向量z2、z3和z4调制剩余块。最后，使用具有tanh激活的卷积层来输出最终图像。鉴别器网络D的灵感来源于SAGAN[6]：4个剩余块，后跟一个线性层和一个输出。为了处理不同宽度的图像生成，D也是完全卷积的，基本上是处理水平重叠的图像块。最后的预测是贴片预测的平均值，它被输入到GAN铰链损耗中[28]。识别网络R的灵感来自CRNN[35]。网络的卷积部分包含六个卷积层和五个池层，所有这些层都具有ReLU激活。最后，使用线性层输出每个窗口的类别分数，并与使用连接主义时间分类（CTC）损失的地面真相注释进行比较[13]。
Conditional Instance Normalization layers [9] are used to modulate the residual blocks using three additional 32 dimensional noise vectors, z2, z3 and z4. Finally, a convolutional layer with a tanh activation is used to output the final image. The discriminator network D is inspired by SAGAN [6]: 4 residual blocks followed by a linear layer with one output. To cope with varying width image generation, D is also fully convolutional, essentially working on horizontally overlapping image patches. The final prediction is the average of the patch predictions, which is fed into a GAN hinge-loss [28]. The recognition network R is inspired by CRNN [35]. The convolutional part of the network contains six convolutional layers and five pooling layers, all with ReLU activation. Finally, a linear layer is used to output class scores for each window, which is compared to the ground truth annotation using the connectionist temporal classification (CTC) loss [13].

在GAN范式[11]中，鉴别器D的目的是区分由G生成的合成图像和真实图像。在我们提出的体系结构中，D的作用也是根据手写输出样式区分这类图像。鉴别器结构必须考虑生成图像的不同长度，因此也被设计为卷积：鉴别器本质上是具有重叠感受野的独立“真/假”分类器的串联。由于我们选择不依赖于字符级注释，因此我们不能对这些分类器中的每一个使用类监督，而不是类条件gan，如[30,6]。这样做的一个好处是，我们现在可以使用未标记的图像来训练D，即使是来自其他看不见的数据语料库。池层将所有分类器的分数聚合到最终的鉴别器输出中。

In the GAN paradigm [11], the purpose of the discriminator D is to tell apart synthetic images generated by G from the real ones. In our proposed architecture, the role of D is also to discriminate between such images based on the handwriting output style. The discriminator architecture has to account for the varying length of the generated image, and therefore is designed to be convolutional as well: The discriminator is essentially a concatenation of separate “real/fake” classifiers with overlapping receptive fields. Since we chose not to rely on character level annotations, we cannot use class supervision for each of these classifiers, as opposed to class conditional GANs such as [30, 6]. One benefit of this is that we can now use unlabeled images to train D, even from other unseen data corpus. A pooling layer aggregates scores from all classifiers into the final discriminator output.

We further handle the multi-scale nature of text styles by extracting layer-specific style information. To this end, we introduce a style mapping network, M , which converts es to layerspecific style representation, ws,i, where i indexes the layers of the generator which are then fed as the AdaIN normalization coefficient [21] to each layer of the generator. Thus, we allow the generator to control both low and high resolution details of the text appearance to match a desired input style.
我们通过提取特定于层的样式信息，进一步处理文本样式的多尺度性质。为此，我们引入了一个样式映射网络M，它将es转换为特定于图层的样式表示ws，i，其中i索引生成器的图层，然后将这些图层作为AdaIN归一化系数[21]馈送到生成器的每一层。因此，我们允许生成器控制文本外观的低分辨率和高分辨率细节，以匹配所需的输入样式。
```









**spatial perception module(SPM) and sequence recognition modul(SRM)**

In addition to our proposed coding method, The synergy between sequence recognition modul(SRM) and spatial perception module(SPM) is our main improvement.

除了我们提出的编码方法，SPM和SRM之间的协同作用是我们的主要改进

```
Many of these methods are inspired by the convolutional recurrent neural network (CRNN) architecture, used originally for scene text recognition by Shi et al. [35].
其中许多方法的灵感来自卷积递归神经网络（CRNN）体系结构，Shi等人最初用于场景文本识别。
```



The generation of alphabetic text usually adopts 1-D sequence prediction architecture to evaluate the content of the generated images.

字母文本的生成通常采用序列预测结构来评估生成图像的内容 **[Adversarial Generation of Handwritten Text Images Conditioned on Sequences，ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation，TextStyleBrush: Transfer of Text Aesthetics from a Single Example，]**。

Chinese characters can also generate text by analogy with letters. However, due to the large number Chinese characters (more than 100000 Chinese characters compared with dozens of letters in English), it is very difficult to  accurately generate 100000 types of glyphs. Due to the reuse mechanism of Chinese component, different glyphs also have problems of near-glyph interference and information redundancy. Therefore, we treat a Chinese glyph as a string of components and unique structure rather than as single character category.

汉字也可以像字母一样生成文本。然而，由于汉字数量庞大(与英语的几十个字母相比，汉字数量超过10万个)，很难准确生成10万种字形。由于中文部件的重用机制，不同的字形也存在着近字形干扰和信息冗余的问题。因此，我们把汉字看作是一串组成成分和独特结构的符号，而不是单一的文字类别。

Compared with all glyphs,  the components are more than 100 times less. Limited combinations of limited components in space form different glyphs, as shown in Figure X. Therefore, if the Chinese text is viewed from the perspective of components, it can be regarded as a very irregular text, that is, there is a highly complex spatial relationship between components on the premise of overall order.

所有部件相比于全部字形类别少100多倍，有限的部件在空间中的有限组合构成了不同的字形，如图X所示的汉字组成。所以若以部件的角度看中文文本，可以看作一种极不规则的文本，即部件之间在整体有序的前提下局部存在高度复杂的空间关系。

Thus, we consider starting from two aspects, introduce a sequence recognition modul(SRM) for promoting the correct character order of handwritten text, and design a spatial perception module(SPM) to refine each type of component in 2-D space. In section 4. Ablation experiments proved that this was necessary.

因此，我们考虑从两个方面入手，引入序列识别模块(SRM)来促进手写体文本的正确字符顺序，并设计空间感知模块(SPM)来细化二维空间中的各类构件。在第四节。消融实验证明这是必要的。

SRM consist of a feature extractor based on convolutional neural network,  followed by a max pooling on the vertical dimension and a full connection layer. We use CTC loss to train the system.

SRM由一个基于卷积神经网络的特征抽取器、一个垂直维度上的最大池和一个全连接层组成。我们用CTC损失来训练系统。

Using SRM alone is not enough to generate clearly detailed handwritten Chinese characters. The reason is that  the SRM using CTC loss is essentially a 1-D prediction system, which is difficult to capture the complex spatial structure of Chinese characters. The same situation is that the performance of CRNN based system is insufficient in irregular text recognition task.

只使用识别器不足以生成细节清楚的手写汉字。原因是基于CTCloss的识别网络本质上是一个一维预测系统，难以捕获到汉字复杂的空间结构。同样的情形是在不规则文本识别任务中CRNN性能不足。

```
, and some handwritten Chinese characters with wrong structure will be generated, as shown in Figure X.

的结果是细节不清楚，会产生一些结构错误的手写汉字，如图X所示。(left:汉字岗，rigt ：non-sense glyph)
```





**空间感知模块(SPM)：**

SPM is Inspired by object detection with inexact supervision where the training data are give with only coarse-grained labels. 

Learning from this idea, we treat it as a non sequential 2-D prediction problem like irregular scene text recognition with image level annotations. 

SPM的灵感来源于具有不精确监督的目标检测，其中训练数据仅使用粗粒度标签。借鉴这一思想，我们将其视为非序列的二维预测问题，类似于带有图像级注释的不规则场景文本识别。



SRM consists of a full convolution network with spatial attention module, followed by a reshape operation.  We use ACE loss to train the system.

SRM由一个具有空间注意模块的全卷积网络和一个reshape操作组成。我们使用ACE损耗来训练系统。

We use SPM to predict the component category at each location in the 2-D feature map, and guide the network to generate the corresponding component at the correct location. 

Note that this is different from 2-D prediction based on attention mechanism. In Section 4, experiments show that improving recognition accuracy by capturing context module is not good for generating content readable text, although it is very important for recognition task. 

我们使用SPM在2-D特征图预测每个位置的部件类别，引导网络在正确的位置生成对应的部件。注意这不同于基于注意力机制的2-D预测，第四节实验证明依靠捕获上下文模块来提高识别准确性对于生成内容可读的文本并不好，尽管这对于识别任务非常重要。





````
SPM不再将部件作为序列预测，通过加入空间注意力模块，注意到“物体”出现的位置，包含一个带空间域注意力机制的全卷积特征提取网络，最终输出是二维特征图，我们使用基于统计的ACEloss在二维空间预测部件分布并计数。实现细化手写文本中部件的目的。
SPM no longer takes components as sequence prediction, and notes the position of "object" by adding spatial attention mechanism,


这有助于引导网络学到不同部件之间的空间依赖关系，根据相邻的部件和结构嵌入创建出该部件位于不同空间位置的不同变体。同时这种方式变相的扩充了样本量少的复合字，比如‘好’这个字，我们识别器

Then, the decoder in RAN generates a  hierarchical composition of Chinese characters based on the knowledge of the extracted radicals and their internal structures.The method of  treating a Chinese character as a composition of radicals rather than as a single character category is a human-like method that can reduce the  size of the vocabulary, ignore redundant information among similar  characters and enable the system to recognize unseen Chinese character  categories, i.e., zero-shot learning. 

然后，RAN中的解码器根据提取的部首及其内部结构的知识生成汉字的分层组合。将汉字视为部首组合而不是单个汉字类别的方法是一种类似人类的方法，它可以减少词汇的大小，忽略相似汉字之间的冗余信息，并使系统能够识别看不见的汉字类别，即零炮学习。



```
the network learns an implicit language model which helps it identify the correct character even if it is not written clearly, by leveraging priors learned from other characters in the text. While this quality is usually desired in a handwriting recognition model, in our case it may lead the network to correctly read characters which were not written clearly by the generator. Therefore, we opted not to use the recurrent ‘head’ of the recognition network
网络学习一种隐式语言模型，通过利用从文本中其他字符学习到的先验知识，该模型有助于识别正确的字符，即使它写得不清楚。虽然手写识别模型通常需要这种质量，但在我们的例子中，它可能会导致网络正确读取生成器未清晰书写的字符。因此，我们选择不使用识别网络的循环“头
```

After the above analysis, we find that for the content recognizer in the generation task, relying on the capture context module to improve the accuracy is not good for generating content readable text, although it is very important for the recognition task.

经过以上分析，我们发现对于生成任务中的内容识别器而言，依靠捕获上下文模块提高准确率对于生成内容可读的文本是没有好处的，尽管这对于识别任务十分重要。



```
然后，按照[10]的思路，要求生成器执行辅助任务。为此，我们引入了一种用于文本识别的辅助网络（R）。然后，我们训练G生成R能够正确识别的图像，从而在与R的“协作”约束下完成其原始对抗目标。我们使用敌对损失[16]和CTC损失[1]的铰链版本来训练该系统。在形式上，D、R、G和φ的培训旨在最小化以下目标：
Then, in the vein of [10], the generator is asked to carry out a secondary task. To this end, we introduce an auxiliary network for text recognition (R). We then train G to produce images that R is able to recognize correctly, thereby completing its original adversarial objective with a “collaboration” constraint with R. We use the hinge version of the adversarial loss [16] and the CTC loss [1] to train this system. Formally, D, R, G and φ are trained to minimize the following objectives:

辅助网络R是一个选通卷积网络，在[2]中介绍（我们使用了“大体系结构”）。该网络由五个卷积层的编码器组成，具有Tanh激活和卷积门，然后是垂直维度上的最大池和由两个堆叠的双向LSTM层组成的解码器
The auxiliary network R is a Gated Convolutional Network, introduced in [2] (we used the “big architecture”). This network consists in an encoder of five convolutional layers, with Tanh activations and convolutional gates, followed by a max pooling on the vertical dimension and a decoder made up of two stacked bidirectional LSTM layers


````









**Losses and optimization settings：**

```
We optimize our model with the Adam algorithm **[A method for stochastic optimization]** (for G,D networks: lr = 2 × 10−4, β1 = 0.5, β2 = 0.999, for R,C networks: lr = 2 × 10−3, β1 = 0.5, β2 = 0.999).

我们使用Adam算法**[随机优化方法]**（对于G，D网络：lr=2×10）优化了我们的模型−4，β1=0.5，β2=0.999，对于R，C网络：lr=2×10−3, β1 = 0.5, β2 = 0.999).


```

We implement the hinge version of the adversarial loss from Geometric GAN**[Geometric GAN]**
$$
\begin{aligned}

L_{G}=&-\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[D(G(\boldsymbol{z}, \boldsymbol{f(e)}))] 
\\
L_{D}=&+\mathbb{E}_{(\boldsymbol{x}) \sim p_{\text {data }}}[\max (0,1-D(\boldsymbol{x}))] \\
&+\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[\max (0,1+D(G(\boldsymbol{z}, \boldsymbol{f(e)})))] \\
\end{aligned}
$$
to promote our generated images look realistic. To encourage the correct character order in the generated handwritten Chinese text image, we use the CTC loss.
$$
\begin{aligned}
L_{SRM_g} = 
&+\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[\operatorname{CTC}(\boldsymbol{e}, SRM(G(\boldsymbol{z}, \boldsymbol{e})))]\\
L_{SRM_d}=&+\mathbb{E}_{\boldsymbol{x} \sim p_{data}, \boldsymbol{e} \sim p_{text}}[\operatorname{CTC}(\boldsymbol{e}, SRM(\boldsymbol{x}))] \\
\end{aligned}
$$
The generated images should retain the correct structure and fine strokes, so we use the ACE loss, 
$$
\begin{aligned}
L_{SPM_g} = 
&+\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[\operatorname{ACE}(\boldsymbol{e}, SPM(G(\boldsymbol{z}, \boldsymbol{e})))]\\
L_{SPM_d}=&+\mathbb{E}_{\boldsymbol{x} \sim p_{data}, \boldsymbol{e} \sim p_{text}}[\operatorname{ACE}(\boldsymbol{e}, SPM(\boldsymbol{x}))] \\

\end{aligned}
$$
Here,  $p_{data}$ denotes the distribution of real Handwritten Chinese text image, $p_z$ is a prior distribution on input noise $z$ and $p_{text}$  refers to a prior distribution of text.

使用pdata表示实[图像，文本]对的联合分布，pz表示输入噪声的先验分布，ptext表示文本的先验分布。



```
stemming from
```



Since the gradients arising from each of the above loss terms can vary greatly in magnitude, we adopt the loss terms balance rule proposed by Alonso et al to updata $L_{R_g}$ and $L_{C_g}$ **[Adversarial Generation of Handwritten Text Images Conditioned on Sequences]**.

由于上述每个损失项产生的梯度在量级上可能有很大差异，因此我们采用了Alonso等人提出的损失项平衡规则更新 $L_{R_g}$ 和 $L_{C_g}$ 。




$$
\boldsymbol{\nabla}_{\boldsymbol{L_G}}= \frac{\partial L_G\left(\boldsymbol G(\boldsymbol{z}, \boldsymbol{e}) ) \right)}{\partial \boldsymbol  G(\boldsymbol{z}, \boldsymbol{e}) } 
\\
\boldsymbol{\nabla}_{\boldsymbol L_{R_g}}=\frac{\partial \mathrm{L_{R_g}}\left(\boldsymbol{e}, R\left(\boldsymbol G(\boldsymbol{z}, \boldsymbol{e})\right)\right)}{\partial \boldsymbol G(\boldsymbol{z}, \boldsymbol{e}) }

\\

\boldsymbol{\nabla}_{\boldsymbol L_{C_g}}=\frac{\partial \mathrm L_{C_g}\left(\boldsymbol{e}, R\left(\boldsymbol G(\boldsymbol{z}, \boldsymbol{e}) \right)\right)}{\partial \boldsymbol G(\boldsymbol{z}, \boldsymbol{e}) }
$$
$\boldsymbol{\nabla}_{\boldsymbol{L_G}}$ ,$\boldsymbol{\nabla}_{\boldsymbol L_{R_g}}$ and $\boldsymbol{\nabla}_{\boldsymbol L_{C_g}}$ are respectively the gradients of $L_{G}$, $L_{R_g}$ and $L_{C_g}$ with respect to the fake image $G(\boldsymbol{z}, \boldsymbol{e}) $. 



In our actual training process, not as  **[Alonso et al]** said, $\boldsymbol{\nabla}_{\boldsymbol{L_{R_g}}}$  is always several orders of magnitude larger than $\boldsymbol{\nabla}_{\boldsymbol L_{D}}$. We take one more step to promote the convergence: $ min(\boldsymbol{\nabla}_{\boldsymbol{L_G}},\boldsymbol{\nabla}_{\boldsymbol L_{R_g}},\boldsymbol{\nabla}_{\boldsymbol L_{C_g}})$ . Suppose that  $\boldsymbol{\nabla}_{\boldsymbol{L_G}}$ is the smallest, we update to obtain blance coefficient $\alpha$ and $\lambda$ , where the parameter $\beta$ and $\gamma$ respectively control the relative importance of $\boldsymbol L_{R_g}$ and $\boldsymbol L_{C_g}$  in the updating network.

在我们训练过程中，并非如 Alonso et al所说，我们采用多一步操作：$ min(\boldsymbol{\nabla}_{\boldsymbol{L_G}},\boldsymbol{\nabla}_{\boldsymbol L_{R_g}},\boldsymbol{\nabla}_{\boldsymbol L_{C_g}})$ 

假设，$\boldsymbol{\nabla}_{\boldsymbol{L_G}}$ 是最小的，更新得到平衡系数，其中$\beta$ and $\gamma$ 分别控制R、C在更新网络占的比重。


$$
\alpha \leftarrow 
\beta

\frac{\sigma\left(\nabla L_G\right)}{\sigma\left(\nabla L_{R_g}\right)\nabla L_{R_g} } \cdot\left[\nabla  L_{R_g}-\mu\left(\nabla L_{R_g}\right)\right]+\mu\left(\nabla L_{G}\right)
$$

$$
\lambda  \leftarrow 
\gamma
\frac{\sigma\left(\nabla L_G\right)}{\sigma\left(\nabla L_{C_g}\right)\nabla L_{C_g}} \cdot\left[\nabla  L_{C_g}-\mu\left(\nabla L_{C_g}\right)\right]+\mu\left(\nabla L_{G}\right)
$$

here σ (·) and μ(·) are respectively the empirical standard deviation and mean.



In G training step, we freeze the weight of D, R and C, let it processes the generated image, and update the weight of G. The total loss is

在G训练步骤中，我们冻结D、R和C的权重，让它处理生成的图像，并更新G的权重。总损失为

$l_g = L_G+\alpha L_{R_g}+\lambda L_{C_g}$

In D training step,  we freeze the weight of G, let D processes the generated images and the real images. For R and C, we only use real images to avoid learning the generated content representation, then update the weight of D, R and C. The total loss is

在D训练步骤中，我们冻结G的权重，让D处理生成的图像和真实图像。对于R和C，我们仅使用真实图像来避免学习生成的内容表示，然后更新D、R和C的权重。总损失为

$l_d = L_D+ L_{R_d}+ L_{C_d}$

The purpose of using the balance coefficient only in the G training step is to prevent G from only paying attention to the handwriting style, global order or local details due to one of loss terms is too large.

仅在G训练步使用平衡系数是为了防止G因其中某一个loss过大，只关注手写风格或整体顺序或局部细节。













```
We used spectral normalization [22] in G and D, following recent works [11], [18], [22] that found that it stabilizes the training. We optimized our system with the Adam algorithm [23] (for all networks: lr = 2 × 10−4, β1 = 0, β2 = 0.999) and we used gradient clipping in D and R. We trained our model for several hundred thousand iterations with mini-batches of 64 images of the same type, either real or generated. While D processes one real batch and one generated batch per training step, R is trained with real data only, to prevent it from learning how to recognize generated (and potentially false) images of text. To train the networks G and φ, we first produce a batch of “fake” images xfake := G(z, φ(s)), and then pass it through D and R. (

我们在G和D中使用了谱归一化[22]，最近的工作[11]、[18]、[22]发现它稳定了训练。我们使用Adam算法[23]优化了我们的系统（对于所有网络：lr=2×10）−4，β1=0，β2=0.999），我们在D和R中使用梯度剪裁。我们用64张相同类型的图像（真实或生成的）小批量训练我们的模型进行数十万次迭代。当D在每个训练步骤中处理一个真实批次和一个生成批次时，R只使用真实数据进行训练，以防止它学习如何识别生成的（可能是错误的）文本图像。为了训练网络G和φ，我们首先生成一批“伪”图像xfake:=G（z，φ（s）），然后通过D和R(

The generator network is optimized by the recognizer loss ℓR and the adversarial loss ℓD. The gradients stemming from each of these loss terms can vary greatly in magnitude. Alonso et al. [2] proposed the following rule to balance the two loss terms
发电机网络通过识别器损耗进行优化ℓR与对抗性损失ℓD.这些损失项中的每一项产生的梯度在量级上可能会有很大的变化。阿隆索等人[2]提出了以下规则来平衡这两个损失项
```









```
然后，按照[10]的思路，要求生成器执行辅助任务。为此，我们引入了一种用于文本识别的辅助网络（R）。然后，我们训练G生成R能够正确识别的图像，从而在与R的“协作”约束下完成其原始对抗目标
Then, in the vein of [10], the generator is asked to carry out a secondary task. To this end, we introduce an auxiliary network for text recognition (R). We then train G to produce images that R is able to recognize correctly, thereby completing its original adversarial objective with a “collaboration” constraint with R. 

如图所示，我们使用ACE loss训练的计数模型能够“注意”关键物体出现的位置。与文本识别问题不同，在文本识别问题中，使用ACE损失函数训练的识别模型倾向于对字符进行预测，而使用ACE损失函数训练的计数模型在对象体上提供了更均匀的预测分布。此外，它将不同程度的“注意力”分配给对象的不同部分。例如，当观察图片中的红色时，我们注意到计数模型更关注一个人的脸。这种现象符合我们的直觉，因为脸是个体最独特的部分。
As shown in the images, our counting model trained with ACE loss manage to pay “attention” to the position where crucial objects occur. Unlike the text recognition problem, where the recognition model trained with the ACE loss function tends to make a prediction for a character, the counting model trained with the ACE loss function provides a more uniform prediction distribution over the body of the object. Moreover, it assigns different levels of “attention” to different parts of an object. For example, when observing the red color in the pictures, we notice that the counting model pays more attention to the face of a person. This phenomenon corresponds to our intuition because the face is the most distinctive part of an individual.


We use a pre-trained text recognition network, R, to evaluate the content of the generated images. We use the output string estimated by R to compute a loss, `R, which reflects how well the generator captured the desired content string, c1, c2 simultaneously on both target images (Os,c1 , Os,c2 ) and their masks (Ms,c1 , Ms,c2 ). Ideally a recognizer should only focus on the text content (foreground element) of the image irrespective of the its background elements. Therefore, constraining it to recognize the same content string on both the target generation and its mask allows us to align it well. In practice, we use an existing word pre-trained recognition model by Baek et al. [40]. Of the ones proposed in that paper, we chose the model with the following configuration, though we did not optimize for this choice: (1) spatial transformation network (STN) using thin-plate spline (TPS) transformation, (2) feature extraction using ResNet network, (3) sequence modelling using BiLSTM, and (4) an attention-based sequence prediction. This OCR was favored above other methods which may be more accurate [41], due to its simplicity and ease of integration as part of our approach.
我们使用预先训练的文本识别网络R来评估生成图像的内容。我们使用由R估计的输出字符串来计算损失，`R，它反映了生成器在两个目标图像（Os，c1，Os，c2）及其掩码（Ms，c1，Ms，c2）上同时捕获所需内容字符串c1，c2的情况。理想情况下，识别器应该只关注图像的文本内容（前景元素），而不考虑其背景元素。因此，限制它识别目标生成和它的掩码上的相同内容字符串允许我们很好地对齐它。在实践中，我们使用Baek等人[40]的现有单词预训练识别模型。在本文提出的模型中，我们选择了具有以下配置的模型，尽管我们没有为此选择进行优化：（1）使用薄板样条（TPS）变换的空间变换网络（STN），（2）使用ResNet网络进行特征提取，（3）使用BiLSTM进行序列建模，以及（4）基于注意的序列预测。由于OCR作为我们方法的一部分，其简单性和易集成性优于其他可能更准确的方法[41]。

```



I：I 能够在二维空间中对component每个类别的分布预测，这不仅约束了单个字形的结构，还细化了单个字形的部件。



ACE：(2) it requires only characters and their numbers in the sequence annotation for supervision, which allows it to advance beyond sequence recognition, e.g., counting problem. 

它只需要序列注释中的字符及其数字进行监督，这允许它超越序列识别，例如计数问题





CTC是典型的一维序列预测loss，对于不规则文本、公式等二维空间的序列，预测效果不好。但是可以促进正确的字符顺序。



ACE loss是为了这种方式的设计缺陷提出的，全称是Aggregation Cross-Entropy聚合交叉熵。文章中描述ACE能够解决2-D文本的识别问题，还在时间复杂度和空间复杂度上优于CTC loss，但是丢失了序列的信息，是一种基于分布的计数loss

结合而二者，CTC会将准确的汉字component 信息固定在序列的准确位置，ACE负责对固定好位置的component添加更多的细节信息。



ACEloss仅仅是在二维空间中估计没有序列的信息，对单个字符生成结构准确、部件清晰的汉字

CTCloss，对全局字符的序列信息监督，保证生成的字符序列准确，对内容生成比较粗略，对字符内部件组件的空间结构有些混乱

ACEloss是基于空间，比如 山冈，没有ACEloss时生成的是错误的汉字，可以起到对字符内组件的空间结构监督，细化组件的细节。恢复轮廓细节，但是对全局字符序列监督能力不够。



在一些二维预测问题中，如带有图像级注释的不规则场景文本识别，定义字符之间的空间关系是一个挑战。字符可以以多行、弯曲或倾斜的方向排列，甚至可以随机分布。幸运的是，所提出的ACE损失函数可以自然地推广到2D预测问题，因为它不需要序列学习过程中的字符顺序信息。假设输出2D预测y具有高度H和宽度W，并且第H行和第W行处的预测表示为yhw k。这需要对yk和N k的计算进行边际调整，如下所示，yk=yk HW=∑H=1∑W W=1 yhw k HW，N k=Nk HW。然后，2D预测的损失函数可以如下转换：

基于attention的方法不好，长文本效果不好，对齐问题、偏移问题

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimage-20211027213405435.png" alt="image-20211027213405435" style="zoom:50%;" />

$$\mathcal{L}_I = \sum_{k=1}^{c} $$​
$$
\mathcal{L}(\mathcal{I}, \mathcal{S})=-\sum_{k=1}^{\left|\mathcal{C}^{\epsilon}\right|} \overline{\mathcal{N}}_{k} \ln \bar{y}_{k}=-\sum_{k=1}^{\left|\mathcal{C}^{\epsilon}\right|} \frac{\mathcal{N}_{k}}{\mathcal{H} \mathcal{W}} \ln \frac{y_{k}}{\mathcal{H} \mathcal{W}}
$$


Suppose that the output 2D prediction y has height H and width W, and the prediction at the h-th line and w-th row is denoted as yhw k . This requires a marginal adaptation of the calculation of yk and N k as follows, yk = yk HW = ∑H h=1 ∑W w=1 yhw k HW , N k = Nk HW . Then, the loss function for the 2D prediction can be transformed as follows:

假设输出2D预测y具有高度H和宽度W，并且第H行和第W行处的预测表示为yhw k。这需要对yk和N k的计算进行边际调整，如下所示，yk=yk HW=∑H=1∑W W=1 yhw k HW，N k=Nk HW。然后，2D预测的损失函数可以如下转换：













在不丧失通用性的情况下，该体系结构设计用于生成和处理固定高度为32像素的图像，此外，G的感受野宽度设置为16像素。如第3.1节所述，生成器网络G具有与字母表一样大的滤波器组F，例如，对于小写英语，F={fa，fb，…，fz}。每个过滤器的尺寸为32×8192。为了生成一个n字符的单词，我们选择并连接这些滤波器中的n个（包括重复，如图2中的字母“e”），将它们与32维噪声向量z1相乘，得到一个n×8192矩阵。接下来，将后一个矩阵重塑为512×4×4n张量，即此时，每个字符的空间大小为4×4。后一个张量被送入三个剩余块中，这三个剩余块提高了空间分辨率，产生了上述感受野重叠，最终图像大小为32×16n。条件实例归一化层[9]用于使用三个额外的32维噪声向量z2、z3和z4调制剩余块。最后，使用具有tanh激活的卷积层来输出最终图像。

鉴别器网络D的灵感来源于SAGAN[6]：4个剩余块，后跟一个线性层和一个输出。为了处理不同宽度的图像生成，D也是完全卷积的，基本上是处理水平重叠的图像块。最后的预测是贴片预测的平均值，它被输入到GAN铰链损耗中[28]。识别网络R的灵感来自CRNN[35]。网络的卷积部分包含六个卷积层和五个池层，所有这些层都具有ReLU激活。最后，使用线性层输出每个窗口的类别分数，并与使用连接主义时间分类（CTC）损失的地面真相注释进行比较[13]。我们的实验是在一台装有v100gpu和16GB内存的机器上进行的。有关架构的更多详细信息，请参阅补充资料。











将在CICAS数据集中汉字类别从7356降到了1000个component，

* 1、生成类别降低
* 2、以component类别进行监督，相当于扩充了复合字体的样本
* 引导生成器学到如何使用component组合一个汉字，并学到每个位置同一部件的大小和风格信息的差异。

The content representation e is computed prior to the average pooling layer so that one can preserve the spatial properties. Here, 512 is the number of channels in last convolutional feature block.

在平均池层之前计算内容表示e，以便可以保留空间属性。这里，512是最后一个卷积特征块中的信道数。





The style of each image is controlled by a noise vector z given as input to the network. In order to generate the same style for the entire word or sentence, this noise vector is kept constant throughout the generation of all the characters in the input.

每个图像的样式由作为网络输入的噪声向量z控制。为了为整个单词或句子生成相同的样式，该噪声向量在输入中所有字符的生成过程中保持不变。



Note that, the encoder are fully convolutional and can handle variable sized inputs and outputs.







每个汉字看做

汉字存在内容重用现象；也就是说，相同的部首可能在不同的位置出现在不同的字符中；参见图2）。我

 在线变焦增强的功能是自适应地探索这种分解

它有两个主要优点：（i）它利用汉字中部首的重复性，引导网络学习汉字的一些基本结构，从而大大减少了训练规模；和（ii）它引导网络在不同的位置和大小学习字符，使网络更加健壮。据我们所知，除SA-VAE外，没有基于CNN的方法利用此领域知识[19]。然而，SA-VAE明确地将其建模为嵌入模型中的预标记133位代码。相反，我们仔细选择750个训练样本作为基本字符b

Chinese characters have the content-reuse phenomenon; that is, the same radicals may present in various characters at various locations; see Fig. 2). I



<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimage-20211101164639804.png" alt="image-20211101164639804" style="zoom:50%;" />

which removes the need for pre-segmented data and external post-processing.

这样就不需要预分割数据和外部后处理。



, i.e., ，即。，





















4.Experiments 



Differently, CalliGAN **[CalliGAN: Style and Structure-aware Chinese Calligraphy Character Generator]** **[Handwritten Chinese Font Generation with Collaborative Stroke Refinement]**and SA-VAE **[Learning to Write Stylized Chinese Characters by Reading a Handful of Examples] [RD-GAN: Few/Zero-Shot Chinese Character Style Transfer via Radical Decomposition and Rendering] and [Multiple Heads are Better than One: Few-shot Font Generation with Multiple Localized Experts]]** are proposed to disentangle the radical of Chinese font as Auxiliary a priori information.

The network presented in **[CalliGAN]** uses LSTM to embed the components sequence of a Glyph into a fixed length representation which can be fed into a generator architecture, and we directly learn the embeddings for each component and structure of Chinese character string. As opposed to our approach, which allows for Handwritten text image generation, this generator is only able to output images of a single Chinese Glyph.

不同的是，**[]** 被提出，去分解汉字字根作为辅助先验信息。CalliGAN中介绍的网络使用LSTM将输入单字的部件序列嵌入到固定长度表示中，而我们直接学习汉字字符串的每个部件和结构的嵌入。与我们允许手写文本图像的方法相反，该生成器只能输出单个汉字字形的图像。



**[Adversarial Generation of Handwritten Text Images Conditioned on Sequences],[ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation]** is similar to our task. However, the generation model of alphabetic text can not be effectively applied to Chinese text for the huge vocabulary and complex spatial structure of Chinese characters.

**[Adversarial Generation of Handwritten Text Images Conditioned on Sequences],[ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation]**与我们的任务相似。然而，汉字庞大的词汇量和复杂的空间结构使得字母文本生成模型不能有效地应用于中文文本。

```
We learn from the full convolution idea in scrabbleGan, but it is different. In order to meet the needs of Chinese text tasks, we do not learn the embeddings for each character directly , but the embeddings of components and structures.

我们借鉴scrabbleGAN中的全卷积思想但有所不同，为了适应中文文本任务需求，我们不是学习单字符的嵌入，而是部件和结构的嵌入。
```



**[RD-GAN: Few/Zero-Shot Chinese Character Style Transfer via Radical Decomposition and Rendering，Multiple Heads are Better than One: Few-shot Font Generation with Multiple Localized Experts]** are involved in component supervision. The former uses LSTM network with attention mechanism to identify component positions, and segments the original image according to the attention map. This structure is not good for complex spatial structure Chinese characters and long text. Although the latter work does not segment the image, it is unnecessary to use multiple feature extraction networks to extract different component information respectively.

涉及对部件监督的工作有 **[RD-GAN: Few/Zero-Shot Chinese Character Style Transfer via Radical Decomposition and Rendering，Multiple Heads are Better than One: Few-shot Font Generation with Multiple Localized Experts]** ，前者采带有注意力机制的LSTM网络识别部件位置，依据注意力图分割原始图像。 这种结构对于复杂空间结构汉字和长文本效果不好。后者的工作虽然没有分割图像，但是使用多个特征提取网络分别提取不同的component信息，这是不必要的。



It is conceivable that the recognition based on RNN does not perform well in the generation task (the network learns an implicit language model which helps it identify the correct character even if it is not written clearly **[implicit language model in LSTM for OCR]**), we use the combination of 1-D sequence and 2-D recognition to promote global character order and local component details respectively. 

可以想到基于rnn的识别在生成任务中表现不好(网络学习一种隐式语言模型，这有助于识别正确的字符，即使它生成的得不清楚**[Implicit language model in lstm for ocr]**)，我们采用一维序列和二维特征图识别相结合的方式分别促进全局有序和局部部件细节。



数据集

* 对比实验：

  * 定量：FID、IS  

  * 定性：

    * 例子：整体好坏

    * 对于无样本，少样本的汉字对比

    * 基于rnn的2-D 注意力机制对于生成不好（

      **[Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer, High performance offline handwritten Chinese character recognition using GoogLeNet and directional feature maps]**

      ）我们考虑采用2D预测的方式，预测“物体”在哪里，是什么。

    * the dependencies between adjacent characters and text-level feature





rewrite、zi2zi、Generating Handwritten Chinese Characters using CycleGAN、scrableGAN、our：取得竞争性结果

插值实验：

消融实验：

one-hot

one-hot+部首ctc

one-hot+ 部首ace

部首编码+部首ctc

部首编码+部首ace

部首编码+部首ctc+部首ace

部首编码+部首ctc+部首ace+attention



拓展试验：

1、

真实的手写文本，同一汉字在不同位置是不同的，单字体生成模型对于打印标准字体效果很好，但是对于手写字体仍未解决这一问题。

. Furthermore, learning the dependencies between adjacent characters allows the network to create different variations of the same character, depending on its neighboring characters. Such examples can be seen in Figure 1 and Figure 3.

. 此外，学习相邻字符之间的依赖关系允许网络根据其相邻字符创建同一字符的不同变体。这些示例如图1和图3所示。

由于没有任何一个现行的研究可以直接生成手写中文文本，为了比较，我们改造scrabbleGAN后，使其生成手写中文文本与我们的框架作比较。

2、

目前手写文本汉字识别中，汉字和文本序列的识别的很多工作都集中于引入能建模上下文序列信息的语言模型以提高识别准确率，而无意义的字形的引入提供了一个不同的思路。





受目标检测领域负样本采样思路的启发，我们为形近字生成了大量的负样本，这些负样本不存在于字库中，是无意义的部件组合，仅看起来像一个汉字。实验证明负样本的引入使得模型对于形近字在特征空间的分类边界更加清晰，提升了模型的识别准确率。

Generating Handwritten Chinese text without character images input

```
受数学表达式识别任务启发，试图采用带有2D注意力机制的循环网络促进空间结构正确的手写中文文本生成时，发现无法拟合。

我们分析主要是由于生成任务中，基于循环网络的结构捕获了其中的上下文关系，从而使网络学习一种隐式语言模型，通过利用从文本中其他字符学习到的先验知识，该模型有助于识别正确的字符，即使它写得不清楚。虽然手写识别模型通常需要这种质量，但在我们的例子中，它可能会导致网络正确读取生成器未清晰书写的字符。同时当文本过长时基于循环网络的识别引起的问题如注意力漂移、训练参数量大，模型难以拟合等问题难以解决。因此，我们选择不使用基于循环网络的2D注意力机制。
```

3、dataset augmentation：

* origin
* 普通增广
* 合成汉字加入
* 合成无意义字形加入
* 二者都加入
* 普通增广+二者都加入





不足：对于可拆分的几千个结构清晰的常用字效果十分好，但是对于字形耦合紧密的复杂汉字，如图X，我们的编码方式难以将其简化。

未来工作，书法草书文本的生成，艺术字的生成