3.Our method





Any appropriate combination of component and structure will correspond to a glyph. More than 100000 Chinese glyphs are a subset of the combination. Since the real handwritten Chinese text image can not be obtained by stacking a single character image, our proposed model aims to learn a way to generate real handwritten Chinese text, which can contain non-sense glyph.

组件和结构的任何适当组合都将对应于图示符。超过100000个汉字字形是这个组合的一个子集。由于真实的手写体中文文本不能通过叠加单个字符来获得，我们提出的模型旨在学习一种生成真实手写体中文文本的方法，该文本可以包含无意义的字形。

**architecture.** Fig. 2 illustrates the architecture proposed str2text model. 

图2示出了str2text系统提出的体系结构。

```
Here, we introduce proposed str2text framework.
```

 

The framework mainly consists of five modules of a Chinese string encoder(CSE), a generator(G), a discriminator(D), a sequence recognition module(SRM), and a spatial perception module(SPM). CSE is mainly responsible for encoding the components and structures of Chinese string into content representation e. The G is constructed based on a full convolution network(FCN)**[]** and can generate variable-sized output. The D promotes realistic-looking handwriting styles. SRM is a 1-D sequence recognizer with connectionist temporal classification (CTC) loss**[]** to ensure that the generated handwritten Chinese text is in the correct character order. SPM is a 2-D prediction model that guides the network to refine each type of component in  appropriate position. Due to the high cost of components spatial relationships annotation, SPM adopts an inexact supervised approach.

这里，我们将介绍我们模型的细节。该模型主要由中文字符串编码器、生成器、鉴别器、序列识别模块(SRM)和空间感知模块(SPM)五个模块组成。其中汉字字符编码器主要负责将汉字字符序列的部件和结构信息编码到内容表示e，生成器基于全卷积网络构建，具备不定长文本的生成能力。鉴别器促进逼真的笔迹风格。SRM是一种具有CTC损失的一维序列识别器[]，以确保生成的手写中文文本符合准确的字符顺序。SPM是一种二维预测模型，用于在空间位置上细化各种类型的部件。SRM确保生成的手写汉字文本顺序准确，内容基本一致。SPM是一个二维预测模型，指导生成器在空间位置上细化每种类型的组件。由于构件空间关系标注成本较高，SPM采用了不精确监督方法。

**encoder：**  

Our input is a Chinese string, e.g., Fig. 2 (“呗员...”). Each Chinese character can be decomposed into a string of components and a unique structure. Meanwhile, the same component may present in various glyphs at various locations. Based on the appealing characteristic, we design an encoding method. Figure 3 shows a few examples of how to encode the information.

我们的输入是一个中文字符串，如图2（“呗员…”）。每个汉字可以分解成一系列的组成部分和一个独特的结构。同时，同一组件可能出现在不同位置的不同图示符中，基于上诉特征，我们设计了一种编码方式，图3显示了一些如何编码信息的示例。

To generate variable-sized output, the encoder needs to send a variable size content representation conforming to the full convolution network into the generator. For this, The CSE directly learns the embedding of each component and structure, then is reshaped to content representation $e\in\mathbb{R}^{512\times 4\times W}$ as shown in Figure 4, instead of mapping the embedding to a fixed-length vector in **[CalliGAN]**, which helps to stimulate the writing process by convolution, that is, the components of Chinese characters are only connected with their local spatial components, and avoid using a recurrent network to learn the embedding of the whole text. Here, 512 is the number of channels, and W corresponds to the width of the generated handwritten Chinese text.

为了生成可变大小的输出，编码器需要向生成器发送符合全卷积网络的可变大小内容表示。为此编码器直接学习每个组件和结构的嵌入，如图5，整合后得到内容表示 $e \in\mathbb{R}^{512\times 4\times W}$，而不是将嵌入映射到***[CalliGAN]***中的固定长度向量，这有助于通过卷积模拟书写过程，即汉字的组件仅与其局部空间组件发生联系，避免使用循环网络学习整个句子文本的嵌入。这里512是通道数，W对应生成手写汉字文本的宽度。





Generator(G) and Discriminator(D)

**生成器和判别器：**



G and D are inspired by SAGAN, and some common techniques in the training, such as self-attention mechanismto refine local area image quality, spectral normalization and hinge loss function to stabilizes the training, was used. FCN is a common technique to deal with variable size input and output problems. In this paper, we inherit this paradigm in SAGAN-based generators. At the same time, FCN allows the network to learn the dependencies between adjacent components, which solves the problem of the differentiated output of the same characters in a single handwriting style. We present the detail of the structure in Table 1.

As shown in Fig. 2, CSE module outputs the content representation $e$. Later, e is concatenated with noise vector $z_1$ in the channel layer and fed into G, and the additional noise vectors $z_2, z_3$ and $ z_4$ are fed into each layer of G through Conditional Batch Normalization (CBN) ayers. This allow G to control the low-resolution and high-resolution details of the appearance of Chinese text to match the required handwritten style. Compared with the common Chinese font generation network, the optimization objective of G is to generate appropriate components in the appropriate position. To cope with varying size image generation, D is also fully convolutional structure to promote the generator output realistic handwritten text. 

G和D受到SAGAN{zhang2019selfattention}的启发，并在训练中使用了一些常用技术，如自我注意机制{vaswani2017attention}来细化局部图像质量，光谱归一化{Miyato2018spectrum}和铰链损失函数{lim2017geometric}来稳定训练。FCN是处理可变大小输入和输出问题的常用技术。在本文中，我们在基于SAGAN的生成器中继承了这一范式。同时，FCN允许网络学习相邻组件之间的依赖关系，从而解决了以单一手写样式区分相同字符输出的问题。我们在表1中给出了结构的详细信息。

”如图2所示，CSE模块输出内容表示$e$。随后，e与信道层中的噪声向量$z_1$串联并馈入G，并且附加噪声向量$z_2、z_3$和$z_4$通过条件批量归一化（CBN）\引用{shell2015bare}层馈入G的每一层。这允许G控制中文文本外观的低分辨率和高分辨率细节，以匹配所需的手写样式。与普通中文字体生成网络相比，G的优化目标是在适当的位置生成适当的组件。为了应对不同大小的图像生成，D也是完全卷积结构，以促进生成器输出逼真的手写文本。



```

D and G are inspired by SAGAN **[Self Attention Gan(SAGAN)]**, and some common techniques in the training Gan are used, such as self-attention mechanism**[[Attention is All You Need]**, to refine local area image quality, spectral normalization**[Spectral normalization for generative adversarial networks]** and hinge loss function to stabilizes the training. We present the detail of the structure in Table 1.

鉴别器网络D和生成器网络G受SAGAN[ ]的启发，如表X所示，并使用训练GAN中的一些常用技术，如注意机制改善局部图像质量、hinge loss和光谱归一化以稳定训练。

As shown in Fig. 2, CSE module outputs the content representation $e$. Later, the $e$ is concatenated with noise vector $z_1$ in the channel layer and fed into G, and the additional noise vectors $z_2, z_3$ and $z_4$ are fed into each layer of G through Conditional Batch Normalization (CBN)**[Modulating early visual processing by language]** layers. Thus we allow the generator to control the low-resolution and high-resolution details of the appearance of Chinese text to match the required handwritten style.

Full convolution networks(FCN) is a common technique to deal with variable size input and output problems **[TextStyleBrush: Transfer of Text Aesthetics from a Single Example, ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation]**. In this paper, we inherit this paradigm in SAGAN-based generators. At the same time, FCN allows the network to learn the dependencies between adjacent components, which solves the problem of the differentiated output of the same characters in a single handwriting style.

如图2所示，汉字字符串编码器模块输出内容表示 e，随后，在通道层中的噪声z1 cat的内容表示后馈入G，同时将附加噪声矢量z2、z3和z4通过条件批量归一化（CBN）馈送至生成器的每一层。因此，我们允许生成器控制中文文本外观的低分辨率和高分辨率细节，以匹配所需的手写样式。

全卷积网络是处理可变大小输入和输出问题的通用技术**[TextStyleBrush：从单个示例中传递文本美学，ScrabbleGAN：半监督变长手写文本生成]**。在本文中，我们在基于SAGAN的生成器中继承了这一范式。同时，全卷积的结构允许网络学习相邻部件之间的依赖关系，这解决了在单一手写样式中相同字符的差异输出的问题

To cope with varying width image generation, D is also fully convolutional to promote G output realistic handwritten text.

为了处理不同宽度的图像生成，D也是完全卷积的，以促进G输出真实的手写文本。

Compared with the common Chinese font generation network, the optimization objective of G is to generate appropriate components in the appropriate position.

```

**spatial perception module(SPM) and sequence recognition modul(SRM)**



In addition to our proposed coding method, The synergy between sequence recognition module(SRM) and spatial perception module(SPM) is our main improvement.
The generation of alphabetic text usually adopts 1-D sequence prediction architecture to evaluate the content of the generated images.
However, due to the huge number of Chinese characters (more than 100,000 Chinese characters compared with dozens of letters in English), it is very difficult to accurately generate 100,000 types of glyphs. The reuse mechanism of Chinese components also brings the problems of near glyph interference and information redundancy. Therefore, we treat a Chinese text as a string of components with complex spatial relationships rather than as a single character category.

除了我们提出的编码方法，SPM和SRM之间的协同作用是我们的主要改进。字母文本的生成通常采用序列预测结构来评估生成图像的内容 **[Adversarial Generation of Handwritten Text Images Conditioned on Sequences，ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation，TextStyleBrush: Transfer of Text Aesthetics from a Single Example，]**。

然而，由于汉字数量巨大（超过100000个汉字，而英语中只有几十个字母），因此很难准确生成100000种字形。中文构件的重用机制也带来了近字形干扰和信息冗余的问题。因此，我们将中文文本视为一组具有复杂空间关系的组件，而不是单个字符类别。





```

Compared with all glyphs,  the components are more than 100 times less. Limited combinations of limited components in space form different glyphs, as shown in Figure X. Therefore, if the Chinese text is viewed from the perspective of components, it can be regarded as a very irregular text, that is, there is a highly complex spatial relationship between components on the premise of overall order.

所有部件相比于全部字形类别少100多倍，有限的部件在空间中的有限组合构成了不同的字形，如图X所示的汉字组成。所以若以部件的角度看中文文本，可以看作一种极不规则的文本，即部件之间在整体有序的前提下局部存在高度复杂的空间关系。
```

We consider starting from two aspects, introducing a sequence recognition module(SRM) for promoting the correct character order of handwritten text, and designing a spatial perception module(SPM) to refine each type of component in 2-D space, as shown in fig X. In section 4. Ablation experiments proved that this was necessary.

我们考虑从两个方面入手，引入序列识别模块(SRM)来促进手写体文本的正确字符顺序，并设计空间感知模块(SPM)来细化二维空间中的各类构件，结构如图X所示。在第四节。消融实验证明这是必要的。

SRM consist of a feature extractor based on a convolutional neural network, followed by a max-pooling on the vertical dimension and a full connection layer. CTC loss is used to train the system. Using SRM alone is not enough to generate detailed handwritten Chinese characters. The reason is that the SRM is essentially a 1-D prediction system, which is difficult to capture the complex spatial structure of Chinese characters. The same situation is that the performance of CRNN based system is insufficient in the irregular text recognition task.

SRM由一个基于卷积神经网络的特征抽取器、一个垂直维度上的最大池和一个全连接层组成。我们用CTC损失来训练系统。只使用识别器不足以生成细节清楚的手写汉字。原因是基于CTCloss的识别网络本质上是一个一维预测系统，难以捕获到汉字复杂的空间结构。同样的情形是在不规则文本识别任务中CRNN性能不足。

SPM is inspired by inexact supervision where the training data are given with only coarse-grained labels. Learning from this idea, we treat it as a non-sequential 2-D prediction problem like irregular scene text recognition with image-level annotations. 
SPM is composed of a full convolution network with a spatial attention module, followed by a reshape operation. SPM constraint the component category at each location in the 2-D feature map, and guide the network to generate the corresponding component at the correct location. Note that this is different from the 2-D prediction based on the attention mechanism. In Section 4, experiments show that improving recognition accuracy by capturing context module is not good for generating content readable text, although it is very important for the recognition task.



SPM的灵感来源于不精确的监督，即只使用粗粒度标签给出训练数据。借鉴这一思想，我们将其视为一个非连续二维预测问题，类似于带有图像级注释的不规则场景文本识别。SPM由一个带空间注意模块的全卷积网络和一个整形操作组成。SPM约束二维特征图中每个位置的组件类别，并引导网络在正确位置生成相应的组件。请注意，这与基于注意机制的二维预测不同。在第4节中，实验表明，通过捕获上下文模块来提高识别精度并不利于生成内容可读的文本，尽管这对于识别任务非常重要。





**Losses and optimization settings：**



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
The generated images should retain the correct structure and fine strokes, so we use the ACE loss.
$$
\begin{aligned}
L_{SPM_g} = 
&+\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[\operatorname{ACE}(\boldsymbol{e}, SPM(G(\boldsymbol{z}, \boldsymbol{e})))]\\
L_{SPM_d}=&+\mathbb{E}_{\boldsymbol{x} \sim p_{data}, \boldsymbol{e} \sim p_{text}}[\operatorname{ACE}(\boldsymbol{e}, SPM(\boldsymbol{x}))] \\

\end{aligned}
$$
Here,  $p_{data}$ denotes the distribution of real Handwritten Chinese text image, $p_z$ is a prior distribution on input noise $z$ and $p_{text}$  refers to a prior distribution of  the text.

使用pdata表示实[图像，文本]对的联合分布，pz表示输入噪声的先验分布，ptext表示文本的先验分布。





Since the gradients arising from each of the above loss terms can vary greatly in magnitude, we adopt the loss terms balance rule to updata $L_{SRM_g}$ and $L_{_SPMg}$ **[Adversarial Generation of Handwritten Text Images Conditioned on Sequences]**.

由于上述每个损失项产生的梯度在量级上可能有很大差异，因此我们损失项平衡规则更新 $L_{R_g}$ 和 $L_{C_g}$ 。




$$
\boldsymbol{\nabla}_{\boldsymbol{L_G}}= \frac{\partial L_G\left(\boldsymbol G(\boldsymbol{z}, \boldsymbol{e}) ) \right)}{\partial \boldsymbol  G(\boldsymbol{z}, \boldsymbol{e}) } 
\\
\boldsymbol{\nabla}_{\boldsymbol L_{SRM_g}}=\frac{\partial \mathrm{L_{SRM_g}}\left(\boldsymbol{e}, R\left(\boldsymbol G(\boldsymbol{z}, \boldsymbol{e})\right)\right)}{\partial \boldsymbol G(\boldsymbol{z}, \boldsymbol{e}) }

\\

\boldsymbol{\nabla}_{\boldsymbol L_{SPM_g}}=\frac{\partial \mathrm L_{SPM_g}\left(\boldsymbol{e}, R\left(\boldsymbol G(\boldsymbol{z}, \boldsymbol{e}) \right)\right)}{\partial \boldsymbol G(\boldsymbol{z}, \boldsymbol{e}) }
$$
$\boldsymbol{\nabla}_{\boldsymbol{L_G}}$ ,$\boldsymbol{\nabla}_{\boldsymbol L_{SRM_g}}$ and $\boldsymbol{\nabla}_{\boldsymbol L_{SPM_g}}$ are respectively the gradients of $L_{G}$, $L_{SRM_g}$ and $L_{SPM_g}$ with respect to the fake image $G(\boldsymbol{z}, \boldsymbol{e}) $. 



In our actual training process, not as  **[Alonso et al]** said, $\boldsymbol{\nabla}_{\boldsymbol{L_{SRM_g}}}$  is always several orders of magnitude larger than $\boldsymbol{\nabla}_{\boldsymbol L_{D}}$. We take one more step to promote the convergence: $ min(\boldsymbol{\nabla}_{\boldsymbol{L_G}},\boldsymbol{\nabla}_{\boldsymbol L_{SRM_g}},\boldsymbol{\nabla}_{\boldsymbol L_{SPM_g}})$ . Suppose that  $\boldsymbol{\nabla}_{\boldsymbol{L_G}}$ is the smallest, we update to obtain blance coefficient $\alpha$ and $\lambda$ , where the parameter $\beta$ and $\gamma$ respectively control the relative importance of $\boldsymbol L_{SRM_g}$ and $\boldsymbol L_{SPM_g}$  in the updating network.

在我们训练过程中，并非如 Alonso et al所说，我们采用多一步操作：$ min(\boldsymbol{\nabla}_{\boldsymbol{L_G}},\boldsymbol{\nabla}_{\boldsymbol L_{R_g}},\boldsymbol{\nabla}_{\boldsymbol L_{C_g}})$ 

假设，$\boldsymbol{\nabla}_{\boldsymbol{L_G}}$ 是最小的，更新得到平衡系数，其中$\beta$ and $\gamma$ 分别控制R、C在更新网络占的比重。


$$
\alpha \leftarrow 
\beta

\frac{\sigma\left(\nabla L_G\right)}{\sigma\left(\nabla L_{SRM_g}\right)\nabla L_{SRM_g} } \cdot\left[\nabla  L_{SRM_g}-\mu\left(\nabla L_{SRM_g}\right)\right]+\mu\left(\nabla L_{G}\right)
$$

$$
\lambda  \leftarrow 
\gamma
\frac{\sigma\left(\nabla L_G\right)}{\sigma\left(\nabla L_{SPM_g}\right)\nabla L_{SPM_g}} \cdot\left[\nabla  L_{SPM_g}-\mu\left(\nabla L_{SPM_g}\right)\right]+\mu\left(\nabla L_{G}\right)
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









