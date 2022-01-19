\title{HCT-GAN: Handwritten Chinese Text-lines Generation with Inexact Supervision}

Text Conditioned Single Shot Handwritten Chinese Text-lines Generation



% author names and affiliations
% use a multiple column layout for up to three different
% affiliations
\author{\IEEEauthorblockN{Michael Shell}
\IEEEauthorblockA{School of Electrical and\\Computer Engineering\\
Georgia Institute of Technology\\
Atlanta, Georgia 30332--0250\\
Email: http://www.michaelshell.org/contact.html}
\and
\IEEEauthorblockN{Homer Simpson}
\IEEEauthorblockA{Twentieth Century Fox\\
Springfield, USA\\
Email: homer@thesimpsons.com}
\and
\IEEEauthorblockN{James Kirk\\ and Montgomery Scott}
\IEEEauthorblockA{Starfleet Academy\\
San Francisco, California 96678--2391\\
Telephone: (800) 555--1212\\
Fax: (888) 555--1212}}

% conference papers do not typically use \thanks and this command
% is locked out in conference mode. If really needed, such as for
% the acknowledgment of grants, issue a \IEEEoverridecommandlockouts
% after \documentclass

% for over three affiliations, or if they all won't fit within the width
% of the page, use this alternative format:
%
%\author{\IEEEauthorblockN{Michael Shell\IEEEauthorrefmark{1},
%Homer Simpson\IEEEauthorrefmark{2},
%James Kirk\IEEEauthorrefmark{3},
%Montgomery Scott\IEEEauthorrefmark{3} and
%Eldon Tyrell\IEEEauthorrefmark{4}}
%\IEEEauthorblockA{\IEEEauthorrefmark{1}School of Electrical and Computer Engineering\\
%Georgia Institute of Technology,
%Atlanta, Georgia 30332--0250\\ Email: see http://www.michaelshell.org/contact.html}
%\IEEEauthorblockA{\IEEEauthorrefmark{2}Twentieth Century Fox, Springfield, USA\\
%Email: homer@thesimpsons.com}
%\IEEEauthorblockA{\IEEEauthorrefmark{3}Starfleet Academy, San Francisco, California 96678-2391\\
%Telephone: (800) 555--1212, Fax: (888) 555--1212}
%\IEEEauthorblockA{\IEEEauthorrefmark{4}Tyrell Inc., 123 Replicant Street, Los Angeles, California 90210--4321}}




% use for special paper notices
%\IEEEspecialpapernotice{(Invited Paper)}




% make the title area
\maketitle

% As a general rule, do not put math, special symbols or citations
% in the abstract
\begin{abstract}
In recent years, researchers have investigated a variety of approaches to the generation of Chinese single character, but the obtainment of handwritten Chinese sequence text images is usually achieved by stacking single character image. Due to the large number of complex glyphs in Chinese characters, it is impossible to generate readable handwritten Chinese sequence text images by directly exploiting the alphabetic handwritten text model. Inspired by the inexact supervision, we propose an effective model named HCT-GAN to generate handwritten Chinese sequence text directly from latent content representation (e.g., learnable embedding vector) conditioned on text labels. Specifically, HCT-GAN is designed as a CGAN-based architecture that additionally integrates a Chinese string encoder (CSE), a sequence recognition module(SRM) and a spatial perception module (SPM). Compared with the one-hot embedding, CSE can obtains the latent priori conditioned on text labels by reusing the structure and component information shared among all the Chinese characters. SRM provides sequence-level constraints to ensure the recognizability of the generated text. SPM can adaptively learn the spatial correlation between the internal components of a character in an inexact supervised manner, which will facilitate the modeling of characters with complicated topological structures. Benefiting from such artful modeling, our model suffices to generate images of sequence Chinese characters with arbitrary length. Extensive experimental results demonstrate that our model achieves state-of-the-art performance in handwritten Chinese text generation.
\end{abstract}

% no keywords




% For peer review papers, you can put extra information on the cover
% page as needed:
% \ifCLASSOPTIONpeerreview
% \begin{center} \bfseries EDICS Category: 3-BBND \end{center}
% \fi
%
% For peerreview papers, this IEEEtran command inserts a page break and
% creates the second title. It will be ignored for other modes.
\IEEEpeerreviewmaketitle





\section{Introduction}
% no \IEEEPARstart
With the development of deep learning, automatic generation of Chinese single font with complicated structure is not a difficult task, but the handwritten Chinese sequence text generation remains underexplored. Compared with single Chinese font generation(hereinafter referred to as Chinese font generation), Handwritten Chinese sequence text generation(hereinafter referred to as Chinese text generation) is a much more challenging task. In addition to the inherent difficulties of Chinese font generation, such as Chinese characters sharing an extremely large glyph with complicated content, handwritten Chinese text generation has the following challengings. (i) handwritten Chinese text contain text-level features, such as the adhesions between adjacent characters. (ii) There are subtle differences in the same Chinese characters of a single writer. For the current Chinese font generation model, handwritten text images are obtained by splicing isolated character images. The method typically has the problem of lack of authenticity for failing to solve the above two critical challengings. Our work alleviates the problem by directly generating variable-sized handwritten Chinese text images.(e.g., see Figure \ref{fig:introduction_show}).


% figue:introduction_show
\begin{figure}
\centering
\includegraphics[width=\linewidth]{introduction.pdf}
\caption{Examples of our model. Top:  Handwritten text images are obtained by splicing independent character images. Bottom:  Generating handwritten text image directly. When we handwrite Chinese text, the same words will be slightly different under the influence of neighbors, and there may be adhesion between characters. The Chinese characters in the box are exactly the same, but there are some differences in the circle. The part marked in green is unique to the bottom. 
}
\label{fig:introduction_show}
\end{figure}


The study of character generation has experienced two stages of development. The first stage is the component assembly method based on computer graphic\cite{mybib:Automatic_Generation_of_Artistic_Chinese_Calligraphy,mybib:Handwritten_Chinese_Character_Font_Generation_Based_on_Stroke_Correspondence,lian2012automatic,lin2014font,lian2016automatic,xu2009automatic,zhou2011easy,zong2014strokebank}. This kind of method regards glyphs as a combination of strokes or radicals, which first extract strokes from Chinese character samples, and then some strokes are selected and assembled into unseen characters by reusing parts of glyphs. The above-mentioned methods are largely dependent on the effect of strokes extraction. The stroke extraction module does not typically perform well when the structure is written in a cursive style.In the second stage, the utilization of U-Net as encoder-decoder architecture enables the font generation problem to be treated as an image-to-image translation problem via learning to map the source style to a target style \cite{azadi2018multi, chang2018generating,jiang2018w,jiang2017dcfont,jiang2019scfont,liu2019fontgan,lyu2017auto,sun2018pyramid,xie2021dg,Yang2019TETGANTE,zhang2018separating,zheng2018coconditional,wen2019handwritten}. Considering blur and ghosting problems, it is Insufficient only to treat a single Chinese character as a glyph category. Recent works such as \cite{wu2020calligan,2020RD,2017Learning,cha2020fewshot,park2020fewshot,2021Multiple} generated Chinese characters by reintroducing prior knowledge of structures and components to alleviate this problem, and helps to generate more diverse fonts with fewer stylized samples. However, all previous Chinese font generation works ignore the basic fact that real Handwritten Chinese text image can not be obtain by stacking single character image.

Handwriting text generation (HTG)\cite{graves2013generating} is originally proposed for the generation of alphabetic text images, which basic structure can naturally model the dependencies between adjacent characters. The most advanced model ScrabbleGAN\cite{fogel2020scrabblegan} uses a one-hot vector to encode a letter/symbol as the conditional information of CGan\cite{mirza2014conditional} and adds a sequence character recognizer to constrain the text content. As shown in Figure\ref{fig:arch}, after applying the ScrabbleGan to the handwritten Chinese text generation task, we can not obtain the readable handwritten Chinese text image. The reason is not only the huge number of Chinese characters (more than 100,000 Chinese characters, but only dozens of letters in English), but also the reuse mechanism of Chinese components brings the problems of near glyph interference and information redundancy, so it is difficult to accurately generate more 100,000 glyphs.

Inspired by the inexact supervision\cite{10.1093/nsr/nwx106}, we propose an effective model named HCT-GAN to generate handwritten Chinese sequence text directly from latent content representation conditioned on text labels. Specifically, HCT-GAN is designed as a CGAN-based architecture that additionally integrates a Chinese string encoder (CSE), a sequence recognition module(SRM) and a spatial perception module (SPM).

With such a large amount of Chinese character classes, if the one-hot embedding method is simply adopted, not only will it lead to an explosive growth of model parameters, but also it is difficult to obtain the content representation conducive to generating tasks. To address this issue, we propose a learnable character encoding method(CSE). By reusing the structure and component information shared among all the Chinese characters, CSE enables every character to be transformed to structure embedding vector and component sequence embedding vector. Compared with the one-hot embedding, CSE is amore informative character encoding method. The alphabetic text generation model generally only uses a sequence character recognizer with a CTC-based recognition loss function, which provides sequence-level constraints to ensure the recognizability of generated text.
Due to the complex spatial structure of Chinese characters, the traditional alphabetic text generation model using only a character sequence recognizer is not enough to generate readable handwritten Chinese text. Thus, we modify the character recognizer to obtain SRM. The former predicts character sequences and the latter predicts component sequences. we further introduce a spatial perception module(SPM) to refine the details of the handwritten Chinese text images to make the content readable.

SPM can guide generator to adaptively learn the spatial correlation between the internal components of a character in an inexact supervised manner, which will facilitate the modeling of characters with complicated topological structures. Benefiting from such artful modeling, our model suffices to generate images of handwritten Chinese text with arbitrary lengths. Extensive experimental results demonstrate that our model achieves state-of-the-art performance in handwritten Chinese text generation.

Unlike the past Chinese font generation paradigm, we only take the latent content representation of Chinese characters(i.e., component and structure learnable embedding vector) as the network input. Since any appropriate combination of component and structure will correspond to a glyph, our model not only generate Chinese text containing arbitrary Chinese characters, but also some non-sense glyphs. As we all know, data augmentation is conducive to the improvement of accuracy in recognition tasks. In this paper, we find that non-sense glyphs can play the same effect. The concrete impact of this discovery is discussed in Section IV-B.

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


% You must have at least 2 lines in the paragraph with the drop letter
% (should never be an issue)


% \hfill mds

% \hfill August 26, 2015

\section{Related Work}

\textbf{Chinese font generation(CFG).} Up to now, large numbers of methods for font generation have been proposed. Generally speaking, those existing methods can be classified into two categories: Computer Graphics based methods and Deep Learning-based methods.Traditional methods are typically based on the basic idea of stroke or radical extraction and reassembly.

\cite{mybib:Automatic_Generation_of_Artistic_Chinese_Calligraphy} proposed a small structural stroke database to represent each character in a multi-level way, and calligraphy is generated via a reasoning-based approach. \cite{mybib:Handwritten_Chinese_Character_Font_Generation_Based_on_Stroke_Correspondence}consider each stroke to be a vector and proposed individual’s handwritten Chinese character font generation method by vector quantization.
In the \cite{xu2009automatic} method, he derive a parametric representation of stroke shapes and generate a new character topology via weighted averaging of a few given character topologies derived from the individual’s previous handwriting. 
\cite{lian2012automatic} and \cite{lian2016automatic} by applying the Coherent Point Drift (CPD) algorithm to achieve non-rigid point set registration between each character to extract stroke/radical.
\cite{lin2014font} exploit a radical placement database of Chinese characters, vector glyphs of some other Chinese characters are automatically created, and these vector glyphs are merged into the user-specific font. 
Later, the Radical composition Model \cite{zhou2011easy} and \cite{zong2014strokebank} were proposed by mapping the standard font component to their handwritten counterparts to synthesize characters. 

After the generative adversarial networks(GANs)\cite{goodfellow2014generative} was proposed, its derivative version\cite{mirza2014conditional,zhang2019selfattention,karras2019stylebased} was widely adopted for style transfe\cite{chen2017stylebank,johnson2016perceptual,isola2017image,zhu2017unpaired,choi2018stargan,choi2020stargan}. Several attempts have been recently made to model font synthesis as an image-to-image translation problem \cite{Rewrite,Zi-to-zi,azadi2018multi, chang2018generating,jiang2018w,jiang2017dcfont,jiang2019scfont,liu2019fontgan,lyu2017auto,sun2018pyramid,xie2021dg,Yang2019TETGANTE,zhang2018separating,zheng2018coconditional,wen2019handwritten}, which transforms the image style while preserving the content consistency.

\textbf{Handwritten text generation(HTG).} Since the high inter-class variability of text styles from writer to writer and intra-class variability of same writer styles\cite{krishnan2021textstylebrush}, the handwritten text generation is challenging.
At present, handwritten text generation mainly focuses on alphabetic text. Alonso et al.\cite{alonso2019adversarial} proposed an offline handwritten text generation model for fixed-size word images. ScrabbleGAN\cite{fogel2020scrabblegan} used a fully-convolutional handwritten text generation model.
For handwritten Chinese text generation(HCTG) tasks, the existing text generation model cannot generate readable content. In contrast, our method is applicable for images of handwritten Chinese text with arbitrary length. 


% figue：architecture
\begin{figure*}
\centering
\includegraphics[width=\textwidth]{architecture.pdf}
\caption{ Left: Examples of the Dictionary. Characters with the same color share the same component index. Color block(c) represents structure index. Right: Overview of the proposed method at training. Given Chinese character string(i.e.,\begin{CJK}{UTF8}{gbsn}呗员...\end{CJK}),it was queried in the Dictionary to obtain corresponding component and structure indexs(i.e.,$s_1s_2,c_1c_2$). Then, The sequences pass through E to obtain content representation e. Later, the e concatenated by noise $z_1$ was fed into G to generate handwritten Chinese text image(i.e., fake). Multi-scale nature of the fake is controlled by additional noise vectors,$z_2,z_3$ and $z_4$ fed to G each layers through Conditional Batch Normalization (CBN) ayers. The output generated images and the real images are transmitted to D, SRM and SPM, which respectively correspond to adversarial loss(),CTC loss(), and ACE loss().
}
\label{fig:arch}
\end{figure*}

\section{Our method}
Any appropriate combination of component and structure will correspond to a glyph. More than 100,000 Chinese glyphs are a subset of the combination. Since the real handwritten Chinese text can not be obtained by stacking a single character, our proposed model aims to learn a way to generate real handwritten Chinese text, which can contain non-sense glyph.

\subsection{architecture}
 The framework mainly consists of five modules of a Chinese string encoder(CSE), a generator(G), a discriminator(D), a sequence recognition module(SRM), and a spatial perception module(SPM). CSE is mainly responsible for encoding the components and structures of Chinese string into latent content representation $e$. G is constructed based on a full convolution network(FCN)\cite{long2015fully} and can generate variable-sized output. D promotes realistic-looking handwriting styles. SRM is a 1-D sequence recognizer with connectionist temporal classification (CTC) loss\cite{graves2006connectionist} to ensure that the generated handwritten Chinese text is in the correct character order. SPM is a 2-D prediction model that guides the network to refine each type of component in correct position. Due to the high cost of components spatial relationships annotation, SPM adopts an inexact supervised approach.



\subsection{Chinese string encoder(CSE)}
Our input is a Chinese character string, e.g., “\begin{CJK}{UTF8}{gbsn}呗员...\end{CJK}”. \cite{wu2020calligan} embed component sequences into fixed-length vectors, which is not suitable for the generation of text image. We propose a intuitive method that decompose each character of Chinese character string into components sequence and structure index, and padding to equal length to fit the generation of text image. 

Since the same components and structure appear repeatedly in various glyphs, the number of indexe is much smaller than the number of character classes in the Dictionary. Figure \ref{fig:arch}(left) shows a few examples of the Dictionary. CSE directly learns the embedding of each index, then which is reshaped into latent content representation tensor $e\in\mathbb{R}^{480\times 4\times 4W}$ , instead of mapping the embedding to a fixed-length vector in\cite{wu2020calligan}, which helps to mimic the writing process that the components of Chinese characters are only connected with their local spatial components, and avoid using a recurrent network to learn the coupling embedding of the whole text. Compared with the one-hot embedding, CSE is a more informative character encoding method. Table \ref{table:encoder}  shows the details of CSE.


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
    \label{table:encoder}
\end{table}



% architecture tabel
\begin{table*}[]
\renewcommand{\arraystretch}{1.5}
\caption{ResBlock is basic structure from BigGAN，}
\label{table:GDRP}
    \centering
\begin{tabular}{c}
\hline \hline$z_1 \in \mathbb{R}^{32\times 4\times 4W} \sim \mathcal{N}(0, I)$ \\
$e\in\mathbb{R}^{480\times 4\times 4W}$\\
\hline $cat(z_1,e) \in\mathbb{R}^{512\times 4\times 4W}$\\
\hline ResBlock up $512 \rightarrow 256$ \\
\hline ResBlock up $256 \rightarrow 128$ \\
\hline ResBlock up $128 \rightarrow 64$ \\
\hline ResBlock up $64ch \rightarrow 32$ \\
\hline Non-Local Block $(32 \times 32W)$ \\
\hline BN, ReLU, Conv $32 \rightarrow 1$ \\
\hline Tanh \\
\hline \hline (a) Generator(G)
\end{tabular}
\begin{tabular}{c}
\hline \hline image $x \in \mathbb{R}^{32 \times 32W \times 1}$ \\
\hline ResBlock down $1 \rightarrow 32$ \\
\hline Non-Local Block $(32 \times 32)$ \\
\hline ResBlock down $32 \rightarrow 64$ \\
\hline ResBlock down $64  \rightarrow 128$
\\
\hline ResBlock down $128  \rightarrow 512$
\\
\hline ResBlock down $512 \rightarrow 1024$ \\
\hline ResBlock $1024 \rightarrow 1024 $ \\
\hline ReLU, Global sum pooling \\
\hline linear(1024) $\rightarrow 1$ \\
\hline \hline (b) Discriminator(D)
\end{tabular}
\begin{tabular}{c}
\hline \hline image $x \in \mathbb{R}^{32 \times 32W \times 1}$ \\
\hline pooling ReLu Conv $1 \rightarrow 64$
\\
\hline pooling ReLu BN Conv $64 \rightarrow 128$ \\
\hline ReLu Conv $128 \rightarrow 256$ \\
\hline BN, ReLU,Conv $256 \rightarrow 256$ \\
\hline pooling ReLu BN Conv $256\rightarrow 512$ \\
\hline pooling ReLu BN Conv $512 \rightarrow 512$ \\
\hline ReLu BN Conv $512 \rightarrow 512$ \\
\hline Max pooling on the vertical dimension\\
\hline Softmax liner(512)$\rightarrow n_{classes+1}$  \\


\hline \hline (d) SRM
\end{tabular}
\begin{tabular}{c}
\hline \hline image $x \in \mathbb{R}^{32 \times 32W \times 1}$ \\
\hline cbam+pooling ReLu Conv $1 \rightarrow 64$
\\
\hline pooling ReLu BN Conv $64 \rightarrow 128$ \\
\hline cbam+ReLu Conv $128 \rightarrow 256$ \\
\hline BN, ReLU,Conv $256 \rightarrow 256$ \\
\hline pooling ReLu BN Conv $256\rightarrow 512$ \\
\hline cbam+ReLu BN Conv $512 \rightarrow 512$ \\
\hline Softmax liner(512)$\rightarrow n_{classes+1}$  \\
\hline reshpe(batch,$n_{classes+1}$,H,W)$ \\
\rightarrow$ (batch,$n_{classes+1}$,H$\times$W) \\
\hline \hline (d) SPM
\end{tabular}
\end{table*}





\subsection{Generator(G) and Discriminator(D)}
G and D are inspired by SAGAN\cite{zhang2019selfattention}, and some common techniques in the training, such as self-attention mechanism\cite{vaswani2017attention} to refine local area image quality, spectral normalization\cite{miyato2018spectral} and hinge loss function\cite{lim2017geometric} to stabilizes the training, was used. FCN is a common technique to deal with variable size input and output problems\cite{krishnan2021textstylebrush,fogel2020scrabblegan}. In this paper, we inherit this paradigm in SAGAN-based generators. At the same time, FCN allows the network to learn the dependencies between adjacent components, which solves the problem of the differentiated output of the same characters in a single handwriting style. 

As shown in Tabel \ref{Tabel:encoder}, CSE module outputs the latent content representation tensor $e\in\mathbb{R}^{480\times 4\times 4W}$.  Later, $e$ is concatenated with noise vector $z_1 \in \mathbb{R}^{32\times 4\times4W} $ in the channel layer and fed into G, and the additional noise vectors $z_2, z_3,z_4$ and $ z_5$ are fed into each layer of G through Conditional Batch Normalization (CBN)\cite{shell2015bare} layers. This allow G to control the low-resolution and high-resolution details of the appearance of Chinese text to match the required handwritten style. Compared with the common Chinese font generation network, the optimization objective of G is to generate appropriate components in the appropriate position. To cope with varying size image generation, D is also fully convolutional structure to promote the generator output realistic handwritten text. 


\subsection{spatial perception module(SPM) and sequence recognition modul(SRM)}

In addition to our proposed coding method, The synergy between sequence recognition module(SRM) and spatial perception module(SPM) is our main improvement.
The generation of alphabetic text usually adopts 1-D sequence prediction architecture to evaluate the content of the generated images. However, due to the huge number of Chinese characters (more than 100,000 Chinese characters compared with dozens of letters in English), it is very difficult to accurately generate 100,000 types of glyphs. The reuse mechanism of Chinese components also brings the problems of near glyph interference and information redundancy. Therefore, we treat a Chinese text as a string of components with complex spatial relationships rather than as a single character category.

We consider starting from two aspects, introducing a sequence recognition module(SRM) for promoting the correct character order of handwritten text, and designing a spatial perception module(SPM) to refine each type of component in 2-D space. In section IV. Ablation experiments prove the necessity. 
SRM consist of a feature extractor based on a convolutional neural network, followed by a max-pooling on the vertical dimension and a full connection layer. CTC loss is used to train the system. Using SRM alone is not enough to generate detailed handwritten Chinese characters. The reason is that the SRM is essentially a 1-D sequence prediction system, which is difficult to capture the complex spatial structure of Chinese characters. The same situation is that the performance of CRNN based system is insufficient in the irregular text recognition task.

SPM is inspired by inexact supervision where the training data are given with only coarse-grained labels. We treat it as a non-sequence-based 2D prediction problem with image-level annotations. 
SPM is composed of a full convolution network with a spatial attention module, followed by a full connection layer. SPM constraint the component category at each location in the 2-D feature map, and guide the network to generate the corresponding component at the correct location. 

Note tha most of recognition model use a recurrent network, typically bidirectional LSTM \cite{shi2016end,he2016reading}, which may predict characters based on linguistic context instead of clear character shapes. In contrast, SRM and SPM only uses local visual features for character recognition and therefore provides better optimization direction for generating characters.

 We present the detail of our proposed model in Table\ref{table:GDRP}.

\subsection{Losses}
We implement the hinge version of the adversarial loss from Geometric GAN\cite{lim2017geometric}
\begin{equation}
\begin{aligned}
L_{G}=&-\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[D(G(\boldsymbol{z}, \boldsymbol{e}))] 
\\
L_{D}=&+\mathbb{E}_{(\boldsymbol{x}) \sim p_{\text {data }}}[\max (0,1-D(\boldsymbol{x}))] \\
&+\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[\max (0,1+D(G(\boldsymbol{z}, \boldsymbol{e})))] \\
\end{aligned}
\end{equation}
to promote our generated images look realistic. To encourage the correct character order in the generated handwritten Chinese text image, we use the CTC loss.
\begin{equation}
\begin{aligned}
L_{SRM} = 
&+\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[\operatorname{CTC}(\boldsymbol{e}, SRM(G(\boldsymbol{z}, \boldsymbol{e}))]\\
\end{aligned}
\end{equation}

The generated images should retain the fine strokes, we use the ACE loss. 
\begin{equation}
    \begin{aligned}
L_{SPM_g} = 
&+\mathbb{E}_{\boldsymbol{z} \sim p_{z}, \boldsymbol{e} \sim p_{text}}[\operatorname{ACE}(\boldsymbol{e}, SPM(G(\boldsymbol{z}, \boldsymbol{e})))]\\
\end{aligned}
\end{equation}
Here,  $p_{data}$ denotes the distribution of real Handwritten Chinese text image, $p_z$ is a prior distribution on input noise $z$ and $p_{text}$  refers to a prior distribution of  the text.

We adopt the loss terms balance rule\cite{alonso2019adversarial} to obtain blance coefficient $\alpha$ and $\lambda$ , where the parameter $\beta$ and $\gamma$ respectively control the relative importance of $\boldsymbol L_{SRM_g}$ and $\boldsymbol L_{SPM_g}$  in the updating generator network. $\nabla L_{G},\nabla L_{SRM}$ and $\nabla L_{SPM}$ are respectively the gradients of $L_G,L_{SRM}$ and $L_{SPM}$ w.r.t. the image.

\begin{equation}
\begin{aligned}
\alpha \leftarrow & \beta \frac{\sigma\left(\nabla L_{G}\right)}{\sigma\left(\nabla L_{S R M_{}}\right) \nabla L_{S R M}} \cdot\left[\nabla L_{S R M}-\mu\left(\nabla L_{S R M}\right)\right]+ \\
& \mu\left(\nabla L_{G}\right)
\end{aligned}
\end{equation}

\begin{equation}
\begin{aligned}
\lambda  \leftarrow & \gamma \frac{\sigma\left(\nabla L_G\right)}{\sigma\left(\nabla L_{SPM}\right)\nabla L_{SPM}} \cdot\left[\nabla  L_{SPM}-\mu\left(\nabla L_{SPM}\right)\right]+\\
&\mu\left(\nabla L_{G}\right)
\end{aligned}
\end{equation}

here $\sigma$ and $\mu$ are respectively the empirical standard deviation and mean.

In D training step, For SRM and SPM, we only use real images to avoid learning the generated content representation. In G training step, we freeze the weight of D, SRM and SPM, let it processes the generated image, and update the weight of G. The total loss is:
\begin{equation}
    L = L_G+\alpha L_{SRM}+\lambda L_{SPM}
\end{equation}


\section{Experiments
}

\subsection{Datasets}

The offline handwritten Chinese database, CASIA-HWDB \cite{liu2011casia} is a widely used database for handwritten Chinese character recognition. It contains samples of isolated characters and handwritten texts that were produced by 1020 writers on dot matrix paper with Anoto pen. Offline handwritten Chinese character sample are divided into three databases: HWDB1.0~1.2(Including 7,185 classes Chinese characters and 171 classes English letters, numbers and symbols). Handwritten texts is also divided into three databases: HWDB2.0~2.2(Its character classes is contained in HWDB1.0~1.2). 

The datasets HWDB1.0-Train and HWDB2.0~2.1-Train is added to HWDB1.1-Train and HWDB2.2-Train respectively for enlarging the training set size to promote generation of characters/texts. The datasets HWDB1.0~1.1-Test and HWDB2.0~2.2-Test are used for inspecting performance. Table \ref{tab:hwdb} shows the dataset we selected, while Figure\ref{fig:hwdb} shows some images from these two datasets.


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


\ref{fig:hwdb}
\ref{tab:hwdb}



% figue: dataset_show
\begin{figure}
\centering
\includegraphics[width=\linewidth]{4A_show_dataset.pdf}
\caption{Examples of the CASIA-HWDB dataset. First line: HWDB2.0(124-P17). Second line: HWDB1.0(156-t).}
\label{fig:hwdb}
\end{figure}


\subsection{Implementation details and evaluation metrics
}
we decompose 7,318 classes of Chinese characters in CICAS-HWDB into 1,179 classes components and 31 classes structure. Some rare character classes almost no longer appear in modern Chinese, e.g.,“\begin{CJK}{UTF8}{gbsn}嚚,橐,畿,曩,爨,黉,夔,龠\end{CJK}”, we remain unchanged for rare-complex character. We use a publicly available Chinese character decomposition document\utl{https://github.com/CoSeCant-csc/chaizi} and modify it to fit the proposed model.

In our experiments, 32 × 32W generated images of handwritten Chinese texts are obtained with the following processing:  Each Chinese character is decomposed into four component indexes and one structure index. Given Chinese indexs sequence encoded by the CSE, each Chinese character can be represented by a tensor of size $480\times 4\times4$, and the latent content representation $e$  is a stack of the tensor in width. Later, the $e$ concatenated by noise $z_1$ is fed into the first residual blocks in generator which upsample the spatial resolution, and lead to the final image size of 32 × 32W. Four additional 32 dimensional noise vectors,$ z_2, z_3,z_4$ and $z_5$ through conditional Instance Normalization layersonditional Instance Normalization layers modulate the other residual blocks.

We resize the real images to a height of 32 pixels, width of 32W pixels, and use 32 × 32W images of handwritten Chinese text/font to better deceive the discriminator, where W is the number of characters contained in labels. We use the Adam optimizer with a fixed learning rate of 0.0002 and batch size of 16. Finally, our method is implemented using the PyTorch framework. Training was performed on 3GPUS with 12GB of RAM each.

We follow the same quantitative evaluation measures as previous handwritten text Images generation methods\cite{alonso2019adversarial,fogel2020scrabblegan}. We compare real handwritten images with generated results using these measures: (1)Fréchet Inception Distance (FID)  is widely used and calculates the distance between the real and generated images; (2)the Geometric Score (GS), which compares the topology between the real and generated manifolds. For the above two indicators, lower is better. We evaluate FID/GS on a sampled set(HWDB1.0~1.1: with HWDB1.0 test set for FID and 7k samples for GS, HWDB2.0~2.2: with 20k samples for FID and 5k samples for GS), due to its computational costs. FID/GS was computed on sampled real and generated images using 32×32W images. 

For promoting handwritten Chinese text/character recognition, we evaluate the performance with recognition accuracy,  character error rate(CER) and edit-distance(ED). we use accuracy to measure handwritten Chinese isolated character recognition accuracy.  we use CER and ED to measure handwritten Chinese texts recognition performance.  CER is the number of misread  out of the number of characters in the test set. The ED is derived from the Levenshtein Distance algorithm, calculated as the minimum edit distance between the predicted and true text. Most of Chinese character generation work focusing on the transfer of font style use different evaluation metrics and so are not directly comparable with our work.

\subsection{Handwritten Chinese generation rsults.}
We report the experimental results on HWDB1.1 and HWDB2.2 respectively. 
\subsubsection{Ablation study.}

We conducted an ablation study by removing key Dmodules of our model: the Chinese string encoder(CSE), the sequence recognition module(SRM), and the spatial perception module(SPM). Without the CSE,  the model produces blurry results in both isolated characters images and handwritten text images. Appling the SRM or SPM, generated samples leads to a improvement in readability, but the character strokes are still not clear compared with HCT-GAN. Compared with SRM, SPM is more critical for generating text images, while it is not obvious in isolated character images. Without the SPM, generated text images can not reflect the true perceived visual quality, which shows that SPM is indispensable. We attempte a model with Replace SRM,  which are not able to obtain better the quality of the images generated.

Compared with the character recognizer, SRM effectively improves the image quality and does not increase the workload (the ground true components sequence comes from the dictionary L in CSE). Meanwhile SRM saves the memory overhead(Component class: 1,179. Character class: 7,356). The character recognizer predicts characters or character sequences, and the SRM predicts component sequences, both using the same network. FID and GS are reported in Table \ref{tab:ablation_study}. Figure \ref{fig:ablation_HWDB1} and Figue \ref{fig:ablation_HWDB2} shows some examples of all versions.


% table:ablation_study
\begin{table}[]
\renewcommand{\arraystretch}{1.2}
    \centering
        \caption{Ablation study. Replace SRM means that replace SRM with character recognizer.}
    \label{tab:ablation_study}
    \begin{tabular}{c|c|ll}
     \text { Dataset } & \text {Model} & \text { FID} $\downarrow$& \text { GS }$\downarrow$ \\
     
\hline \hline   & HCT-GAN(without CSE) &                      21.08 & $1.30\times 10^{-2}$ \\
                & HCT-GAN(without SPM)& 15.47& $5.82\times 10^{-4}$ \\
\text {HWDB1.1} & HCT-GAN(without SRM) &  16.23                 & $6.20 \times 10^{-3}$   \\
                & HCT-GAN  & \textbf{15.14} &
        \textbf{4.10 \times $10^{-4}$} \\
                & HCT-GAN(Replace SRM) &  18.62 & $8.30\times 10^{-3}$   \\

\hline          & HCT-GAN(without CSE)	 &                   22.76 & $1.50\times 10^{-1}$  \\
                & HCT-GAN(without SPM)	 &                 20.42 & $4.86\times 10^{-3}$ \\
\text {HWDB2.2} & HCT-GAN(without SRM)	 &                   18.61 & $4.10\times 10^{-3}$ \\
                & HCT-GAN                & \textbf{17.82} & \bf{3.33 \times $10^{-3}$ }\\
                 & HCT-GAN(Replace SRM)   & 20.69 & $4.00\times 10^{-3}$  \\
\hline
    \end{tabular}
\end{table}




\subsubsection{Showing generation rsults}


\textbf{Gradient balancing study.} Gradient balancing is an important trick in multi-task learning. To compare with \cite{fogel2020scrabblegan}, we apply the gradient balancing scheme suggested in \cite{alonso2019adversarial}. Table \ref{tab:gradient_balancing} reports FID, GS for different gradient balancing settings. 

\textbf{ Comparison to scrabbleGAN.} We trained the HCT-GAN on the two datasets described in Section IV-A, HWDB1.1 and HWDB2.2. Figure \ref{fig:comparison} represent results trained by ScrabbleGAN\cite{fogel2020scrabblegan}  alongside results of our method on the same characters/texts images. It is obvious from the figure that our network produces images that are much clearer, whether for isolated characters or variable-size text images. 

When the character strokes becomes complex, the details of the top-left character images begin to blur. On the contrary, our network produces images that are still clear.Directly generating variable-size handwritten text image is more challenging. The bottom-left text have lost readability, and the bottom-right text still has realistic-looking handwriting and diverse writing styles. Later, we will further represent generated arbitrarily text images with various styles, not only in the training set.

\textbf{Showing diverse handwriting.} We are able to generate different handwriting styles by changing the noise vector z. Figure \ref{fig:comparison} shows examples of randomly selected character/text generated in different handwriting styles.

\textbf{Interpolation between different style.} We are able to capture variations of the different handwriting styles by interpolating between the two noise vector. Figure \ref{fig:style_interpolation} shows the interpolation between two styles obtained from the randomly noises. We note that interpolation hardly changes the shape of the character and focuses more on the overall global style elements(e.g., tilt, ink thickness).

\textbf{Generating unsee text.} Each the character in the Figure \ref{fig:show_test_unsee}(top) text is not in the training set, which indicates that HCT-GAN can generate handwritten text containing unsee characters in through CSE. Figure \ref{fig:show_test_unsee}(down) shows generated text in the HWDB2.2-Test.

\textbf{Generating non-sense glyph.} Figure \ref{fig:show_nonsense} shows glyphs formed by the random combination of components and structures. These glyphs look like Chinese characters, but they are not. 

\subsubsection{Improving recognition performance}

Our proposed HCT-GAN not only promotes generating more realistic handwritten Chinese text images but also improves the handwritten Chinese text recognition(HCTR) performance. For all experiments in this section, we use the code provided by \cite{shi2016end} as our HCTR framework. The purpose of using the basic CRNN model is to prove that HCTR performance can be improved by simply appending the generated image to the training set. Note that like the random affine transformations, we only use the training set. 

CER, ED and ACC are reported in Table \ref{tab:recognition_performance}. As can be seen in the table, using the HCT-GAN generated samples further improve the recognition performance compared to only using affine augmentation. Moreover, we find that appling non-sense glyphs as negative samples is more conductive to improvement in performance than equivalent character augmentation.

\section{conclusion}
We have presented a model capable of directly generating handwriting Chinese text image of arbitrary length. In our model, we design a Chinese sequence encoder(CSE) suitable for Chinese text image generation, and introduce the spatial perception module(SPM) into the text image generation model. Experimental results show that the proposed method generates high-quality images of handwritten Chinese text.

Our research is still ongoing and there are some deficiencies. Generated characters with many strokes and close coupling are still blurred. However, these Chinese characters almost no longer appear in modern commonly used Chinese, which has a limited impact on our methods. In the future, we additionally plan to address handwritten text image generation with controllable style.



% figue:comparison
\begin{figure*}
\centering
\includegraphics[width=\linewidth]{comparison.pdf}
\caption{Comparing to our results(right side) to those from ScrabbleGAN \cite{fogel2020scrabblegan}(left side) on the CASIA-HWDB \cite{liu2011casia} dataset. Top: training on HWDB1.1, down: training on HWDB2.2. 
}
\label{fig:comparison}
\end{figure*}

% table:comparison
\begin{table}[]
\renewcommand{\arraystretch}{1.3}
    \centering
        \caption{FID and GS scores in comparison to ScrabbleGAN.
}
    \label{tab:comparison}
    \begin{tabular}{c|c|ll}
     \text { Dataset } & \text {Model} & \text { FID} $\downarrow$& \text { GS } $\downarrow$\\
     
\hline \hline\text {HWDB1.1}   
& ScrabbleGAN &   22.83 & $3.49\times 10^{-2}$ \\
& HCT-GAN	  &\textbf{15.14}  & \textbf{4.10 \times $10^{-4}}$ \\

\hline\text {HWDB2.2}          
& ScrabbleGAN	 & 32.04 & $1.39\times10^{-1}$  \\
& HCT-GAN	 & \textbf{17.81} & \textbf{3.33 \times $10^{-3}$} \\
\hline
    \end{tabular}
\end{table}


% table:gradient_balancing
\begin{table}[]
\renewcommand{\arraystretch}{1.1}
    \centering
    \caption{Gradient balancing study on HCT-GAN. $\alpha$ and $\beta$ correspond to SPM and SRM respectively}
   \begin{tabular}{l|ll|ll}

 Dataset  & $\alpha$ &$\beta $&\text{FID}$\downarrow$ &\text{ GS}$\downarrow$ \\
\hline \hline  
        & 1 & 0 & $16.23 $ & $6.20\times 10^{-3}$ \\
HWDB1.1 & 0 & 1 & $15.47 $ & $5.82\times             10^{-4}$ \\
        & 1 & 1 & \textbf{15.14}$&
        \bf{4.10 \times $10^{-4}$} \\
        &0.5&0.5&$21.71$ & $3.87 \times
        10^{-3}$ \\
\hline
        & 1 & 0 & $18.61$ & $4.10 \times
        10^{-3}$  \\
        & 0 & 1 & $20.48$ & $4.86 \times
        10^{-3}$  \\
HWDB2.2 & 1 & 1 & $31.24$ & $1.87 \times
        10^{-1}$  \\
        & 0.1 & 0.5 & $30.75$ & $1.09 \times
        10^{-1}$  \\
        & 0.5 & 0.1 & $20.35$ & $9.82 \times
        10^{-3}$ \\
        & 0.5 & 0.5 & \bf{17.82} & \bf{3.33 \times $10^{-3}$}  \\
\hline
\end{tabular}
    \label{tab:gradient_balancing}
\end{table}









% figue:style_interpolation
\begin{figure*}
\centering
\includegraphics[width=\linewidth]{style_interpolation.pdf}
\caption{ An interpolation between two different styles of handwriting generated by HCT-GAN. 
}
\label{fig:style_interpolation}
\end{figure*}


% figue:different_style_show
\begin{figure*}
\centering
\includegraphics[width=\linewidth]{different_style_show.pdf}
\caption{ Showing diverse handwriting. Left: The characters generated are:\begin{CJK}{UTF8}{gbsn} 跋袄肮扒\end{CJK}. Right: The text generated are:\begin{CJK}{UTF8}{gbsn}泛树织做只迁挺椒淅枳森哒达沃送玳代边代认做\end{CJK}. All characters are selected randomly, and each row represents one style.
}
\label{fig:different_style_show}
\end{figure*}



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

% table:Improving recognition performance
\begin{table}[]
\renewcommand{\arraystretch}{1.3}
    \caption{Extending the CICAS-HWDB dataset with generated images. Impact on the text recognition performance in terms of Edit Distance (ED),character Error Rate (WER) and accuracy(ACC) on the Test set.}
    \centering
    \begin{tabular}{c|cl|ll}
    Set & Aug & HCT-GAN &CER[\%] $\downarrow$& ED
    $\downarrow$\\
\hline \hline 
         & \times & \times & $31.21 \pm 0.13$ & $8.18$ \\
         &\checkmark & \times& $11.13 \pm 0.11$  & $2.94$ \\
HWDB2.2  &\times &20k& $23.17 \pm 0.21$ & $6.01$ \\
         &\times &40k& $20.12 \pm 0.15$ & $5.30$\\
         &\checkmark &40k& \bf{7.43}\pm \bf{0.17} & \bf{1.90}  \\
\hline
% \end{tabular}
%       \begin{tabular}{c|cl|ll}
    Set & Aug & HCT-GAN &  ACC[\%] $\uparrow$& \\
\hline \hline 
         & \times & \times & $85.65\pm 0.05$ & \\
         &\checkmark & \times& $91.49 \pm 0.22$  & \\
HWDB1.1  &\times &200k& $86.44 \pm 0.18$ &  \\
         &\times &400k& $88.74 \pm 0.31$ &  \\
         &\checkmark &400k& \bf{94.43} \pm \bf{0.19} & \\
          &\times &100k(non-sense) &\bf{90.90} \pm \bf{0.08} & \\
        &\checkmark &100k(non-sense)& \bf{94.50} \pm \bf{0.22} & \\
\hline
\end{tabular}
    \label{tab:recognition_performance}
\end{table}