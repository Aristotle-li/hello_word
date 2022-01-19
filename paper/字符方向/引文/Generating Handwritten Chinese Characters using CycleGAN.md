







The handwritten Chinese character recognition (HCCR) problem has been extensively studied for more than 40 years [1]-[7]. Nevertheless, HCCR is still a challenge unsolved problem owing to its large scale vocabulary (say, as many as 6763 classes in GB2312-80 standard, as large as 27533 in GB18010-2000 standard, or as huge as 70244 in GB180102005 standard), great diversity in handwriting styles (imaging there are more than 1 billion Chinese people), too many similar and confusable Chinese characters, and so on. It seems that traditional offline HCCR approaches, such as the modified quadratic discriminant function (MQDF) based methods, meet a bottleneck since no significant progress has been observed by us in recent years. The best traditional methods, such as MQDF or DLQDF [1][4][5], achieved a fairly good performance with less than 93% accuracy on a challenge offline HCCR database, CASIA-HWDB [1], leaving great margin with human performance.

手写汉字识别（HCCR）问题已经被广泛研究了40多年[1]-[7]。尽管如此，HCCR由于其庞大的词汇量（如GB2312-80标准中多达6763个类，GB18010-2000标准中多达27533个类，或GB180102005标准中多达70244个类），书写风格的多样性（中国人口超过10亿），仍然是一个挑战性的未解决问题，太多相似和易混淆的汉字，等等。传统的离线HCCR方法，如基于修正二次判别函数（MQDF）的方法，似乎遇到了瓶颈，因为近年来我们没有观察到重大进展。最好的传统方法，如MQDF或DLQDF[1][4][5]，在挑战离线HCCR数据库CASIA-HWDB[1]上取得了相当好的性能，准确率不到93%，在人为绩效方面留下了很大的差距





以前的大多数作品都依赖于简单笔画的层次表示[27,26,17]。他们将汉字分解成笔画，然后结合笔画来模仿个性化的书写风格。因此，这些方法只关注字符的局部表示，而不是整体样式，因此需要为每个新字符调整笔划的形状、大小和位置。相反，zi2zi[25]学习使用pix2pix[9]和成对字符图像作为训练数据来转换字体

。然而，在生成手写汉字的任务中，由于要求用户书写大量字符是不可行的，因此很难获得大量成对的训练示例。手写样本通常也与用户的书写分离，而不知道字符的真实标签。此外，即使是用户编写的同一个字符每次都会发生变化，这使得学习整体风格而不是模仿每个字符变得更加重要。因此，对于手写体中文生成问题，使用不成对的汉字而不是成对的数据更合适。图2提供了成对和非成对训练数据的示例。成对的训练数据包含两种字体中相同字符的图像，而未成对的数据不一定包含相同的字符集。



在这项工作中，我们将中文手写字符生成描述为一个从现有打印字体到个性化手写样式的映射问题。针对这一问题，我们进一步提出了一种基于非成对图像到图像的转换方法。我们的主要贡献是：•我们建议使用DenseNet CycleGAN生成个性化手写风格的汉字。•我们建议将内容准确性和风格差异作为评估标准，以评估生成字符的质量我们在CASIA数据集[16]和新引入的Lanting书法数据集上证明了我们提出的方法的有效性。

In this work, we formulate the Chinese handwritten character generation as a problem that learns a mapping from an existing printed font to a personalized handwritten style. We further propose a method based on unpaired image-toimage translation to solve this problem. Our main contributions are: • We propose to generate Chinese characters in a personalized handwritten style using DenseNet CycleGAN. • We propose content accuracy and style discrepancy as the evaluation metrics to assess the quality of the generated characters. • We demonstrate the efficacy of our proposed method on both CASIA dataset [16] and our newly introduced Lanting calligraphy dataset.



### Related work

从数字时代开始，汉字的生成就得到了研究[20]。

在文献中，汉字生成主要表现为艺术书法、排版生成问题[27,25,28,14]，或个人手写生成问题[26,15,22]。





（汉字生成主要表现为排版生成问题，或艺术书法和个人手写生成问题，目前两者的生成大都基于auto encode结构，其中以zi-to-zi和w-net为代表，对于单个字体的风格转换效果十分惊艳，但是这种结构若生成带有连体的手写文本行或者草书就会出现单张文本图像上多个字符encode编码混叠等现象，例如我和你encode后，是我和你三个字的表示，你和他encode后得到的是你和他三个字的表示，G可以很好地生成我和你，你和他。但是对于生成我和他就无能为力了，而字符组成文本的方式成千上万，不可能将所有组合都训练一个遍，我们的解决方式是使用传统GAN的结构，对文本进行字根级的编码，但是进行文本级的生成和监督。采用字根级的编码还有一个好处是对于书法或者手写汉字，样本不均衡，有些汉字的书法样本十分稀少，甚至只有两三个样本，汉字字符总量有两万多，而字根只有一千多，对于复合字而言，可以看做两个样本，增加了样本量，使得样本量少的书法字体达到很好的生成效果，不同于传统将汉字拆分组合的方式，这里仅简单的将汉字编码为左右或上下，这是基于GAN中内容鉴别器决定的。）





以前的大多数作品都以简单笔画的层次表示[27,26]为基础来表示汉字。例如，StrokeBank[32]将汉字分解为一个组件树。FlexiFont[19]扫描并处理相机捕获的手写字符图像，然后将这些字符格式化为个性化的TrueType字体文件。自动形状变形[14]首先为每个字符生成形状模板，然后将两个给定汉字分解为笔划，以在笔划之间建立精确的对应关系，从而实现非刚性点集配准。最近，awesome排版[28]探索了为排版生成特殊效果的问题，并利用文本效果空间分布高度规律性的统计数据来指导合成过程。Zi2zi[25]将每个汉字视为一个整体，并利用成对的训练数据学习字体之间的转换。然而，在生成个性化汉字手写字体的任务中，很难获得大量成对的训练样本。



### . Image style transfer

甘斯。通用对抗网络（GAN）[4]是一种强大的生成模型，在许多计算机视觉任务（如图像修复[7]和图像到图像翻译[31]）以及自然语言处理任务（如语音合成[18]和跨语言学习[11]）中取得了令人印象深刻的成果。GANs将生成性建模描述为两个竞争网络之间的博弈：给定一些输入噪声，生成器网络生成合成数据，而鉴别器网络区分生成器的输出和真实数据。形式上，生成器G和鉴别器D之间的博弈具有极大极小目标

GANs. General adversarial networks (GANs) [4] are powerful generative models which have achieved impressive results in many computer vision tasks such as image inpainting [7] and image-to-image translation [31], as well as natural language processing tasks such as speech synthesis [18] and cross-language learning [11]. GANs formulate generative modeling as a game between two competing networks: a generator network produces synthetic data given some input noise and a discriminator network distinguishes between the generator’s output and true data. Formally, the game between the generator G and the discriminator D has the minimax objective:



cGANs和pix2pix。与学习从随机噪声向量到输出图像的映射的GANs不同，条件GANs（CGAN）在附加信息的条件下学习从随机噪声向量到输出图像的映射。CGAN能够进行图像到图像的转换，因为它们可以对输入图像进行调节并生成相应的输出图像。Pix2pix[9]是一种使用CGAN的通用图像到图像转换算法。它可以在各种各样的问题上产生合理的结果。给定一个包含成对相关图像的训练集，pix2pix将学习如何将一种类型的图像转换为另一种类型的图像，反之亦然。

cGANs and pix2pix. Unlike GANs which learn a mapping from a random noise vector to an output image, conditional GANs (cGANs) learn a mapping from a random noise vector to an output image conditioning on additional information. cGANs are capable of image-to-image translation since they can condition on an input image and generate a corresponding output image. Pix2pix[9] is a generic image-to-image translation algorithm using cGANs. It can produce reasonable results on a wide variety of problems. Given a training set which contains pairs of related images, pix2pix learns how to convert an image of one type into an image of another type, or vice versa.



Zi2zi。Zi2zi[25]使用GAN以端到端的方式在字体之间转换汉字，假设没有笔划标签或任何其他通常难以获得的辅助信息。zi2zi的网络结构基于pix2pix，增加了多字体的类别嵌入。这使zi2zi能够使用一个经过训练的模型将字符转换为几种不同的字体。Zi2zi使用源字体和目标字体的成对汉字作为训练数据。然而，由于获取大量成对的训练样本用于个性化手写汉字生成是不切实际的，因此zi2zi不适用于我们的问题。基基莱根。循环一致性GANs（CycleGANs）在没有成对示例的情况下学习图像翻译[31]。相反，它在输入和输出图像之间按周期训练两个生成模型。除了对抗性损失外，周期一致性损失还用于防止两个生成模型相互矛盾。CycleGAN的默认生成器体系结构是ResNet[5]，而默认鉴别器体系结构是PatchGAN分类器[9]。

Zi2zi. Zi2zi [25] uses GAN to transform Chinese characters between fonts in an end-to-end fashion, assuming no stroke label or any other auxiliary information which is usually difficult to obtain. The network structure of zi2zi is based on pix2pix with the addition of category embedding for multiple fonts. This enables zi2zi to transform characters into several different fonts with one trained model. Zi2zi uses paired Chinese characters of the source font and the target font as the training data. However, since it is impractical to obtain a large set of paired training examples for personalized handwritten Chinese character generation, zi2zi is not applicable to our problem. CycleGAN. Cycle-consistent GANs (CycleGANs) learn the image translation without paired examples [31]. Instead, it trains two generative models cycle-wise between the input and output images. In addition to the adversarial losses, cycle consistency loss is used to prevent the two generative models from contradicting each other. The default generator architecture of CycleGAN is ResNet [5], while the default discriminator architecture is a PatchGAN classifier [9].