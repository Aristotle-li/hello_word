## 格式刷

## 6 EXPERIMENTS 

We rigorously tested our proposed TSB architecture and report qualitative and quantitative results, as well as a user study, comparing our results with previous work.

我们严格地测试了我们提出的TSB架构，并报告了定性和定量的结果，以及用户研究，将我们的结果与以前的工作进行了比较



### 6.1 Implementation details 

We use a fixed-size localized style image of size 256 × 256. The localized image is cropped out from the larger scene image along with its context while preserving the original aspect ratio of the word present inside it. We also assume the ground-truth rectangular word bounding box information available. The synthetically rendered content image is of dimension 64 × 256 for training, and during inference it uses a variable width image of dimension 64 × W . The style and content encoders, (Fs) and (Fc), respectively, (Sec. 3.1), use ResNet34 [35], following the format proposed by Zhanzhan et al. [47] for deep text features and the modification as explained in Sec. 3.1. We use StyleGAN2 [28] for our generator (Sec. 3.2). We base our generator on the StyleGAN2 variant with skip connections, a residual discriminator, and without progressive growing. We adapt this architecture to also produce foreground masks for the generated image as shown in the Fig. 2. We also modified the input dimensions to generate output images of size 64 × 256. The learned content representation ec ∈ R4×16 is given as input to the first layer of the generator. As explained in Sec. 3.2, we do not use noise inputs, instead conditioning the output on our latent style and content vectors. Our models were all trained end-to-end, with the exclusion of the pre-trained networks – the typeface classifier, C, of Sec. 4.1 and recognizer, R of Sec. 4.2 – which were kept frozen. We use the Adam optimizer with a fixed learning rate of 0.002 and batch size of 64. We empirically set the relative weights of the different loss functions as: λ1 = 1.0, λ2 = 500.0, λ3 = 1.0, λ4 = 1.0, λ5 = 10.0, λ6 = 1.0. Finally, our method is implemented using the PyTorch distributed framework [48]. Training was performed on 8GPUS with 16GB of RAM each.

我们使用尺寸为256 × 256的固定尺寸的本地化样式图像。局部图像从更大的场景图像中连同它的上下文一起裁剪出来，同时保留其内部单词的原始纵横比。我们还假设ground-truth矩形字包围框信息可用。综合渲染的内容图像用于训练，尺寸为64 × 256，推理时使用尺寸为64 × W的变宽图像。风格编码器(Fs)和内容编码器(Fc)分别使用ResNet34[35](第3.1节)，遵循zhan等人提出的格式，[47]用于深度文本特征和第3.1节中解释的修改。我们使用StyleGAN2[28]作为生成器(第3.2节)。我们将生成器建立在带有跳跃连接的StyleGAN2变体的基础上，它是一个残差鉴别器，并且没有渐进增长。我们采用这种架构来为生成的图像生成前景蒙版，如图2所示。我们还修改了输入尺寸，生成了64 × 256的输出图像。将学到的内容表示ec∈R4×16作为生成器第一层的输入。正如第3.2节所解释的，我们不使用噪声输入，而是根据我们的潜在风格和内容向量调节输出。我们的模型都是端到端的训练，排除了预先训练的网络-第4.1节的字体分类器C和第4.2节的识别器R -它们被冻结。我们使用Adam优化器，固定学习率为0.002，批量大小为64。实验确定不同损失函数的相对权重为:λ1 = 1.0， λ2 = 500.0， λ3 = 1.0， λ4 = 1.0， λ5 = 10.0， λ6 = 1.0。最后，我们的方法是使用PyTorch分布式框架[48]实现的。训练在8gpu上进行，每个gpu有16GB内存。



### 6.2 Datasets 

Our experiments use a variety of datasets, representing real and synthetic photos, scene text and handwriting. Below we list all the datasets used in this work. The sets and annotations collected as part of this work shall be publicly released, as well as any test splits used in our experiments. Synthetic data. We use three different synthetic datasets in this work. SynthText [49]: We use the SynthText in the wild dataset for training our architecture on synthetic data in a self-supervised manner. The synthetically-trained model is later used for purpose of performing an ablation study (Sec. 6.4) on our architecture. Synth-Paired dataset: This is the test synthetic dataset prepared for the ablation study (Sec. 6.4) using the pipeline presented by SRNet [8] where we can produce target (new content) style word images following the source style. Note that, the above two datasets and the synthetically TSB trained model is only used for the ablation study. All other experiments use real world datasets where the supervision of target style is unavailable. Synth-Font dataset: This is a separate word level synthetic set of around 250K word images, sampled from ∼2K different typefaces using again the pipeline presented by SRNet [8]. This separate set was used in pre-training our typeface classification network, C, as mentioned in Sec. 4.1. In addition to the synthetic sets, we used collections of real images. These sets are described below, with the exception of our own, Imgur5K, detailed in Sec. 5.

我们的实验使用各种数据集，代表真实的和合成的照片，场景文本和手写。下面我们列出了在这项工作中使用的所有数据集。作为这项工作的一部分收集的集合和注释将被公开发布，以及在我们的实验中使用的任何测试分割。合成数据。在这项工作中，我们使用三种不同的合成数据集。SynthText[49]:我们在野生数据集中使用SynthText，以自我监督的方式训练我们的架构处理合成数据。综合训练的模型后来用于对我们的建筑进行消融研究(第6.4节)。Synth-Paired dataset:这是为消融研究(第6.4节)准备的测试合成数据集，使用SRNet[8]提供的管道，在这里我们可以按照源风格生成目标(新内容)风格的单词图像。需要注意的是，上述两个数据集和TSB综合训练模型仅用于消融研究。所有其他的实验都使用真实世界的数据集，其中目标样式的监督是不可用的。Synth-Font数据集:这是一个单独的词级合成集，包含约250K个单词图像，再次使用SRNet[8]提供的管道从约2K种不同的字体中采样。这个独立的集合用于对我们的字体分类网络C进行预训练，如第4.1节所述。除了合成集，我们还使用了真实图像的集合。下面描述了这些集合，除了我们自己的Imgur5K，详细在第5节。

ICDAR 2013. [50] This set is part of the ICDAR 2013 Robust Reading Competition. Compared to other real datasets, ICDAR 2013 images are of higher resolution with prominent text. There are 848 and 1,095 word images in the original train and test sets. 

这套是ICDAR 2013年鲁棒阅读比赛的一部分。与其他真实数据集相比，ICDAR 2013图像具有更高的分辨率和突出的文本。原始列车和测试集分别有848和1095个单词的图像

ICDAR 2015. [51] Released as part of the ICDAR 2015 Robust Reading competition, this set was designed to be more challenging than ICDAR 2013. Most of the images in this set have low resolution and viewpoint irregularities (e.g., perspective distortions, curved text). 

作为ICDAR 2015稳健阅读竞赛的一部分，该系列的设计比ICDAR 2013更具挑战性。这个集合中的大多数图像都有低分辨率和不规则的视点(例如，透视扭曲，弯曲的文本)。

TextVQA. [52] The dataset was collected for the task of visual question answering (VQA) in images. It contains 28,408 scene photos sourced from the OpenImages set [53], with 21,953 images for training, 3,166 for validation, and 3,289 for testing. While the purpose of the dataset is VQA, it contains a large variety of scene text in different, challenging styles, more so than other sets. We annotated this set with word polygons and recognition labels. 

采集数据集用于图像的可视化问题回答(VQA)任务。它包含28408张来自OpenImages集合[53]的场景照片，其中21,953张用于训练，3,166张用于验证，3,289张用于测试。虽然该数据集的目的是VQA，但它包含大量不同的、具有挑战性的场景文本，比其他集合更甚。我们用单词多边形和识别标签注释这个集合。

IAM Handwriting Database. [46] This set contains 1,539 handwritten forms, written by 657 authors. IAM offers sentence-level labels, lines, and words. In our work, we only use word-level annotations. We use the official partition for writer independent text line recognition which splits forms into writer-exclusive training, validation, and testing sets. 

这一套包含了1539份手写表格，由657位作者撰写。IAM提供句子级别的标签、行和单词。在我们的工作中，我们只使用文字级别的注释。我们使用官方分区来进行独立于作者的文本行识别，它将表单划分为作者专用的训练集、验证集和测试集。

### 6.3 Evaluation measures 

We follow the same quantitative evaluation measures as previous scene text editing methods [8], [9]. We compare target style images with generated results using these measures: (1) Mean square error (MSE), the l2 distance between two images; (2) structural similarity index measure (SSIM) [54]; (3) Peak signalto-noise ratio (PSNR). Low MSE scores and high SSIM and PSNR scores are best. These metrics can only be used with a synthetic image set, where we can generate a corresponding target style image. 

To test on real photos, where we do not have a prediction of how new text would look with an example style, **we measure text recognition accuracy.** **It is computed as $$
A c c=\frac{1}{\# \text { test }}\left(\sum_{i} \mathbb{I}\left(R\left(\mathcal{O}_{s_{i}, c_{i}}\right)==c_{i}\right)\right)
$$**(6) or, **the number of times a predicted string is identical to the actual string ci.** To this end, we use our pre-trained text recognition network, R, (Sec. 4.2). To evaluate real handwritten images, we use methods from the GAN literature, also adopted by previous handwritten text generation methods [13], [14], [34]: (**1) Fr` echet Inception Distance (FID**) [55] **calculates the distance between the feature vectors of real and generated images, and,** (2), **the Geometric Score (GS) [56], which compares geometrical properties between the real and generated manifolds.** **Similar to others [14], we evaluate GS on a reduced set, due to its computational costs, randomly sampling 5K images in both real and generated sets.** **We resized the images to a fixed size (64×256) for GS computation.** **FID was computed on sampled real and generated images the size of the test sets using variable-width images (299×W ) following the protocol presented in [1**4]. **Lower scores on both metrics are better.** Finally, we report results of a user study which compares the visual quality of our results and those of previous works. Details of this study are discussed in Sec. 6.5

我们遵循与之前的场景文本编辑方法[8]，[9]相同的定量评估措施。我们将目标风格的图像与生成的结果进行比较，使用以下措施:(1)均方误差(MSE)，即两幅图像之间的l2距离;(2)结构相似度指标(SSIM) [54];(3)峰值信噪比。MSE分数低，SSIM和PSNR分数高的效果最好。这些指标只能用于合成图像集，其中我们可以生成相应的目标样式图像。

为了在真实的照片上测试，我们无法预测新文本在示例样式下会是什么样子，我们测量文本识别的准确性。它被计算为Acc = 1 #test(∑i i (R(Osi,ci) == ci))，(6)或，预测字符串与实际字符串ci相同的次数。为此，我们使用预先训练好的文本识别网络R(第4.2节)。为了评价真实的手写图像，我们使用GAN文献中的方法，也采用了之前的手写文本生成方法[13]，[14]，[34]:(1) Fr ' echet Inception Distance (FID)[55]计算真实图像和生成图像的特征向量之间的距离，(2)几何评分(GS)[56]，比较真实流形和生成流形之间的几何属性。与其他的[14]相似，我们在一个减少的集合上评估GS，由于它的计算成本，在真实和生成集合中随机采样5K图像。我们将图像大小调整为固定大小(64×256)进行GS计算。根据[14]中的协议，在采样的真实图像和生成的图像上计算使用可变宽度图像(299×W)的测试集的大小。在这两个指标上得分越低越好。最后，我们报告了一个用户研究的结果，比较了我们的结果的视觉质量和那些先前的工作。这项研究的细节将在第6.5节中讨论





![image-20211123111451708](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123111451708.png)

Word-level style transfer results. Each image pair shows input source style on the left and output with novel content (string) on the right. All examples are real photos (no synthetic data) taken from ICDAR13 [50], TextVQA [52], IAM handwriting [46], and the Imgur5K set collected for this work. Note that the handwritten results are of variable widths, but resized to fixed dimension in this figure for better visualization. See supplemental for more results.

字级样式转换结果。每个图像对在左边显示输入源样式，在右边显示新的内容(字符串)。所有的例子都是真实的照片(没有合成数据)，从ICDAR13 [50]， TextVQA [52]， IAM手写[46]，以及为这项工作收集的Imgur5K集。请注意，手写结果的宽度是可变的，但为了更好地可视化，在此图中将大小调整为固定尺寸。更多结果见补充。



![image-20211123111515776](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123111515776.png)

TABLE 2: Ablation study comparing the influence of different loss functions on our TSB results.

消融研究比较了不同损失函数对TSB结果的影响。



![image-20211123111533262](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123111533262.png)

TABLE 3: Text recognition accuracy on images from three datasets. Real is provided as the baseline recognition of R on the real photos. * Denotes results reported by SWAPText [9] on an undisclosed subset of ICDAR 13. Since they did not release code, their numbers are not comparable with others and provided here only for reference.

文本识别精度图像从三个数据集。Real作为真实照片上R的基线识别。*表示SWAPText[9]在ICDAR 13的一个未公开子集上报告的结果。因为他们没有发布代码，所以他们的数据不能与其他数据进行比较，这里只提供参考。



### 6.4 Text generation results Ablation study. 文本生成结果消融研究。

Table 2 provides an ablation study evaluating the effects of the different loss functions, scale of style features and the role of having masks while training TSB (Sec. 4). The first three quantitative metrics (MSE, SSIM and PSNR) focus on pixels level differences and may not reflect the true perceived visual quality. Hence, we also add the FID metric which is adopted in many GAN based generative models. The first setting (`D) mimics the original style GAN training with the difference that the noise inputs are replaced with conditional style and content vectors. Although the images are produced in a realistic manner, it performs worst in terms of metrics since there are no losses which captures the respective content and style. Using `D + `R while training also produces realistic results but the output style is still inconsistent with the input style. Only when we add the reconstruction loss, `rec, do the input and output styles became consistent. We improve results even more by adding the cyclic reconstruction loss, `cyc, which contributed to even better consistency between source and target styles. Note that, the above settings used the multi-scale (M) style representation and utilizes the foreground mask while training.

表2提供了一个烧蚀研究评估不同的损失函数的影响,风格特色和规模的角色面具在训练TSB(秒。4)。前三个量化指标(MSE, SSIM和PSNR)专注于像素水平差异,可能不能反映真实的视觉感知质量。因此，我们还加入了许多基于GAN的生成模型中采用的FID指标。第一个设置(' D)模拟了原始风格GAN训练，不同的是，噪声输入被条件风格和内容向量替换。尽管图像是以一种真实的方式产生的，但它在度量方面表现最差，因为没有损失捕获各自的内容和风格。在训练中使用“D +”R也能产生现实的结果，但输出风格仍然与输入风格不一致。只有当我们添加重建损失，' rec，输入和输出风格变得一致。通过添加循环重建损失，我们进一步改进了结果，这有助于源和目标风格之间更好的一致性。注意，上面的设置使用了多尺度(M)样式表示，并在训练时利用了前景蒙版。

The last three rows compares the model trained with the perceptual loss, `type in addition to the other loss functions presented before. Here we notice a change in performance where the FID metric starts improving while there is a drop in other metrics such as MSE, SSIM and PSNR. This behaviour is because the latter metrics are biased towards preferring output images which blurry or smooth and penalize the sharp images. We asserted this by performing Gaussian filtering on the output images and noticed the drop in performance across these metrics. Here we consider the FID metric to be more reliable and consistent with human visual inspection. Our best results in terms of FID metric (last row), are produced when we have full text-specific perceptual loss, `type, of Sec. 4.1 with multi-scale style feature and trained using foreground masks. The perceptual and texture losses, `per, `tex, respectively, constrain the activations of the initial layers and the stationary properties of the generated image, making them consistent with input style image. The embedding loss, `emb, penalises the results based on typeface-level features.

最后三行比较了经过训练的模型与知觉损失，“类型以及前面提到的其他损失功能。”在这里，我们注意到性能的变化，FID指标开始改善，而其他指标，如MSE、SSIM和PSNR下降。这种行为是因为后一指标偏向于更倾向于输出图像模糊或平滑，并惩罚锐利的图像。我们通过对输出图像执行高斯滤波来断言这一点，并注意到这些指标的性能下降。这里，我们认为FID指标更可靠，更符合人眼目测。我们在FID指标(最后一行)方面的最佳结果，是在我们拥有第4.1节的完整文本特异性感知损失类型时产生的，该类型具有多尺度风格特征，并使用前景蒙版进行训练。感知和纹理损失，“per，”tex，分别约束初始层的激活和生成图像的静态属性，使它们与输入风格的图像一致。嵌入损失，' emb，惩罚结果基于字体级别的特征。



##### Recognition accuracy on real photos.

 Following others, we compare machine recognition accuracy on generated images. Accuracy is measured using Eq. (6) and we compare with SRNet [8] and SWAPText [9]. For these tests, we did not fine-tune our model on the ICDAR 2013 and ICDAR 2015 sets since these provide very few training images. Importantly, the test splits used by SWAPText [9] and SRNet [8] were not disclosed. Furthermore, SWAPText did not share their code. We consequently provide SWAPText numbers from their paper, only for reference, although they are not directly comparable with ours. We use a third party implementation [58] of SRNet along with a pre-trained model trained in a supervised setting for comparing it with our TSB model on the same test images. Table 3 reports these recognition results. The first row provides baseline accuracy of our recognizer, R (Sec. 4.2), on the original photos. This accuracy is computed by comparing its output with the human labels available for these sets. Evidently, the recognition engine is far from optimal, yet despite this, serves very well in training our model. Our method clearly generates images with better recognizable text compared with images generated by SRNet [8] and SWAPText [9]. Qualitative results. Fig. 4 presents word-level qualitative samples generated by our TSB. We show both the source (input) style box

![image-20211123111656594](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123111656594.png)

and our generated results with new content. Our method clearly captures the desired source style, from just a single sample. To our knowledge, this is the first time one-shot text style transfer is demonstrated for both scene text and handwriting domain. Fig. 5, presents scene text editing results from ICDAR13 [50] and TextVQA [52] datasets. The left image is the original scene image along with words marked for replacement (shown in blue bounding boxes) and the right one is edited image using new content by the TSB. We demonstrate these results by selectively stitching back the generated word image (foreground and background separately using masks) back to the source bounding box using Poisson blending. See supplemental for more details.

接下来，我们比较了机器识别生成图像的准确性。精度通过Eq.(6)进行测量，并与SRNet[8]和SWAPText[9]进行比较。对于这些测试，我们没有对ICDAR 2013和ICDAR 2015集的模型进行微调，因为它们提供的训练图像很少。重要的是，SWAPText[9]和SRNet[8]使用的测试拆分没有公开。此外，SWAPText不共享它们的代码。因此，我们提供了他们论文中的SWAPText编号，仅供参考，尽管它们不能与我们的直接比较。我们使用SRNet的第三方实现[58]，以及在监督设置中训练的预训练模型，将其与我们的TSB模型在相同的测试图像上进行比较。表3报告了这些识别结果。第一行提供了我们的识别器R(第4.2节)对原始照片的基线精度。这种精度是通过将其输出与这些集合中可用的人工标签进行比较来计算的。显然，识别引擎远不是最优的，但尽管如此，在训练我们的模型非常好。与SRNet[8]和SWAPText[9]生成的图像相比，我们的方法生成的图像具有更好的文本可识别性。定性的结果。图4给出了TSB生成的词级定性样本。我们同时显示源(输入)样式框和带有新内容的生成结果。我们的方法从一个样本中清楚地捕获了所需的源样式。据我们所知，这是第一次为场景文本和手写领域演示一次性文本样式转换。图5为ICDAR13[50]和TextVQA[52]数据集的场景文本编辑结果。左边的图像是原始场景图像，以及标记为替换的单词(在蓝色边框中显示)，右边的图像是TSB使用新内容编辑的图像。我们通过选择性地使用泊松混合将生成的单词图像(前景和背景分别使用掩码)拼接回源边界框来演示这些结果。更多细节见补充。

##### Offline handwritten generation results. 

**Table 4, provides quantitative comparisons of generated handwritten texts, comparing our method with Davis et al**. [14] the recent SotA method designed specifically for generating handwritten text. Since they did not report FID/GS metrics for IAM dataset at word level, we took their official implementation (available from the author’s Github repository) along with the pre-trained model for validating these results. **We report FID scores** where the lower the values better the generation quality. Evidently, here too, our method outperforms the previous work. Qualitative examples of handwritten output are provided in Fig. 4. We emphasize again, that handwritten styles are learned in a one-shot manner, from the single word example provided for the sourced style while previous methods such as Davis et al. [14] uses a much larger image (a sentence or two) to extract the style representation.

表4提供了生成手写文本的定量比较，将我们的方法与Davis等人的[14](最近专为生成手写文本而设计的SotA方法)进行了比较。由于他们没有在单词级别上报告IAM数据集的FID/GS指标，我们采用了他们的官方实现(可从作者的Github存储库获得)和预先训练的模型来验证这些结果。我们报告FID评分，值越低，生成质量越好。显然，在这里，我们的方法也优于前面的工作。图4给出了手写输出的定性例子。我们再次强调，手写体样式是通过一种一次性的方式来学习的，从为源样式提供的单个单词示例中学习，而先前的方法，如Davis等人的[14]使用更大的图像(一两个句子)来提取样式表示。



![image-20211123111748159](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123111748159.png)

TABLE 5: User study. The user study shows clear margin in favor of our results, across scene text and handwriting domains. See Sec. 6.5 for details.





##### 6.5 User study results 、

Text styles and aesthetics are ultimately qualitative concepts, often designed to appeal to human aesthetics. Recognizing that, we also evaluate generated images in a user study, comparing our results with those of SotA methods. Specifically, we tested the quality of scene text generation methods by randomly sampling 60 word images from the ICDAR 2013 set, and used them as source styles for comparing our method with SRNet [8]. We compared the quality of generated handwritten word images by sampling 60 images from different authors present in the IAM dataset. These were used as style source images when comparing our results with those produced by Davis et al. [14]. These style images were line instances for Davis et al. [14] since the original method showed results using lines as style examples, whereas our TSB is applied at the word level. We used just a single word instance from the same author, as the style example. The target generation was set at word-level. We asked eight participants to rate generated images in two separate tasks. In the first task, we presented participants with randomlyordered real and generated images, from our method and its baselines (SRNet [8] for scene text; Davis et al. [14] for handwriting). Users were asked to classify images as either Original (sampled from real datasets) or Generated. We present our findings in Table 5. Numbers are the percent of times users considered an image to be real. Rows labeled as Real report the frequency of correctly identifying real photos. In both scene text and handwriting, users misidentified our results with genuine images nearly twice as often as the images generated by our baselines, testifying to the improved photo-realism of our results. The second task presented a source style image alongside photos with the same style but new content, generated by our method and SRNet [8] for scene text, and Davis et al. [14] for handwriting. Participants were asked to select the more realistic generated photo. The results, reported in the last row of Table 5, show that the participants preferred our method over SRNet, 77.70% of the times and over Davis et al. [14], 77.65% of the times. These results again testify to the heightened photo-realism of our method compared to existing work.

文本风格和美学最终是定性的概念，通常是为了迎合人类美学而设计的。认识到这一点，我们也在用户研究中评估生成的图像，将我们的结果与SotA方法的结果进行比较。具体来说，我们从ICDAR 2013集中随机抽取60幅单词图像，测试场景文本生成方法的质量，并将其作为源样式与SRNet[8]进行比较。通过对IAM数据集中来自不同作者的60幅图像进行采样，我们比较了生成的手写文字图像的质量。当将我们的结果与Davis等人制作的结果进行比较时，这些图像被用作样式源图像。这些样式图像是Davis等人的线条实例，因为原始方法显示的结果使用线条作为样式示例，而我们的TSB是应用在单词层面。我们只使用来自同一作者的单个单词实例作为样式示例。目标代被设置为文字级。我们让8名参与者在两个独立的任务中对生成的图像进行评分。在第一个任务中，我们向参与者展示了随机排序的真实图像和生成的图像，这些图像来自我们的方法及其基线(SRNet[8]表示场景文本;戴维斯等人的[14]笔迹)。用户被要求将图像分为原始图像(从真实数据集中采样)和生成图像。我们在表5中展示了我们的发现。数字是用户认为图像是真实的百分比。标记为真实的行报告正确识别真实照片的频率。无论是场景文本还是手写，用户误认真实图像的频率几乎是基线生成的图像的两倍，这证明了我们结果的照片真实感得到了改善。第二个任务展示了一个源风格的图像和具有相同风格但新内容的照片，由我们的方法和SRNet[8]生成的场景文本，和Davis等人生成的[14]用于手写。参与者被要求选择更真实的生成的照片。表5最后一行报告的结果显示，与SRNet相比，77.70%的参与者更喜欢我们的方法，77.65%的参与者更喜欢Davis等人的[14]。这些结果再次证明，与现有的工作相比，我们的方法提高了照片的真实感。

![image-20211123111854720](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123111854720.png)

Fig. 6: Limitations of our approach. Qualitative example scene text (top) and handwritten (bottom) failures. These failures are due to local style variations (e.g., colors varying among characters), metallic colors which were not well represented in the training, and uniquely complex calligraphy for handwriting.

我们方法的局限性。定性示例场景文本(上)和手写(下)失败。这些失败的原因是由于地方风格的变化(如不同字符的颜色不同)，在训练中没有很好的体现金属颜色，以及书法独特的复杂。



#### 7 LIMITATIONS OF OUR APPROACH 

Fig. 6 offers sample generation failures in scene text (top) and handwriting domains (bottom). Most of these examples share complex styles where: (a) The target foreground font color/style is inconsistent with the input style image, (b) Cases where the generation is not photo-realistic. Some of these complex scenario includes the text is written in metallic objects, different colors for different characters, etc. Regardless, in all these cases, our method managed to correctly generate the target content. In case of handwriting, there are few instances where some of the characters in the target content are blurred or not generated in a realistic manner. This happens mostly in scenarios when the source style image is too short (< 3 characters). Row 4 & 5, Col. 1 presents failure cases where the input style is a very complex form of calligraphy. The other failure examples shown in the figure belong to the scenarios where the system didn’t capture the cursive property of the input style or got the shear incorrectly. In our current set-up we used a pre-trained text recognizer which is trained only on scene text images. We believe that many such issues could be mitigated if we pre-trained model from handwritten domain itself. 

8 CONCLUSION We present a method for disentangling real photos of text, extracting an opaque representation of all appearance aspects of the text and allowing transfer of this appearance to new content strings. To our knowledge, our method claims a number of novel capabilities. Most notably, our TSB requires only a single source style example: Style transfer is one-shot. Our disentanglement approach is further trained in a self-supervised manner, allowing the use of real photos for training, without style labels. Finally, unlike previous work we show synthetically-generated results on both scene text and handwritten text whereas existing methods were tailored to one domain or the other, but not both. Our method aims at use cases involving creative self expression and augmented reality (e.g., photo-realistic translation, leveraging multi-lingual OCR technologies [59]). Our method can be used for data generation and augmentation for training future OCR systems, as successfully done by others [49], [60] and in other domains [61], [62]. We are aware, however, that like other technologies, ours can be misused, possibly the same as deepfake faces can be used for misinformation. We see it as imperative that the abilities we describe are published, to facilitate research into detecting such misuse, e.g., by moving beyond fake faces to text, in benchmarks such as FaceForensics++ [63] and the Deepfake Detection Challenge (DFDC) [64]. Our method can also be used to create training data for detecting fake text from images. Finally, we hope our work will encourage regulators and educators to address the inexorable rise of deepfake technologies.





## Adversarial Generation of Handwritten Text Images Conditioned on Sequences



III. RESULTS 

### A. Experimental setup 

In our experiments, we use 128 × 512 images of handwritten words obtained with the following preprocessing: we isometrically resize the images to a height of 128 pixels, then remove the images of width greater than 512 pixels and finally, pad them with white to reach a width of 512 pixels for all the images (right-padding for French, left-padding for Arabic). Table I summarizes the meaningful characteristics of the two datasets we work with, namely the RIMES [15] and the OpenHaRT [24] datasets, while Fig. 4 shows some images from these two datasets

在我们的实验中，我们使用128 × 512幅手写体文字图像，经过以下预处理得到:我们将图像的高度等距调整为128像素，然后删除宽度大于512像素的图像，最后，用白色填充它们，使所有图像的宽度达到512像素(右填充为法语，左填充为阿拉伯语)。表1总结了我们使用的两个数据集，即RIMES[15]和OpenHaRT[24]数据集的有意义特征，而图4显示了这两个数据集的一些图像

![image-20211123112043880](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123112043880.png)

To reflect the distribution found in natural language, the words to be generated are sampled from a large list of words (French Wikipedia for French, OpenHaRT for Arabic). **For the text recognition experiments on the RIMES dataset (Section III-D), we use a separate validation dataset of 6060 images**. **We evaluate the performance with Fr ́ echet Inception Distance (FID) [25] and Geometry Score (GS) [26]. FID is widely used and gives a distance between real and generated data. GS compares the topology of the underlying real and generated manifolds and provides a way to measure the mode collapse.** **For these two indicators, lower is better.** In general, we observed that FID correlates with visual impression better than GS. **For each experiment, we computed FID (with 25k real and 25k generated images) and GS (with 5k real and 5k generated images, 100 repetitions and default settings for the other parameters) every 10000 iterations and trained the system with different random seeds.** **We then chose independently the best FID and the best GS among the different runs**. **To verify the textual content, we relied on visual inspection. To measure the impact of data augmentation on the text recognition performance, we used Levenshtein distance at the character level (Edit Distance) and Word Error Rate**.

为了反映在自然语言中发现的分布，要生成的单词是从一个很大的单词列表中取样(法语Wikipedia代表法语，OpenHaRT代表阿拉伯语)。在RIMES数据集上进行文本识别实验(Section III-D)，我们使用单独的6060张图像验证数据集。我们用Fŕechet Inception Distance (FID)[25]和Geometry Score (GS)[26]来评估性能。FID被广泛应用，它给出了真实数据和生成数据之间的距离。GS比较底层实流形和生成流形的拓扑结构，并提供了一种测量模态折叠的方法。对于这两个指标，越低越好。总的来说，我们观察到FID与视觉印象的相关性比GS更好。对于每个实验，我们每10000次迭代计算FID (25k真实图像和25k生成图像)和GS (5k真实图像和5k生成图像，100次重复，其他参数的默认设置)，并用不同的随机种子训练系统。然后，我们在不同的运行中独立选择最好的FID和最好的GS。为了验证文本内容，我们依靠视觉检查。为了测量数据增强对文本识别性能的影响，我们使用了字符级别的Levenshtein距离(编辑距离)和单词错误率。

###  B. Ablation study 

For all the experiments in this section, we used the RIMES database described in Section III-A. 

B.消融研究对于本部分的所有实验，我们使用了section III-A中描述的RIMES数据库。

1. Gradient balancing: When training the networks (G, φ), the norms of the gradients coming from D and R may differ by several orders of magnitudes. As mentioned in Section II, we found it useful to balance these two gradients. Table II reports FID, GS and a generated image for different gradient balancing settings. 

   梯度平衡:在训练网络(G， φ)时，来自D和R的梯度的规范可能相差几个数量级。正如在第二节中提到的，我们发现平衡这两个梯度是很有用的。表II报告了不同梯度平衡设置的FID、GS和生成的图像

![image-20211123112110814](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123112110814.png)

Without gradient balancing, we observed that ||∇R||2 was typically 102 to 103 times greater than ||∇D||2, meaning that the learning signal for (G, φ) is biased toward satisfying R. As a result, the word “r ́ eparer” is clearly readable, but the FID is high (141.35) and the generated image is not realistic (the background is noisy, the letters are too far apart). With α = 0.1, ||∇R||2 is much smaller than ||∇D||2, meaning that G and φ take little account of the auxiliary recognition task. As illustrated by the second image in Table II, we lose control of the textual content of the generated image. FID is better than before, but still high (72.93). In a way, the generated image is quite realistic, since the background is whiter and the writing more cursive. On the contrary, when setting α to 10, G and φ mostly learn from the feedback of R and the generation is thus successfully conditioned on the textual content. In fact, we can distinguish the letters of “r ́ eparer” in the third generated image in Table II. However, as we are focusing on optimizing the generation process to have a minimal CTC cost, we observe strong visual artifacts that remind of the one obtained by Deep Dream generators [27]. FID is much higher (222.47) and the resulting images are very noisy, as demonstrated by the third image in Table II. The best compromise corresponds to α = 1. We obtain the best FID of 23.94 and GS of 8.58 × 10−4, while the generated image is both readable and realistic. For all other experiments, we set α to 1.

没有梯度平衡,我们观察到,| |∇R | 2通常是102到103倍| |∇D | 2,这意味着学习信号(G,φ)是偏向满足R .因此,这个词“Ŕeparer”显然是可读的,但支撑材高(141.35)和生成的图像是不现实的(背景噪声,字母之间的距离太远)。α = 0.1时，||∇R| 2比||∇D| 2小得多，这意味着G和φ几乎不考虑辅助识别任务。如表II中的第二个图像所示，我们失去了对生成图像的文本内容的控制。FID较前有所改善，但仍较高(72.93)。从某种程度上说，生成的图像相当真实，因为背景更白，文字更草书。相反，当α设置为10时，G和φ主要是从R的反馈中学习，从而成功地以文本内容为条件进行生成。事实上，我们可以在表II中生成的第三张图像中区分出“ŕeparer”的字母。然而，当我们专注于优化生成过程以拥有最小的CTC成本时，我们观察到强烈的视觉伪影，这让我们想起深梦生成器[27]所获得的。FID要高得多(222.47)，得到的图像非常嘈杂，如表II中的第三张图像所示。最好的折中对应于α = 1。得到了最佳FID值为23.94,GS值为8.58 × 10−4，生成的图像具有较高的可读性和真实感。对于所有其他的实验，我们将α设为1。

2) Adversarial loss: Using the network architecture described in Section II, we test three different adversarial training procedures: the “vanilla” GAN [8] (GAN), the Least Squares GAN [28] (LSGAN) and the Geometric GAN [11], [16], [18], used in our model. FID and GS are reported in Table III.

   对抗损失:使用第二部分描述的网络架构，我们测试了三种不同的对抗训练程序:“普通”GAN [8] (GAN)，最小二乘GAN [28] (LSGAN)和几何GAN[11]，[16]，[18]，在我们的模型中使用。FID和GS见表III。
   
   ![image-20211123112213612](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123112213612.png)

As shown in Table III, Geometric GAN leads to the best performance in terms of FID and GS. LSGAN fails to produce text-like outputs in three out of five trials. The low FID for vanilla GAN indicates that it produces realistic images. The high GS in Table III shows that both GAN and LSGAN suffer from a style collapse, and we observed that the textual content was not controlled. The trends given by FID and GS have been successfully confirmed by visual inspection of the generated samples. 

3) Self-attention: We use a self-attention layer [18], in both the generator and the discriminator, as it may help to keep coherence across the full image. We trained our model with and without this module to measure its impact.

如表三所示，几何GAN在FID和GS方面的性能最好。在五分之三的试验中，LSGAN无法产生类似文本的输出。香草甘的低FID表示它生成真实的图像。表III中的高GS表明，GAN和LSGAN都遭受了风格崩溃，我们观察到文本内容没有得到控制。FID和GS给出的趋势已通过对生成样品的目视检查成功确认。3） 自我注意：我们在生成器和鉴别器中都使用了自我注意层[18]，因为它可能有助于保持整个图像的一致性。我们对模型进行了培训，无论是否使用该模块，以测量其影响。

![image-20211123112237945](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123112237945.png)

Without self-attention, we still obtain realistic samples with correct textual content, but using self-attention improves performance both in terms of FID and GS, as shown in Table IV. 

4) Conditional Batch Normalization: As described in Section II, G is provided a noise chunk and φ(s) through each CBN layer. Another reasonable option, closer to [10], is to concatenate the whole noise z with φ(s), and feed it to the first linear layer of G (in this scenario, CBN is replaced with standard Batch Normalization). Table V reports FID and GS for these two solutions.

在没有自我注意的情况下，我们仍然可以获得具有正确文本内容的真实样本，但使用自我注意可以提高FID和GS的性能，如表IV所示。

4）条件批量归一化：如第二节所述，G通过每个CBN层提供一个噪声块和φ（s）。另一个更接近于[10]的合理选择是将整个噪声z与φ（s）连接起来，并将其馈送到G的第一个线性层（在这种情况下，CBN被标准批量归一化所取代）。表V报告了这两种解决方案的FID和GS。

![image-20211123112300024](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123112300024.png)

FID and GS in Table V indicates that feeding the generator inputs through CBN layers improves realism and reduces mode collapse. The visual inspection of the generated samples confirmed these trends and showed that the other solution prevents from correctly conditioning on the textual content.

表V中的FID和GS表明，通过CBN层供给发生器输入提高了真实感，减少了模态坍塌。对生成的样本进行的目视检查证实了这些趋势，并表明其他解决方案阻止了对文本内容的正确处理。

![image-20211123112353244](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123112353244.png)

C. Generation of handwritten text images 

We trained the model detailed in Section II on the two datasets described in Section III-A, RIMES and OpenHaRT. Fig. 5 and Fig. 6 display some randomly generated (not cherry-picked) samples in French and Arabic respectively. For these two languages, we observe that our model is able to produce images of cursive handwriting, successfully conditioned on variable-length words (even if some words remain barely readable, e.g. le and olibrius in Fig. 5). The typography of the individual characters is varied, but we can detect a slight collapse of writing style among the images. For French, as we trained the generator to produce words from all Wikipedia, we are able to successfully synthesize words that are not present in the training dataset. In Fig. 5 for instance, the words olibrius, inventif, ionique, gorille and ski are not in RIMES, while Dimanche, bonjour, malade and golf appear in the corpus but with a different case.

我们在第II节中详述的模型上训练了第III-A节中描述的两个数据集，RIMES和openenhart。图5和图6分别显示了一些随机生成的(非挑选的)法语和阿拉伯语样本。对于这两种语言，我们观察到，我们的模型能够产生草书的图像，成功地以可变长度的单词为条件(即使一些单词仍然难以读懂，例如图5中的le和olibrius)。但我们可以在这些图像中发现轻微的写作风格崩溃。对于法语，当我们训练生成器生成来自所有维基百科的单词时，我们能够成功地合成训练数据集中不存在的单词。例如，在图5中，单词olibrius, inventif, ionique, gorille和ski不在RIMES中，而Dimanche, bonjour, malade和golf出现在语料库中，但情况不同。

D. Data augmentation

for handwritten text recognition We aim at evaluating the benefits of generated data to train a model for handwritten text recognition. To this end, we trained from scratch a Gated Convolutional Network [2] (identical to the network R described in Section II-B) with the CTC loss, RMSprop optimizer [29] and a learning rate of 10−4. We used the validation data described in III-A for early stopping

我们的目标是评估生成的数据的好处来训练一个手写文本识别模型。为此，我们从零开始训练了一个带CTC损耗、RMSprop优化器[29]和10−4学习率的门选卷积网络[2](与第二节- b中描述的网络R相同)。我们使用III-A中描述的验证数据进行早期停



![image-20211123112409609](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123112409609.png)

Table VI: Extending the RIMES dataset with 100k generated images. Impact on the text recognition performance in terms of Edit Distance (ED) and Word Error Rate (WER) on the validation set.

使用生成的100k图像扩展RIMES数据集。编辑距离(ED)和单词错误率(WER)对验证集文本识别性能的影响。



Table VI shows that extending the RIMES dataset with data generated with our adversarial model brings a slight improvement in terms of Edit Distance and Word Error Rate. Note that using only GAN-made synthetic images for training the text recognition model does not yield competitive results. 

IV. CONCLUSION 

We presented an adversarial model to produce synthetic images of handwritten word images, conditioned on the sequence of characters to render. Beyond the classical use of a generator and a discriminator to create plausible images, we employ recurrent layers to embed the word to condition on, and add an auxiliary recognition network in order to generate an image with legible text. Another crucial component of our model lies in balancing the gradients coming from the discriminator and from the recognizer when training the generator. We obtained realistic word images in both French and Arabic. Our experiments showed a slight reduction in error rate for the French model trained on combined data. An immediate continuation of our experiments would be to train the described model on more challenging datasets, with textured background for instance. Furthermore, deeper investigation to reduce the observed phenomenon of style collapse would be a significant improvement. Another important line of work is to extend this system to the generation of line images of varying size.

## scrabbleGAN



### 4. Results 4.1. Implementation details

Without loss of generality, the architecture is designed to generate and process images with fixed height of 32 pixels, in addition, the receptive field width of G is set to 16 pixels. As mentioned in Section 3.1, the generator network G has a filter bank F as large as the alphabet, for example, F = { fa, fb, . . . , fz} for lowercase English. Each filter has a size of 32 × 8192. To generate one n-character word, we select and concatenate n of these filters (including repetitions, as with the letter ‘e’ in Figure 2), multiplying them with a 32 dimensional noise vector z1, resulting in an n × 8192 matrix. Next, the latter matrix is reshaped into a 512 × 4 × 4n tensor, i.e. at this point, each character has a spatial size of 4 × 4. The latter tensor is fed into three residual blocks which upsample the spatial resolution, create the aforementioned receptive field overlap, and lead to the final image size of 32 × 16n. Conditional Instance Normalization layers [9] are used to modulate the residual blocks using three additional 32 dimensional noise vectors, z2, z3 and z4. Finally, a convolutional layer with a tanh activation is used to output the final image. The discriminator network D is inspired by BigGAN [6]: 4 residual blocks followed by a linear layer with one output. To cope with varying width image generation, D is also fully convolutional, essentially working on horizontally overlapping image patches. The final prediction is the average of the patch predictions, which is fed into a GAN hinge-loss [28]. The recognition network R is inspired by CRNN [35]. The convolutional part of the network contains six convolutional layers and five pooling layers, all with ReLU activation. Finally, a linear layer is used to output class scores for each window, which is compared to the ground truth annotation using the connectionist temporal classification (CTC) loss [13]. Our experiments are run on a machine with one V100 GPU and 16GB of RAM. For more details on the architecture, the reader is referred to the supplemental materials.

在不丧失一般性的前提下，设计架构生成和处理固定高度为32像素的图像，G的接受域宽度设置为16像素。如3.1节所述，发生器网络G有一个与字母大小相同的滤波器组F，例如F = {fa, fb，…， fz}表示小写英语。每个过滤器的尺寸为32 × 8192。为了生成一个n个字符的单词，我们选择并连接这些滤波器的n个(包括重复，如图2中的字母“e”)，将它们与32维噪声向量z1相乘，得到一个n × 8192矩阵。然后将后一个矩阵重塑为512 × 4 × 4n张量，即此时每个字符的空间大小为4 × 4。后一个张量被送入三个剩余的块中，对空间分辨率进行采样，产生上述接受场重叠，最终得到32 × 16n的图像大小。条件实例归一化层[9]被用来调制剩余的块使用三个额外的32维噪声向量，z2, z3和z4。最后，使用带有tanh激活的卷积层输出最终图像。鉴别器网络D的灵感来自BigGAN[6]: 4个残差块，后跟一个线性层，只有一个输出。为了处理变化宽度的图像生成，D也是完全卷积的，本质上是在水平重叠的图像补丁上工作。最后的预测是补丁预测的平均值，它被送入GAN铰链损失[28]。识别网络R的灵感来源于CRNN[35]。网络的卷积部分包含6个卷积层和5个池化层，所有这些层都使用ReLU激活。最后，使用一个线性层输出每个窗口的类得分，并将其与使用连接主义时态分类(CTC)损失[13]的ground truth注释进行比较。我们的实验是在一台拥有一个V100 GPU和16GB RAM的机器上运行的。关于架构的更多细节，读者可以参考补充材料。

![image-20211123143341921](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123143341921.png)

Figure 3: Generating different styles. Each row in the figure is generated by the same noise vector and results in the same handwriting style. The words generated in each column from left to right are: retrouvailles,  ́ ecriture, les,  ́ etoile, feuilles, soleil, p ́ eripat ́ eticien and chaussettes

产生不同的风格。图中的每一行都是由相同的噪声向量生成的，结果是相同的笔迹风格。每一栏中生成的单词从左到右依次是:retrouvailles、́ecriture、les、́etoile、feuilles、soleil、ṕeripat́eticien和chaussettes

![image-20211123143402231](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123143402231.png)

Figure 4: Results of the work by Alonso et al. [2] (left column) vs our results (right column) on the words: olibrius, inventif, bonjour, ionique, malade, golf, ski, Dimanche, r ́ eparer, famille, gorille, certes, des, le

阿隆索等人的工作结果[2](左栏)与我们的结果(右栏)的词:olibrius, inventif, bonjour, ionique, malade, golf, ski, Dimanche, ŕeparer, famille, gorille, certes, des, le

### 4.2. Datasets and evaluation metrics 

To evaluate our method, we use three standard benchmarks: RIMES[14], IAM [29], and CVL [25]. The RIMES dataset contains words from the French language, spanning about 60k images written by 1300 different authors. The IAM dataset contains about 100k images of words from the English language. The dataset is divided into words written by 657 different authors. The train, test and validation set contain words written by mutually exclusive authors. The CVL dataset consists of seven handwritten documents, out of which we use only the six that are English. These doc-uments were written by about 310 participants, resulting in about 83k word crops, divided into train and test sets. 



All images were resized to a fixed height of 32 pixels while maintaining the aspect ratio of the original image. For the specific case of GAN training, and only when labels were used (supervised case), we additionally scaled the image horizontally to make each character approximately the same width as the synthetic ones, i.e. 16 pixels per character. This was done in order to challenge the discriminator by making real samples more similar to the synthesized ones. We evaluate our method We evaluate our method using two common gold standard metrics. **First, word error rate (WER) is the number of misread words out of the number of words in the test set. Second, normalized edit-distance (NED) is measured by the edit-distance between the predicted and true word normalized by the true word length.** Whenever possible, we repeat the training session five times and report the average and standard deviation thereof. 

为了评估我们的方法，我们使用三个标准基准:RIMES[14]、IAM[29]和CVL[25]。RIMES数据集包含来自法语的单词，涵盖了由1300名不同作者撰写的约6万幅图像。IAM数据集包含大约10万张英语词汇的图片。该数据集由657位不同作者的文字组成。训练、测试和验证集包含由互斥作者编写的单词。CVL数据集由7个手写文档组成，我们只使用其中的6个英文文档。这些文件由大约310名参与者撰写，结果产生了大约83k字的农作物，分为火车和测试集。

所有图像都被调整为32像素的固定高度，同时保持原始图像的纵横比。对于GAN训练的具体情况，只有在使用标签的情况下(监督的情况下)，我们在图像水平方向上进行缩放，使每个字符的宽度近似于合成字符的宽度，即每个字符16个像素。这样做是为了通过使真实样品更接近合成样品来挑战鉴别器。我们使用两个常见的黄金标准指标来评估我们的方法。首先，单词错误率(WER)是测试集中单词数中的误读单词数。其次，归一化编辑距离(NED)是通过由真单词长度归一化的预测值与真单词之间的编辑距离来度量的。只要有可能，我们会重复训练5次，并报告其平均值和标准差。

### 4.3. Comparison to Alonso el al.

 [2] Since no implementation was provided, we focus on qualitative comparison to [2] using images and metrics presented therein. Figure 4 contains results shown in [2] alongside results of our method on the same words. As can be seen in the figure, our network produces images that are much clearer, especially for shorter words. More generally, our results contain fewer artifacts, for example, the letter ‘m’ in the fifth row, the redundant letter ‘i’ in the sixth row and the missing ‘s’ in the row before last. Table 4 compares the two methods using standard metrics for GAN performance evaluation**, namely Fr ́ echet Inception Distance (FID) [16] and geometric-score (GS)** [23]. Using a similar setting1 to the ones described in [2], our method shows slightly better performance on both metrics. Note, however, that since we do not have access to the data from [2], both metrics for that method are copied from the paper, and hence cannot be used to directly compare to our results.

由于没有提供实现，我们专注于定性比较[2]使用图像和指标在其中提出。图4包含[2]中显示的结果以及我们的方法对相同单词的结果。从图中可以看出，我们的网络生成的图像更加清晰，尤其是对于较短的单词。更一般地说，我们的结果包含更少的工件，例如，第五行中的字母“m”，第六行中的冗余字母“i”，以及前一行中缺失的“s”。表4比较了两种方法使用GAN性能评估的标准指标，即Fŕechet Inception Distance (FID)[16]和geometry -score (GS)[23]。使用与[2]中描述的设置1相似的设置1，我们的方法在这两个指标上显示出稍微更好的性能。但是请注意，由于我们不能访问来自[2]的数据，该方法的两个指标都是从本文中复制的，因此不能用于直接比较我们的结果。

### 4.4. Generating different styles

 We are able to generate different handwriting styles by changing the noise vector z that is fed into ScrabbleGAN. Figure 3 depicts examples of selected words generated in different handwriting styles. Each row in the figure represent a different style, while each column contains a different word to synthesize. As can be seen in the figure, our network is able to generate both cursive and non-cursive text, with either a bold or thin pen stroke. This image provides a good example of character interaction: while all repetitions of a character start with identical filters fi, each final 1We ran this experiment once, as opposed to [2] who presented the best result over several runs instantiation might be different depending on the adjacent characters.

我们可以通过改变输入ScrabbleGAN的噪声向量z来生成不同的笔迹风格。图3描述了以不同手写风格生成的选定单词的示例。图中的每一行代表不同的风格，而每一列包含要合成的不同单词。如图所示，我们的网络既可以生成草书文本，也可以生成非草书文本，可以使用粗体或细笔。这张图片提供了一个很好的角色交互的例子:当所有重复的角色都以相同的过滤器fi开始时，每个最后的1We运行这个实验一次，而不是[2]，它在几次运行中呈现最好的结果，实例化可能取决于相邻的角色。



![image-20211123143526042](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123143526042.png)

Table 1: Comparison of our method to Alonso et al.[2] using Fr ́ echet Inception Distance and geometric-score metrics. Lower values are better.

将我们的方法与Alonso等人使用Fŕechet Inception Distance和几何得分指标进行比较。数值越低越好。

**Figure 5 shows interpolations between two different styles** on the IAM dataset. In each column we chose two random noise vectors for the first and last row, and interpolated between them linearly to generate the noise vectors for the images in between. The size of each letter, the width of the pen strokes and the connections between the letters change gradually between the two styles. The gray background around the letters is a property of the original IAM dataset and can be found in most of the images in the dataset. As a result, the generator also learns to generate variations of the background.

图5显示了IAM数据集上两种不同风格之间的插值。在每一列中，我们为第一行和最后一行选择两个随机噪声向量，并在它们之间线性插值，以生成中间图像的噪声向量。在两种风格中，每个字母的大小、笔画的宽度以及字母之间的联系都在逐渐变化。字母周围的灰色背景是原始IAM数据集的属性，可以在数据集中的大多数图像中找到。因此，生成器也学习生成背景的变化。

### 4.5. Boosting HTR performance

 Our primary motivation to generate handwriting images is to improve the performance of an HTR framework compared to the “vanilla” supervised setting. For all experiments in this section, we use the code provided by [3] as our HTR framework, as it contains all the improvements presented in [10] (for which no implementation was provided), as well as some other recent advances that achieve state of the art performance on the scene text recognition problem for printed text. We show that training the best architecture in [3] on the handwritten data yields performance close to state of the art on HTR, which should be challenging to improve upon. Specifically, our chosen HTR architecture is composed of a thin plate spline (TPS) transformation model, a ResNet backbone for extracting the visual features, a bi-directional LSTM module for sequence modeling, and an attention layer for the prediction. In all the experiments, we used the validation set to choose the best performing model, and report the performance thereof on its associated test set.

Train set augmentation is arguably the most straightforward application of a generative model in this setting: by simply appending generated images to the train set, we strive to improve HTR performance in a bootstrap manner. Table 2 shows WER and NED of the HTR network when trained on various training data agumentations on the training data, for both RIMES and IAM datasets, where each row adds versatility to the process w.r.t. its predecessor. For each dataset, the first row shows results when using the original training data, which is the baseline for comparison. Next, the second row shows performance when the data is augmented with a random affine transformations. The third row shows results using the original training data and an additional 100k synthetic handwriting image generated by ScrabbleGAN. The last row further fine-tunes the latter model using the original training data. As can be seen in the table, using the ScrabbleGAN generated samples during training leads to a significant improvement in performance compared to using only off-the-shelf affine augmentations

我们生成手写图像的主要动机是，与“普通的”监督设置相比，提高HTR框架的性能。对所有实验在本节中,我们使用[3]提供的代码作为我们HTR框架,因为它包含所有[10]中给出的改进提供了(没有实现),以及一些其他最近的进步,获得先进的性能在现场对印刷文本文字识别的问题。我们表明，在[3]中对手写数据进行最佳架构训练，可以产生接近HTR技术水平的性能，这应该是有挑战性的改进。具体来说，我们选择的HTR架构由一个薄板样条(TPS)转换模型、一个用于提取视觉特征的ResNet主干、一个用于序列建模的双向LSTM模块和一个用于预测的注意层组成。在所有的实验中，我们使用验证集来选择最佳的执行模型，并报告其在其关联的测试集上的性能。
列车集增强可以说是生成模型在这种情况下最直接的应用:通过简单地将生成的图像附加到列车集，我们努力以bootstrap方式提高HTR性能。表2显示了在训练数据的各种训练数据汇总上训练HTR网络的WER和NED，包括RIMES和IAM数据集，其中每一行增加了过程w.r.t.的多功能性。对于每个数据集，第一行显示使用原始训练数据时的结果，这是比较的基线。接下来，第二行显示使用随机仿射转换扩充数据时的性能。第三行显示了使用原始训练数据和由ScrabbleGAN生成的额外的100k合成手写图像的结果。最后一行使用原始训练数据进一步微调后一种模型。从表中可以看出，在训练过程中使用ScrabbleGAN生成的样本，与只使用现成的仿射增强相比，性能有了显著的提高

![image-20211123143632504](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123143632504.png)

Figure 5: Style interpolation. Each column contains an interpolation between two different styles of handwriting generated by ScrabbleGAN. Note that the GAN captures the background noise typical to the IAM dataset [29].

![image-20211123143651847](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123143651847.png)

Table 2: HTR experiments on RIMES and IAM. For each dataset we report four results with gradually increasing versatility to the dataset w.r.t. its predecessor. The second column (‘Aug’) indicates usage of random affine augmentation in train time. The third column (‘GAN’) indicates whether synthetic images were added to the original train set, and how many. The fourth column (‘Refine’) indicates whether another pass of fine tuning was performed using the original data. See text for more details.

Domain adaptation, sometimes called transductive transfer learning, is the process of applying a model on data from a different distribution than the one it was trained on. We test this task by transferring from IAM to CVL as they both use the same alphabet and are somewhat visually similar. One naive solution for this is training a model on the IAM dataset, and testing its performance on the CVL test set. This will be our baseline for comparison. Since ScrabbleGAN can be trained on unlabeled data, it can adapt to the style of CVL images without using the ground truth. We synthesize data according three different flavors: using either CVL style, CVL lexicon, or both (as opposed to IAM). Data generated from each of these three flavors is appended to the IAM training set, as we find this helps stabilize HTR training. Finally, we set a “regular” supervised training session of CVL train set, to be used as an oracle, i.e. to get a sense of how far we are from using the train labels. Table 3 summarizes performance over the CVL test set of all the aforementioned configurations, ranging from the naive case, through the flavors of using data from ScrabbleGAN, to the oracle. First, we wish to emphasize the 17% WER gap between the naive approach and the oracle, showing how hard it is for the selected HTR to generalize in this case. Second, we observe that synthesizing images with CVL style and IAM lexicon (second row) does not alter the results compared to the naive approach. On the other hand, synthesizing images with IAM style and CVL lexicon (third row) boosts WER performance by about 5%. Finally, synthesizing images with both CVL style and lexicon (fourth row) yields another 5% boost in WER, with NED score that is better than the oracle.

领域适应，有时称为转导迁移学习，是将一个模型应用于来自不同分布的数据的过程，而不是训练它的那个分布。我们通过从IAM转移到CVL来测试这个任务，因为它们都使用相同的字母，而且在视觉上有些相似。一个简单的解决方案是在IAM数据集上训练一个模型，然后在CVL测试集上测试它的性能。这将是我们进行比较的基准。由于ScrabbleGAN可以在未标记的数据上进行训练，它可以适应CVL图像的风格，而不使用地面真相。我们根据三种不同的方式合成数据:使用CVL风格、CVL词典或两者都使用(与IAM相反)。这三种方法产生的数据都会被附加到IAM训练集中，因为我们发现这有助于稳定HTR训练。最后，我们设置了一个CVL列车集的“常规”监督训练会话，以用作一个oracle，即了解我们距离使用列车标签有多远。表3总结了上述所有配置在CVL测试集上的性能，范围从简单的情况到使用ScrabbleGAN数据的方式，再到oracle。首先，我们希望强调天真的方法和oracle之间17%的WER差距，表明在这种情况下，选择HTR进行概括是多么困难。其次，我们观察到，与原始方法相比，使用CVL样式和IAM词汇(第二行)合成图像不会改变结果。另一方面，使用IAM风格和CVL词汇(第三行)合成图像将WER性能提高约5%。最后，综合使用CVL风格和词汇(第四行)的图像在WER中又提高了5%，NED得分比oracle好。

![image-20211123143730720](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123143730720.png)

Table 3: Domain adaptation results from the IAM dataset to the CVL dataset. First row is naive approach of using a net trained on IAM. Next three rows show the effect of 100k synthetic images having either CVL style, CVL lexicon or both. The bottom row shows the oracle performance of supervised training on the CVL train set, just for reference. No CVL labels were used to train HTR, except for the oracle.



### 4.6. Gardient balancing ablation study

Several design considerations regarding parameter selection were made during the conception of ScrabbleGAN. We focus on two main factors: First, the effect of gradient balancing (GB) presented below, and second, the surprising effect of the architecture of the recognizer R which we leave to the supplementary material. Table 4 compares HTR results on the RIMES dataset using three different variations of gradient balancing during training: First, we show results when no gradient balancing is used whatsoever. Second, we apply the gradient balancing scheme suggested in [2], which is shown in Eq. (2). Finally, we show how our modified version performs for different values of the parameter α, as described in Eq. (3). For all the above options we repeat the experiment shown in the third row of Table 2, and report WER and NED scores. Clearly, the best results are achieved using samples synthesized from a GAN trained using our gradient balancing approach with α = 1. Figure 6 further illustrates the importance of balancing between ℓR and ℓD and the effect of the parameter α. Each column in the figure represents a different value starting from training only with ℓR on the left, to training only with ℓD on the right. The same input text, “ScrabbleGAN”, is used in all of the images and the same noise vector is used to generate each row. As expected, using only the recognizer loss results in images which look noisy and do not contain any readable text. On the other hand, using only the adversarial loss results in real-looking handwriting images, but do not contain the desired text but rather gibberish. A closer look at this column reveals that manipulating the value of z changes the letter itself, rather than only the style. From left to right, the three middle columns contain images generated by a GAN trained with α values of 10, 1, and 0.1. The higher the value of α is, the higher the weight of the ℓR is. The results using α = 10 are all readable, but contain much less variability in style. Conversely, using α = 0.1 yields larger variability in style at the expense of the text readability, as some of the letters become unrecognizable. The images depicted in Figure 6 provide another explanation for the quantitative results shown in Table 4. Training an HTR network with images generated by a GAN trained with larger α deteriorates the results on diverse styles, while training with images generated by a GAN trained with a smaller α value might lead to recognition mistakes caused by training on unclear text images.



![image-20211123143834902](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123143834902.png)

Table 4: GB ablation study, comparing HTR performance trained on different synthetic datasets. Each such set was generated by a GAN with different GB scheme. See text for details.

![image-20211123143848995](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123143848995.png)

Figure 6: Comparison of different balancing levels between ℓD and ℓR, the discriminator and recognizer loss terms, respectively. Setting α’s value to ∞ or 0 means training only with R or D, respectively. All examples are generation of the word “ScrabbleGAN”, where each row was generated with the same noise vector z.

5. ### Conclusion and Future Work 


We have presented a new architecture to generate offline handwritten text images, which operates under the assumption that writing characters is a local task. Our generator architecture draws inspiration from the game “Scrabble”. Similarly to the game, each word is constructed by assembling the images generated by its characters. The generated images are versatile in both stroke widths and general style. Furthermore, the overlap between the receptive fields of the different characters in the text enables the generation of cursive as well as non-cursive handwriting. We showed that the large variability of words and styles generated, can be used to boost performance of a given HTR by enriching the training set. Moreover, our approach allows the introduction of an unlabeled corpus, adapting to the style of the text therein. We show that the ability to generate words from a new lexicon is beneficial when coupled with the new style. An interesting avenue for future research is to use a generative representation learning framework such as VAE [24] or BiGAN [7, 8], which are more suitable for few shot learning cases like author adaptation. Additionally, disentanglement approaches may allow finer control of text style, such as cursive-ness or pen width. In the future, we additionally plan to address the fact that generated characters have the same receptive field width. This is, of course, not the case for most scripts, as ‘i’ is usually narrower than ‘w’, for example. One possible remedy for this is having a different width for each character filter depending on its average width in the dataset. Another option is to apply STN [21] as one of the layers of G, in order to generate a similar effect. J





## Handwritten Chinese Font Generation with Collaborative Stroke Refinement

4. ### Experiments

    We conduct extensive experiments to validate the proposed method and compare it with two state-of-the-art methods, HAN[7] and EMD[24]

![image-20211123144049139](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123144049139.png)

Figure 5. Examples of “compound” and “single-element” characters. Some radicals are marked in red.

![image-20211123144104244](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123144104244.png)

Figure 6. In the user study, our method achieves the close subjective score with much less training set compared with baselines.



### 4.1. Training Sample Selection

 For Chinese characters, “compound” characters appear as obvious layout structure (e.g. left-right, top-bottom, surrounding, left-middle-right or top-middle-bottom), while “single-element” characters cannot be structural decomposed. Both radicals and “single-element” characters are known as basic units to construct all characters (see Fig. 5). Many compound Chinese characters share the same basic units in themselves, which means though over 8000 characters in Chinese language, there are only rather limited basic units (including 150 radicals and about 450 “singleelement” characters). Based on this prior knowledge, a small set of characters are selected as training samples. We select 450 “single-element” characters and 150×2 compound characters covering all 150 radicals, to create a small dataset SD totally including 750 training samples.

###  4.2. Comparison with Baseline Methods

 Two state-of-the-art Chinese font synthesis methods we compared with are HAN and EMD. HAN[7] is especially proposed for handwritten fonts synthesis. It proposes hierarchical adversarial discriminator to improve the generated performance. Experiment shows it relies on about 2500 paired training samples to achieve good performance; EMD[24] can achieve style transfer from a few samples by disentangling the style/content representation of fonts, while it relies on a large font library for training and performs not well on handwritten fonts. 

#### 4.2.1 Performance on Small Dataset 

SD We first experimentally compare our method with HAN[7] and EMD[24] under the selected small dataset SD(see Fig. 7). Specially, EMD (few-shot) denotes we generate characters using a few reference inputs, just like what the paper has done. However, for fair comparison, EMD (finetune) denotes we further fine tune EMD with SD. For HAN, we directly train it on SD. We specially choose both printed-style and handwritten-style fonts to fairly demonstrate our model generally outperforms baselines. According to Fig. 7, our model slightly outperforms baseline methods on printed-style fonts (1st row) since printedstyle fonts are always featured with regular structure and wide strokes, which makes the model take no advantages of proposed collaborative stroke refinement. However, our method achieves impressive results on handwritten fonts featured with thin strokes(2nd row) or even irregular structure (3rd rows). Compared with baselines, our model synthesizes more details without missing or overlapped strokes. While these defections happen in baseline methods, we can barely recognize some synthesized characters of them. More experimental results are displayed in appendix. 

#### 4.2.2 Performance on Different Training-set Size 

We further change the size of training set to 1550 and 2550 by randomly adding additional samples to SD. we compare our model with baselines under various sizes of training set on handwritten font synthesis tasks (see Fig. 9). Besides RMSE, to rationally measure the fidelity of synthesized characters, we conduct a user Study (see Fig. 6). 100 volunteers are invited to subjectively rate the synthesized characters from score 1 to 5, where 1 is the worst and 5 is the best. The user study results show that the performance of all methods is improved with the increase of the training set. However, when the size is larger than 1550, the increasing trend of our method stops and the score begins to float up and down. Thus we conclude that 1550 training samples have completely covered radicals/single-element characters with different shapes so that more training samples will not bring more improvement. Additionally, User Study result demonstrates that when the size of training set is 750, our method achieves equal even higher subjective score compared with HAN trained by 2550 characters. 

### 4.3. Ablation Study 

Effect of Adaptive Pre-deformation We conduct experiments by removing adaptive pre-deformation or not. As shown in the second row of Fig. 10, some strokes are missing compared with the ground truth, while the results of ours are significantly better, which means pre-deformation guarantees that the generated strokes are complete. When absolute locations of a certain stroke between the source character and the target character are seriously discrepant, the transfer network may be confused about whether this stroke should be abandoned or be mapped to another position. Our adaptive pre-deformation roughly align the strokes essentially relieves the transfer network of learning the mapping of stroke location. 

Effect of online zoom-augmentation The results after removing online augmentation module are shown in the third row of Fig. 10 from which we can see the generated strokes are so disordered that we even cannot recognize the characters. Without zoom-augmentation, the model





## Generating Handwritten Chinese Characters using CycleGAN



4. ### Experiments

In this section, we evaluate our proposed method on two publicly available datasets. Furthermore, we propose content accuracy and style discrepancy as complementary evaluation metrics in addition to visual appearance. Main results are shown in this section.

### 4.1. Datasets

##### CASIA-HWDB dataset. 

The Chinese handwriting database, CASIA-HWDB [16] is a widely used database for Chinese handwritten character recognition [30, 29]. This database contains samples of isolated characters and handwritten texts that were produced by 1020 writers using Anoto pen on papers. In this study, we use the HWDB1.1 dataset from the CASIA-HWDB. It contains 300 files (240 in HWDB1.1 training set and 60 in HWDB1.1 test set). Each file contains about 3000 isolated gray-scale Chinese character images written by one writer, as well as their corresponding labels. The isolated character images are resized to 128 × 128 pixels. Other than resizing, no other data preprocessing method is performed. For the task of generating handwritten characters, we use the file HW252 (1252-c.gnt) in the HWDB1.1 dataset as the target style, and SIMHEI font as the source style. SIMHEI is a commonly used Chinese font. Figure 4 shows the Chinese character “yong” in 5 different styles. The first two are printed fonts, and the last three are handwritten.

中文手写体数据库CASIA-HWDB[16]是一个广泛用于中文手写体字符识别的数据库[30,29]。这个数据库包含了1020位作家使用anto笔在纸上创作的孤立字符和手写文本的样本。本研究使用CASIA-HWDB中的HWDB1.1数据集。它包含300个文件(240个在HWDB1.1训练集，60个在HWDB1.1测试集)。每个文件包含大约3000个独立的灰度汉字图像，这些图像是由一个作者写的，以及对应的标签。孤立的字符图像被调整为128 × 128像素。除了调整大小，没有执行其他数据预处理方法。对于手写字符的生成任务，我们使用HWDB1.1数据集中的文件HW252 (1252-c.gnt)作为目标样式，SIMHEI字体作为源样式。SIMHEI是一种常用的汉字字体。图4显示了5种不同风格的汉字“勇”。前两种是印刷字体，后三种是手写字体。



![image-20211123155848016](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123155848016.png)

Figure 4: The character “yong” in 5 different fonts. (a) SIMHEI; (b) SIMKAI; (c) character in Lanting calligraphy dataset; (d) handwritten character from HW252 (1252-c.gnt) in HWDB1.1; (e) handwritten character from HW292 (1292-c.gnt) in HWDB1.1.



### 4.2. Performance metrics

Generative models usually lack objective evaluation criteria, which makes it difficult to quantitatively assess the quality of the images generated. To measure the performance of our handwritten character generation method on the CASIA-HWDB dataset, we propose two complementary evaluation metrics: the content accuracy and the style discrepancy. Both evaluations are based on a pre-trained network: the HCCR-GoogLeNet [30], which is a handwritten Chinese character classification model based on GoogLeNet [23].

生成模型通常缺乏客观的评价标准，难以对生成的图像质量进行定量评价。为了测量我们的手写字符生成方法在CASIA-HWDB数据集上的性能，我们提出了两个互补的评价指标:内容准确性和风格差异。这两种评估都是基于预先训练的网络:hcrc -GoogLeNet[30]，这是一个基于GoogLeNet[23]的手写汉字分类模型。

Content accuracy. The HCCR-GoogLeNet model is trained using the CASIA-HWDB handwritten character database with 1, 020 writers in total, including HW252. It achieved the state-of-the-art accuracy of Chinese character classification. Inspired by the idea of the Inception score [21], the pre-trained HCCR-GoogLeNet model can be used to evaluate the quality of the generated handwritten characters. The intuition is that if the generated characters are realistic, the pre-trained HCCR-GoogLeNet will also be able to classify the generated characters correctly. In our case, characters in the target style are generated from available images in the source style. Therefore, the true labels of the generated characters are known. If the characters generated can be accurately classified by pre-trained character recognition models, to some extent it indicates that the generative model is of high quality. Table 2 shows the test accuracy of HCCR-GoogLeNet on the HW252 handwritten characters and the SIMHEI font characters as a baseline. The high recognition accuracy indicates that the HCCR-GoogLeNet accuracy is a reliable metric for quality measurement. However, the equally good performance on HW252 and SIMHEI implies that this metric measures the generation quality from a single perspective: it only assesses the content quality, not the style quality. Therefore, the recognition accuracy of HCCRGoogLeNet on the generated handwritten characters is referred to as the content accuracy.

Style discrepancy. To measure the discrepancy in style between the true characters in the target domain and the generated characters, we borrow the style loss in neural style transfer algorithm [2]. The idea is to use the correlations between different filter activations at one layer as a style representation. The feature correlations are given by the Gram matrix G` ∈ RN`×N` , where N ` is the number of filters in the `-th layer, and G` ij is the inner product between the vectorized feature map i and j in layer `: G` ij = ∑ k F` ikF ` jk. (7) The style discrepancy is thus defined as the root-meansquare difference between the style representations of the target characters and the generated characters. Lower discrepancy corresponds to better style quality. In our experiments, the input of Inception module 3 in HCCRGoogLeNet is used as layer ` to calculate the style discrepancy. We run two baseline experiments to get an approximate of the range of the style discrepancy. (a) The style discrepancy between two randomly and equally split subsets of the HW252 handwritten dataset. Since these two subsets are written by the same person and have the same style, the result represents the lower bound of the style discrepancy. The style discrepancy lower bound is 503.77. (b) The style discrepancy between HW252 and SIMHEI. This is the style difference between the source style and the target style. It thus represents the most possible disagreement in style, and it measures the style quality of a trivial identity style transfer model. Therefore it can be regarded as an upper bound of the style discrepancy. The style discrepancy upper bound is 3006.03.

##### 4.3. Implementation details

In the experiments, we consider two types of transfer modules: ResNet with 6 blocks (ResNet-6) and DenseNet with 5 blocks (DenseNet-5). The DenseNet-5 transfer module has roughly the same number of parameters as the ResNet-6 transfer module. The only preprocessing procedure we used is to resize the training images to 128 × 128 pixels; no other preprocessing methods (e.g. crop and flip) are used. For all the experiments, the regularization strength is set to λ = 10, and the Adam optimizer [12] with a batch size of 1 is used. The learning rate is set to 0.0002 for the first 100 epochs and then linearly decays to 0 over the next 100 epochs. The number of iterations in each epoch in the experiments is the larger number of the training examples in the two styles.

![image-20211123160048248](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123160048248.png)



Table 2: The top-1 and top-5 test accuracy of HCCRGoogLeNet on the HW252 handwritten characters and the SIMHEI font characters. The equally good performance on HW252 and SIMHEI implies that this metric only measures the content quality. 



##### 4.4. Handwritten characters results

We use SIMHEI font as the source style and handwritten characters in HW252 as the target style. SIMHEI and HW252 are both split to unpaired training and validation sets. In real applications, we would like the users to only write a few Chinese characters, based on which the remaining handwritten characters can be generated with his/her personal style using our proposed method. Therefore, the goal of this experiment is to use small training sets to train a CycleGAN model, and perform style transfer on the validation sets. Let rA be the split ratio of HW252, that is, rA of the characters in HW252 are randomly assigned to the training set, and the remaining characters are in the validation set. Similarly, rB denotes the split ratio of SIMHEI. Both rA and rB take values in {5%, 10%, 15%, 30%}, which gives 16 combinations. Table 3 and 4 show the top-5 content accuracy and style discrepancy for ResNet-6 and DenseNet-5, respectively. The results indicate that both content and style quality improves with the number of increasing training data. In particular, when rA and rB are greater than 10%, the content accuracy is always greater than 80%. The performance of ResNet-6 and DenseNet-5 are comparable. Furthermore, since the style discrepancy ranges between 503.77 and 3006.03, the style discrepancies are on the low end of the spectrum. Figure 5 shows the generated handwritten characters using DenseNet-5 and ResNet-6, as well as the source and target styles. The background of the generated characters is clear and the contents are perfectly recognizable. The style of the strokes is noticeably different from that of the source font. The composition of the radicals and the blurry boundaries of the strokes highly resemble the writer’s handwriting style. Figure 6 shows a famous Chinese poem “On the Stork Tower” composed of characters generated by ResNet-6 CycleGAN. All the characters are clearly recognizable with personalized style.







ACE



Owing to its large character set (7,357 classes), diverse writing style, and character-touching problem, the offline HCTR problem is highly complicated and challenging to solve. Therefore, it is a favorable testbed to evaluate the robustness and effectiveness of the ACE loss in 1D predictions.

由于其庞大的字符集(7357类)，多样的书写风格，以及字符的触摸问题，离线HCTR问题是非常复杂和具有挑战性的解决。因此，评估ACE损耗在1D预测中的鲁棒性和有效性是一个很好的实验平台。



## Download Feature Data

liu2011casia



To enable the evaluation of machine learning and classification algorithms on standard feature data, we provide the feature data of offline handwriting datasets HWDB1.0 and HWDB1.1, online handwriting datasets OLHWDB1.0 and OLHWDB1.1. The samples fall in 3,755 classes of Chinese characters in GB2312-80 level-1 set. The datasets HWDB1.1 and OLHWDB1.1 (300 writers) are proposed to be used for preliminary experiments of Chinese character recognition of standard category set. The datasets HWDB1.0 and OLHWDB1.0 (420 writers) can be added to HWDB1.1 and OLHWDB1.1 for enlarging the training set size. 

The feature extraction methods are specified in the reference below, and the results reported there can be used for fair comparison: 

C.-L. Liu, F. Yin, D.-H. Wang, Q.-F. Wang, Online and Offline Handwritten Chinese Character Recognition: Benchmarking on New Databases, Pattern Recognition, 46(1): 155-162, 2013. 

For offline characters, the feature extracted is the 8-direction histogram of normalization-cooperated gradient feature (NCGF), combined with pseudo 2D normalization method line density projection interpolation. The resulting feature is 512D. 

For online characters, the feature extracted is the 8-direction histogram of original trajectory direction combined with pseudo 2D bi-moment normalization. The resulting feature is 512D. 

The feature data of each dataset is partitioned into two subsets for training and testing, respectively. The numbers of writers and samples of the files are shown in the Table below.

为了对标准特征数据进行机器学习和分类算法评估，我们提供了离线手写数据集HWDB1.0和HWDB1.1、在线手写数据集OLHWDB1.0和OLHWDB1.1的特征数据。样本分为GB2312-80一级集3755类汉字。本文提出将HWDB1.1和OLHWDB1.1（300个作者）数据集用于标准分类集汉字识别的初步实验。可以将数据集HWDB1.0和OLHWDB1.0（420个写入程序）添加到HWDB1.1和OLHWDB1.1中，以扩大训练集的大小。”特征提取方法在下面的参考文献中有详细说明，报告的结果可用于公平比较：“刘春霖，尹芳芳，王德宏，王秋芳，联机和脱机手写汉字识别：新数据库的基准测试，模式识别，46（1）：155-162，2013。”对于脱机字符，提取的特征是标准化协同梯度特征（NCGF）的8方向直方图，结合伪2D标准化方法线密度投影插值。生成的功能是512D。”对于在线字符，提取的特征是原始轨迹方向的8方向直方图，并结合伪2D双矩归一化。生成的功能是512D。”“每个数据集的特征数据被划分为两个子集，分别用于训练和测试。下表显示了编写器的数量和文件的示例。

![image-20211123190553443](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211123190553443.png)

The format of the feature data files is described in [fileFormat-mpf.pdf](http://www.nlpr.ia.ac.cn/databases/download/feature_data/FileFormat-mpf.pdf). In brief, each file has a header with the header size given as the first 4-byte integer number in the file. The last two integer numbers in the header give the number of samples in the file and the feature dimensionality. Following the header are the records of all samples, each sample including a 2-byte label (GB code) and the feature vector, each dimension in a unsigned char byte. 

For the possibility of writer-specific data analysis, the feature data of each writer is stored in a file named after the writer index. The training or test data files of a dataset are packed in a ZIP archive. Please click the links below for download.

fileFormat-mpf.pdf中描述了要素数据文件的格式。简言之，每个文件都有一个标头，标头大小作为文件中的第一个4字节整数。标题中的最后两个整数表示文件中的样本数和特征维度。标题后面是所有样本的记录，每个样本包括一个2字节的标签（GB代码）和特征向量，每个维度包含一个无符号字符字节。”为了进行特定于写入程序的数据分析，每个写入程序的功能数据都存储在一个以写入程序索引命名的文件中。数据集的培训或测试数据文件打包在ZIP存档中。请点击下面的链接下载。







## 4 Experiments Text and Style Conditioned GAN for Generation of Offline Handwriting Lines

 We first discuss the data used. Then we compare to prior methods. We then show exploration into our method with an ablation study (Sec. 4.1) and an examination of the style space (Sec. 4.2). We finally discuss a user study we performed (Sec. 4.3). We use the IAM handwriting dataset [25] and the RIMES dataset [11], which contain segmented images of handwriting words and lines with accompanying transcriptions. We developed our method exclusively with the IAM training (6,161 lines) and validation (1,840 lines) sets, and reserved the test sets for experiments (FID/GS scores use training images). Note that IAM consists of many authors, but authors are disjoint across train/val/test splits. We resize all images to a fixed height of 64 pixels, maintaining aspect ratio. We apply a random affine slant transformation to each image in training (-45°, 45° uniform). Fig. 5 compares our results to those from Alonso et al. [2] and ScrabbleGAN [6]. Our results appear to have similar quality as [6]. It can be seen in Fig. 6 that [6] (left) lacks diversity in horizontal spacing; despite the style changing, the images are always the same length. This is due to their architectural choice to have the length dependant on content, not style. Our method takes both content and style into consideration for spacing, leading to variable length images for the same text. 

We report Fréchet Inception Distance (FID) [12] and Geometry Score (GS) [22] in Table 1 using a setup similar to [6]. There exist some intricacies to the FID and GS calculation which are included in the Supplementary Materials (S.1). Fig. 7 shows interpolation between three sets of two styles taken from test set images. These images look realistic, even on the interpolated styles. Notice the model even predicts faint background textures similar to dataset images. We note that while styles vary, it mostly varies in terms of global style elements (e.g., slant, ink thickness); the variation rarely comes from character shapes. Figs. 7 and 6 were generated with text not present in the training set; We notice no difference when generating with text from the dataset compared to other text.

我们首先讨论使用的数据。然后，我们比较了以前的方法。然后，我们通过消融研究（第4.1节）和风格空间检查（第4.2节）对我们的方法进行了探索。最后，我们讨论了我们进行的用户研究（第4.3节）。我们使用IAM手写数据集[25]和RIMES数据集[11]，其中包含手写单词和线条的分段图像以及随附的抄本。我们专门使用IAM训练集（6161行）和验证集（1840行）开发了我们的方法，并为实验保留了测试集（FID/GS分数使用训练图像）。请注意，IAM由许多作者组成，但作者在列车/val/测试拆分中是不相交的。我们将所有图像调整为64像素的固定高度，保持纵横比。我们对训练中的每个图像应用随机仿射倾斜变换（-45°，45°均匀）。图5将我们的结果与Alonso等人[2]和ScrabbleGAN[6]的结果进行了比较。我们的结果似乎具有与[6]相似的质量。从图6中可以看出，[6]（左）在水平间距上缺乏多样性；尽管样式发生了变化，但图像的长度始终相同。这是因为他们选择的架构长度取决于内容，而不是样式。我们的方法同时考虑了内容和样式的间距，从而为相同的文本生成可变长度的图像。

我们使用类似于[6]的设置在表1中报告了Fréchet起始距离（FID）[12]和几何分数（GS）[22]。补充材料（S.1）中包含的FID和GS计算存在一些复杂性。图7显示了从测试集图像中获取的三组两种样式之间的插值。即使在插值样式上，这些图像看起来也很逼真。请注意，该模型甚至可以预测类似于数据集图像的微弱背景纹理。我们注意到，虽然样式各不相同，但它主要根据全局样式元素（例如，倾斜、墨水厚度）而变化；这种变化很少来自角色的形状。无花果。7和6生成的文本不在训练集中；我们注意到，与其他文本相比，使用数据集中的文本生成时没有任何差异。







