

> 题目：Multi-scale multi-class conditional generative adversarial network for handwritten character generation
>
> 来源：2019
>
> 作者：

### motivation

它是一种基于条件生成对抗网络（CGAN）的神经网络，设计用于真实的多尺度角色生成。MSMC-CGAN将全局和局部图像信息结合起来作为条件，该条件还可以帮助我们生成多类手写字符。我们的模型设计了独特的神经网络结构、图像特征和训练方法。为了验证该模型的性能，我们将其应用于中文笔迹生成，并使用了一种称为平均意见分数（MOS）的评估方法。MOS结果表明，MSMC-CGAN具有良好的性能



Different from the previous research, our method focuses on the realistic and natural handwritten character generation. Thus, we provide our model both the global information such as basic shapes of characters and partial information like the writing style of some strokes or the combination points between two strokes. These information are also used when we judge the quality of the generated characters. The next section will introduce our model structure and the method for global and partial feature extraction.

与以往的研究不同，我们的方法侧重于真实自然的手写体字符生成。因此，我们为我们的模型提供了全局信息（如字符的基本形状）和局部信息（如某些笔画的书写风格或两个笔画之间的结合点）。当我们判断生成字符的质量时，也会使用这些信息。下一节将介绍我们的模型结构以及全局和局部特征提取的方法。





### structure

Before introducing our generative model network structure, discriminate model structure, and feature extraction, we first give our definition for “multi-class” and “multi-scale.” Multi-class A type of neural network model which can generate different classes of objects. There are a large number of Chinese characters, if we want to generate them in one generator, we should make sure that our model can generate different characters by giving it different condition y. Multi-scale A type of neural network model which can accept different sizes of inputs. Since our model is to generate realistic and natural characters, we need to give it more detailed information like images of same character in different scales.

在介绍生成模型网络结构、判别模型结构和特征提取之前，我们首先给出了“多类”和“多尺度”的定义。多类是一种可以生成不同类别对象的神经网络模型。有大量的汉字，如果我们想在一个生成器中生成它们，我们应该确保我们的模型可以通过给定不同的条件y来生成不同的字符。多尺度神经网络模型是一种能够接受不同大小输入的神经网络模型。由于我们的模型是生成真实自然的角色，我们需要给它更多的详细信息，比如相同角色在不同尺度下的图像。
