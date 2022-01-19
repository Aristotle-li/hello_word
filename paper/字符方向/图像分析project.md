

## GAN及其变体用于手写数字图像生成

代码来自于 https://github.com/znxlwm/pytorch-generative-model-collections，缩放像素到[0,1]区间后，原始代码做了三通道归一化而mnist数据集是单通道图像，需改成单通道且均值方差必须为0.5，或者直接不做transforms.Normalize也可以。

### 一、实验环境配置

#### 环境配置

| torch       | 1.8.1+cu111 |
| ----------- | ----------- |
| torchvision | 0.9.1+cu111 |
| numpy       | 1.17.3      |
| Python      | 3.6         |
| matplotlib  | 3.1.1       |

#### Dataset

- GANs are notoriously finicky with hyperparameters, and also require many training epochs. In order to make this assignment approachable without a GPU, we will be working on the MNIST dataset, which is 60,000 training and 10,000 test images. Each picture contains a centered image of white digit on black background (0 through 9). This was one of the first datasets used to train convolutional neural networks and it is fairly easy -- a standard CNN model can easily exceed 99% accuracy.

  To simplify our code here, we will use the PyTorch MNIST wrapper, which downloads and loads the MNIST dataset.  The default parameters will take 5,000 of the training examples and place them into a validation dataset. The data will be saved into a folder called `data`.


## 二、理论方法

In 2014, [Goodfellow et al.](https://arxiv.org/abs/1406.2661) presented a method for training generative models called Generative Adversarial Networks (GANs for short). In a GAN, we build two different neural networks. Our first network is a traditional classification network, called the **discriminator**. We will train the discriminator to take images, and classify them as being real (belonging to the training set) or fake (not present in the training set). Our other network, called the **generator**, will take random noise as input and transform it using a neural network to produce images. The goal of the generator is to fool the discriminator into thinking the images it produced are real.

We can think of this back and forth process of the generator ($G$) trying to fool the discriminator ($D$), and the discriminator trying to correctly classify real vs. fake as a minimax game:
$$\underset{G}{\text{minimize}}\; \underset{D}{\text{maximize}}\; \mathbb{E}_{x \sim p_\text{data}}\left[\log D(x)\right] + \mathbb{E}_{z \sim p(z)}\left[\log \left(1-D(G(z))\right)\right]$$
where $z \sim p(z)$ are the random noise samples, $G(z)$ are the generated images using the neural network generator $G$, and $D$ is the output of the discriminator, specifying the probability of an input being real. In [Goodfellow et al.](https://arxiv.org/abs/1406.2661), they analyze this minimax game and show how it relates to minimizing the Jensen-Shannon divergence between the training data distribution and the generated samples from $G$.

To optimize this minimax game, we will aternate between taking gradient *descent* steps on the objective for $G$, and gradient *ascent* steps on the objective for $D$:
1. update the **generator** ($G$) to minimize the probability of the __discriminator making the correct choice__. 
2. update the **discriminator** ($D$) to maximize the probability of the __discriminator making the correct choice__.

While these updates are useful for analysis, they do not perform well in practice. Instead, we will use a different objective when we update the generator: maximize the probability of the **discriminator making the incorrect choice**. This small change helps to allevaiate problems with the generator gradient vanishing when the discriminator is confident. This is the standard update used in most GAN papers, and was used in the original paper from [Goodfellow et al.](https://arxiv.org/abs/1406.2661). 

In this assignment, we will alternate the following updates:
1. Update the generator ($G$) to maximize the probability of the discriminator making the incorrect choice on generated data:
$$\underset{G}{\text{maximize}}\;  \mathbb{E}_{z \sim p(z)}\left[\log D(G(z))\right]$$
2. Update the discriminator ($D$), to maximize the probability of the discriminator making the correct choice on real and generated data:
$$\underset{D}{\text{maximize}}\; \mathbb{E}_{x \sim p_\text{data}}\left[\log D(x)\right] + \mathbb{E}_{z \sim p(z)}\left[\log \left(1-D(G(z))\right)\right]$$

### Generative Adversarial Networks (GANs)

#### Discriminator

Our first step is to build a discriminator. Fill in the architecture as part of the `nn.Sequential` constructor in the function below. All fully connected layers should include bias terms. The architecture is:
 * Fully connected layer with input size 784 and output size 256
 * LeakyReLU with alpha 0.01
 * Fully connected layer with input_size 256 and output size 256
 * LeakyReLU with alpha 0.01
 * Fully connected layer with input size 256 and output size 1

Recall that the Leaky ReLU nonlinearity computes $f(x) = \max(\alpha x, x)$ for some fixed constant $\alpha$; for the LeakyReLU nonlinearities in the architecture above we set $\alpha=0.01$.

The output of the discriminator should have shape `[batch_size, 1]`, and contain real numbers corresponding to the scores that each of the `batch_size` inputs is a real image.

####  Generator

Now to build the generator network:

- Fully connected layer from noise_dim to 1024
- `ReLU`
- Fully connected layer with size 1024
- `ReLU`
- Fully connected layer with size 784
- `TanH` (to clip the image to be in the range of [-1,1])

#### GAN Loss

Compute the generator and discriminator loss. The generator loss is:
$$\ell_G  =  -\mathbb{E}_{z \sim p(z)}\left[\log D(G(z))\right]$$
and the discriminator loss is:
$$ \ell_D = -\mathbb{E}_{x \sim p_\text{data}}\left[\log D(x)\right] - \mathbb{E}_{z \sim p(z)}\left[\log \left(1-D(G(z))\right)\right]$$

#### Optimizing our loss

Make a function that returns an `optim.Adam` optimizer for the given model with a 1e-3 learning rate, beta1=0.55, beta2=0.999. You'll use this to construct optimizers for the generators and discriminators for the rest of the notebook.

#### Variants of GAN structure

![image-20210811152524227](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20210811152524227.png)



#### LOSS Lists

| *Name*      | *Paper Link*                      | *Value Function*                                             |
| ----------- | --------------------------------- | ------------------------------------------------------------ |
| **GAN**     | https://arxiv.org/abs/1406.2661   | <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210701155813034.png" alt="image-20210701155813034" style="zoom:50%;" /> |
| **LSGAN**   | https://arxiv.org/abs/1611.04076  | <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210701155830237.png" alt="image-20210701155830237" style="zoom:50%;" /> |
| **WGAN**    | https://arxiv.org/abs/1701.07875  | <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210701155843055.png" alt="image-20210701155843055" style="zoom:50%;" /> |
| **WGAN_GP** | https://arxiv.org/abs/1704.00028  | <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210701155854175.png" alt="image-20210701155854175" style="zoom:50%;" /> |
| **DRAGAN**  | https://arxiv.org/abs/1705.07215  | <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210701155903683.png" alt="image-20210701155903683" style="zoom:50%;" /> |
| **CGAN**    | https://arxiv.org/abs/1411.1784   | <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210701155913787.png" alt="image-20210701155913787" style="zoom:50%;" /> |
| **ACGAN**   | [https://arxiv.org/abs/1610.09585 | <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210701155925134.png" alt="image-20210701155925134" style="zoom:50%;" /> |
| **EBGAN**   | https://arxiv.org/abs/1609.03126  | <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210701155938304.png" alt="image-20210701155938304" style="zoom:50%;" /> |
| **BEGAN**   | https://arxiv.org/abs/1703.10717  | <img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210701155545039.png" alt="image-20210701155545039" style="zoom:50%;" /> |

## 三、实验结果展示

#### Results for mnist

The following results can be reproduced with command:

```
python main.py --dataset mnist --gan_type <TYPE> --epoch 50 --batch_size 64
```

#### random generation

All results are generated from the random noise vector.

| *Name*  | *Epoch 1*                                                    | *Epoch 15*                                                   | *Epoch 30*                                                   |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GAN     | ![GAN_epoch001](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgGAN_epoch001.png) | ![GAN_epoch015](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgGAN_epoch015.png) | ![GAN_epoch030](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgGAN_epoch030.png) |
| LSGAN   | ![LSGAN_epoch001](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgLSGAN_epoch001.png) | ![LSGAN_epoch015](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgLSGAN_epoch015.png) | ![LSGAN_epoch030](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgLSGAN_epoch030.png) |
| WGAN    | ![WGAN_epoch001](/Volumes/Macintosh HD/pycharm/GAN/pytorch-generative-model-collections-master/results/mnist/WGAN/WGAN_epoch001.png) | ![WGAN_epoch015](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgWGAN_epoch015.png) | ![WGAN_epoch030](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgWGAN_epoch030.png) |
| WGAN_GP | ![WGAN_GP_epoch001](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgWGAN_GP_epoch001.png) | ![WGAN_GP_epoch015](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgWGAN_GP_epoch015.png) | ![WGAN_GP_epoch030](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgWGAN_GP_epoch030.png) |
| DRAGAN  | ![DRAGAN_epoch001](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgDRAGAN_epoch001.png) | ![DRAGAN_epoch015](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgDRAGAN_epoch015.png) | ![DRAGAN_epoch030](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgDRAGAN_epoch030.png) |
| EBGAN   | ![EBGAN_epoch001](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgEBGAN_epoch001.png) | ![EBGAN_epoch015](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgEBGAN_epoch015.png) | ![EBGAN_epoch030](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgEBGAN_epoch030.png) |
| BEGAN   | ![BEGAN_epoch001](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgBEGAN_epoch001.png) | ![BEGAN_epoch015](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgBEGAN_epoch015.png) | ![BEGAN_epoch030](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgBEGAN_epoch030.png) |



#### Conditional generation

Each row has the same noise vector and each column has the same label condition.

| *Name* | *Epoch 1*                                                    | *Epoch 15*                                                   | *Epoch 30*                                                   |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| CGAN   | ![CGAN_epoch001](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgCGAN_epoch001.png) | ![CGAN_epoch015](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgCGAN_epoch015.png) | ![CGAN_epoch030](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgCGAN_epoch030.png) |
| ACGAN  | ![ACGAN_epoch001](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgACGAN_epoch001.png) | ![ACGAN_epoch015](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgACGAN_epoch015.png) | ![ACGAN_epoch030](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgACGAN_epoch030.png) |

#### Loss plot

| *Name*  | *Loss*                                                       | *Name* | *Loss*                                                       |
| ------- | ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| GAN     | <img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgGAN_loss.png" alt="GAN_loss" style="zoom:50%;" /> | ACGAN  | <img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgACGAN_loss.png" alt="ACGAN_loss" style="zoom:50%;" /> |
| LSGAN   | <img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgLSGAN_loss.png" alt="LSGAN_loss" style="zoom:50%;" /> | CGAN   | <img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgCGAN_loss.png" alt="CGAN_loss" style="zoom:50%;" /> |
| WGAN    | <img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgWGAN_loss.png" alt="WGAN_loss" style="zoom:50%;" /> | BEGAN  | <img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgBEGAN_loss.png" alt="BEGAN_loss" style="zoom:50%;" /> |
| WGAN_GP | <img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgWGAN_GP_loss.png" alt="WGAN_GP_loss" style="zoom:50%;" /> | EBGAN  | <img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgEBGAN_loss.png" alt="EBGAN_loss" style="zoom:50%;" /> |
| DRAGAN  | <img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgDRAGAN_loss.png" alt="DRAGAN_loss" style="zoom:50%;" /> |        |                                                              |



WGAN_GP和EBGAN没有达到代码demo中的效果，进一步分析原因。

在30个EPOCH内，可以看出只有WGAN_GP和EBGAN的discriminator和generator的loss快速重合后，迅速分开。观察应该是对于mnist数据集这两个的discriminator太强了，以至于让generator学不到什么东西。对学习率调整，扩大十倍后WGAN_GP达到预期效果，EBGAN没训练出来，调整了很多参数还是不行。

| *Name*  | *Epoch 1*                                                    | *Epoch 15*                                                   | *Epoch 30*                                                   | *loss*                                                       |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| WGAN_GP | ![WGAN_epoch001](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgWGAN_epoch001.png) | ![WGAN_epoch015](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgWGAN_epoch015.png) | ![WGAN_epoch030](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgWGAN_epoch030.png) | <img src="/Volumes/Macintosh HD/pycharm/GAN/pytorch-generative-model-collections-master/models/mnist/WGAN_GP/WGAN_GP_loss_WORK-0.01-0.55-0.99.png" alt="WGAN_GP_loss_WORK-0.01-0.55-0.99" style="zoom:50%;" /> |
| EBGAN   |                                                              |                                                              |                                                              |                                                              |





```
Inputs: input, h_0
    - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
      of the input sequence. The input can also be a packed variable length
      sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
      or :func:`torch.nn.utils.rnn.pack_sequence`
      for details.
    - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
      containing the initial hidden state for each element in the batch.
      Defaults to zero if not provided. If the RNN is bidirectional,
      num_directions should be 2, else it should be 1.

Outputs: output, h_n
    - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
      containing the output features (`h_t`) from the last layer of the RNN,
      for each `t`.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
      been given as the input, the output will also be a packed sequence.

      For the unpacked case, the directions can be separated
      using ``output.view(seq_len, batch, num_directions, hidden_size)``,
      with forward and backward being direction `0` and `1` respectively.
      Similarly, the directions can be separated in the packed case.
    - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
      containing the hidden state for `t = seq_len`.

      Like *output*, the layers can be separated using
      ``h_n.view(num_layers, num_directions, batch, hidden_size)``.

Shape:
    - Input1: :math:`(L, N, H_{in})` tensor containing input features where
      :math:`H_{in}=\text{input\_size}` and `L` represents a sequence length.
    - Input2: :math:`(S, N, H_{out})` tensor
      containing the initial hidden state for each element in the batch.
      :math:`H_{out}=\text{hidden\_size}`
      Defaults to zero if not provided. where :math:`S=\text{num\_layers} * \text{num\_directions}`
      If the RNN is bidirectional, num_directions should be 2, else it should be 1.
    - Output1: :math:`(L, N, H_{all})` where :math:`H_{all}=\text{num\_directions} * \text{hidden\_size}`
    - Output2: :math:`(S, N, H_{out})` tensor containing the next hidden state
      for each element in the batch

Attributes:
    weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
        of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
        `(hidden_size, num_directions * hidden_size)`
    weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
        of shape `(hidden_size, hidden_size)`
    bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
        of shape `(hidden_size)`
    bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
        of shape `(hidden_size)`

.. note::
    All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
    where :math:`k = \frac{1}{\text{hidden\_size}}`

.. include:: ../cudnn_rnn_determinism.rst

.. include:: ../cudnn_persistent_rnn.rst

Examples::

    >>> rnn = nn.RNN(10, 20, 2)
    >>> input = torch.randn(5, 3, 10)
    >>> h0 = torch.randn(2, 3, 20)
    >>> output, hn = rnn(input, h0)
```
