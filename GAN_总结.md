[CVPR 2020 论文大盘点-文本图像篇 - 极市社区 (cvmart.net)](https://bbs.cvmart.net/articles/2778)

[CVPR 2021 论文大盘点-文本图像篇 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247575870&idx=3&sn=12f6147dbeb9bc2f56e7445041eccaa7&chksm=ec1d44c7db6acdd1d808b84a4ebb52450786b68f198b032208e7bba11a9e10dbfa2a01c43cc8&mpshare=1&scene=1&srcid=09239AvYSBi2sB5LaXxRzjiQ&sharer_sharetime=1632383624698&sharer_shareid=643a8cca9b2f3019b51d64715c30b9ce&exportkey=Afp%2BXCjplUl1TJ0MyizNt1I%3D&pass_ticket=Rmpgc6bKhzx9xIY%2B314ydrzytrAy0NcXD8IA%2FPfqvwymObKSDI7ywAASID3ROrRB&wx_header=0#rd)



# GAN概述：

[从双层优化视角理解对抗网络GAN - 知乎 (zhihu.com

# Adam：

pytorch 中 torch.optim.Adam 方法的使用和参数的解释

现在很多深度网络都优先推荐使用Adam做优化算法，我也一直使用，但是对它的参数一知半解，对它的特性也只是略有耳闻，今天我终于花时间看了一下论文和网上的资料。整理如下。

Adam是从2个算法脱胎而来的：AdaGrad和RMSProp，它集合了2个算法的主要优点，同时也做了自己的一些创新，大概有这么几个卖点：

1. 计算高效，方便实现，内存使用也很少。
2. 更新步长和梯度大小无关，只和alpha、beta_1、beta_2有关系。并且由它们决定步长的理论上限。
3. 对目标函数没有平稳要求，即loss function可以随着时间变化
4. 能较好的处理噪音样本，并且天然具有退火效果
5. 能较好处理稀疏梯度，即梯度在很多step处都是0的情况

它在Adam: A Method for Stochastic Optimization中被提出。

参数：

params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
lr (float, 可选) – 学习率（默认：1e-3）
betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
 个人理解：

lr：同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）。较大的值（如 0.3）在学习率更新前会有更快的初始学习，而较小的值（如 1.0E-5）会令训练收敛到更好的性能。

betas = （beta1，beta2）

beta1：一阶矩估计的指数衰减率（如 0.9）。

beta2：二阶矩估计的指数衰减率（如 0.999）。该超参数在稀疏梯度（如在 NLP 或计算机视觉任务中）中应该设置为接近 1 的数。

eps：epsilon：该参数是非常小的数，其为了防止在实现中除以零（如 10E-8）。


# Conditional Batch Normalization



[(32条消息) Conditional Batch Normalization 详解（SFT思路来源）_Arthur_Holmes的博客-CSDN博客](https://blog.csdn.net/Arthur_Holmes/article/details/103934892)

# lmdb 数据库

[lmdb 数据库 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/70359311)



# 尝试过且有效的技巧

### 梯度惩罚

GAN的对抗训练机制让Generator和Discriminator的梯度极不稳定，很容易出现训练发散的情况。

因此需要对梯度进行限制，早期研究中常常会使用梯度剪裁来限制梯度变化，但是简单的剪裁可能会带来梯度消失或者爆炸的情况出现。

近些年来很多关于GAN的论文都使用到了名为梯度惩罚的技术，即将模型对于输入的梯度作为loss中的惩罚项，

使得模型输入有微小变化的时候，网络权重不会产生太大的变化。

## 梯度裁剪的使用

常见的梯度裁剪有两种

- 确定一个范围，如果参数的gradient超过了，直接裁剪
- 根据若干个参数的gradient组成的的vector的L2 Norm进行裁剪

第一种方法，比较直接，对应于pytorch中的nn.utils.clip_grad_value(parameters, clip_value). 将所有的参数剪裁到 [ -clip_value, clip_value]

第二中方法也更常见，对应于pytorch中clip_grad_norm_(parameters, max_norm, norm_type=2)。 如果所有参数的gradient组成的向量的L2 norm 大于max norm，那么需要根据L2 norm/max_norm 进行缩放。从而使得L2 norm 小于预设的 clip_norm



## 梯度裁剪的使用位置

在backward得到梯度之后，step()更新之前，使用梯度剪裁。从而完成计算完梯度后，进行裁剪，然后进行网络更新的过程。





### 优先训练Discriminator

这个策略下大致有如下三种不同的实现方式：

1. 在Generator开始训练之前，先训练一个能判别真假的Discriminator；
2. 每训练n（n>=1）次Discriminator，训练一次Generator；
3. 在Discriminator中使用更大的学习率（Heusel, Martin et al. “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.” NIPS (2017)）

### 标签平滑或者添加噪声

在Discriminator和Generator的loss中都有不少的分类loss，使用标签平滑或者合理地对标签添加噪声都可以降低训练难度。

### 使用更多的标签信息

在训练过程中，除了图片的真假信息外，如果数据集中有其他信息，尽量利用起来，能够提升模型训练效果。

### 利用分类网络建立图片的重建loss

在Generator的损失函数中，通常会加入一个重建损失，用于评估生成图片和真实图片之间的差距。

在一些对生成图片的细节要求不高的任务中，可以直接使用L1Loss作为重建损失,

为了得到更细致的生成结果，可以i利用分类的特征提取能力，将生成图片和真实图片在分类网络中得到的特征图之间的差距加入到重建损失中。

## 资料中常提到的技巧

### Batch normalization

在绝大部分的深度学习任务中，Batch normalization都有比较好的效果

Batch normalization对Generator的作用尚有争议，有研究认为Batch normalization在Generator中有负面作用（Gulrajani et al., 2017, [http://arxiv.org/abs/1704.00028](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1704.00028).）

不过一般都认为Batch normalization对Discriminator有积极作用（“Tutorial on Generative Adversarial Networks—GANs in the Wild,” by Soumith Chintala, 2017, [https://www.youtube.com/watch?v=Qc1F3-Rblbw](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DQc1F3-Rblbw).
）

> 在Generator的loss中使用了梯度惩罚的情况下，Discriminator尽量避免使用Batch normalization，可以考虑使用Layer normalization、Weight Normalization或者Instance Normalization等。

### 避免梯度稀疏以及信息丢失

ReLU或者MaxPool产生的稀疏梯度会造成训练困难，生成任务与一般的分类回归任务不同的是，生成任务需要非常完整的细节信息，因此，这些操作中产生的信息丢失会影响到Generator的训练。

因此，在GAN中，因尽量避免使用池化层（MaxPool、AvgPool等），可以使用Leaky-ReLU替代ReLU。

### 指数平均参数

通过对不同epoch下的参数求指数平均可以让训练过程变得更加平稳（Yazici, Yasin et al. “The Unusual Effectiveness of Averaging in GAN Training.” CoRRabs/1806.04498 (2018): n. pag.）
不过指数平均中有一个超参需要设置，不想调这个超参的话，直接只用滑动平均参数也可以获得不错的效果。

末尾贴一个我自己复现的名为W-Net的项目

[https://github.com/Arctanxy/W-Net-PyTorchgithub.com/Arctanxy/W-Net-PyTorch](https://link.zhihu.com/?target=https%3A//github.com/Arctanxy/W-Net-PyTorch)

目前使用过，且有效的训练技巧有：梯度惩罚、标签平滑、在Discriminator中使用更大的学习率、利用分类网络建立重建loss。

后面会陆续补充其他训练技巧及相应的代码实现。





在pytorch中停止梯度流的若干办法，避免不必要模块的参数更新

**避免不需要更新的模型模块被参数更新**。

一般来说，截断梯度流可以有几种思路：

1. 停止计算某个模块的梯度，在优化过程中这个模块还是会被考虑更新，然而因为梯度已经被截断了，因此不能被更新。

- 设置`tensor.detach()`： 完全截断之前的梯度流
- 设置参数的`requires_grad`属性：单纯不计算当前设置参数的梯度，不影响梯度流
- `torch.no_grad()`：效果类似于设置参数的`requires_grad`属性



## **tensor.detach()**

`tensor.detach()`的作用是：

> `tensor.detach()`会创建一个与原来张量共享内存空间的一个新的张量，不同的是，这个新的张量将不会有梯度流流过，这个新的张量就像是从原先的计算图中脱离(detach)出来一样，对这个新的张量进行的任何操作都不会影响到原先的计算图了。因此对此新的张量进行的梯度流也不会流过原先的计算图，从而起到了截断的目的。

以GAN举例

```python
 def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_B
        # fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = self.real_B # GroundTruth
        # real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = self.fake_B
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()


    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

    # 先调用 forward, 再 D backward， 更新D之后； 再G backward， 再更新G
    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
```

**我们注意看第六行，`self.pred_fake = self.netD.forward(fake_AB.detach())`使得在反向传播`D_loss`的时候不会更新到`self.netG`，因为`fake_AB`是由`self.netG`生成的，代码如`self.fake_B = self.netG.forward(self.real_A)`。**

## **设置requires_grad**

**`tensor.detach()`是截断梯度流的一个好办法，但是在设置了`detach()`的张量之前的所有模块，梯度流都不能回流了（不包括这个张量本身，这个张量已经脱离原先的计算图了），如以下代码所示：**

```
x = torch.randn(2, 2)
x.requires_grad = True

lin0 = nn.Linear(2, 2)
lin1 = nn.Linear(2, 2)
lin2 = nn.Linear(2, 2)
lin3 = nn.Linear(2, 2)
x1 = lin0(x)
x2 = lin1(x1)
x2 = x2.detach() # 此处设置了detach，之前的所有梯度流都不会回传了
x3 = lin2(x2)
x4 = lin3(x3)
x4.sum().backward()
print(lin0.weight.grad)
print(lin1.weight.grad)
print(lin2.weight.grad)
print(lin3.weight.grad)
```

**我们发现`lin0.weight.grad`和`lin0.weight.grad`都为`None`了，因为通过脱离中间张量，原先计算图已经和当前回传的梯度流脱离关系了。**

**这样有时候不够理想，因为我们可能存在只需要某些中间模块不计算梯度，但是梯度仍然需要回传的情况，在这种情况下，如下图所示，我们可能只需要不计算`B_net`的梯度，但是我们又希望计算`A_net`和`C_net`的梯度，这个时候怎么办呢？当然，通过`detach()`这个方法是不能用了。**

**事实上，我们可以通过设置张量的`requires_grad`属性来设置某个张量是否计算梯度，而这个不会影响梯度回传，只会影响当前的张量。修改上面的代码，我们有：**

```
x = torch.randn(2, 2)
x.requires_grad = True

lin0 = nn.Linear(2, 2)
lin1 = nn.Linear(2, 2)
lin2 = nn.Linear(2, 2)
lin3 = nn.Linear(2, 2)
x1 = lin0(x)
x2 = lin1(x1)
for p in lin2.parameters():
    p.requires_grad = False
x3 = lin2(x2)
x4 = lin3(x3)
x4.sum().backward()
print(lin0.weight.grad)
print(lin1.weight.grad)
print(lin2.weight.grad)
print(lin3.weight.grad)
```

**只有设置了`requires_grad=False`的模块没有计算梯度，但是梯度流又能够回传。**

**另外，设置`requires_grad`经常用在对输入变量和输入的标签进行新建的时候使用，如：**

## **torch.no_grad()**

**在对训练好的模型进行评估测试时，我们同样不需要训练，自然也不需要梯度流信息了。我们可以把所有参数的`requires_grad`属性设置为`False`，事实上，我们常用`torch.no_grad()`上下文管理器达到这个目的。即便输入的张量属性是`requires_grad=True`, `torch.no_grad()`可以将所有的中间计算结果的该属性临时转变为`False`。**

**如例子所示：**

```text
x = torch.randn(3, requires_grad=True)
x1 = (x**2)
print(x.requires_grad)
print(x1.requires_grad)

with torch.no_grad():
    x2 = (x**2)
    print(x1.requires_grad)
    print(x2.requires_grad)
```

**注意到只是在`torch.no_grad()`上下文管理器范围内计算的中间变量的属性`requires_grad`才会被转变为`False`，在该管理器外面计算的并不会变化。**

**不过和单纯手动设置`requires_grad=False`不同的是，在设置了`torch.no_grad()`之前的层是不能回传梯度的，延续之前的例子如：**

**此处如果我们打印`lin1.weight.requires_grad`我们会发现其为`True`，但是其中间变量`x2.requires_grad=False`。**

**一般来说在实践中，我们的`torch.no_grad()`通常会在测试模型的时候使用，而不会选择在选择性训练某些模块时使用[1]，例子如：**

```text
model.train()
# here train the model, just skip the codes
model.eval() # here we start to evaluate the model
with torch.no_grad():
 for each in eval_data:
  data, label = each
  logit = model(data)
  ... # here we just skip the codes
```

## **注意**

**通过设置属性`requires_grad=False`的方法（包括`torch.no_grad()`）很多时候可以避免保存中间计算的buffer，从而减少对内存的需求，但是这个也是视情况而定的，比如如[2]的所示**

```text
graph LR;
	input-->A_net;
	A_net-->B_net;
	B_net-->C_net;
```

**如果我们不需要`A_net`的梯度，我们设置所有`A_net`的`requires_grad=False`，因为后续的`B_net`和`C_net`的梯度流并不依赖于`A_net`，因此不计算`A_net`的梯度流意味着不需要保存这个中间计算结果，因此减少了内存。**

**但是如果我们不需要的是`B_net`的梯度，而需要`A_net`和`C_net`的梯度，那么问题就不一样了，因为`A_net`梯度依赖于`B_net`的梯度，就算不计算`B_net`的梯度，也需要保存回传过程中`B_net`中间计算的结果，因此内存并不会被减少。**

**但是通过`tensor.detach()`的方法并不会减少内存使用，这一点需要注意。**

## **设置优化器的更新列表**

**这个方法更为直接，即便某个模块进行了梯度计算，我只需要在优化器中指定不更新该模块的参数，那么这个模块就和没有计算梯度有着同样的效果了。如以下代码所示:**

```text
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_1 = nn.linear(10,10)
        self.model_2 = nn.linear(10,20)
        self.fc = nn.linear(20,2)
        self.relu = nn.ReLU()

    def foward(inputv):
        h = self.model_1(inputv)
        h = self.relu(h)
        h = self.model_2(inputv)
        h = self.relu(h)
        return self.fc(h)
```

**在设置优化器时，我们只需要更新`fc层`和`model_2层`，那么则是:**

```text
curr_model = model()
opt_list = list(curr_model.fc.parameters())+list(curr_model.model_2.parameters())
optimizer = torch.optim.SGD(opt_list, lr=1e-4)
```

**当然你也可以通过以下的方法去设置每一个层的学习率来避免不需要更新的层的更新[3]：**

```text
optim.SGD([
                {'params': model.model_1.parameters()},
                {'params': model.mode_2.parameters(), 'lr': 0},
         {'params': model.fc.parameters(), 'lr': 0}
            ], lr=1e-2, momentum=0.9)
```

**这种方法不需要更改模型本身结构，也不需要添加模型的额外节点，但是需要保存梯度的中间变量，并且将会计算不需要计算的模块的梯度（即便最后优化的时候不考虑更新），这样浪费了内存和计算时间。**