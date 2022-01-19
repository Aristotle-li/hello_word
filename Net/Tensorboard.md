# PyTorch下的Tensorboard 使用

[MouLo](https://www.zhihu.com/people/loath-11)

炼丹博士在读

本文主要介绍PyTorch框架下的可视化工具Tensorboard的使用

*面向第一次接触可视化工具的新手<其实是备忘>*

之前用了几天visdom，用起来很方便，但是画的图显得很乱，所以花了一晚上把代码里的visdom都改成了tensorboard。

## Tensorboard 

### 安装

原本是tensorflow的可视化工具，pytorch从1.2.0开始支持tensorboard。之前的版本也可以使用tensorboardX代替。

在使用1.2.0版本以上的PyTorch的情况下，一般来说，直接使用pip安装即可。

```text
pip install tensorboard
```

这样直接安装之后，**有可能**打开的tensorboard网页是全白的，如果有这种问题，解决方法是卸载之后安装更低版本的tensorboard。

```text
pip uninstall tensorboard
pip install tensorboard==2.0.2
```

### Tensorboard的使用逻辑

Tensorboard的工作流程简单来说是

-  将代码运行过程中的，某些你关心的数据保存在一个**文件夹**中：

```text
这一步由代码中的writer完成
```

-  再读取这个**文件夹**中的数据，用浏览器显示出来：

```text
这一步通过在命令行运行tensorboard完成。
```

### 代码体中要做的事

首先导入tensorboard

```text
from torch.utils.tensorboard import SummaryWriter   
```

这里的SummaryWriter的作用就是，将数据以特定的格式存储到刚刚提到的那个**文件夹**中。

首先我们将其实例化

```text
writer = SummaryWriter('./path/to/log')
```

这里传入的参数就是指向文件夹的路径，之后我们使用这个writer对象“拿出来”的任何数据都保存在这个路径之下。

这个对象包含多个方法，比如针对数值，我们可以调用

```text
writer.add_scalar(tag, scalar_value, global_step=None, walltime=None)
```

这里的tag指定可视化时这个变量的名字，scalar_value是你要存的值，global_step可以理解为x轴坐标。

举一个简单的例子： 

```text
for epoch in range(100)
    mAP = eval(model)
    writer.add_scalar('mAP', mAP, epoch)
```

这样就会生成一个x轴跨度为100的折线图，y轴坐标代表着每一个epoch的mAP。这个折线图会保存在指定的路径下（但是现在还看不到）

同理，除了数值，我们可能还会想看到模型训练过程中的图像。

```text
 writer.add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
 writer.add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
```

### **可视化**

我们已经将关心的数据拿出来了，接下来我们只需要在命令行运行：

```text
tensorboard --logdir=./path/to/the/folder --port 8123
```

然后打开浏览器，访问地址http://localhost:8123/即可。这里的8123只是随便一个例子，用其他的未被占用端口也没有任何问题，注意命令行的端口与浏览器访问的地址同步。

如果发现不显示数据，注意检查一下路径是否正确，命令行这里注意是

```text
--logdir=./path/to/the/folder 
```

而不是

```text
--logdir= './path/to/the/folder '
```

另一点要注意的是tensorboard并不是实时显示（visdom是完全实时的），而是默认30秒刷新一次。



### 细节

### 1.变量归类

命名变量的时候可以使用形如

```text
writer.add_scalar('loss/loss1', loss1, epoch)
writer.add_scalar('loss/loss2', loss2, epoch)
writer.add_scalar('loss/loss3', loss3, epoch)
```

的格式，这样3个loss就会被显示在同一个section。

### 2.同时显示多个折线图

假如使用了两种学习率去训练同一个网络，想要比较它们训练过程中的loss曲线，只需要将两个日志文件夹放到同一目录下，并在命令行运行

```text
tensorboard --logdir=./path/to/the/root --port 8123
```