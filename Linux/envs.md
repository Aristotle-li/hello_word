## 下载anaconda3

下载anaconda主要是为了管理环境更方便，可能不同的项目需要不同版本的python，这个时候只需要切换不同的虚拟环境跑程序就可以啦！

建议从清华镜像网站上下载比较快（自己下载对应需要的版本即可），命令：

```bash
wget -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.05-Linux-x86_64.sh
```

此处下载到了/home/leo/目录下，接着安装anaconda3，命令：

```text
bash Anaconda3-2021.05-Linux-x86_64.sh
```

接着，一直回车，最后输入yes表示同意该许可证，然后继续回车进行安装，再输yes，直到安装完毕。

重启终端，可以看到进入了base环境下，输入conda命令，可以检查是否安装成功。

## 创建要用的虚拟环境

一般是conda+空格+create+-n+环境名+python=3.6（看你需要用哪个版本，一般3系以上差别不大，可能还需要看自己需要用的包支持哪个版本），命令：

```text
conda create -n py3.6 python=3.6
```

后面输入y就行，3个done代表创建完毕。

切换到新创建的虚拟环境中，命令：

```text
conda activate py3.6
```

| 命令                         | 意义               |
| ---------------------------- | ------------------ |
| conda info --env             | 查看所有的虚拟环境 |
| conda activate 环境名        | 进入指定环境下     |
| conda deactivate 环境名      | 退出指定环境       |
| conda remove -n 环境名 --all | 删除指定环境       |

## 安装pytorch

深度学习框架我用pytorch（其实是目前只会用pytorch），所以下面介绍安装pytorch的步骤。

先用命令nvidia-smi查看cuda版本， 服务器版本是CUDA Version:  xx.x，去pytorch官网上下载对应版本的即可，直接在官网复制特定的命令到终端执行，后面输入y直到3个done出现代表安装成功。（如果一直下载不下来可以去搜一下如何添加清华镜像的下载路径）

输入python进入python终端，执行下面代码，输出True则代表安装成功了。

```python3
import torch
print(torch.cuda.is_available())
```

后面代码中需要用什么包，就直接用conda install 包名就可以了，或者用pip，但具体不同包安装可能有不同的坑，建议装不上就百度看别人的经验。

linux中常用的命令

下载anaconda3

创建要用的虚拟环境

安装pytorch

## linux中常用的命令

先列一些经常用到的命令，（建议需要用什么直接百度更方便），文件夹类的操作用可视化软件Xftp操作就很方便。

| 命令                                                         | 意义                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| pwd                                                          | 显示当前的工作目录                                           |
| ls -a                                                        | 显示当前路径下所有文件，包括隐藏文件                         |
| ls -l                                                        | 显示文件的详细信息                                           |
| cd ..                                                        | 切换到上一级目录                                             |
| cd ~                                                         | 切换到用户的home目录下                                       |
| cd+空格+文件名                                               | 切换到指定的文件中                                           |
| mkdir+空格+文件名                                            | 创建新文件夹                                                 |
| rmdir+空格+文件名                                            | 删除指定文件夹                                               |
| wget+空格+参数+下载路径及文件名                              | 从指定路径下载指定文件，参数常用的为-c代表断点续传，连接中断重连后可继续接着下载；默认下载到当前文件目录里 |
| nvidia-smi                                                   | 查看当前GPU使用情况                                          |
| tar -zcvf /data/xxx/Generatefile_name.tar.gz OridinaryFile_name | Generatefile_name.tar.gz是形成压缩文件的名称（需要具体地址）；OridinaryFile_name为需要压缩的文件夹名称，需要先进入需要压缩文件夹的上一层。 |
| tar -zxvf  xxx.tar.gz -C 解压目录名                          | 解压tar.gz格式压缩包                                         |
| zip -q -r /data/xxx/Generatefile_name.zip OridinaryFile_name/ | 压缩成zip格式的压缩包                                        |
| unzip Generatefile_name.zip                                  | 解压                                                         |

linux中常用的命令

下载anaconda3

创建要用的虚拟环境

安装pytorch

## linux中常用的命令

先列一些经常用到的命令，（建议需要用什么直接百度更方便），文件夹类的操作用可视化软件Xftp操作就很方便。

| 命令                                                         | 意义                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| pwd                                                          | 显示当前的工作目录                                           |
| ls -a                                                        | 显示当前路径下所有文件，包括隐藏文件                         |
| ls -l                                                        | 显示文件的详细信息                                           |
| cd ..                                                        | 切换到上一级目录                                             |
| cd ~                                                         | 切换到用户的home目录下                                       |
| cd+空格+文件名                                               | 切换到指定的文件中                                           |
| mkdir+空格+文件名                                            | 创建新文件夹                                                 |
| rmdir+空格+文件名                                            | 删除指定文件夹                                               |
| wget+空格+参数+下载路径及文件名                              | 从指定路径下载指定文件，参数常用的为-c代表断点续传，连接中断重连后可继续接着下载；默认下载到当前文件目录里 |
| nvidia-smi                                                   | 查看当前GPU使用情况                                          |
| tar -zcvf /data/xxx/Generatefile_name.tar.gz OridinaryFile_name | Generatefile_name.tar.gz是形成压缩文件的名称（需要具体地址）；OridinaryFile_name为需要压缩的文件夹名称，需要先进入需要压缩文件夹的上一层。 |
| tar -zxvf  xxx.tar.gz -C 解压目录名                          | 解压tar.gz格式压缩包                                         |
| zip -q -r /data/xxx/Generatefile_name.zip OridinaryFile_name/ | 压缩成zip格式的压缩包                                        |
| unzip Generatefile_name.zip                                  | 解压                                                         |





## Python中 list, numpy.array, torch.Tensor 格式相互转化           

**1.1 list 转 numpy**

ndarray = np.array(list)

 

**1.2 numpy 转 list**

list = ndarray.tolist()

 

**2.1 list 转 torch.Tensor**

tensor=torch.Tensor(list)

 

**2.2 torch.Tensor 转 list**

先转numpy，后转list

list = tensor.numpy().tolist()

 

**3.1 torch.Tensor 转 numpy**

ndarray = tensor.numpy()

*gpu上的tensor不能直接转为numpy

ndarray = tensor.cpu().numpy()

 

**3.2 numpy 转 torch.Tensor**

tensor = torch.from_numpy(ndarray) 





### pytorch 两种冻结层的方式

### 一、设置requires_grad为False

```python
for param in model.named_parameters():
    if param[0] in need_frozen_list:
        param[1].requires_grad = False
```

这种方法需要注意的是层名一定要和model中一致，model经过.cuda后往往所用层会添加module.的前缀，会导致后面的冻结无效。

还需要注意的是加上filter：

```python
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
```

### 二、使用 torch.no_grad()

这种方式只需要在网络定义中的forward方法中，将需要冻结的层放在 torch.no_grad()下，**强力推这种方式**。

```python
class xxnet(nn.Module):
    def __init__():
        ....
        self.layer1 = xx
        self.layer2 = xx
        self.fc = xx

    def forward(self.x):
        with torch.no_grad():
            x = self.layer1(x)
            x = self.layer2(x)
        x = self.fc(x)
        return x
```

这种方式则是将layer1和layer2定义的层冻结，只训练fc层的参数。



## PyTorch张量维度操作（拼接、扩展、压缩、重复、变形、置换维度）


涉及到的方法有：
拼接torch.cat();torch.stack
扩展torch.Tensor.expand()
压缩torch.squeeze(); torch.Tensor.narrow()
重复torch.Tensor.repeat(); torch.Tensor.unfold()
变形torch.Tensor.view()；torch.Tensor.resize_()
置换维度torch.Tensor.permute()

torch.cat(seq, dim=0, out=None) → Tensor
在指定的维度dim上对序列seq进行连接操作
torch.stack(seq, dim=0, out=None) → Tensor
增加新的维度进行堆叠
torch.Tensor.expand(*sizes) → Tensor
返回张量的一个新视图，可以将张量的单个维度扩大为更大的尺寸。
传入-1则意味着维度扩大不涉及这个维度。
扩大张量不需要分配新内存，仅仅是新建一个张量的视图。

torch.squeeze(input, dim=None, out=None) → Tensor
除去输入张量input中数值为1的维度，并返回新的张量。
当通过dim参数指定维度时，维度压缩操作只会在指定的维度上进行。
如果一个张量只有1个维度，那么它不会受到上述方法的影响。
输出的张量与原张量共享内存，如果改变其中的一个，另一个也会改变。

torch.Tensor.repeat(*sizes)
沿着指定的维度重复张量。不同于expand()方法，本函数复制的是张量中的数据。

torch.Tensor.unfold(dim, size, step) → Tensor
返回一个新的张量，其中元素复制于有原张量在dim维度上的数据，复制重复size次，复制时的步进值为step。

torch.Tensor.narrow(dimension, start, length) → Tensor
返回一个经过缩小后的张量。操作的维度由dimension指定。缩小范围是从start开始到start+length。执行本方法的张量与返回的张量共享相同的底层内存。

torch.Tensor.view(*args) → Tensor
返回一个有相同数据但形状不同的新的张量。
返回的装两必须与原张量有相同的数据和相同的元素个数，但是可以有不同的尺寸。

torch.Tensor.resize_(*sizes)
将张量的尺寸调整为指定的大小。如果元素个数比当前的内存大小大，就将底层存储大小调整为与新元素数目一致的大小。
如果元素个数比当前内存小，则底层存储不会被改变。原来张量中被保存下来的元素将保持不变，但新内存将不会被初始化。

torch.Tensor.permute(*dims)
将执行本方法的张量的维度换位。
