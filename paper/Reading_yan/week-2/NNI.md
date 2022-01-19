## 上新了，NNI！微软开源自动机器学习工具NNI概览及新功能详解



2018年9月，微软亚洲研究院发布了第一版 NNI (Neural Network Intelligence) ，目前已在 GitHub 上获得 3.8K 星，成为最热门的自动机器学习（AutoML）开源项目之一。



作为为研究人员和算法工程师量身定制的自动机器学习工具和框架，NNI 在过去一年中不断迭代更新，我们发布了稳定 API 的 1.0 版本，并且不断将最前沿的算法加入其中，加强对各种分布式训练环境的支持。



最新版本的 NNI 对机器学习生命周期的各个环节做了更加全面的支持，**包括特征工程、神经网络架构搜索（NAS）、超参调优和模型压缩在内的步骤，你都能使用自动机器学习算法来完成**。



无论你是刚刚入门机器学习的小白，还是身经百战的“调参大法师”，NNI 都能助你一臂之力。在这篇文章中，我们会全方位解读 NNI 最新版本中的各项功能，让大家了解这个简单易用的自动机器学习工具。



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210427125327190.png" alt="image-20210427125327190" style="zoom:67%;" />





**概述**



自动机器学习是近年来火热的应用和研究方向，各种自动机器学习工具也层出不穷，它们各有优点与局限性。有的聚焦于算法，但不支持分布式训练；有的功能强大，但没有易用的用户界面，学习成本较高；有的只支持特定领域，不提供通用功能；还有的只能在云端使用。



微软自动深度学习工具 NNI 具备以下优势：



• **支持多种框架**：提供基于 Python 的 SDK，支持 PyTorch、TensorFlow、scikit-learn、LightGBM 等主流框架和库；



• **支持多种训练平台**：除在本机直接运行外，还能通过 SSH 调度一组 GPU 服务器，或通过 FrameworkController、KubeFlow、OpenPAI 等在 Kubernetes 下调度大规模集群；



• **支持机器学习生命周期中的多环节**：特征工程、神经网络架构搜索（NAS）、超参调优和模型压缩等；



• **提供易用的命令行工具和友好的 WEB 用户界面**；



• **大量的示例**能帮助你很快上手；



• 最后划重点，**NNI的所有文档都有中文版**！ 



完整中文文档请参考：https://aka.ms/nnizh



![图片](https://mmbiz.qpic.cn/mmbiz_png/HkPvwCuFwNM7tdB2sQTHv99EUIImz63UWBJrD0kuDNvacvQGCNFdHibJn7Bk98qy8ibJFm6FZyEvx4iandjlWfLwQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

自动机器学习工具对比

### NNI 入门与超参优化



机器学习和人工智能通过近些年的厚积薄发，已经形成不少经典的机器学习算法和深度学习网络，这些算法各有特点，在不同的数据集上所需要的超参也有所不同。而自动机器学习中的超参优化就是为了解决这个问题，通过启动多个实例来找到调优结果较好的组合。



NNI 的超参调优功能就是为这样的场景准备的。在超参搜索算法上，NNI 不仅提供了 TPE、SMAC、进化算法等优秀算法，还提供了遍历、批处理、随机、Hyperband 等十多种算法。另外，还支持自动终止低效实例，加速学习过程。



NNI 的安装基于 Python pip 命令，“pip install nni”即可一步完成。



NNI 的使用也非常简单：首先，定义好需要搜索的超参空间；然后，在需要调参的网络启动之前，通过 NNI 的接口读取参数并在训练中将精确度等指标传入 NNI；最后，配置好要使用的调参算法等，即可开始。



具体过程可参考入门教程：https://aka.ms/nnizq



你也可以在这里找到所有示例：https://aka.ms/nnize

![image-20210427125432092](/Users/lishuo/Library/Application Support/typora-user-images/image-20210427125432092.png)



NNI 的超参调优不仅能用于机器学习，对于各类系统、数据库的繁杂参数都可以根据实际场景进行有针对性的调优。使用过程和超参调优非常类似，通过 Python 为系统传入不同的参数配置，然后将确定的调优指标（如读写速度，磁盘空间大小等）回调给 NNI 即可。



更多信息请访问：https://aka.ms/nnizrd



![图片](https://mmbiz.qpic.cn/mmbiz_png/HkPvwCuFwNM7tdB2sQTHv99EUIImz63US7X8ibIcFfyfHKdzpmEJVYPfFT6WfibOqicgYBfUOCRjHIMSEABlDTYfw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

NNI 在运行中，可随时通过界面了解进度



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210427125509953.png" alt="image-20210427125509953" style="zoom:67%;" />



分析超参之间的关联关系，快速发现规律





**自动特征工程**



特征工程是应用经典机器学习算法的前置步骤，通过特征工程，能让机器学习过程更快得到较好的结果。



前面介绍的 NNI 的超参调优功能，可直接应用于特征增强、自动特征选择等特征工程中的各个子领域。为使新手轻松上手，NNI 还内置了基于梯度和决策树的自动特征选择算法，同时还提供了扩展其它算法的接口。



NNI 团队还对自动特征工程的效果进行了对比，在流行的 colon-cancer、gisette、rcv1、neews20.binary、real-sim 等数据集上进行了测试。我们发现如果在成千上万个特征中仅选择前20个，决策树基本都能取得较好的结果，如果选出更多特征，会得到更好的效果。



更多信息请访问：https://aka.ms/nnizfe





***\*神经网络结构搜索（NAS）\****



神经网络搜索（Neural Architecture Search，简称 NAS）通过自动搜索网络结构来获得较好的性能，在今年涌现了大量研究成果。NAS 算法多种多样，实现也各不相同。



为了促进 NAS 的创新，我们对 NAS 算法抽象与实现进行了探索，让用户不仅能在自己的数据集上直接应用算法，还能很容易地横向比较不同 NAS 算法的效果。



NNI 中实现了 ENAS、DARTS、P-DARTS 算法，并提供了 one-shot 算法的接口。另外，还支持了网络模态（Network Morphism）这样的经典搜索方法。



算法介绍及用法：https://aka.ms/nnizn





**模型压缩**



随着深度学习的发展，模型也越来越大。虽然精确度有了很大的提升，但较大的模型尺寸不仅会影响推理速度，还对部署的硬件要求较高。因此，模型压缩也是一个热门话题。



主要的模型压缩算法可分为两类，一类是压缩网络结构的剪枝算法，另一类是减小模型精度的量化算法。NNI 目前提供了 AGP、L1Filter、Slim、Lottery Ticket、FPGM、QAT、DoReFa 等9种模型压缩算法。用户也可根据需要，通过 NNI 的模型压缩接口来实现自己的压缩算法。



相关算法介绍及用法：https://aka.ms/nnizc





***\*结语\****



随着人工智能的发展，理论和建模方法也始终在不断演进。立足于研究与应用的最前线，我们希望将最好用的工具提供给每一位研究员和算法工程师，加速人工智能领域的发展进步。



2020年，我们将加速创新，力图让 NNI 能够提供全生命周期的自动化框架、更丰富的算法、更高效的分布式调参效率，进一步提升 NNI 的易用性和用户体验。



作为一个开源项目，我们期待大家为 NNI 添加新算法、功能、示例，也希望大家为 NNI 提出建议、报告问题，让我们为大家提供更好的工具，如果您有任何反馈与建议，欢迎在 GitHub 社区中告知我们。



NNI 的 GitHub 社区：https://aka.ms/nniis



扫描下方二维码或点击**阅读原文**，访问 NNI 的中文文档。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210427125548589.png" alt="image-20210427125548589" style="zoom:67%;" />

NNI 中文文档链接：https://aka.ms/nnizh





# 深度解析AutoML工具——NNI：带上超参一起训练

[![松桦](https://pic1.zhimg.com/v2-fe8648487a9445deb8aa0723e33f2f2a_xs.jpg?source=172ae18b)](https://www.zhihu.com/people/wu-chun-sheng-71)

[松桦](https://www.zhihu.com/people/wu-chun-sheng-71)

31 人赞同了该文章



![img](https://pic2.zhimg.com/v2-383d7ee1dfa71a63d72c4a94d19444e1_r.jpg)

NNI (Neural Network Intelligence) 是自动机器学习（AutoML）的工具包。 它通过多种调优的算法来搜索最好的神经网络结构和（或）超参，并支持单机、本地多机、云等不同的运行环境。

------

## 安装指南

### 兼容性

- Linux Ubuntu 16.04 或更高版本
- MacOS 10.14.1
- Windows 10.1809

### 安装

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple nni
```

推荐使用清华源

------

## 工作流程&快速上手

为了理解NNI的工作流程，我们不妨来训练一个**Mnist**手写体识别，网络结构确定之后，NNI可以来帮你找到最优的超参。一个朴素的想法是：**在有限的时间内，NNI测试一定量的超参，返回给你最优的参数组合。**

为了更好地理解NNI的工作流程，首先需要认识几个概念：

- Trial：Trial 是一次尝试，它会使用某组配置（例如，一组超参值，或者特定的神经网络架构）来进行训练，并返回该配置下的得分。本质上就是加入了NNI_API的用户的原始代码。
- Experiment：实验是一次找到模型的最佳超参组合，或最好的神经网络架构的任务。 它由Trial和自动机器学习算法所组成。
- Searchspace：搜索空间是模型调优的范围。 例如，超参的取值范围。
- Configuration：配置是来自搜索空间的一个参数实例，每个超参都会有一个特定的值。
- Tuner：Tuner是一个自动机器学习算法，会为下一个Trial生成新的配置。新的 Trial 会使用这组配置来运行。
- Assessor：Assessor分析Trial的中间结果（例如，测试数据集上定期的精度），来确定 Trial 是否应该被提前终止。
- Training Platform：训练平台是Trial的执行环境。根据Experiment的配置，可以是本机，远程服务器组，或其它大规模训练平台（例如，OpenPAI，Bitahub）。

那么你的实验（Experiment）便是在一定的搜索空间（Searchspace）内寻找最优的一组超参数（Configuration），使得该组参数对应的Mnist（Trail）有最大的准确率，在有限的时间和资源限制下，Tuner和Assessor帮助你更快更好的找到这组参数。

为了更更好地理解NNI的工作流程，我们一起来完成一个在本地（Training Platform）训练的，针对Mnist手写体识别（Trail）的，最佳超参搜索的实验（Experiment）。

1. 下载[Mnist TensorFlow example](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni/tree/master/examples/trials/mnist)（选择tf是因为他自带mnist数据集无需下载）。
2. 打开*mnist.py*，Ctrl+F，搜索nni，得到四个结果，分别在12、198、207、231行，换句话说，**一个原本的训练\*Mnist\*手写体识别代码，加上4行代码，就可以由nni来完成超参优化的工作！**原始代码见*mnist_before.py*。

```python
1     """A deep MNIST classifier using convolutional layers."""
......    
12    import nni # import不解释 （1/4）    
......    
198   nni.report_intermediate_result(test_acc)    # 记录网络训练过程中的准确率 （3/4）    
......    
207   nni.report_final_result(test_acc)    # 返回该超参数组合下网络最终准确率 （4/4）    
......    
231   tuner_params = nni.get_next_parameter()    # 生成下一组超参数组合 （2/4）    
......    
234   params.update(tuner_params)    # 更新params字典    
235   main(params)    # 将超参数组合传入主函数训练
 
```

nni嵌入原始代码生成trail的逻辑如上所示：获取超参数组合，测试该组合的效果并记录，产生下一组超参数，直到达到时间或尝试次数上限。

需要注意的是，nni对于需要搜索的超参数是根据变量名称匹配的，因此网络中超参变量名要和搜索空间中定义的一致！本例使用NNI API的方式进行嵌入，亦可使用Annotation的方式嵌入，详情参考[Annotation](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni/blob/master/docs/zh_CN/TrialExample/Trials.md%23nni-python-annotation)。

\3. 打开*search_space.json*，指定需要进行搜索的变量，以及变量的搜索空间。可以看到**dropout_rate**是在**[0.5, 0.9]**的均匀分布中取值，**conv_size**是在**[2,3,5,7]**四个值中选取。关于更多的采样策略和参数设置可以参考[SearchSpace](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni/blob/master/docs/zh_CN/Tutorial/SearchSpaceSpec.md)。

\4. 打开对应系统的配置文件——Win:*config_windows.yml*；Linux,macOS:*config.yml*，以Win为例，我在重要配置后添加了注释，其余配置使用默认即可。

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1            			# 并行trail数量
maxExecDuration: 1h            			# 实验执行的时间上限
maxTrialNum: 10                			# Trial任务的最大数量
#choice: local, remote, pai
trainingServicePlatform: local			# 训练平台，一般为local
searchSpacePath: search_space.json		# 搜索空间，一般为search_space.json
#choice: true, false
useAnnotation: false	# 本例中我们使用的NNI API的方法进行集成，此处选择false 
                        # 如果设置了 useAnnotation=True，searchSpacePath 字段必须被删除
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner, GPTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE                         # 优化算法
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python mnist.py             
  codeDir: .
  gpuNum: 0					 # GPU可见性设置
```

那么以上配置便意味着我们给nni一个小时的时间、十次尝试机会，在同时只能运行一个trail以及使用TPE优化算法的情况下来寻找最优的超参组合。

除实例教程的基本配置文件外，还有一些配置推荐使用：

```text
logDir: 默认为<user home directory>/nni/experiment    # 指定log输出路径    
logLevel: 支持trace, debug, info, warning, error, fatal，默认为info # 指定log信息输出登记
```

详情请参考[配置](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni/blob/master/docs/zh_CN/Tutorial/ExperimentConfig.md)。

\5. 打开Windows命令行，进入安装了nni的环境，然后启动**MNIST Experiment。**

```
nnictl create --config config_windows.yml
```

nnictl是一个命令行工具，用来控制NNI Experiment，如启动、停止、继续Experiment，启动、停止 NNIBoard 等等。更多用法请参考[nnictl](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni/blob/master/docs/zh_CN/Tutorial/Nnictl.md)。

\6. 启动Experiment 后，可以在命令行界面找到WebUI的地址，在浏览器打开地址就可以看到实验的详细信息，详细教程可参考[WebUI](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni/blob/master/docs/zh_CN/Tutorial/WebUI.md)。

![img](https://pic1.zhimg.com/v2-f0d576ad9874a4cf4d6dae6e60bb80a4_r.jpg)



\7. 实验完成后，WebUI一直可以访问，直到我们在命令行停止实验`nnictl stop`。

至此，NNI的基本使用情况已介绍完毕。

------

## 系统架构

![img](https://pic4.zhimg.com/v2-77e21dbb9e64dfa9a591e49350ab0c07_r.jpg)

- NNIManager是系统的核心管理模块，负责调用TrainingService来管理Trial，并负责不同模块之间的通信。
- Dispatcher是消息处理中心。
- TrainingService是平台管理、任务调度相关的模块，它和 NNIManager 通信，并且根据平台的特点有不同的实现。NNI 支持[本机](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni/blob/master/docs/zh_CN/TrainingService/LocalMode.md)，[远程平台](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni/blob/master/docs/zh_CN/TrainingService/RemoteMachineMode.md)(SSH方法调用多台GPU协同实验)，[OpenPAI 平台](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni/blob/master/docs/zh_CN/TrainingService/PaiMode.md)，[Kubeflow 平台](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni/blob/master/docs/zh_CN/TrainingService/KubeflowMode.md) 以及 [FrameworkController 平台](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni/blob/master/docs/zh_CN/TrainingService/FrameworkControllerMode.md)。
- DB是DateBase管理训练数据.

------

AutoML我懂，NNI我也懂，代码我都写好了，就差一块GPU了？

没有GPU是问题吗？能用钱解决的问题都不是问题！问题是没钱？

**BitaHub**了解一下。

**BitaHub**（[http://www.bitahub.com](https://link.zhihu.com/?target=http%3A//www.bitahub.com)）面向AI开发者提供快速构建、训练模型的能力，让开发者专注于业务和科研。**新用户注册即免费赠送20算力！**

点击[https://forum.bitahub.com/views/page-registe.html?inviteid=_b1f099806011453ebaaec2e2b2bb8cec](https://link.zhihu.com/?target=https%3A//forum.bitahub.com/views/page-registe.html%3Finviteid%3D_b1f099806011453ebaaec2e2b2bb8cec)完成注册！

注册后请先阅读一下BitaHub的[帮助手册](https://link.zhihu.com/?target=https%3A//www.bitahub.com/help/index.html)。

------

## Bitahub&NNI

以上文提到的mnist实验为例，下面我们展示如何在Bitahub上优雅的使用NNI。

实验中相关文件如Dockerfile、trail代码请在（[https://github.com/SonghuaW/bita_nni](https://link.zhihu.com/?target=https%3A//github.com/SonghuaW/bita_nni)）下载。

1. **新建镜像**：NNI提供了官方Docker镜像，为了适配Bitahub平台，我对NNI镜像做了一些修改，并上传至dockerhub：*wushhub/nni_bitahub*，在Bitahub平台上，选择我的镜像（[https://www.bitahub.com/personalmirror](https://link.zhihu.com/?target=https%3A//www.bitahub.com/personalmirror)）-新建镜像（编程语言算法框架可任意选择，但需要与后文创建的项目一致）-上传Dockerfile-提交生成。等待镜像生成成功。
2. **创建项目**：此时项目的编程语言算法框架应选择与新建镜像时的选择一致。代码选择Github上的mnist文件夹。
3. **运行任务**：Bitahub上的任务分两种，debug类型和非debug类型。debug类型提供jupyterlab给用户调试。非debug类型无法调试，直接提交任务，输出结果。由于Docker端口映射的限制，只有在debug模式下，用户才可以使用WebUI。下面详细介绍两种任务类型下如何使用NNI。

- debug模式：

1. 点击新建任务

![img](https://pic2.zhimg.com/v2-b9d5afdaf4b54a92b90d308893bd8d35_r.jpg)

\2. 配置参数，镜像选择我们刚刚制作好的镜像；GPU类型选择dubug；启动命令可以在/mnist/nni.txt中复制。

![img](https://pic3.zhimg.com/v2-aada72e282d183a2b482c6bf235635ae_r.jpg)

debug模式下启动命令为：停止ssh服务；创建/output/nni文件夹作为文件输出目录；启动Experiment并将WebUI端口指定为22；cat占位等待，否则任务将直接结束。

```text
service ssh stop; mkdir -p /output/nni && nnictl create --config /code/mnist/config.yml --port 22 && cat
```

\3. 查看WebUI，当任务状态变成运行中时，点击查看。

![img](https://pic2.zhimg.com/v2-f777c3639f0faabae0199ca6746082b1_r.jpg)

在查看页面下拉至底部

![img](https://pic3.zhimg.com/v2-71d18a11b8490b952d763da6ddd5c94e_r.jpg)

打开202.38.95.226:13304即可访问WebUI。

\4. 在任务查看界面选择Output，即可查看输出文件。

![img](https://pic3.zhimg.com/v2-2b17b4244c87d882695158e9c40aa54e_r.jpg)

\5. 任务结束后，点击停止即可，此后WebUI无法访问，但输出文件一直可以查看。

![img](https://pic1.zhimg.com/v2-9363726d58bc06d0b9d5ff5b3bc602bc_r.jpg)



- 非debug模式：点击新建任务，配置参数，镜像选择我们刚刚制作好的镜像；GPU类型选择dubug；启动命令可以在/mnist/nni.txt中复制。

![img](https://pic1.zhimg.com/v2-a84673de7828ed1d33238c8d65bfa3f0_r.jpg)

```text
mkdir -p /output/nni && nnictl create --config /code/mnist/config.yml --port 22 && python3 /code/mnist/watch.py && nnictl stop
```

非debug模式下我们加入了一个监视器**watch.py**，实时监控experiment的信息，当在nni的log文件中找到‘Experiment done’的字符串时，return并执行nnictl stop。注意，在此模式下我们无法访问WebUI，但仍可读取输出文件。任务在执行nnictl stop后会自行结束，无需手动停止。

------

Open for comments and suggestions!

[wsh0913@mail.ustc.edu.cn](mailto:wsh0913@mail.ustc.edu.cn)

鹏城实验室人工智能中心

------

**Reference**

[https://github.com/microsoft/nni](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni)