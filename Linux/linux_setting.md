# Ubuntu 18.04 深度学习环境搭建

主要是在 `Ubuntu 18.04` 上安装 `Anaconda` + `CUDA` + `cuDNN` + `PyTorch` + `JupyterLab` + `TensorBoard`。

这里做下记录，给以后的小伙伴作参考。

**服务器的环境配置：**

- `OS`：Ubuntu 18.04.5
- `CPU`：Intel Xeon Gold 6130
- `GPU`：NVIDIA TITAN RTX
- `CUDA`：10.1.243
- `cuDNN`：8.0.4
- `Python`：3.7.9
- `PyTorch`：1.5.1
- `TorchVision`：0.6.1

本文在 `Win 10` 上使用 `MobaXterm`，通过 `SSH` 远程连接服务器进行操作。

该软件可以[点击这里](https://links.jianshu.com/go?to=https%3A%2F%2Fmobaxterm.mobatek.net%2Fdownload-home-edition.html)下载。

连接到服务器后，就可以进行接下来的安装。

## 1 Anaconda

推荐使用 `Anaconda`，能够提供丰富的科学计算包，还有便捷的虚拟环境管理功能。

### 1.1 下载 Anaconda

`wget` 下载：



```cpp
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
```

> --2020-10-12 11:24:59-- [https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh](https://links.jianshu.com/go?to=https%3A%2F%2Frepo.anaconda.com%2Farchive%2FAnaconda3-2020.07-Linux-x86_64.sh)
> Resolving repo.anaconda.com (repo.anaconda.com)... 104.16.130.3, 104.16.131.3, 2606:4700::6810:8303, ...
> Connecting to repo.anaconda.com (repo.anaconda.com)|104.16.130.3|:443... connected.
> HTTP request sent, awaiting response... 200 OK
> Length: 576830621 (550M) [application/x-sh]
> Saving to: ‘Anaconda3-2020.07-Linux-x86_64.sh’
>
> Anaconda3-2020.07-Linux-x86_64.sh 100%[=================================================================================================================>] 550.11M 3.03MB/s in 2m 13s
>
> 2020-10-12 11:27:14 (4.13 MB/s) - ‘Anaconda3-2020.07-Linux-x86_64.sh’ saved [576830621/576830621]

### 1.2 安装 Anaconda

1. `bash` 安装：



```css
bash Anaconda3-2020.07-Linux-x86_64.sh
```

安装时，按要求按 `ENTER` 或输入 `yes` 即可。

> Welcome to Anaconda3 2020.07
>
> In order to continue the installation process, please review the license
> agreement.
> Please, press **ENTER** to continue
>
> Do you accept the license terms? [yes|no]
> [no] >>> **yes**
>
> Anaconda3 will now be installed into this location:
> /home/user/anaconda3
>
> - Press **ENTER** to confirm the location
> - Press **CTRL-C** to abort the installation
> - Or specify a different location below
>
> PREFIX=/home/user/anaconda3
> WARNING: md5sum mismatch of tar archive
> expected: 8a581514493c9e0a1cbd425bc1c7dd90
> got: 762ba0b8c09541d8b435b939ec3b58ff -
> Unpacking payload ...
> Collecting package metadata (current_repodata.json): done
> Solving environment: done
>
> \## Package Plan ##
>
> environment location: /home/user/anaconda3
>
> added / updated specs:
>
> The following NEW packages will be INSTALLED:
>
> Preparing transaction: done
> Executing transaction: done
> installation finished.
>
> Do you wish the installer to initialize Anaconda3
> by running conda init? [yes|no]
> [no] >>> **yes**
>
> no change /home/user/anaconda3/condabin/conda
> no change /home/user/anaconda3/bin/conda
> no change /home/user/anaconda3/bin/conda-env
> no change /home/user/anaconda3/bin/activate
> no change /home/user/anaconda3/bin/deactivate
> no change /home/user/anaconda3/etc/profile.d/conda.sh
> no change /home/user/anaconda3/etc/fish/conf.d/conda.fish
> no change /home/user/anaconda3/shell/condabin/Conda.psm1
> no change /home/user/anaconda3/shell/condabin/conda-hook.ps1
> no change /home/user/anaconda3/lib/python3.8/site-packages/xontrib/conda.xsh
> no change /home/user/anaconda3/etc/profile.d/conda.csh
> modified /home/user/.bashrc

1. 查看 `python` 版本：



```undefined
python --version
```

> Python 3.8.3

## 2 CUDA 和 cuDNN

### 2.1 检查版本

1. 查看 `CUDA` 版本：



```bash
cat /usr/local/cuda/version.txt
```

> CUDA Version 10.1.243

1. 查看 `cuDNN` 版本：



```php
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

> cat /usr/include/cudnn_version.h: No such file or directory

发现服务器已经配置了 `CUDA`，只需安装 `cuDNN` 就好。

### 2.2 安装 CUDA

可参考：[ubuntu 18.04 安装 cuda10.1](https://links.jianshu.com/go?to=https%3A%2F%2Fzhuanlan.zhihu.com%2Fp%2F112138261)

### 2.3 安装 cuDNN

1. 访问 [cuDNN 官网](https://links.jianshu.com/go?to=https%3A%2F%2Fdeveloper.nvidia.com%2Fcudnn)，并注册帐号。
2. 登录后，根据 `CUDA` 版本，选择 `Download cuDNN v8.0.4 (September 28th, 2020), for CUDA 10.1`。
3. 根据系统版本，下载以下三个文件：
   - cuDNN Runtime Library for Ubuntu18.04 (Deb)
   - cuDNN Developer Library for Ubuntu18.04 (Deb)
   - cuDNN Code Samples and User Guide for Ubuntu18.04 (Deb)
4. 上传到服务器后，`dpkg` 依次安装：



```css
sudo dpkg -i libcudnn8_8.0.4.30-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn8-dev_8.0.4.30-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn8-samples_8.0.4.30-1+cuda10.1_amd64.deb
```

> [sudo] password for user:********
> Selecting previously unselected package libcudnn8.
> (Reading database ... 348142 files and directories currently installed.)
> Preparing to unpack libcudnn8_8.0.4.30-1+cuda10.1_amd64.deb ...
> Unpacking libcudnn8 (8.0.4.30-1+cuda10.1) ...
> Setting up libcudnn8 (8.0.4.30-1+cuda10.1) ...
> Processing triggers for libc-bin (2.27-3ubuntu1.2) ...
>
> Selecting previously unselected package libcudnn8-dev.
> (Reading database ... 348160 files and directories currently installed.)
> Preparing to unpack libcudnn8-dev_8.0.4.30-1+cuda10.1_amd64.deb ...
> Unpacking libcudnn8-dev (8.0.4.30-1+cuda10.1) ...
> Setting up libcudnn8-dev (8.0.4.30-1+cuda10.1) ...
> update-alternatives: using /usr/include/x86_64-linux-gnu/cudnn_v8.h to provide /usr/include/cudnn.h (libcudnn) in auto mode
>
> Selecting previously unselected package libcudnn8-samples.
> (Reading database ... 348174 files and directories currently installed.)
> Preparing to unpack libcudnn8-samples_8.0.4.30-1+cuda10.1_amd64.deb ...
> Unpacking libcudnn8-samples (8.0.4.30-1+cuda10.1) ...
> Setting up libcudnn8-samples (8.0.4.30-1+cuda10.1) ...

1. 查看 `cuDNN` 版本：



```php
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

> \#define CUDNN_MAJOR 8
> \#define CUDNN_MINOR 0
> \#define CUDNN_PATCHLEVEL 4

可以看出，版本为 `8.0.4`。

### 2.4 测试 cuDNN

1. 复制样例目录到 `home`：



```bash
cp -r /usr/src/cudnn_samples_v8/ $HOME
```

1. 打开样例目录：



```bash
cd ~/cudnn_samples_v8/mnistCUDNN
```

1. `make` 编译：



```go
make clean && make
```

> rm -rf *o
> rm -rf mnistCUDNN
> Linking agains cublasLt = true
> CUDA VERSION: 10010
> TARGET ARCH: x86_64
> HOST_ARCH: x86_64
> TARGET OS: linux
> SMS: 35 50 53 60 61 62 70 72 75
> /usr/local/cuda/bin/nvcc -ccbin g++ -I/usr/local/cuda/include -I/usr/local/cuda/targets/ppc64le-linux/include -IFreeImage/include -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o fp16_dev.o -c fp16_dev.cu
> g++ -I/usr/local/cuda/include -I/usr/local/cuda/targets/ppc64le-linux/include -IFreeImage/include -o fp16_emu.o -c fp16_emu.cpp
> g++ -I/usr/local/cuda/include -I/usr/local/cuda/targets/ppc64le-linux/include -IFreeImage/include -o mnistCUDNN.o -c mnistCUDNN.cpp
> /usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o mnistCUDNN fp16_dev.o fp16_emu.o mnistCUDNN.o -I/usr/local/cuda/include -I/usr/local/cuda/targets/ppc64le-linux/include -IFreeImage/include -L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/ppc64le-linux/lib -lcublasLt -LFreeImage/lib/linux/x86_64 -LFreeImage/lib/linux -lcudart -lcublas -lcudnn -lfreeimage -lstdc++ -lm

1. 运行结果：



```undefined
./mnistCUDNN
```

> Executing: mnistCUDNN
> cudnnGetVersion() : 8004 , CUDNN_VERSION from cudnn.h : 8004 (8.0.4)
> Host compiler version : GCC 7.5.0
>
> There are 4 CUDA capable devices on your machine :
> device 0 : sms 72 Capabilities 7.5, SmClock 1770.0 Mhz, MemSize (Mb) 24217, MemClock 7001.0 Mhz, Ecc=0, boardGroupID=0
> device 1 : sms 72 Capabilities 7.5, SmClock 1770.0 Mhz, MemSize (Mb) 24220, MemClock 7001.0 Mhz, Ecc=0, boardGroupID=1
> device 2 : sms 72 Capabilities 7.5, SmClock 1770.0 Mhz, MemSize (Mb) 24220, MemClock 7001.0 Mhz, Ecc=0, boardGroupID=2
> device 3 : sms 72 Capabilities 7.5, SmClock 1770.0 Mhz, MemSize (Mb) 24220, MemClock 7001.0 Mhz, Ecc=0, boardGroupID=3
> Using device 0
>
> Result of classification: 1 3 5
>
> **Test passed!**

出现 `Test passed!`，即表示测试通过。

## 3 PyTorch

### 3.1 搭建虚拟环境

1. 创建虚拟环境：



```undefined
conda create -n pytorch python=3.7
```

1. 激活虚拟环境：



```undefined
conda activate pytorch
```

相关命令：



```php
conda env list  # 查看已创建的虚拟环境
conda deactivate  # 退出虚拟环境
conda create -n <new_envname> --clone <old_envname>  # 克隆虚拟环境
conda remove -n <envname> --all  # 删除虚拟环境
```

> **注意：**
> 接下来均在该虚拟环境下操作。

1. 查看 `python` 版本：



```undefined
python --version
```

> Python 3.7.9

### 3.2 安装常用的包

`conda` 安装：



```undefined
conda install numpy scipy pandas matplotlib seaborn scikit-learn
```

其他的包，可根据需要，另行安装。

相关命令：



```php
conda list  # 查看已安装的包
conda update <package>  # 更新包
conda uninstall <package>  # 卸载包
```

### 3.3 安装 PyTorch

1. `conda` 安装：



```swift
conda install pytorch=1.5 torchvision cudatoolkit=10.1 -c pytorch
```

1. 在 `Ipython` 查看 `pytorch` 版本，并检测 `cuda` 是否可用：



```css
In [1]: import torch
In [2]: print(torch.__version__)
In [3]: print(torch.cuda.is_available())
```



```css
Out [2]: 1.5.1
Out [3]: True
```

没有报错，即证明安装成功。

## 4 JupyterLab

### 4.1 安装 JupyterLab

`pip` 安装：



```undefined
pip install jupyterlab
```

### 4.2 生成 hash 密码

1. 在 `Ipython` 输入：



```css
In [1]: from notebook.auth import passwd
In [2]: passwd()
```

1. 连续输入两次密码：



```undefined
Enter password:********
Verify password:********
```

得到 `sha1` 开头的字符串，即为 `hash` 密码：



```css
Out [2]: sha1:2a2736453bda:3a339d14ae9a890bea91cab3f0ad713fdfb22688
```

### 4.3 修改配置文件

1. 生成配置文件：



```undefined
jupyter lab --generate-config
```

1. 打开 `~/.jupyter/jupyter_notebook_config.py`
2. 找到以下参数，取消注释，并修改赋值：



```python
c.NotebookApp.allow_remote_access = True  # 允许远程访问
c.NotebookApp.allow_root = True  # 允许以 root 的方式运行
c.NotebookApp.ip = '0.0.0.0'  # 允许任意 ip 访问
c.NotebookApp.notebook_dir = "/home/user"  # 设置页面的根目录，user 改为自己的用户名
c.NotebookApp.open_browser = False  # 运行时不启动浏览器
c.NotebookApp.password = u"sha1:********"  # 设置之前生成的 hash 密码，记得在字符串前加 u
c.NotebookApp.port = 8880  # 设置访问端口，可改为任意不冲突的端口号
```

1. 保存文件即可。

### 4.4 运行 JupyterLab

1. 在终端输入：



```undefined
jupyter lab
```

> [I 16:54:58.623 LabApp] JupyterLab extension loaded from /home/user/anaconda3/envs/pytorch/lib/python3.7/site-packages/jupyterlab
> [I 16:54:58.624 LabApp] JupyterLab application directory is /home/user/anaconda3/envs/pytorch/share/jupyter/lab
> [I 16:54:58.630 LabApp] Serving notebooks from local directory: /home/user
> [I 16:54:58.630 LabApp] Jupyter Notebook 6.1.4 is running at:
> [I 16:54:58.630 LabApp] [http://poweredge:8880/](https://links.jianshu.com/go?to=http%3A%2F%2Fpoweredge%3A8880%2F)
> [I 16:54:58.630 LabApp] Use **Control-C** to stop this server and shut down all kernels (twice to skip confirmation).

1. 打开浏览器，在地址栏输入 `<host_ip>:8880`，并在页面内输入密码。

> **Tips:**
> `host_ip` 为服务器的 `IP` 地址。

1. 进入 `JupyterLab` 界面。

### 4.5 结束 JupyterLab

1. 关闭 `JupyterLab` 页面。
2. 在终端连续按两次 `Ctrl+C`，结束 `JupyterLab` 进程：

> **^C**[I 17:02:12.279 LabApp] interrupted
> Serving notebooks from local directory: /home/user
> 0 active kernels
> Jupyter Notebook 6.1.4 is running at:
> [http://poweredge:8880/](https://links.jianshu.com/go?to=http%3A%2F%2Fpoweredge%3A8880%2F)
> Shutdown this notebook server (y/[n])? **^C**[C 17:02:12.487 LabApp] received signal 2, stopping
> [I 17:02:12.488 LabApp] Shutting down 0 kernels
> [I 17:02:12.489 LabApp] Shutting down 0 terminal

## 5 TensorBoard

### 5.1 安装 TensorBoard

`pip` 安装：



```undefined
pip install tensorboard
```

### 5.2 运行 TensorBoard

1. 在终端输入：



```xml
tensorboard --logdir=<tf_log_dir> --host=<host_ip> --port=6660
```

命令参数：

- `--logdir`：设置 `tfevents` 日志的目录路径
- `--host`：设置服务器的 `IP` 地址
- `--port`：设置访问端口，默认值为 `6006`

1. 打开浏览器，在地址栏输入 `<host_ip>:6660`。
2. 进入 `TensorBoard` 界面。

> **Tips:**
> 如果打不开界面，可尝试以下操作：
>
> 1. 检查终端是否按了 `Ctrl+C`，导致结束了 `TensorBoard` 进程
> 2. 检查浏览器是否为 `Chrome` 内核
> 3. 检查日志目录路径是否存在 `tfevents` 日志文件
> 4. 检查端口号是否冲突
> 5. 使用 `--host` 设置服务器的 `IP` 地址

### 5.3 结束 TensorBoard

1. 关闭 `TensorBoard` 页面。
2. 在终端按 `Ctrl+C`，结束 `TensorBoard` 进程。

## 6 结语

好啦，安装到这里就结束了，有帮助的话，请点个赞，谢谢~

接下来会说下 `MMDetection` 的安装和使用。





[
码农家园](https://www.codenong.com/)

## ubuntu18.04:安装nvidia-440驱动+cuda10.2+cudnn8.0+pytorch





一、安装nvidia驱动：https://blog.csdn.net/u014754541/article/details/97108282

1.检查电脑gpu是否CUDA-capable:

```bash
 lspci | grep -i nvidia   #没有lspci就安装  
 apt install pciutils
```



2.禁用nouveau并重启：

```bash
sudo vim /etc/modprobe.d/blacklist.conf  #打开禁用列表 #在文本最后一行添加 
blacklist nouveau 
options nouveau modeset=0
```



3.更新后重启：

```bash
sudo update-initramfs -u 
sudo reboot
```



4.查看是否禁用nouveau：执行后没有任何输出证明禁用成功；

```bash
lsmod | grep nouveau
```



5.删除旧的NVIDIA驱动：无旧的NVIDIA可忽略这步

```bash
sudo apt-get remove nvidia-* 
sudo apt-get autoremove 
sudo apt-get update #更新系统软件仓库列表
```



6.使用下面的命令查看系统推荐安装哪个版本的N卡驱动:发现推荐nvidia-440

```bash
ubuntu-drivers devices
```



下表是官方cuda和驱动对应版本：因为安装cuda10.2,因此也是安装nvidia-440

![img](https://i2.wp.com/img-blog.csdnimg.cn/20200616215728936.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODY2MTQ0Nw==,size_16,color_FFFFFF,t_70)

如果直接用下行命令可能安装的不是自己想要的驱动版本：

```bash
sudo ubuntu-drivers autoinstall
```



因此安装新版本的驱动前需要添加源：

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa sudo apt update
```

7.安装驱动：

```bash
sudo apt install nvidia-driver-440
```



8.安装完成后重启，nvidia-smi测试是否安装成功。

二、安装cuda10.2：

下载地址：https://developer.nvidia.com/cuda-toolkit-archive

![img](https://i2.wp.com/img-blog.csdnimg.cn/20200616220259941.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODY2MTQ0Nw==,size_16,color_FFFFFF,t_70)

1.选择要下载的cuda版本，然后执行安装命令行。

2.在执行第二步时，在出现的提示中选择continue和accept,直到出现下图：

![img](https://i2.wp.com/img-blog.csdnimg.cn/20200616222237236.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODY2MTQ0Nw==,size_16,color_FFFFFF,t_70)

nvidia的显卡驱动刚才安装过了，那么只需要移动到Driver，按enter键，将＂［］＂中的Ｘ去掉即是不选择．然后移动到Install再回车，等待后出现下图表示安装成功：

![img](https://i2.wp.com/img-blog.csdnimg.cn/20200616222416313.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODY2MTQ0Nw==,size_16,color_FFFFFF,t_70)

3.添加环境变量：

```bash
vi ~/.bashrc
```



在文件末尾添加：

```bash
export PATH="/usr/local/cuda-10.2/bin:$PATH" 
export LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH"
export PATH="/home1/hehao/cuda-10.1/bin:$PATH" 
export LD_LIBRARY_PATH="/home1/hehao/cuda-10.1/lib64:$LD_LIBRARY_PATH"
```

使其生效：

```bash
source ~/.bashrc
```



输入 nvcc -V，显示版本信息即安装成功。

![img](https://i2.wp.com/img-blog.csdnimg.cn/2020061622305596.png)

三、安装cudnn

下载地址：https://developer.nvidia.com/cudnn（需要注册账号才能下载）

![img](https://i2.wp.com/img-blog.csdnimg.cn/2020061623154570.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODY2MTQ0Nw==,size_16,color_FFFFFF,t_70)

```bash
# 安装runtime库 
sudo dpkg -i libcudnn8_8.0.0.180-1+cuda10.2_amd64.deb # 安装developer库 
sudo dpkg -i libcudnn8-dev_8.0.0.180-1+cuda10.2_amd64.deb # 安装代码示例和《cuDNN库用户指南》 
sudo dpkg -i libcudnn8-doc_8.0.0.180-1+cuda10.2_amd64.deb
```



验证cuDNN在Linux上是否安装成功。为了验证cuDNN已经安装并正确运行，需要编译位于/usr/src/cudnn_samples_v8目录下的mnistCUDNN样例:

```bash
# 将cuDNN示例复制到可写路径
$ cp -r /usr/src/cudnn_samples_v8/ $HOME
# 进入到可写路径
$ cd  $HOME/cudnn_samples_v8/mnistCUDNN
# 编译mnistCUDNN示例
$ make clean && make
# 运行mnistCUDNN示例
$ ./mnistCUDNN
```



![img](https://i2.wp.com/img-blog.csdnimg.cn/2020061623285828.png)

四、安装pytorch

进入网址https://pytorch.org/，选择合适的配置后执行命令：

![img](https://i2.wp.com/img-blog.csdnimg.cn/20200616233518264.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODY2MTQ0Nw==,size_16,color_FFFFFF,t_70)

测试pytorch和cuda加速是否安装成功：

```bash
python3 
import torch 
torch.cuda.is_available() #返回True代表cuda加速成功
```



![img](https://i2.wp.com/img-blog.csdnimg.cn/20200616234438473.png)

至此，配置nvidia驱动+cuda+cudnn+pytorch完成。

