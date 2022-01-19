linux系统下终端创建虚拟环境

首先查看你是否安装了虚拟环境使用的第三方使用工具virtualenv：virtualenv --version ，安virtualenv：pip install virtualenv。创建虚拟环境，virtualenv venv。激活你创建的虚拟环境，

。推出虚拟环境。deactivate

virtualenv 

```
创建

conda create -n XXX

激活/切换

conda activate XXX
source activate XXX
退出
conda deactivate
删除

python的virtualenv (venv)可以直接删除。 如果virtualenv是一个独立的文件夹： rm -rf .venv如果virtualenv和源码在一个目录： rm -rf bin/ include/ lib/ local/

`pwd` 显示当前目录

nvidia-smi # 用来查看显卡的驱动版本可以通过命令查询

再输入nvcc -V即可看到nvidia cuda compiler的版本信息。

```



### 根据已有环境名复制生成新的环境

在服务器上想要使用别人搭好的环境，但是又怕自己对环境的修改更新会影响他人的使用，这个时候可以使用conda命令进行复制环境。
首先假设已经安装了Anaconda。


假设已有环境名为A，需要生成的环境名为B：

```
conda create -n B --clone A根据已有环境路径复制生成新的环境
假设已有环境路径为D:\A，需要生成的新的环境名为B：
```

```
conda create -n B --clone D:\A
```


生成的新的环境的位置在anaconda的安装路径下，一般情况在D:\Anaconda3\envs\文件夹下。

/tmp/pycharm_project_944/PycharmProjects/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn

```
brew install --cask unity
```



                初步介绍几个brew命令
    
        本地软件库列表：brew ls
        查找软件：brew search google（其中google替换为要查找的软件关键字）
        查看brew版本：brew -v  更新brew版本：brew update

现在可以输入命令open ~/.zshrc -e 或者 open ~/.bash_profile -e 整理一下重复的语句(运行 echo $SHELL 可以查看应该打开那一个文件修改)

```csharp
sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
sh -c "$(curl -fsSL https://gitee.com/mirrors/oh-my-zsh/raw/master/tools/install.sh)"
wget https://gitee.com/mirrors/oh-my-zsh/raw/master/tools/install.sh


/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
  
  echo $ZSH_THEME 查看当前主题名称
```

## 创建一个新的conda环境 

用于ML-agent插件包。conda主要用于管理package。你创建了这个conda环境也就意味着你接下来安装的package都会住进这个环境里。创建新的conda环境，终端输入。**conda create -n ml-agents python=3.6**

然后需要激活这个环境 输入下面的内容**activate ml-agents**

trainer_config

```
conda install tensorflow-cpu==2.0
```

 mlagents-learn config/trainer_config.yam --run-id=test --train

 mlagents-learn config/trainer_config.yaml  --tensorflow --run-id=test --train

```
[Ctrl] + d   ## 相当于输入 exit（）
[Ctrl] + c ##如果在Linux 底下输入了错误的指令或参数，想让当前的程序『停掉』的话
```

```bash
cuda/version.txt
```

  pytorch            pytorch/linux-64::pytorch-1.7.1-py3.8_cuda10.1.243_cudnn7.6.3_0

### [Linux指令 vi编辑，保存及退出](https://www.cnblogs.com/tanshaoxiaoji/p/vi.html)

编辑模式
　　使用vi进入文本后，按i开始编辑文本

退出编辑模式 
　　按ESC键，然后：
　　　　退出vi
　　　:q!  不保存文件，强制退出vi命令
　　　 :w   保存文件，不退出vi命令
　　　 :wq  保存文件，退出vi命令

cat/home1/hehao/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2




首发于[Deep Learning Lab](https://www.zhihu.com/column/c_118891383)

![做深度学习需要知道哪些linux指令？（持续更新）](https://pic1.zhimg.com/v2-ac3b7a89ff077a58854e44327f2d3958_1440w.jpg?source=172ae18b)



```
usage: git [--version] [--help] [-C <path>] [-c <name>=<value>]
           [--exec-path[=<path>]] [--html-path] [--man-path] [--info-path]
           [-p | --paginate | -P | --no-pager] [--no-replace-objects] [--bare]
           [--git-dir=<path>] [--work-tree=<path>] [--namespace=<name>]
           <command> [<args>]

These are common Git commands used in various situations:

start a working area (see also: git help tutorial)
   clone     Clone a repository into a new directory
   init      Create an empty Git repository or reinitialize an existing one

work on the current change (see also: git help everyday)
   add       Add file contents to the index
   mv        Move or rename a file, a directory, or a symlink
   restore   Restore working tree files
   rm        Remove files from the working tree and from the index

examine the history and state (see also: git help revisions)
   bisect    Use binary search to find the commit that introduced a bug
   diff      Show changes between commits, commit and working tree, etc
   grep      Print lines matching a pattern
   log       Show commit logs
   show      Show various types of objects
   status    Show the working tree status

grow, mark and tweak your common history
   branch    List, create, or delete branches
   commit    Record changes to the repository
   merge     Join two or more development histories together
   rebase    Reapply commits on top of another base tip
   reset     Reset current HEAD to the specified state
   switch    Switch branches
   tag       Create, list, delete or verify a tag object signed with GPG

collaborate (see also: git help workflows)
   fetch     Download objects and refs from another repository
   pull      Fetch from and integrate with another repository or a local branch
   push      Update remote refs along with associated objects

'git help -a' and 'git help -g' list available subcommands and some
concept guides. See 'git help <command>' or 'git help <concept>'
to read about a specific subcommand or concept.
See 'git help git' for an overview of the system.
```

# 做深度学习需要知道哪些linux指令？（持续更新）

### **系统操作**

linux磁盘管理常用的三个命令df、du和fdisk

- df：列出文件系统的整体文件使用量
- du：检查磁盘空间使用量
- fdisk：用于磁盘分区

```bash
df -hl  #查看磁盘剩余空间
df -h #查看每个根路径的分区大小
du -sh [目录名] #返回该目录的大小
du -sm [文件夹] #返回该文件夹总M数
# 挂载/卸载磁盘
sudo fdisk -l
mount -o rw src_dir dst_dir
umount dir

#即刻关机
sudo shutdown
#定时关机
sudo shutdown -h xx:xx
#重启
sudo reboot
sudo shutdown -r now
sudo init 6
# 
ps aux | grep ssh
```

### 文件操作

```bash
# 文件权限
chmod 777 dir # read,write,executable for all
chmod 777 -R dir # 递归到所有子文件夹
chmod 755 dir  # rwx for owner, rx for group
chmod 755 -R dir  

# ls文件夹下所有文件
ls dir -hl 
# ls文件夹中的前/后N个文件
ls dir | head -N
ls dir | head -n N
ls dir | tail -N
ls dir | tail -n N
# 查找文件
find dir -name "*.xxx"
find dir -name "xxx*"

# 统计文件夹下所有文件个数
ls dir -l  | grep "^-" | wc -l # 只统计一层
ls dir -l  | grep -c "^-"      # 只统计一层
ls dir -lR | gerp "^-" | wc -l # 递归统计所有子文件夹
ls dir -lR | gerp -c "^-"      # 递归统计所有子文件夹
find dir -type f | wc -l   # 递归统计所有子文件夹
find dir -maxdepth 1 -type f | wc -l # 只统计一层
########## 推荐使用find！！！############
# 统计文件夹下某后缀文件个数
ls dir -l | grep ".xxx" | wc -l  #注意不是“*.xxx”
find dir -type f -name "*.xxx" | wc -l
# copy特定后缀文件
find src_dir -type f -name "*.xxx"  | xargs -i cp {} dst_dir
find src_dir -type f -name "*.xxx"  | xargs cp -t dst_dir
# copy前/后N个特定后缀文件
find src_dir -type f -name "*.xxx" | head -n N | xargs -i cp {} dst_dir
find src_dir -type f -name "*.xxx" | head   -N | xargs -i cp {} dst_dir
find src_dir -type f -name "*.xxx" | head -n N | xargs cp -t dst_dir
find src_dir -type f -name "*.xxx" | head   -N | xargs cp -t dst_dir

find src_dir -type f -name "*.xxx" | tail -n N | xargs -i cp {} dst_dir
find src_dir -type f -name "*.xxx" | tail   -N | xargs -i cp {} dst_dir
find src_dir -type f -name "*.xxx" | tail -n N | xargs cp -t dst_dir
find src_dir -type f -name "*.xxx" | tail   -N | xargs cp -t dst_dir
# copy前N个文件,控制一下文件夹搜索的depth
find src_dir -maxdepth 1 -type f | head -N | xargs cp -t dst_dir
find src_dir -maxdepth 1 -type f -name ".xxx"| head -N | xargs cp -t dst_dir
# 随机copy N个文件
find src_dir -type f -name "*.xxx" | shuf -n N | xargs -i cp {} dst_dir
find src_dir -type f -name "*.xxx" | shuf -n N | xargs cp -t dst_dir

# sort and copy the first N files.
find src_dir -type f -name "*.xxx" | sort | head -n N | xargs -i cp {} dst_dir
find src_dir -type f -name "*.xxx" | sort -r | head -n N | xargs cp -t dst_dir

# 从远程服务器往本地服务器copy文件
scp <username>@<ip>:src_dir dst_dir
scp -r <username>@<ip>:src_dir dst_dir # 递归拷贝
# 从本地服务器往远程服务器copy文件
scp src_dir <username>@<ip>:dst_dir 
scp -r src_dir <username>@<ip>:dst_dir  # 递归拷贝

# 把文件夹下所有文件名字写到一个txt里
find src_dir -type f -name "*.xxx" > dst_dir/xxx.txt
# 把所有文件内容写到一个文件里
cat src_dir/xxx* >> dst_file
find src_dir -type f -name "xxx*" | sort | xargs -i cat {} >> dst_file

# 输出文件的前/后N行
head -n N xxx.xxx
head -N xxx.xxx
tail -n N xxx.xxx
tail -N xxx.xxx
# 计算文件行数
wc -l xxx.xxx
# 计算文件words数
wc -w xxx.xxx
# 输出文件中包含某种pattern的行
grep -n "some_pattern" xxx.xxx
# 输出文件中包含某种pattern的行数
grep -c "some_pattern" xxx.xxx
# 计算文件大小
du -h xxx.xxx
# 计算文件夹大小
du -sh dir
# 文件行打乱
shuf -o src_file dst_file # src_file和dst_file同名时，原地执行操作
# 文件切分
split -l N src_file out_file_prefix # 按行切分，每个文件N行，生成的文件个数 = src_file总行数 / N
split -n N src_file out_file_prefix # 指定切分N个文件，但一行可能会被切分到两个文件中

# 文件批量重命名
for file in "*.xxx"; do mv "$file" "${file/.xxx/_123.xxx}";done # a.xxx -> a_123.xxx

# zip包的压缩和解压
zip -r dst_zip_file src_dir
unzip zip_file
# tar包的压缩和解压
tar -cvf dst.tar src_dir/  # 压缩
tar -xvf tar_file  # 解压
# 
```

### 后台运行指令，保存log

```text
nohup  指令 > xxx.log  2>&1 &
```

### ！和sh

做深度学习，我们经常使用的一个强大又非常方便的工具——jupyter notebook，里面有一个非常magic的操作，就是在命令前加入叹号（！），可以直接调用系统命令，例如：

![img](https://pic1.zhimg.com/80/v2-3559dbbd3988721b12558fab773ece10_1440w.jpg)

再结合上一个板块“文件操作”中我们讲到的find操作，就能碰撞出更大的火花：

![img](https://pic2.zhimg.com/80/v2-2ac991f669a41bc2764a9de63068cf59_1440w.jpg)

试想一个场景，你需要用卷积神经网络做分类任务，首先要做的就是对本地的几万张图片做预处理操作，那么怎样方便快捷地获取它们呢？

```python3
image_dir = "/home/abc/image"
imags = !find image_dir -type f -name "*.jpg"
images = list(images)
for i in images:
    # do your own operation.
```

如上，核心就只有一行代码，简便、高效而且功能非常强大。

这里又提出了一个问题，使用！在jupyter notebook里没问题，但是如果我要把代码写在py文件里，那就不行了，好在有强大的替换工具：**sh**。示例如下：

![img](https://pic3.zhimg.com/80/v2-73d4967241be6afc90028164d6cfff8a_1440w.jpg)

在sh里，find，ls等命令变成了它的库函数，相关参数也变成了函数的参数，且都按照字符串的形式传入。

上面示例有一个小问题，检测到的文件名字串最后都是带有“\n”换行符的，用strip()函数可以去掉：

![img](https://pic4.zhimg.com/80/v2-294cd560920e10ded5b0fe704577a80f_1440w.jpg)

或者通过列表表达式：

```python3
import sh
file_list1 = sh.find('./test', '-type', 'f', '-name', '*.txt')
files_list1 = [f.strip() for f in file_list1]
```

命令行里的管道操作变成了sh里的嵌套调用：

![img](https://pic3.zhimg.com/80/v2-937bfcf21dc4e3b22a55e1509a718e6e_1440w.jpg)

### linux三剑客——sed

```bash
## 通用格式：
sed "some pattern" target_file
# 查找替换——只在屏幕打印输出时替换，不改变原文件
sed "s/xx/oo/g" a.txt  # 把a.txt中的xx替换为oo，并打印在屏幕
# 查找替换——在原文件替换
sed -i "s/xx/oo/g" a.txt
# 多处查找替换(原文替换)
sed -i "s/11/22/g;s/33/44/g;s/55/66/g" a.txt #将a.txt中11替换为22，33替换为44，55替换为66
## 总结："s/xx/oo/g"中的s代表substitute，子串模式匹配，g代表整行。xx是源字符，oo是结果字符。

# 删除空行
sed -i "/^\s*$/d" a.txt # d代表删除 
# 复制行
sed -n "m,np" a.txt > b.txt  # 将a.txt的m到n行复制到b.txt中
```



### 文件创建、修改与查看

先理解下重定向 > ,>>,<, <<。

![img](https://pic2.zhimg.com/80/v2-ff3d5d4dd4d3bb34fe60db38a5512421_1440w.jpg)

> 文件描述符：
> 0：标准输入
> 1：标准输出
> 2：标准错误输出

```bash
# 创建并打开文件(存在时打开，不存在时创建并打开)
 ## vi和vim可远程ssh使用，gedit不可以。
vi f1.txt
vim f1.txt
gedit f1.txt

# 只创建文件
touch f2.txt
cd > f2.txt 
cd >> f2.txt # 同上

# 文件内容查看：cat，tac，more，less，head，tail，nl
cat f3.txt # 从第一行开始显示全部文件内容
tac f3.txt # 从最后一行开始倒着显示全部文件内容
more f3.txt # 一页一页显示文件内容，在more运行中，有几个键可以按：space：向下翻一页；Enter:向下一行；q:离开；b:向上翻页；:f :立即显示文件名和行数； /字串：向下搜索字串；n：重复前一个搜寻
less f3.txt # 一页一页显示文件内容，在less运行中，可以输入的命令：space：向下翻一页；Enter:向下一行；q:离开；b:向上翻页；:f :立即显示文件名和行数， /字串：向下搜索字串；？字串：向上搜索字串。n：同上； N：反向重复前一个搜寻
## 如果是为了搜索字串，推荐less，less支持高亮，more不高亮。
head -n N f3.txt # 显示前N行
tail -n N f3.txt # 显示最后N行
nl f3.txt # 显示全部内容，带行号

# 创建并修改文件：echo， cat
echo "some information" > f4.txt # 把“some information”写入文件f4.txt, 如果文件不存在，则创建；如果文件存在，则覆盖原有内容。
echo "some information" >> f4.txt # 同上，但文件存在时不覆盖，而是在原有文件后面进行追加。
cat > f4.txt # 输入此命令后，进入交互模式，此时从键盘输入的字符会输出到文件f4.txt, 覆盖原有内容。按ctrl + c 结束输入。
cat >> f4.txt # 同上，但是不覆盖原有内容，只是在末尾追加。

# 文件内容合并
cat f5.txt f6.txt > f7.txt
cat f5.txt f6.txt >> f7.txt

# 
ls -l > file 2 > /dev/null # 将输出重定向到file,且将错误输出重定向到/dev/null中。 /dev/null：在类Unix系统中，/dev/null，或称空设备，是一个特殊的设备文件，它丢弃一切写入其中的数据（但报告写入操作成功），读取它则会立即得到一个EOF。被称作黑洞。
```

### GPU

```bash
# 查看gpu使用情况及进程号
sudo nvidia-smi
# 每N秒查看一次gpu使用情况及进程号
watch -n N nvidia-smi
# 查看cuda版本
nvcc -V
# 查看使用gpu的进程
sudo fuser -v /dev/nvidia*
# 关闭gpu进程
sudo kill -9 xxx

# 安装cuda & cudnn
# 安装cuda（https://developer.nvidia.com/cuda-downloads）
Ctrl-Alt+F1
sudo service lightdm stop/start
sudo apt remove --purge nvidia*
sudo apt autoremove
sudo ./cuda_10.0.xxx_xxx.xx_linux.run
# cuda测试
cd /usr/local/cuda-10.0/samples/1_Utilities/deviceQuery
make
./deviceQuery
# 安装cudnn（https://developer.nvidia.com/rdp/cudnn-download）
tar -zxvf cudnn-10.0-linux-x64-v7.4.2.24.tgz 
cp cuda/lib64/* /usr/local/cuda-10.0/lib64/
cp cuda/include/* /usr/local/cuda-10.0/include/  

# 添加环境变量
gedit ~/.bashrc
## 文本最后添加以下内容：
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
## 保存退出，打开新终端激活
source ~/.bashrc

# 参考：https://blog.csdn.net/qq_36999834/article/details/107589779?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param
```

### jupyter notebook

远程访问 方法一：

```bash
# 配置jupyter notebook可远程访问
jupyter-notebook --generate-config
jupyter-notebook password # 输入密码并确认，这就是以后的登陆密码
vi /home/username/.jupyter/jupyter_notebook_config.json # 复制里面的sha1码
vi /home/username/.jupyter/jupyter_notebook_config.py
 # 在jupyter_notebook_config.py 文件填入下面配置：
 # 设置默认目录
c.NotebookApp.notebook_dir = u'/home/username/jupyterdata'
 # 允许通过任意绑定服务器的ip访问
c.NotebookApp.ip = '*'
 # 用于访问的端口
c.NotebookApp.port = 8888
 # 不自动打开浏览器
c.NotebookApp.open_browser = False
 # 设置登录密码
c.NotebookApp.password = u'sha1:xxxxxxxxxxxxxxxx' # 上面复制的sha1码

# 使用
 # 在远程服务器输入：
jupyter notebook
 # 打开本地浏览器，输入：
192.168.x.xxx:8888
```

远程访问 方法二（推荐）：

```bash
# 配置jupyter notebook可远程访问
jupyter-notebook --generate-config
jupyter-notebook password # 输入密码并确认，这就是以后的登陆密码

# 使用
jupyter notebook --ip 0.0.0.0
```

conda虚拟环境相关操作：

```bash
# 查看conda环境
conda info -e # 查看所有安装的conda环境
conda info  # 查看当前环境的信息
conda list # 查看当前环境安装了哪些包
# 创建conda环境
conda create -n env_name #创建conda环境
conda create -n env_name python=3.5.2  #创建基于python3.5.2的conda环境，并且安装依赖包。
# 切换conda环境
conda activate env_name #切换到新环境
conda activate root  #切换到root环境
conda deactivate      #退出当前环境
# 删除conda环境
conda remove -n env_name --all
# 清理pkgs下的安装包，节省磁盘空间
conda clean -a
# 查看python版本
import sys
print(sys.version)

# conda安装库
conda install xxx
conda install xxx -c conda-forge # 比上一条拥有更多的库 
# 一个例子
conda create -n tensorflow python=3.5.2
conda activate tensorflow
   #Tensorflow based on GPU
pip install sh
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-win_amd64.whl 
pip install tensorflow==2.0.0-alpha0  #同上一行
conda install opencv=3.4.2
pip install opencv-python
conda install scipy #在当前环境下安装scipy，keras需要
conda install jupyter #
conda install pandas  #
conda install scikit-learn
pip install scikit-learn
pip install scikit-image
pip install lightgbm
pip install xgboost
pip install catboost
conda install seaborn
pip install keras
jupyter notebook


# 安装pytorch
#默认 使用 cuda10.1
pip install torch===1.3.0 torchvision===0.4.1 -f https://download.pytorch.org/whl/torch_stable.
#cuda 9.2
pip install torch==1.3.0+cu92 torchvision==0.4.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
# cpu
pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### 相关库的安装：

- openslide

```bash
pip install openslide-python==1.1.2
sudo apt install python-openslide
```

在jupyter notebook中进行虚拟环境切换：

```bash
# 1. 安装nb_conda
conda install nb_conda
# 2. 启动虚拟环境，假设名字为test
conda activate test
# 3. 在虚拟环境中安装ipykernel模块（nb_conda有时候会工作异常。一个方法是通过ipykernel来注册）
pip install ipykernel
# 4.(可选)将环境添加到kernel中
python -m ipykernel install --user --name test --display-name test 
## 上述命令将把通过conda create创建的名为test的虚拟环境注册到notebook当中，并且其显示名也为test

# 5. 重启jupyter notebook,效果如下图所示
jupyter notebook --ip 172.18.32.195
```

![img](https://pic4.zhimg.com/80/v2-5ed166a9dea2c4fc19074c332897a337_1440w.jpg)

![img](https://pic4.zhimg.com/80/v2-5c508d51cb7d0737c7285ae758913c2b_1440w.jpg)

安装cpp kernel：

```text
conda create -n cpp
conda activate cpp
conda install xeus-cling -c conda-forge
pip install ipykernel
python -m ipykernel install --user --name cpp --display-name cpp

# 重启jupyter notebook
jupyter notebook --ip 0.0.0.0

# 效果如下图所示
```

![img](https://pic4.zhimg.com/80/v2-d2a623fc881fbc029a16c062b65baae3_1440w.jpg)

![img](https://pic3.zhimg.com/80/v2-d7c5a804c81d5dc2618ddf157d3acada_1440w.jpg)

这样，就可以在jupyter notebook里写c++的代码，飞起！

jupyter主题设置

[天意帝：怎么安装Jupyter Notebook主题皮肤并设置教程](https://zhuanlan.zhihu.com/p/54397619)

## SSH

```bash
sudo apt update
sudo apt install openssh-server
set fencs=utf-8,GB18030,ucs-bom,default,latin1
```

## 用户环境变量与系统环境变量

```bash
# 修改用户环境变量
vim ~/.bashrc
     export PATH=xxx:$PATH
更新：
source ~/.bashrc

# 修改系统环境变量
sudo vim /etc/profile
     export PATH=xxx:$PATH
更新：
source /etc/bashrc
设置完后，退出root再重进，root的环境变量又还原了，仍没有改变
解决办法：
上面设置完成后，再执行下述操作：
vim /root/.bashrc ， 在文末添加一句话：
source /etc/profile

然后更新：
source /root/.bashrc
```

### pip使用国内源

```bash
# 使用默认源
pip install xxx
# 使用清华源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xxx
# 使用阿里源
pip install -i http://mirrors.aliyun.com/pypi/simple/ xxx
# 或者配置默认源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 安装samba

```bash
#step1:
sudo apt install samba
sudo apt install smbclient
#step2:
sudo vim /etc/samba/smb.conf
 # 在smb.conf文件的行尾添加：
[xxx] # xxx为samba的共享文件夹
comment = Share Folder require password
browseable = yes
path = /home/xxx
create mask = 0777
directory mask = 0777
valid users = xxx
public = yes
writable = yes
available = yes
:wq
#step3:
sudo /etc/init.d/samba restart 
#step4:
cd /home
sudo mkdir xxx
sudo chmod 777 xxx
#step5:
 # 添加账户
sudo groupadd xxx -g 6000
sudo useradd xxx -g 6000 -d /home/xxx
sudo passwd xxx
sudo usermod -aG users xxx
sudo smbpasswd -a xxx
#step6:
 # 转到windows电脑，添加一个网络位置
\\ip_address\xxx  # ip_address为创建samba的ubuntu电脑的本地ip
```

## **清理空间**

```bash
# 查看apt缓存
sudo du -sh /var/cache/apt
# 清理apt缓存
sudo apt clean
# 查看缩略图缓存
du -sh ~/.cache/thumbnails
#清理缩略图缓存
rm -rf ~/.cache/thumbnails/*
```

## 错误处理

(1) The package *** needs to be reinstalled, but I can't find an archive for it.

```text
$ sudo dpkg --remove --force-all ***
$ sudo apt-get update
```

## **Git**

![img](https://pic4.zhimg.com/80/v2-5d09f5baca51eed5ac270deb9088cbfb_1440w.jpg)



```bash
# 1. 安装完git以后,第一步:
git config --global user.name "xxx"
git config --global user.email "xxx @xxx.com "
# 2. 创建版本库Repository(本地仓库)
mkdir abc
cd abc
git init #执行后在文件夹abc下面会生成.git文件夹,说明成功

# 3. 版本库操作
# 1) 添加文件进暂存区
git add readme.txt # 把文件readme.txt添加进暂存区
git add .  # 把所有已改动文件添加进暂存区.
# 2) 从暂存区提交到版本库
git commit -m "some information"
# 3) 查看当前状态:
    # Changes not staged for commit: 有文件修改未提交暂存区; 
    # Changes to be committed: 已经提交暂存区,但未提交版本库; 
    # nothing to commit, working directory clean: 无需提交,工作区干净.
git status
# 4) 查看历史记录信息
git log # 显示当前版本及当前版本之前的记录信息.
git log --pretty=oneline # 只显示一行
git reflog # 显示所有历史信息, 包含回退版本信息.
# 5) 版本回退
git reset --hard=HEAD^   # 回退一个版本
git reset --hard=HEAD^^  # 回退两个版本
git reset --hard=HEAD~N  # 回退N个版本
## 如果文件未add到暂存区
git checkout -- xxx.xxx  #如果文件xxx.xxx未add到暂存区,checkout上一个版本替换当前更改
git checkout -- . # checkout所有未add到暂存区的文件,用上一个版本替换当前更改
# 6) 删除版本库里的文件
    # 注意,删除文件跟增加文件一样,也需要先add(注意add后面的.),再commit.
rm test.txt
git add .
git commit -m "delete test.txt" 

# 4. 连接远程仓库
## 由于本地git库和远程github仓库之间的传输通过ssh加密,所以需要一点设置:
# 1)创建SSH key.
## 看看本地是否有.ssh文件夹,有的话,里面是否有id_rsa和id_rsa.pub这两个文件,有的话跳过,没有的话输入如下:
ssh-keygen -t rsa –C “youremail@xxx.com”
# 2) 登陆github -> Settings -> SSH and GPG keys -> New SSH key, 
# 在 title部分起个名字,Key部分把id_rsa.pub的内容粘贴进去.点击add key, 完成.
# 3) 添加远程库
## 点击"+" -> new repository -> 在Repository name * 中添加仓库的名字. -> 点击 Create repository
## 把本地仓库和刚创建的远程仓库关联:
git remote add origin https://github.com/your_github_name/your_repository_name.git
git push -u origin master # 本地仓库分支master推送到远程仓库. 第一次需要加 -u 参数,以后都不需要加-u参数.

# 5. 从远程仓库克隆
git clone https://github.com/xxx/xxx.git

# 6. 创建于合并分支
# 1) 创建并切换dev分支
git checkout -b dev
# 2) 创建分支 dev
git branch dev
# 3) 切换分支dev
git checkout dev  # 1) = 2) + 3)
# 4) 查看所有分支
git branch # 带星号*为当前所在分支
# 5) 合并分支
git merge dev
# 6) 删除分支
git branch -d dev
```

## Eclipse快捷键

```bash
ctrl + alt + up/down: 复制选中的代码块，并粘贴选中代码块的上面/下面
alt + / : 补全  # ma ， sysout.
```

## Pytorch

```python3
# 
import torch
import torch.nn as nn
import torchvision

# pytorch, cuda, cudnn 版本信息
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

# gpu type, count, capability, available or not.
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_capability(0))
print(torch.cuda.is_available())
```

## Docker(待更新)

## MySQL(待更新)

## Nginx(待更新)

编辑于 2020-12-31

[深度学习（Deep Learning）](https://www.zhihu.com/topic/19813032)

[Linux](https://www.zhihu.com/topic/19554300)

赞同 662

18 条评论

分享

喜欢收藏



### 文章被以下专栏收录

- [![Deep Learning Lab](https://pic1.zhimg.com/4b70deef7_xs.jpg?source=172ae18b)](https://www.zhihu.com/column/c_118891383)

- ## [Deep Learning Lab](https://www.zhihu.com/column/c_118891383)

- [进入专栏](https://www.zhihu.com/column/c_118891383)

### 推荐阅读

- # 【第一篇】Ubuntu 16.04 服务器深度学习环境共用方案（搭建、分配、克隆、远程）

- 目录（2018-12-30版）方案试验的配置介绍服务器系统安装深度学习基础环境搭建快速配置Jupyter远程交互环境服务器用户分配与DL环境共用用户定制个人DL环境并切换DL环境的高效克隆复现先啰嗦…

- 小白Licko

- # 一篇文章带你配置个人linux深度学习服务器

- 内容较长建议先码后看，文章最后提供可下载的pdf版本。 最近这两天重装了一下系统，安装了我个人必要常用的软件，才了不少坑，在此记录一下，希望能给有需要的朋友一点帮助。 目前配置环境…

- 小哲发表于小哲AI

- # 12 个非常有趣的 Linux 命令！

- \1. sl 命令你会看到一辆火车从屏幕右边开往左边…… 安装 $ sudo apt-get install sl 运行 $ sl 命令有 -a l F e 几个选项， -a An accident seems to happen. You&#39;ll feel pity for pe…

- 芋道源码发表于芋道源码

- ![我的ubuntu连vi都没有？？那在命令行怎么编辑文件？？](https://pic4.zhimg.com/v2-86979c3e300280d1f2db397c5814e922_250x0.jpg?source=172ae18b)

- # 我的ubuntu连vi都没有？？那在命令行怎么编辑文件？？

- 纠结的绳子

## 18 条评论

切换为时间排序

写下你的评论...



发布

- ![知乎用户](https://pic4.zhimg.com/da8e974dc_s.jpg?source=06d4cd63)知乎用户2019-04-21

  有些操作是不需要 sudo 的，比如说 nvidia-smi．另外，正则表达式要求太高了点．

  3回复踩举报

- ](https://pic1.zhimg.com/9478b8318b2d6cbc774144483b4ac13e_s.jpg?source=06d4cd63)](https://www.zhihu.com/people/huo-hua-de-41)[霍华德](https://www.zhihu.com/people/huo-hua-de-41)2019-04-21

  你这么写，记不住的

  1回复踩举报

- 

  哈哈 手打一遍就记住了。深度学习在处理上百万以上的数据以及做各种增强时，这些指令还是比较实用的

  赞回复踩举报


  已改 谢谢

- 

- 

  1.文件操作可以用python,比shell好记
  2.建议添加docker相关

  

## cat

$dsd$

`jkl`

> 

* 

把 textfile1 的文档内容加上行号后输入 textfile2 这个文档里：

```
cat -n textfile1 > textfile2
```

把 textfile1 和 textfile2 的文档内容加上行号（空白行不加）之后将内容附加到 textfile3 文档里：

```
cat -b textfile1 textfile2 >> textfile3
```

清空 /etc/test.txt 文档内容：

```
cat /dev/null > /etc/test.txt
```

cat 也可以用来制作镜像文件。例如要制作软盘的镜像文件，将软盘放好后输入：

```
cat /dev/fd0 > OUTFILE
```

相反的，如果想把 image file 写到软盘，输入：

```
cat IMG_FILE > /dev/fd0
```

**注**：

- \1. OUTFILE 指输出的镜像文件名。
- \2. IMG_FILE 指镜像文件。
- \3. 若从镜像文件写回 device 时，device 容量需与相当。
- \4. 通常用制作开机磁片。