[5个必须记住的SSH命令        ](https://www.cnblogs.com/weafer/archive/2011/06/10/2077852.html)      

### **1.目录操作**

```text
cd                                      // 前进
cd ..                                   // 后退一级
ls                                      // 查看当前目录下的所有目录和文件
mkdir new_dir                           // 新建名为"new_dir"的文件夹
pwd                                     // 显示当前位置路径
```

### **2. 文件操作**

```text
touch a.txt                             // 在当前目录下新增文件a.txt
rm a.txt                                // 删除文件a.txt
tar -zcvf test.zip test                 // 文件打包，将文件夹test打包为文件包test.zip
unzip test.zip                          // 解压文件test.zip
mv a.txt b.txt                          // 将文件a.txt重命名为b.txt 
mv /a /b /c                             // 将目录a移动到目录b下，并重新命名为目录c
```

##    **二、通过ssh远程访问GPU的Jupyter Notebook**

我们编写python经常会遇到jupyter  notebook的格式，即ipynb，实际上这也是经常使用的机器学习训练方法，当然做工程pycharm应该更合适。这里我们讲解如何通过ssh远程访问GPU上的Jupyter Notebook，并将它在本地电脑上可视化，分为两个步骤即可。

### **1. 首先在远程GPU服务器的terminal上启动Jupyter Notebook的服务**

在终端输入以下代码：

```text
jupyter notebook --no-browser --port=8889
```

将远端的Jupyter端口设置为8889.

### **2. 然后在本地terminal上启动ssh，对接端口**

在本地终端cmd输入以下代码：

```text
ssh -N -f -L localhost:8888:localhost:8889 username@serverIP
```

-N 告诉SSH没有命令要被远程执行； -f 告诉SSH在后台执行； -L 是指定port forwarding的配置，远端端口是8889，本地的端口号的8888。

### **4. 最后启动本地端口，并输入指令**

最后打开浏览器访问：http://localhost:8888/

如果是第一次访问，他会让你输入远程端给出的指令，即token密码，可由ssh工具终端的界面上复制粘贴获得。

![img](https://pic4.zhimg.com/80/v2-514be6746a6ff9d7014456f84fd4116f_1440w.jpg)

输入token即可远程访问jupyter的内容了。

至于如何使用Pycharm本地训练DL代码，我更新在了专栏另一篇文章中。

[阿尔法杨XDU：Pycharm通过ssh远程连接GPU服务器训练深度学习代码](https://zhuanlan.zhihu.com/p/259683970)

OpenSSH是SSH连接工具的免费版本。telnet，rlogin和ftp用户可能还没意识到他们在互联网上传输的密码是未加密的，但SSH是加密的，OpenSSH加密所有通信（包括密码），有效消除了窃听，连接劫持和其它攻击。此外，OpenSSH提供了安全隧道功能和多种身份验证方法，支持SSH协议的所有版本。

SSH是一个非常伟大的工具，如果你要在互联网上远程连接到服务器，那么SSH无疑是最佳的候选。下面是通过网络投票选出的25个最佳SSH命令，你必须牢记于心。

（注：有些内容较长的命令，在本文中会显示为截断的状态。如果你需要阅读完整的命令，可以把整行复制到您的记事本当中阅读。）

**1、复制SSH密钥到目标主机，开启无密码SSH登录**

```
第一步：
ssh-Keygen

第二步
ssh-copy-id -i ~/.ssh/id_rsa.pub 172.18.32.195

```

如果还没有密钥，请使用ssh-keygen命令生成。

**2、从某主机的80端口开启到本地主机2001端口的隧道**

```
ssh -N -L2001:localhost:80 somemachine
```

现在你可以直接在浏览器中输入http://localhost:2001访问这个网站。

**3、将你的麦克风输出到远程计算机的扬声器**

```
dd if=/dev/dsp | ssh -c arcfour -C username@host dd of=/dev/dsp
```

这样来自你麦克风端口的声音将在SSH目标计算机的扬声器端口输出，但遗憾的是，声音质量很差，你会听到很多嘶嘶声。

**4、比较远程和本地文件**

```
ssh user@host cat /path/to/remotefile | diff /path/to/localfile –
```

在比较本地文件和远程文件是否有差异时这个命令很管用。

**5、通过SSH挂载目录/文件系统**

```
sshfs name@server:/path/to/folder /path/to/mount/point
```

从http://fuse.sourceforge.net/sshfs.html下载sshfs，它允许你跨网络安全挂载一个目录。

**6、通过中间主机建立SSH连接**

```
ssh -t reachable_host ssh unreachable_host
```

Unreachable_host表示从本地网络无法直接访问的主机，但可以从reachable_host所在网络访问，这个命令通过到reachable_host的“隐藏”连接，创建起到unreachable_host的连接。

**7、将你的SSH公钥复制到远程主机，开启无密码登录 – 简单的方法**

```
ssh-copy-id username@hostname
```

**8、直接连接到只能通过主机B连接的主机A**

```
ssh -t hostA ssh hostB
```

当然，你要能访问主机A才行。

**9、创建到目标主机的持久化连接**

```
ssh -MNf <user>@<host>
```

在后台创建到目标主机的持久化连接，将这个命令和你~/.ssh/config中的配置结合使用：

```
Host host
ControlPath ~/.ssh/master-%r@%h:%p
ControlMaster no
```

所有到目标主机的SSH连接都将使用持久化SSH套接字，如果你使用SSH定期同步文件（使用rsync/sftp/cvs/svn），这个命令将非常有用，因为每次打开一个SSH连接时不会创建新的套接字。

**10、通过SSH连接屏幕**

```
ssh -t remote_host screen –r
```

直接连接到远程屏幕会话（节省了无用的父bash进程）。

**11、端口检测（敲门）**

```
knock <host> 3000 4000 5000 && ssh -p <port> user@host && knock <host> 5000 4000 3000
```

在一个端口上敲一下打开某个服务的端口（如SSH），再敲一下关闭该端口，需要先安装knockd，下面是一个配置文件示例。

```
[options]
logfile = /var/log/knockd.log
[openSSH]
sequence = 3000,4000,5000
seq_timeout = 5
command = /sbin/iptables -A INPUT -i eth0 -s %IP% -p tcp –dport 22 -j ACCEPT
tcpflags = syn
[closeSSH]
sequence = 5000,4000,3000
seq_timeout = 5
command = /sbin/iptables -D INPUT -i eth0 -s %IP% -p tcp –dport 22 -j ACCEPT
tcpflags = syn
```

**12、删除文本文件中的一行内容，有用的修复**

```
ssh-keygen -R <the_offending_host>
```

在这种情况下，最好使用专业的工具。

**13、通过SSH运行复杂的远程shell命令**

```
ssh host -l user $(<cmd.txt)
```

更具移植性的版本：

```
ssh host -l user “`cat cmd.txt`”
```

**14、通过SSH将MySQL数据库复制到新服务器**

```
mysqldump –add-drop-table –extended-insert –force –log-error=error.log -uUSER -pPASS OLD_DB_NAME | ssh -C user@newhost “mysql -uUSER -pPASS NEW_DB_NAME”
```

通过压缩的SSH隧道Dump一个MySQL数据库，将其作为输入传递给mysql命令，我认为这是迁移数据库到新服务器最快最好的方法。

**15、删除文本文件中的一行，修复“SSH主机密钥更改”的警告**

```
sed -i 8d ~/.ssh/known_hosts
```

**16、从一台没有SSH-COPY-ID命令的主机将你的SSH公钥复制到服务器**

```
cat ~/.ssh/id_rsa.pub | ssh user@machine “mkdir ~/.ssh; cat >> ~/.ssh/authorized_keys”
```

如果你使用Mac OS X或其它没有ssh-copy-id命令的*nix变种，这个命令可以将你的公钥复制到远程主机，因此你照样可以实现无密码SSH登录。

**17、实时SSH网络吞吐量测试**

```
yes | pv | ssh $host “cat > /dev/null”
```

通过SSH连接到主机，显示实时的传输速度，将所有传输数据指向/dev/null，需要先安装pv。

如果是Debian：

```
apt-get install pv
```

如果是Fedora：

```
yum install pv
```

（可能需要启用额外的软件仓库）。

**18、如果建立一个可以重新连接的远程GNU screen**

```
ssh -t user@some.domain.com /usr/bin/screen –xRR
```

人们总是喜欢在一个文本终端中打开许多shell，如果会话突然中断，或你按下了“Ctrl-a  d”，远程主机上的shell不会受到丝毫影响，你可以重新连接，其它有用的screen命令有“Ctrl-a  c”（打开新的shell）和“Ctrl-a  a”（在shell之间来回切换），请访问http://aperiodic.net/screen/quick_reference阅读更多关于screen命令的快速参考。

**19、继续SCP大文件**

```
rsync –partial –progress –rsh=ssh $file_source $user@$host:$destination_file
```

它可以恢复失败的rsync命令，当你通过VPN传输大文件，如备份的数据库时这个命令非常有用，需要在两边的主机上安装rsync。

```
rsync –partial –progress –rsh=ssh $file_source $user@$host:$destination_file local -> remote
```

或

```
rsync –partial –progress –rsh=ssh $user@$host:$remote_file $destination_file remote -> local
```

**20、通过SSH W/ WIRESHARK分析流量**

```
ssh root@server.com ‘tshark -f “port !22″ -w -' | wireshark -k -i –
```

使用tshark捕捉远程主机上的网络通信，通过SSH连接发送原始pcap数据，并在wireshark中显示，按下Ctrl+C将停止捕捉，但也会关闭wireshark窗口，可以传递一个“-c  #”参数给tshark，让它只捕捉“#”指定的数据包类型，或通过命名管道重定向数据，而不是直接通过SSH传输给wireshark，我建议你过滤数据包，以节约带宽，tshark可以使用tcpdump替代：

```
ssh root@example.com tcpdump -w – ‘port !22′ | wireshark -k -i –
```

**21、保持SSH会话永久打开**

```
autossh -M50000 -t server.example.com ‘screen -raAd mysession’
```

打开一个SSH会话后，让其保持永久打开，对于使用笔记本电脑的用户，如果需要在Wi-Fi热点之间切换，可以保证切换后不会丢失连接。

**22、更稳定，更快，更强的SSH客户端**

```
ssh -4 -C -c blowfish-cbc
```

强制使用IPv4，压缩数据流，使用Blowfish加密。

**23、使用cstream控制带宽**

```
tar -cj /backup | cstream -t 777k | ssh host ‘tar -xj -C /backup’
```

使用bzip压缩文件夹，然后以777k bit/s速率向远程主机传输。Cstream还有更多的功能，请访问http://www.cons.org/cracauer/cstream.html#usage了解详情，例如：

```
echo w00t, i’m 733+ | cstream -b1 -t2
```

**24、一步将SSH公钥传输到另一台机器**

```
ssh-keygen; ssh-copy-id user@host; ssh user@host
```

这个命令组合允许你无密码SSH登录，注意，如果在本地机器的~/.ssh目录下已经有一个SSH密钥对，ssh-keygen命令生成的新密钥可能会覆盖它们，ssh-copy-id将密钥复制到远程主机，并追加到远程账号的~/.ssh/authorized_keys文件中，使用SSH连接时，如果你没有使用密钥口令，调用ssh user@host后不久就会显示远程shell。

**25、将标准输入（stdin）复制到你的X11缓冲区**

```
ssh user@host cat /path/to/some/file | xclip
```

你是否使用scp将文件复制到工作用电脑上，以便复制其内容到电子邮件中？xclip可以帮到你，它可以将标准输入复制到X11缓冲区，你需要做的就是点击鼠标中键粘贴缓冲区中的内容



## **如何连上远程服务器**

### 连上服务器

首先，当然得是要用自己的PC连接上服务器。如果你的PC是linux系统，那么可以直接通过ssh指令进行远程访问，这里不详细说明。接下来我们主要说PC是windows上的操作。首先，我们要下一个用于ssh连接的工具，个人推荐的是Xshell。其打开后的界面如下图所示：

然后我们依次文件->新建得到下图：

按图中红色字操作，注意一下，以上的应用场景主要是在你的PC和你的服务器是在同一个局域网下的（通俗说连的是同一个wifi），如果不在同一个局域网下，本文不做叙述。

确定之后，会弹出一个会话框，选中你新建的会话名称然后点击连接，就会让你先输入你在服务器上的用户名和密码，确定之后显示如下界面就表示连接成功了。

服务器与本地电脑之间传输文件

如果想往服务器上传文件，可以点击下图红色方框中的按钮。

得到

左侧是本地文件，右侧是服务器文件，两边文件拖动就可以实现文件互传了。真的很方便。

## 如何让代码在后台运行

由于实验室的网实在是不稳定，所以经常遇到跑了好几个小时的代码快要出结果的时候却断网了，导致与服务器的连接中断，代码也就自然而然的停止运行了（至于其中的具体原因可以自行百度）。这点真的让人很苦恼。同时，当你的ssh在执行一个代码时，你如果不新建一个连接，你在这个连接中是无法干其他的事情的，这一点也很不好。于是，考虑可以把代码放到服务器后台运行。

### 第一种 nohup

最开始的做法是

```
$ nohup python test.py
1
```

这样执行的时候会将代码放在服务器后台执行，你的终端是看不到运行过程的，期间运行的结果（代码运行过程中打印出来的）会在一个生成的nohup.out文件中保存。

### 第二种 screen

后来接触到了screen命令，觉得着实好用，在这里极力推荐。

可以简单的认为用这个命令你可以为不同的任务开不同的窗口，这个窗口之间是可以切换的，同时，窗口和你的会话连接基本上没有任何区别，这样你可以在开一个连接的时候同时干多件事情，并且在终端看得到运行过程的同时而不会由于断网而导致代码停止运行。其常用命令如下：

##### 创建一个窗口

screen -S name #创建一个窗口，并且为这个窗口命名$screen -S yolo
 当你执行完以上命令后,就会自动跳入名为yolo的窗口，在这个窗口里可以干你想干的事情。

当你不想呆在这个窗口时，你可以通过快捷键Ctrl+a+D断开这个窗口的连接而回到连接会话界面。即显示如下

```
[detached from 28113.yolo]user@ubuntu-Super-Server:~/code$
1
```

说明从yolo这个窗口断开回到了会话界面。但是这个断开只是不显示那个窗口，而窗口对应的任务是在后台运行的。

##### 查看已创建的所有窗口

$screen -ls #可以查看已创建的所有窗口
 执行上述指令后，出现如下结果，说明创建了两个窗口，可以看到窗口的名字和id，Detached说明窗口是断开的，再次强调这里的断开是指没有让他显示，其对应的任务是在后台执行的。

```
user@ubuntu-Super-Server:~/code$ screen -lsThere are screens on:	28475.ssd	(2017年11月27日 20时07分41秒)	(Detached)	28113.yolo	(2017年11月27日 19时57分26秒)	(Detached)
1
```

##### 重新连接Detached的窗口

如果想看其中一个窗口任务的执行状态，可以通过如下指令：

$screen -r ssd #重新连接到yolo窗口，显示其运行过程

##### 杀死某个窗口

1. 如果想直接停止某个窗口任务的运行，可以直接通过杀死id的方式 $kill -9 28475
    \#终止ssd窗口对应任务的运行，同时杀死该窗口 执行完以上指令再看存在的窗口时后会发现只剩名为yolo的窗口了
2. 在该窗口内敲`exit`,就可以彻底删除该窗口以及窗口内的作业
3. -wipe 　检查目前所有的screen作业，并删除已经无法使用的screen作业。

##### 窗口连接不上的情况

用 screen -ls, 显式当前状态为Attached， 但当前没有用户登陆些会话。screen此时正常状态应该为(Detached)

此时用screen -r ，怎么也登不上。最后找到解决方法：screen -D -r ＜session-id>

-D -r 先踢掉前一用户，再登陆。
 `user@ubuntu-Super-Server:~/code$ screen -lsThere is a screen on: 28113.yolo (2017年11月27日 19时57分26秒) (Detached)`

总结一下，screen可以实现代码在后台运行时的可视化，同时，能在开一个会话连接时创建多个窗口处理不同的任务。用起来也很方便。

## 服务器上tensorboard的使用

对于使用tensorflow的同学，tensorboard是个很好的工具。通过以下的命令可以打开tensorboard，这个port可以自己改，默认的是6006。

```
$tensorboard --logdir=log --port=6006
1
```

尴尬的是，当你在服务器上执行上段指令时，他会给你个地址让你在浏览器上打开。但是，使用服务器上的浏览器基本是件不可能的事情！我怎么将服务器上的信息转到自己的电脑上呢？这时就需要用到端口转发。我们最近实现这个有两种方式：

### 第一种 针对win10用户

想必大家都知道2017年win10系统内嵌了一个linux的子系统。首先我们要启用这个子系统。具体步骤是：打开 所有设置->更新与安全->针对开发人员然后选择下图中的开发人员模式。

执行完上述步骤win10会自动下载相关库和包，下载完后在控制面板中找到程序和功能然后选择启用或关闭Windows功能，出现下图菜单：

勾选适用于Linux的Windows子系统（Beta），然后重启电脑。

重启电脑后win+R输入cmd，然后在dos界面输入bash之后就会下载对应的组件，这个时间可能比较长需要耐心等待。下载完成后，输入相应的用户名和密码就好了。这样，我们就完成了win10上的linux子系统的安装。下次打开时，直接通过win+R输入bash即可。

接下来就是用自己电脑的linux与服务器进行ssh连接并进行端口转发。具体步骤是：

首先win+R 输入bash打开win10自带的linux的子系统，在命令行输入以下指令

```
$ ssh -L 16006:127.0.0.1:6006 usr@192.168.1.115
```

解释下上面的指令，127.0.0.1是自己PC的本地地址(localhost),16006代表的本地的16006端口，192.168.1.115是服务器的ip地址（同样服务器和PC在一个局域网下）usr是在服务器上想登陆的用户名，6006是服务器上的6006端口，上面的指令就可以实现将服务器上6006端口的信息转发到本地主机的16006端口上。

当你建立了以上连接后，在你的xshell建立的会话连接中输入打开tensorboard的指令后，我们在自己的电脑浏览器上输入127.0.0.1:16006或者localhost:16006就可以访问到服务器上的tensorboard的信息了。

### 第二种 依然是使用xshell

具体做法是当你用xshell建立好连接后，点击下图红框中的属性按钮

然后点击属性中的SSH下的隧道，得到如下界面

点击添加：

将侦听端口改为16006（当然也可以是其他的,就是本机的端口），目标主机和源主机保持localhost不变，目标端口就是服务器上打开tensorboard对应的端口6006然后点击确定之后，就建立好了服务器端口16006与自己电脑端口6006的转发。然后按之前的步骤打开tensorboard，在本地浏览器中输入127.0.0.1:16006或者localhost:16006就可以访问到服务器上的tensorboard的信息了，所以说xshell真的特别好用！

### 远程服务器上jupyter notebook的使用

在服务器上jupyter notebook也是经常被用到的，一般输入如下命令便可以打开

$ jupyter notebook --port=8888
  这里的port默认是8888。同样，在不能使用服务器浏览器的情况下想本地PC能读到服务器8888端口的信息，也需要建立端口转发。整个过程同tensorboard的设置类似，不过是把端口号更改就好。当端口转发建立后。我们在服务器输入上述指令，会得到以下输出：

```
[I 23:17:08.227 NotebookApp] Writing notebook server cookie secret to /run/user/1002/jupyter/notebook_cookie_secret[I 23:17:08.275 NotebookApp] JupyterLab alpha preview extension loaded from /home/yuwei/anaconda3/lib/python3.6/site-packages/jupyterlabJupyterLab v0.27.0Known labextensions:[I 23:17:08.277 NotebookApp] Running the core application with no additional extensions or settings[I 23:17:08.280 NotebookApp] Serving notebooks from local directory: /home/yuwei[I 23:17:08.280 NotebookApp] 0 active kernels [I 23:17:08.280 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/?token=d4846a751b41bb288ac8c38b1da7976c0677b6aa51430705[I 23:17:08.280 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).[C 23:17:08.280 NotebookApp] Copy/paste this URL into your browser when you connect for the first time, to login with a token: http://localhost:8888/?token=d4846a751b41bb288ac8c38b1da7976c0677b6aa51430705
1
```

需要做的只是将最后一行的地址复制到本地浏览器然后将8888（服务器端口）改为转发的本地端口（如：16006或8888）就可以了。这样我们在自己的电脑上也可以使用服务器上的jupyter notebook了，如下图所示。

当然还有一种方法，详细见这篇文章吧。

[如何在window访问ubuntu服务器的jupyter notebook](https://zhuanlan.zhihu.com/p/30845372)

## 其他常用命令

使用如下指令的前提是安装好了NVIDIA的驱动（cuda。。）。

$ CUDA_VISIBLE_DEVICES=2 python [test.py](http://test.py)
 在使用gpu版本的tensorflow时，跑对应的代码（如果代码中没有明确指定用哪个gpu跑）默认会调用所有的gpu，使用如上命令后可以指定所使用的gpu。

$ nvidia-smi
 执行上述指执行上述指执行上述指执行上述指执行上述指令可以查看服务器上gpu的使用状况。

Tue Nov 28 09:20:42 2017  ±----------------------------------------------------------------------------+| NVIDIA-SMI 384.90 Driver Version: 384.90  ||-------------------------------±---------------------±---------------------+| GPU Name Persistence-M| Bus-Id Disp.A | Volatile Uncorr. ECC || Fan  Temp Perf Pwr:Usage/Cap| Memory-Usage | GPU-Util Compute M. ||=++==============|| 0 GeForce GTX TIT… Off | 00000000:05:00.0 On | N/A || 22% 45C P8 17W /  250W | 443MiB / 12204MiB | 0% Default  |±------------------------------±---------------------±---------------------+| 1 GeForce GTX TIT… Off | 00000000:06:00.0 Off | N/A || 22% 46C P8 16W / 250W | 2MiB / 12207MiB | 0% Default  |±------------------------------±---------------------±---------------------+| 2 GeForce GTX TIT… Off | 00000000:09:00.0 Off | N/A || 22% 42C P8 15W / 250W | 2MiB / 12207MiB | 0% Default  |±------------------------------±---------------------±---------------------+| 3 GeForce GTX TIT… Off | 00000000:0A:00.0 Off | N/A || 22% 35C P8 14W / 250W | 2MiB / 12207MiB | 0% Default  |±------------------------------±---------------------±---------------------+
 总结一下，上述的种种操作是我反复使用的操作，还是比较实用的。但是本文中的关于ssh和端口转发的内容都有一个前提是自己的PC和服务器都在同一个局域网下，如果不在同一个局域网里上述的操作可能不太适用，需要进行其他的操作，这些是本文中没有提到的。

以上纯属经验之谈，如有错误之处，望大家批评指正。