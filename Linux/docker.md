镜像：相当于代码

容器：就像运行起来的代码，叫做程序

镜像仓库：就像github

# 实验室公用服务器之Docker配置： 

对于计算机科研，需要使用服务器进行必要的实验。由于使用同一服务器的同学很多，因此为了保证每位同学在使用时能够在服务器上拥有自己的环境，于是使用docker进行管理。其原理就是为每位同学建立属于自己的容器，与宿主机隔离，形成独立的沙箱，这样每个人就可以在自己的docker中乱搞，而且不会影响到其他同学的使用了。

下面将以Ubuntu 18.04.4 LTS为例说明配置属于自己的docker的流程。

本实验服务器环境及配置如下：

- Ubuntu 18.04
- GTX2080Ti 4
- cuda 10.1

## 步骤

我们将物理服务器上操作系统称之为**宿主机**，保证在宿主机上安装有docker，如没有安装，请自行安装docker。

1.查看是否安装成功：`docker -v`

2.然后从[Docker Hub](https://hub.docker.com/)上寻找并拉取所需要的镜像，这里拉取了`nvidia/cuda`：`docker pull nvidia/cuda:10.0-base`

3.通过`docker run`命令创建一个新的容器，其命令格式为：`docker run [OPTIONS] IMAGE [COMMAND] [ARG...]`

其中`OPTIONS`有如下配置：

| 参数名                  | 含义                                                         |
| :---------------------- | :----------------------------------------------------------- |
| -d, --detach=false      | 指定容器是否运行于前台，默认为false，当设置为true容器会在后台运行并返回容器id |
| -i, --interactive=false | 打开STDIN，用于控制台交互                                    |
| -t, --tty=false         | 分配tty设备，该可以支持终端登录，默认为false                 |
| -u, --user=""           | 指定容器的用户                                               |
| -a, --attach=[]         | 登录容器（必须是以docker run -d启动的容器）                  |
| -w, --workdir=""        | 指定容器的工作目录                                           |
| -c, --cpu-shares=0      | 设置容器CPU权重，在CPU共享场景使用                           |
| -e, --env=[]            | 指定环境变量，容器中可以使用该环境变量                       |
| --name=""               | 为容器指定一个名称                                           |
| -m, --memory=""         | 指定容器的内存上限                                           |
| -P, --publish-all=false | 随机指定容器暴露的端口                                       |
| -p, --publish=[]        | 指定容器暴露的端口                                           |
| -h, --hostname=""       | 指定容器的主机名                                             |
| -v, --volume=[]         | 给容器挂载存储卷，挂载到容器的某个目录                       |
| --volumes-from=[]       | 给容器挂载其他容器上的卷，挂载到容器的某个目录               |
| --cap-add=[]            | 添加权限，权限清单详见：http://linux.die.net/man/7/capabilities |
| --cap-drop=[]           | 删除权限，权限清单详见：http://linux.die.net/man/7/capabilities |
| --cidfile=""            | 运行容器后，在指定文件中写入容器PID值，一种典型的监控系统用法 |
| --cpuset=""             | 设置容器可以使用哪些CPU，此参数可以用来容器独占CPU           |
| --device=[]             | 添加主机设备给容器，相当于设备直通                           |
| --dns=[]                | 指定容器的dns服务器                                          |
| --shm-size=             | 设置共享内存大小，默认是64M                                  |
| --gpus gpu-request      | 要添加到容器的GPU设备（`all`用于传递所有GPU）                |

 本例中使用`docker run --gpus all -itd --name=xxx --shm-size 8G -p 9099:22 -v /media/work/xxx:/data nvidia/cuda:10.0-base`创建容器，其中选项可查阅上表自行配置。

4.运行起容器后，会返回容器的id，可通过`docker ps`查看该容器的运行情况。同样在宿主机中，使用`docker exec -it xxx /bin/bash`进入容器内部进行交互，`xxx`可以是容器的name也可以是容器id。

5.通常新机器是没有安装`ssh`的，可以通过`service ssh start`或`/etc/init.d/ssh`来启动ssh服务，如果报错为`bash: /etc/init.d/ssh: No such file or directory`类似的信息，则需要重新安装`ssh`。对于Ubuntu系统，具体方法如下：

```shell
sudo apt-get remove openssh-server  openssh-client --purge -y
sudo apt-get autoremove
sudo apt-get autoclean
sudo apt-get update
sudo apt-get install openssh-server openssh-client
```

安装完成后即可启动ssh。

6.依然在宿主机中，还是进入容器内部，使用`passwd`命令更改登录密码，此密码用于`ssh`登录。
7.使用`exit`退出容器，再使用一次`exit`登出宿主机。现在通过`ssh root@ip -p 9099`发现并不能直接连上，即使是给容器设置了密码。
8.回到步骤6，进入容器后，`vi /etc/ssh/sshd_config`，将`PermitRootLogin`的选项设置为`yes`。保存后使用`service ssh restart`重启ssh服务。
9.回到步骤7，这时就可以成功通过`ssh`命令连接到属于自己的docker了。

**大功告成！**







![pre](https://images.gitbook.cn/FhYlVBzAujawd4pPelhBRsXUHWQu)![next](https://images.gitbook.cn/Fo7y6SSMJjNvbUIBsNPgRYJlyjZn)

​                ![第08课：用](https://images.gitbook.cn/FuCncARztzu_dmvH3GRvbi_rKcZD)            

# 第08课：用 Docker 建立一个公用 GPU 服务器 



- - - [为什么要使用 Docker 来建立服务器？](https://gitchat.csdn.net/columntopic/5a13c07375462408e0da8e72#undefined)
    - [服务器配置思路](https://gitchat.csdn.net/columntopic/5a13c07375462408e0da8e72#undefined)
    - [宿主主机配置](https://gitchat.csdn.net/columntopic/5a13c07375462408e0da8e72#undefined)
    - [使用 Dockerfile 定制镜像](https://gitchat.csdn.net/columntopic/5a13c07375462408e0da8e72#undefined)
    - [简易服务器监控网站](https://gitchat.csdn.net/columntopic/5a13c07375462408e0da8e72#undefined)
    - [服务器管理方案小结](https://gitchat.csdn.net/columntopic/5a13c07375462408e0da8e72#undefined)
    - [接下来做什么？](https://gitchat.csdn.net/columntopic/5a13c07375462408e0da8e72#undefined)



> 首先声明一下，Docker 本来被设计用来部署应用（**一次配置，到处部署**），但是在这篇文章里面，我们是把 Docker 当做一个虚拟机来用的，虽然这稍微有悖于 Docker 的使用哲学，但是，作为一个入门教程的结课项目，我们通过这个例子复习之前学到的 Docker 指令还是很好的。
>
> 本文我们主要使用容器搭建一个可以供小型团队（10人以下）使用的 GPU 服务器，用来进行 Deep Learning  的开发和学习。如果读者不是深度学习研究方向的也不要担心，本文的内容依旧是讲解 Docker  的使用，不过提供了一个应用场景。另外，本文会涉及到一些之前没有提到过的   Linux 指令，为了方便 Linux 初学者，会提供详细解释或者参考资料。
>
> 本文参考了以下资料的解决思路，将 LXC 容器替换成 Docker 容器，并针对实际情况作了改动：
>
> https://abcdabcd987.com/setup-shared-gpu-server-for-labs/

### 为什么要使用 Docker 来建立服务器？

深度学习目前火出天际（2017年），我所在的实验室也有相关研究。但是，深度学习模型的训练需要强悍的显卡，由于目前显卡的价格还是比较高的，所以不可能给每个同学都配备几块显卡。因此，公用服务器就成了唯一的选择。但是，公用服务器有一个问题：如果大家都直接操作宿主主机，直接在宿主主机上配置自己的开发环境的话肯定会发生冲突。

实验室一开始就是直接在宿主机配置账号，也允许每个人配置自己需要的开发环境，结果就是慢慢地大家的环境开始发生各种冲突，导致谁都没有办法安安静静地做研究。于是，我决定使用 Docker 把服务器容器化，每个人都直接登录自己的容器，所有开发都在自己的容器内完成，这样就避免了冲突。并且，Docker  容器的额外开销小得可以忽略不计，所以也不会影响服务器性能。

### 服务器配置思路

服务器的配置需要满足一些条件：

- 用户可以方便地登录
- 用户可以自由安装软件
- 普通用户无法操作宿主主机
- 用户可以使用 GPU 资源
- 用户之间互不干扰

我的解决思路是，在服务器安装显卡驱动后，使用 **nvidia-docker** 镜像运行容器。

为什么使用 nvidia-docker 呢？因为 **Docker 是平台无关的**（也就是说，无论镜像的内容是什么，只要主机安装了 Docker，就可以从镜像运行容器），这带来的问题就是——当需要使用一些专用硬件的时候就会无法运行。

因此，Docker 本身是不支持容器内访问 NVIDIA GPU 资源的。早期解决这个问题的办法是在容器内安装 NVIDIA  显卡驱动，然后映射与 NVIDIA 显卡相关的设备到容器（Linux  哲学：硬件即文件，所以很容易映射）。这种解决办法很脆弱，因为这样做之后就要求容器内的显卡驱动与主机显卡硬件型号完全吻合，否则即使把显卡资源映射到容器也无法使用！所以，使用这种方法，容器显然无法再做到平台无关了。

为了解决这些问题，nvidia-docker 应运而生。nvidia-docker 是专门为需要访问显卡资源的容器量身定制的，它对原始的 Docker 命令作了封装，只要使用 **nvidia-docker run** 命令运行容器，容器就可以访问主机显卡设备（只要主机安装了显卡驱动）。nvidia-docker 的使用规则和 Docker 是一致的，只需要把命令里的“docker”替换为“nvidia-docker”就可以了。

然后，为了方便大家使用，为每个容器做一些合适的端口映射，为了方便开发，我还配置了图形界面显示功能！

最后，为了实时监控服务器的资源使用情况，使用 WeaveScope 平台监控容器运行状况（当然，这部分内容和 Docker 入门使用关系不大，大家随意看一下就好了）。

> 如果你没有 GPU 服务器，并且自己的电脑显卡也比较差，你可以不用 nvidia-docker，仅仅使用普通的 Docker 就好了。当然，你可能需要恩距自己的实际情况对后文提供的 Dockerfile 进行修改。

### 宿主主机配置

首先，服务器主机需要安装显卡驱动，你可以使用 NVIDIA 官网提供的 “.run” 文件安装，也可以图方便使用 apt 安装：

```
sudo apt install nvidia-387 nvidia-387-dev 
```

接下来，我们安装 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quick-start)：

```
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
```

我们以“tensorflow/tensorflow:latest-gpu”为基础镜像定制自己的镜像，所以先 pull 这个镜像：

```
sudo docker pull tensorflow/tensorflow:latest-gpu
```

### 使用 Dockerfile 定制镜像

这部分内容参考了[这个项目](https://github.com/fcwu/docker-ubuntu-vnc-desktop)。配置可以在浏览器显示的远程桌面：

```
FROM tensorflow/tensorflow:latest-gpu
MAINTAINER Shichao ZHANG <***@gmail>

ENV DEBIAN_FRONTEND noninteractive

RUN sed -i 's#http://archive.ubuntu.com/#http://tw.archive.ubuntu.com/#' /etc/apt/sources.list

# built-in packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common curl \
    && sh -c "echo 'deb http://download.opensuse.org/repositories/home:/Horst3180/xUbuntu_16.04/ /' >> /etc/apt/sources.list.d/arc-theme.list" \
    && curl -SL http://download.opensuse.org/repositories/home:Horst3180/xUbuntu_16.04/Release.key | apt-key add - \
    && add-apt-repository ppa:fcwu-tw/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
        supervisor \
        openssh-server openssh-client pwgen sudo vim-tiny \
        net-tools \
        lxde x11vnc xvfb \
        gtk2-engines-murrine ttf-ubuntu-font-family \
        libreoffice firefox \
        fonts-wqy-microhei \
        language-pack-zh-hant language-pack-gnome-zh-hant firefox-locale-zh-hant libreoffice-l10n-zh-tw \
        nginx \
        python-pip python-dev build-essential \
        mesa-utils libgl1-mesa-dri \
        gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine pinta arc-theme \
        dbus-x11 x11-utils \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'root:root' |chpasswd
# tini for subreap                                   
ENV TINI_VERSION v0.9.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini

ADD image /
RUN pip install setuptools wheel && pip install -r /usr/lib/web/requirements.txt

EXPOSE 80
WORKDIR /root
ENV HOME=/home/ubuntu \
    SHELL=/bin/bash
ENTRYPOINT ["/startup.sh"]  
```

然后，由此 Dockerfile 构建镜像：

```
sudo docker build -t gpu:v0.1 .
```

等待镜像构建完成。

现在，从这个镜像运行一个容器：

```
sudo nvidia-docker run -d -ti --rm --name gputest -p 9999:80 -e VNC_PASSWORD=1234 gpu:v0.1
```

> 说明：
>
> -e VNC_PASSWORD 设置登录密码。

我的服务器网址是“223.3.43.127”，端口是我们指定的9999，会要求我们输入密码：



![enter image description here](http://images.gitbook.cn/00478020-cc08-11e7-a1aa-212b011b68f6)





输入你设置的密码，即可进入桌面环境：



![enter image description here](http://images.gitbook.cn/18306940-cc08-11e7-bf41-c9c5b004a42a)



好了，这样的话团队成员就可以方便地使用 GPU 服务器了！

### 简易服务器监控网站

我们使用一个开源项目来监控容器的运行——[Weave Scope](https://www.weave.works/docs/scope/latest/introducing/)。

首先，在宿主主机执行以下命令来安装和启动 Weave Scope：

```
sudo curl -L git.io/scope -o /usr/local/bin/scope
sudo chmod a+x /usr/local/bin/scope
scope launch
```

然后浏览器打开服务器 IP 地址，端口号4040，就可以实时监控了：



![enter image description here](http://images.gitbook.cn/fbdb5a50-cc31-11e7-bf41-c9c5b004a42a)



点击对应的容器即可查看容器信息，包括 CPU 占用率，内存占用，端口映射表，开机时间，IP 地址，进程列表，环境变量等等。并且，通过这个监控网站，可以对容器做一些简单操作：停止，重启，attach，exec 等。

这是一个很简单的 Docker 容器监控方案，**使用者可以通过这个监控网站直接操作容器，所以无需登录宿主主机来进行相关操作，完美实现资源隔离**。

但是，这个方案也有一些缺点，比如每个用户都可以看到其它用户的密码，甚至可以直接进入其他用户的容器！不过，由于我们的使用背景是“实验室或者小团队的私密使用”，作为关系紧密的内部小团体，建立在大家相互信任的基础上，相信这也不是什么大问题。

### 服务器管理方案小结

现在总结一下我们的 GPU 服务器容器化的全部工作：

- 宿主主机配置 Docker 和 nvidia-docker，安装显卡驱动；
- 使用 Dockerfile 定制镜像；
- 为每个用户运行一个容器，注意需要挂载需要的数据卷；

```
sudo nvidia-docker run -ti -d --name ZhangShichao -v /home/berry/dockerhub/zsc/:/root/zsc -v /media/zhangzhe/data1:/root/sharedData -p 6012:22 -p 6018:80 -p 6010:6000 -p 6011:6001 -p 6019:8888 -e VNC_PASSWORD=ZhangShichao     gpu:v0.1
```

- 使用 WeaveScope 监控容器的运行情况；

在此之后，如果团队成员需要启动新的容器，管理员可以通过宿主主机为用户运行需要的容器。普通用户无法操作宿主主机，完美实现隔离！

### 接下来做什么？

本文主要是一个实战例子，相信通过这个例子复习 Docker 的使用，读者一定对 Docker 操作更加熟悉。下一篇文章，也是最后一篇文章，我们将简单介绍 Docker 生态以及 Docker 在实际中的大型应用。

​    

​                                    

[课程详情](https://gitchat.csdn.net/column/5a13be9775462408e0da8d9d)

[                                                               1                                                                                                                     第01课：初遇 Docker                                                               ](https://gitchat.csdn.net/columnTopic/5a13bf0175462408e0da8dc1)[                                                               2                                                                                                                     第02课：面向新手-Linux 命令行初探                                                               ](https://gitchat.csdn.net/columnTopic/5a13bf4775462408e0da8de4)[                                                               3                                                                                                                     第03课：安装 Docker 到你的电脑                                                               ](https://gitchat.csdn.net/columnTopic/5a13bf6475462408e0da8df3)[                                                               4                                                                                                                     第04课：Docker 使用入门（上）                                                               ](https://gitchat.csdn.net/columnTopic/5a13bf8475462408e0da8e02)[                                                               5                                                                                                                     第05课：Docker使用入门（下）                                                               ](https://gitchat.csdn.net/columnTopic/5a13bfdb75462408e0da8e19)[                                                               6                                                                                                                     第06课：学习编写 Dockerfile                                                               ](https://gitchat.csdn.net/columnTopic/5a13bffd75462408e0da8e31)[                                                               7                                                                                                                     第07课：Docker 底层原理初探                                                               ](https://gitchat.csdn.net/columnTopic/5a13c04575462408e0da8e58)[                                                               8                                                                                                                     第08课：用 Docker 建立一个公用 GPU 服务器                                                                ](https://gitchat.csdn.net/columnTopic/5a13c07375462408e0da8e72)[                                                               9                                                                                                                     第09课：后记-Docker 生态                                                               ](https://gitchat.csdn.net/columnTopic/5a13c0af75462408e0da8e8d)