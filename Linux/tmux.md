Tmux 使用教程



作者： [阮一峰](http://www.ruanyifeng.com)

日期： [2019年10月21日](http://www.ruanyifeng.com/blog/2019/10/)

Tmux 是一个终端复用器（terminal multiplexer），非常有用，属于常用的开发工具。

本文介绍如何使用 Tmux。

![img](https://www.wangbase.com/blogimg/asset/201910/bg2019102005.png)

## 一、Tmux 是什么？

### 1.1 会话与进程

命令行的典型使用方式是，打开一个终端窗口（terminal window，以下简称"窗口"），在里面输入命令。**用户与计算机的这种临时的交互，称为一次"会话"（session）** 。

会话的一个重要特点是，窗口与其中启动的进程是[连在一起](http://www.ruanyifeng.com/blog/2016/02/linux-daemon.html)的。打开窗口，会话开始；关闭窗口，会话结束，会话内部的进程也会随之终止，不管有没有运行完。

一个典型的例子就是，[SSH 登录](http://www.ruanyifeng.com/blog/2011/12/ssh_remote_login.html)远程计算机，打开一个远程窗口执行命令。这时，网络突然断线，再次登录的时候，是找不回上一次执行的命令的。因为上一次 SSH 会话已经终止了，里面的进程也随之消失了。

为了解决这个问题，会话与窗口可以"解绑"：窗口关闭时，会话并不终止，而是继续运行，等到以后需要的时候，再让会话"绑定"其他窗口。

### 1.2 Tmux 的作用

**Tmux 就是会话与窗口的"解绑"工具，将它们彻底分离。**

> （1）它允许在单个窗口中，同时访问多个会话。这对于同时运行多个命令行程序很有用。
>
> （2） 它可以让新窗口"接入"已经存在的会话。
>
> （3）它允许每个会话有多个连接窗口，因此可以多人实时共享会话。
>
> （4）它还支持窗口任意的垂直和水平拆分。

类似的终端复用器还有 GNU Screen。Tmux 与它功能相似，但是更易用，也更强大。

## 二、基本用法

### 2.1 安装

Tmux 一般需要自己安装。

> ```bash
> # Ubuntu 或 Debian
> $ sudo apt-get install tmux
> 
> # CentOS 或 Fedora
> $ sudo yum install tmux
> 
> # Mac
> $ brew install tmux
> ```

### 2.2 启动与退出

安装完成后，键入`tmux`命令，就进入了 Tmux 窗口。

> ```bash
> $ tmux
> ```

上面命令会启动 Tmux 窗口，底部有一个状态栏。状态栏的左侧是窗口信息（编号和名称），右侧是系统信息。

![img](https://www.wangbase.com/blogimg/asset/201910/bg2019102006.png)

按下`Ctrl+d`或者显式输入`exit`命令，就可以退出 Tmux 窗口。

> ```bash
> $ exit
> ```

### 2.3 前缀键

Tmux 窗口有大量的快捷键。所有快捷键都要通过前缀键唤起。默认的前缀键是`Ctrl+b`，即先按下`Ctrl+b`，快捷键才会生效。

举例来说，帮助命令的快捷键是`Ctrl+b ?`。它的用法是，在 Tmux 窗口中，先按下`Ctrl+b`，再按下`?`，就会显示帮助信息。

然后，按下 ESC 键或`q`键，就可以退出帮助。

## 三、会话管理

### 3.1 新建会话

第一个启动的 Tmux 窗口，编号是`0`，第二个窗口的编号是`1`，以此类推。这些窗口对应的会话，就是 0 号会话、1 号会话。

使用编号区分会话，不太直观，更好的方法是为会话起名。

> ```bash
> $ tmux new -s <session-name>
> ```

上面命令新建一个指定名称的会话。

### 3.2 分离会话

在 Tmux 窗口中，按下`Ctrl+b d`或者输入`tmux detach`命令，就会将当前会话与窗口分离。

> ```bash
> $ tmux detach
> ```

上面命令执行后，就会退出当前 Tmux 窗口，但是会话和里面的进程仍然在后台运行。

`tmux ls`命令可以查看当前所有的 Tmux 会话。

> ```bash
> $ tmux ls
> # or
> $ tmux list-session
> ```

### 3.3 接入会话

`tmux attach`命令用于重新接入某个已存在的会话。

> ```bash
> # 使用会话编号
> $ tmux attach -t 0
> 
> # 使用会话名称
> $ tmux attach -t <session-name>
> ```

### 3.4 杀死会话

`tmux kill-session`命令用于杀死某个会话。

> ```bash
> # 使用会话编号
> $ tmux kill-session -t 0
> 
> # 使用会话名称
> $ tmux kill-session -t <session-name>
> ```

### 3.5 切换会话

`tmux switch`命令用于切换会话。

> ```bash
> # 使用会话编号
> $ tmux switch -t 0
> 
> # 使用会话名称
> $ tmux switch -t <session-name>
> ```

### 3.6 重命名会话

`tmux rename-session`命令用于重命名会话。

> ```bash
> $ tmux rename-session -t 0 <new-name>
> ```

上面命令将0号会话重命名。

### 3.7 会话快捷键

下面是一些会话相关的快捷键。

> - `Ctrl+b d`：分离当前会话。
> - `Ctrl+b s`：列出所有会话。
> - `Ctrl+b $`：重命名当前会话。

## 四、最简操作流程

综上所述，以下是 Tmux 的最简操作流程。

> 1. 新建会话`tmux new -s my_session`。
> 2. 在 Tmux 窗口运行所需的程序。
> 3. 按下快捷键`Ctrl+b d`将会话分离。
> 4. 下次使用时，重新连接到会话`tmux attach-session -t my_session`。

## 五、窗格操作

Tmux 可以将窗口分成多个窗格（pane），每个窗格运行不同的命令。以下命令都是在 Tmux 窗口中执行。

### 5.1 划分窗格

`tmux split-window`命令用来划分窗格。

> ```bash
> # 划分上下两个窗格
> $ tmux split-window
> 
> # 划分左右两个窗格
> $ tmux split-window -h
> ```

![img](https://www.wangbase.com/blogimg/asset/201910/bg2019102007.jpg)

### 5.2 移动光标

`tmux select-pane`命令用来移动光标位置。

> ```bash
> # 光标切换到上方窗格
> $ tmux select-pane -U
> 
> # 光标切换到下方窗格
> $ tmux select-pane -D
> 
> # 光标切换到左边窗格
> $ tmux select-pane -L
> 
> # 光标切换到右边窗格
> $ tmux select-pane -R
> ```

### 5.3 交换窗格位置

`tmux swap-pane`命令用来交换窗格位置。

> ```bash
> # 当前窗格上移
> $ tmux swap-pane -U
> 
> # 当前窗格下移
> $ tmux swap-pane -D
> ```

### 5.4 窗格快捷键

下面是一些窗格操作的快捷键。

> - `Ctrl+b %`：划分左右两个窗格。
> - `Ctrl+b "`：划分上下两个窗格。
> - `Ctrl+b <arrow key>`：光标切换到其他窗格。`<arrow key>`是指向要切换到的窗格的方向键，比如切换到下方窗格，就按方向键`↓`。
> - `Ctrl+b ;`：光标切换到上一个窗格。
> - `Ctrl+b o`：光标切换到下一个窗格。
> - `Ctrl+b {`：当前窗格与上一个窗格交换位置。
> - `Ctrl+b }`：当前窗格与下一个窗格交换位置。
> - `Ctrl+b Ctrl+o`：所有窗格向前移动一个位置，第一个窗格变成最后一个窗格。
> - `Ctrl+b Alt+o`：所有窗格向后移动一个位置，最后一个窗格变成第一个窗格。
> - `Ctrl+b x`：关闭当前窗格。
> - `Ctrl+b !`：将当前窗格拆分为一个独立窗口。
> - `Ctrl+b z`：当前窗格全屏显示，再使用一次会变回原来大小。
> - `Ctrl+b Ctrl+<arrow key>`：按箭头方向调整窗格大小。
> - `Ctrl+b q`：显示窗格编号。

## 六、窗口管理

除了将一个窗口划分成多个窗格，Tmux 也允许新建多个窗口。

### 6.1 新建窗口

`tmux new-window`命令用来创建新窗口。

> ```bash
> $ tmux new-window
> 
> # 新建一个指定名称的窗口
> $ tmux new-window -n <window-name>
> ```

### 6.2 切换窗口

`tmux select-window`命令用来切换窗口。

> ```bash
> # 切换到指定编号的窗口
> $ tmux select-window -t <window-number>
> 
> # 切换到指定名称的窗口
> $ tmux select-window -t <window-name>
> ```

### 6.3 重命名窗口

`tmux rename-window`命令用于为当前窗口起名（或重命名）。

> ```bash
> $ tmux rename-window <new-name>
> ```

### 6.4 窗口快捷键

下面是一些窗口操作的快捷键。

> - `Ctrl+b c`：创建一个新窗口，状态栏会显示多个窗口的信息。
> - `Ctrl+b p`：切换到上一个窗口（按照状态栏上的顺序）。
> - `Ctrl+b n`：切换到下一个窗口。
> - `Ctrl+b <number>`：切换到指定编号的窗口，其中的`<number>`是状态栏上的窗口编号。
> - `Ctrl+b w`：从列表中选择窗口。
> - `Ctrl+b ,`：窗口重命名。

## 七、其他命令

下面是一些其他命令。

> ```bash
> # 列出所有快捷键，及其对应的 Tmux 命令
> $ tmux list-keys
> 
> # 列出所有 Tmux 命令及其参数
> $ tmux list-commands
> 
> # 列出当前所有 Tmux 会话的信息
> $ tmux info
> 
> # 重新加载当前的 Tmux 配置
> $ tmux source-file ~/.tmux.conf
> ```

## 八、参考链接

- [A Quick and Easy Guide to tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/)
- [Tactical tmux: The 10 Most Important Commands](https://danielmiessler.com/study/tmux/)
- [Getting started with Tmux](https://linuxize.com/post/getting-started-with-tmux/)

（完）

mailto:yifeng.ruan@gmail.com)

#### Tmux 的快捷键前缀（Prefix）

为了使自身的快捷键和其他软件的快捷键互不干扰，Tmux 提供了一个快捷键前缀。当想要使用快捷键时，需要先按下快捷键前缀，然后再按下快捷键。Tmux 所使用的快捷键前缀默认是组合键 Ctrl-b（同时按下 Ctrl 键和 b 键）。

但是，由于键盘上 Ctrl 键和 b 键距离太远了，操作起来特别不方便，所以经常需要修改快捷键前缀：只需将以下配置加入到 Tmux 的配置文件 ~/.tmux.conf 中（没有此文件就创建一个）：

```
#个人喜欢吧快捷键前缀设置为 Ctrl + a
unbind C-b
set -g prefix C-a
```

### **Tmux 中的特殊功能**

#### **会话（session）**

一个 Tmux 会话中可以包含多个窗口。在会话外创建一个新的会话：

```
tmux new -s <name-of-my-session>
```

进入会话后创建新的会话：只需要按下 Ctrl-b : ，然后输入如下的命令：

```
Ctrl-b
 :new -s <name-of-my-new-session>
```

在 Tmux 的会话间切换

在会话内获取会话列表，可以按下Ctrl-b s。下图所示的就是会话的列表：

```
Ctrl-b s
```

![img](https://segmentfault.com/img/remote/1460000007427968?w=531&h=78)
列表中的每个会话都有一个 ID，该 ID 是从 0 开始的。按下对应的 ID 就可以进入会话。

在会话外获取会话列表：

```
tmux ls
```

在会话外进入会话：

```
tmux attach -t <name-of-my-session>
或
tmux a -t <name-of-my-session>

#进入列表中第一个会话
tmux attach
或
tmux a
```

临时退出但不删除会话：

```
Ctrl + b d
```

在会话内退出并删除session

```
Ctrl+b 
:kill-session

#删除所有session
Ctrl+b 
:kill-server
```

在会话外删除指定session

```
tmux kill-session -t <name-of-my-session>
```

#### **窗口（Window）**

一个 Tmux 会话中可以包含多个窗口。一个窗口中有可以防止多个窗格。
在 Tmux 的会话中，现有的窗口将会列在屏幕下方。下图所示的就是在默认情况下 Tmux 列出现有窗口的方式。这里一共有三个窗口，分别是“server”、“editor”和“shell”。
![img](https://segmentfault.com/img/remote/1460000007427969?w=628&h=388)

创建窗口：

```
Ctrl-b c
```

查看窗口列表

```
Ctrl-b w
```

切换到指定窗口，只需要先按下Ctrl-b，然后再按下想切换的窗口所对应的数字。

```
Ctrl-b 0
```

切换到下一个窗口

```
Ctrl+b n
```

切换到上一个窗口

```
Ctrl+b p
```

在相邻的两个窗口里切换

```
Ctrl+b l
```

重命名窗口

```
Ctrl+b ,
```

在多个窗口里搜索关键字

```
Ctrl+b f
```

删除窗口

```
Ctrl+b &
```

#### **窗格(Panes)**

一个tmux窗口可以分割成若干个格窗。并且格窗可以在不同的窗口中移动、合并、拆分。

创建pane横切split pane horizontal

```
Ctrl+b "
```

竖切split pane vertical

```
Ctrl+b %
```

按顺序在pane之间移动

```
Ctrl+b o
```

上下左右选择pane

```
Ctrl+b 方向键上下左右
```

调整pane的大小(我发现按住Ctrl+b 再按 [上|下|左|右] 键也可以实现相同的效果)

```
Ctrl+b 
:resize-pane -U #向上

Ctrl+b 
:resize-pane -D #向下

Ctrl+b 
:resize-pane -L #向左

Ctrl+b 
:resize-pane -R #向右
```

在上下左右的调整里，最后的参数可以加数字 用以控制移动的大小，例如：

```
Ctrl+b 
:resize-pane -D 5 #向下移动5行
```

在同一个window里上下左右移动pane

```
Ctrl+b { （往左边，往上面）
Ctrl+b } （往右边，往下面）
```

删除pane

```
Ctrl+b x
```

更换pane排版（上下左右分隔各种换）

```
Ctrl+b “空格”
```

移动pane至新的window

```
Ctrl+b !
```

移动pane合并至某个window

```
Ctrl+b :join-pane -t $window_name
```

按顺序移动pane位置

```
Ctrl+b Ctrl+o
```

显示pane编号

```
Ctrl+b q
```

显示时间

 

```
Ctrl+b t
```

  