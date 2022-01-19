Mac 终端默认 shell 为 bash。
 zsh 可能是目前最好的 shell ，至于好在哪里可自行百度。
 本文主要介绍使用 zsh 以及 oh-my-zsh 的配置。

查看当前使用的 shell

```bash
echo $SHELL

/bin/bash
```

查看安装的 shell

```undefined
cat /etc/shells

/bin/bash
/bin/csh
/bin/ksh
/bin/sh
/bin/tcsh
/bin/zsh
```

使用 brew 更新 zsh

```php
brew install zsh

==> Downloading https://homebrew.bintray.com/bottles/zsh-5.5.1.high_sierra.bottle.tar.gz
######################################################################## 100.0%
==> Pouring zsh-5.5.1.high_sierra.bottle.tar.gz
/usr/local/Cellar/zsh/5.5.1: 1,444 files, 12MB
```

重启终端即可使用 zsh

## oh-my-zsh

执行从 oh-my-zsh 的 GitHub 下载的安装脚本

```csharp
sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

  ____  / /_     ____ ___  __  __   ____  _____/ /_  
 / __ \/ __ \   / __ `__ \/ / / /  /_  / / ___/ __ \ 
/ /_/ / / / /  / / / / / / /_/ /    / /_(__  ) / / / 
\____/_/ /_/  /_/ /_/ /_/\__, /    /___/____/_/ /_/  
                        /____/                       ....is now installed!
Please look over the ~/.zshrc file to select plugins, themes, and options.
p.s. Follow us at https://twitter.com/ohmyzsh.
p.p.s. Get stickers and t-shirts at http://shop.planetargon.com.
```

注意：oh-my-zsh国内镜像安装和更新方法：

```
sh -c "$(curl -fsSL https://gitee.com/mirrors/oh-my-zsh/raw/master/tools/install.sh)"
```



```
$ ls ~/.oh-my-zsh
cache  custom  lib  log  MIT-LICENSE.txt  oh-my-zsh.sh  plugins  README.markdown  templates  themes  tools
```

> lib 提供了核心功能的脚本库
>  tools 提供安装、升级等功能的快捷工具
>  plugins 自带插件的存在放位置
>  templates 自带模板的存在放位置
>  themes  自带主题文件的存在放位置
>  custom 个性化配置目录，自安装的插件和主题可放这里

]

#### 设置主题

安装完毕后，我们就可以使用了，咱们先来简单配置一下。
 Oh My Zsh 提供了很多主题风格，我们可以根据自己的喜好，设置主题风格，主题的配置在 ~/.zshrc 文件中可以看到，用一个自己熟悉的编辑器打开这个文件，可以找到这一项：

```bash
vim ~/.zshrc
ZSH_THEME="robbyrussel"
```

可以看到，我们默认使用的主题叫做robbyrussel。 它的显示效果嘛，大概是这样

![img](https://img-blog.csdn.net/20170604161404271?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3pnMTM1NDg5MzAxODY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
 Oh My Zsh默认自带了一些默认主题，存放在~/.oh-my-zsh/themes目录中。我们可以查看这些主题

#### 六、启用插件

 Oh My Zsh 默认自带了一些默认主题，存放在~/.oh-my-zsh/plugins目录中。我们可以查看这些插件

Oh My Zsh默认是只启用git插件(在~/.zshrc文件中)

```bash
plugins=(git)
```

如需启用更多插件，可加入需启用插件的名称。如下

```bash
plugins=(git wd web-search history history-substring-search)
```

推荐几个好用插件

```bash
zsh-history-substring-search
https://github.com/robbyrussell/oh-my-zsh/tree/master/plugins/history-substring-search
zsh-syntax-highlighting
https://github.com/zsh-users/zsh-syntax-highlighting
zsh-autosuggestions
https://github.com/zsh-users/zsh-autosuggestions
```

一些小技巧
 给history命令增加时间

.zshrc中加入以下行

```bash
$ vim ~/.zshrc
HIST_STAMPS="yyyy-mm-dd" 
source ~/.zshrc
```

#### 七、更新oh-my-zsh

设置自动更新oh-my-zsh

默认情况下，当oh-my-zsh有更新时，都会给你提示。如果希望让oh-my-zsh自动更新，在~/.zshrc 中添加下面这句

```bash
DISABLE_UPDATE_PROMPT=true
```

要手动更新，可以执行

```bash
$ upgrade_oh_my_zsh
```

#### 八、卸载oh my zsh

直接在终端中，运行uninstall_oh_my_zsh既可以卸载。



## 自动补全插件

下载 incr 自动补全插件 http://mimosa-pudica.net/src/incr-0.2.zsh
 将插件放在 oh-my-zsh 自定义插件目录中。

打开 oh-my-zsh 配置文件

```undefined
vim ~/.zshrc
```

在 `plugins` 中添加 `incr` 并重命名为incr.plugin.zsh
 在配置文件结束添加：plugins=(git incr )

更新配置:

```bash
source ~/.zshrc
```