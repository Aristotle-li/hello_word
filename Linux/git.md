# 常见指令：



```
git config --global user.name "Aristotle-li"
git config --global user.email "1326150359@qq.com"
ssh-keygen -t rsa -C "1326150359@qq.com"
```

## 一、创建版本库

𝑚𝑘𝑑𝑖𝑟  𝑙𝑒𝑎r𝑛𝑔𝑖t创建文件夹

 cd learngit 进入文件夹
*𝑝𝑤𝑑*显示当前目录 git init把这个目录变成Git可以管理的仓库
*𝑙𝑠*−*𝑎ℎ*显示所有文件，包括隐藏文件

 git add a.txt 把a文件添加到仓库(一次只能添加一个)
 $ git commit -m “wrote a readme file” 把文件提交到仓库，后面的文字是本次提交文件的介绍文字(一次添加多个)

## 二、时光机穿梭

*𝑔**𝑖**𝑡**𝑠**𝑡**𝑎**𝑡**𝑢**𝑠*查看文档修改状态

 git diff 查看修改的有哪些变化

## 三、版本回退

*𝑔**𝑖**𝑡**𝑙**𝑜**𝑔*查看提交历史

 git log  —pretty=oneline 查看提交历史简洁版
*𝑔**𝑖**𝑡**𝑟**𝑒**𝑠**𝑒**𝑡*−−*ℎ**𝑎**𝑟**𝑑**𝐻**𝐸**𝐴**𝐷*回退到上一个版本//或*𝑔**𝑖**𝑡**𝑟**𝑒**𝑠**𝑒**𝑡*—*ℎ**𝑎**𝑟**𝑑**𝐻**𝐸**𝐴**𝐷* 100(会退到前100版本) cat readme.txt 查看文件里面的内容
*𝑔**𝑖**𝑡**𝑟**𝑒**𝑠**𝑒**𝑡*−−*ℎ**𝑎**𝑟**𝑑*3628164回退到指定版本(如果终端没有关闭)

 git reflog 记录每一次提交的命令

## 四、工作区和暂存区

## 五、管理修改

## 六、撤销修改

*𝑔**𝑖**𝑡**𝑐**ℎ**𝑒**𝑐**𝑘**𝑜**𝑢**𝑡*—*𝑟**𝑒**𝑎**𝑑**𝑚**𝑒*.*𝑡**𝑥**𝑡*把*𝑟**𝑒**𝑎**𝑑**𝑚**𝑒*.*𝑡**𝑥**𝑡*文件在工作区的修改全部撤销

 git reset HEAD readme.txt 把暂存区的撤销掉，重新放回工作区

## 七、删除文件

$ rm test.txt 删除文件(删除后commit、删除后未commit)

## 八、初始化仓库：

## Create a new repository on the command line

```
touch README.md
git init
git add README.md
git commit -m “first commmit”
git remote add origin https://github.com..
git push -u origin master
```

## Push an existion repository from the command line

```
git push add origin http://….
git push -u origin master
```

### 第一次提交失败

报错：

```
error: failed to push some refs to 'git@github.com:xxxxxxx/xxxxxxxx.git

hint: Updates were rejected because the tip of your current branch is behin

hint: its remote counterpart. Integrate the remote changes (e.g.

hint: 'git pull ...') before pushing again.

hint: See the 'Note about fast-forwards' in 'git push --help' for details.
```

分析与解决：
 出现错误的主要原因是github中的README.md文件不在本地代码目录中
 可以通过如下命令进行代码合并【注：pull=fetch+merge]

```
git pull --rebase origin master
```

执行上面代码后可以看到本地代码库中多了README.md文件
 此时再执行语句 git push -u origin master即可完成代码上传到github

## 九、撤销已经push到远端的文件的文件

在使用git时，push到远端后发现commit了多余的文件，或者希望能够回退到以前的版本。

先在本地回退到相应的版本：

```
git reset --hard <版本号>
// 注意使用 --hard 参数会抛弃当前工作区的修改
// 使用 --soft 参数的话会回退到之前的版本，但是保留当前工作区的修改，可以重新提交
```

如果此时使用命令：

```
git push origin <分支名>
```

会提示本地的版本落后于远端的版本；
 ![img](https://images2015.cnblogs.com/blog/1017946/201707/1017946-20170713173042618-96460405.png)

为了覆盖掉远端的版本信息，使远端的仓库也回退到相应的版本，需要加上参数--force

```
git push origin <分支名> --force
```

## 十、git拉取远程分支并创建本地分支

#### 一、查看远程分支

使用如下Git命令查看所有远程分支：

```
git branch -r
```

#### 二、拉取远程分支并创建本地分支

##### 方法一

使用如下命令：

```
git checkout -b 本地分支名x origin/远程分支名x
```

使用该方式会在本地新建分支x，并自动切换到该本地分支x。

##### 方式二

使用如下命令：

```
git fetch origin 远程分支名x:本地分支名x
```

使用该方式会在本地新建分支x，但是不会自动切换到该本地分支x，需要手动checkout。

------

划重点：每次我都要忘记！
 1.
 如果fork别人的项目或者是参与开源项目的开发，修改好了代码之后，一定要看看自己远端的版本是不是跟原项目的版本一致，如果不是请更新你的远端仓库，如果你在没有更新的情况下push上去了，再去pull request的时候，会出现冲突。
 为了不必要的麻烦，请保持自己的远端仓库与fork的远端仓库版本一致。![img](https://images.cnblogs.com/cnblogs_com/Yfling/1050723/o_git1.png)



H5mobile中用到的git流程：

```
// step1：拉取远端分支
git remote -v
// git fetch origin dev:dev(错误)
git fetch origin dev
git checkout -b dev origin/dev
git log
git checkout master
git log
git checkout dev
gst
clear

// step2：提交代码
git pull origin dev // 当前在dev分支，拉取远端分支，与远端同步
git status
git add .
git commit -m ''
git status
git push origin dev

// step3：部署到测试环境
ssh master@10.8.8.8
->输入密码
-> yes
cd mobileH5

// 方案1：
git checkout dev
git pull origin dev
npm run testprod
// 方案2：
git pull origin dev
npm run dev
// 之后打开http://10.8.8.8/mobile_......html（注意这里有没有端口号8001？80？。。）
exit // 退出服务器

// merge!!!!这里是合并到master!!!!!
// step4：merge到master分支，打补丁，push到master
cd mobileH5
git pull origin dev  // 一定要检查一下是否是最新版本
git checkout master
git pull origin master  // 版本检查
git merge dev // 将dev分支合并到当前分支（这里会进入vim）退出：shift+冒号 输入：wq（回车）
git log   // 查看版本是否正确
gulp patch  // gulp打补丁
git push origin master  // push到master

// step5：部署到test环境
git merge dev



PS:
// 在本地npm run build文件
 3561  ls
 3562  cd deist
 3563  cd dist
 3564  ls
 3565  python -m SimpleHTTPServer 8080

在Mac环境下执行Python脚本
cd 到文件夹目录
在文件首行添加#!/usr/bin/env python
添加权限 chmod 777 filename.py
执行脚本./filename.py
简单脚本:files.py
http://blog.csdn.net/hi_chen_xingwang/article/details/51569514
```

### mobileH5V2迭代之后的流程

```
// 进入某一目录
cd /var/folders/6y/kb5tt1qd6x56f90s180y6y0m0000gn/T/phantomjs
// 将某一文件copy到当前目录
cp ~/Desktop/phantomjs-2.1.1-macosx.zip


npm run cli-create  // 输入这句之后后面会有提示让你输入文件名
//启动项目
npm start
git pull guanghe mobileH5V2
git remote add guanghe https://github.com/guanghetv/mobileH5V2.git
git pull guanghe develop
```

尚未整理

```
 5525  cd desktop
 5526  git clone https://github.com/Yfling/mobileH5V2.git
 5527  cd mobileH5V2
 5528  npm install
 5529* cd /var/folders/6y/kb5tt1qd6x56f90s180y6y0m0000gn/T/phantomjs
 5530* ll
 5531* cp ~/Desktop/phantomjs-2.1.1-macosx.zip .
 5532* ll
 5533  npm i
 5534  npm start
 5535  atom .
 5536  npm run cli-create
 5537  npm start
 5538  git pull guanghe mobileH5V2
 5539  git remote
 5540  git remote add https://github.com/guanghetv/mobileH5V2.git
 5541  git remote add guanghe https://github.com/guanghetv/mobileH5V2.git
 5542  git remote
 5543  git pull guanghe devlop
 5544  git pull guanghe develop
 5545  git checkout mind-review
 5546  git pull guanghe develop
 5547  git checkout mind-review
 5548  git merge guanghe/develop develop
 5549  git bransh
 5550  git branch --all
 5551  git checkout mind-review
 5552  git merge guanghe/develop origin/develop
 5553  git fetch
 5554  git show
 5555  git checkout guanghe/feature/mind-review
 5556  git checkout -b mind-review
 5557  git status
 5558  git checkout develop
 5559  git fetch guanghe/develop
 5560  git fetch remotes/guanghe/develop
 5561  git fetch remotes/guanghe
 5562  git pull guanghe/develop
 5563  git remotes
 5564  git remotes --list
 5565  git remote --list
 5566  git remote
 5567  git pull guanghe develop
 5568  git merge guanghe/develop develop
 5569  git status
 5570  git branch
 5571  npm run cli-create
 5572  git status
 5573  git add .
 5574  git status
 5575  git commit -m '新增期中复习运营页面'
 5576  git push origin feature/mind-review
 5577  npm start
 5578* git checkout master
 5579* git remote -v
 5580* git pull guanghe master
```

### 运营平台

#### 测试环境部署

```
// step3：部署到测试环境
ssh master@10.8.8.8  // 输入密码：u..m..

cd Shadow
git fetch Yfling h5-backstage:h5-backstage  // 当前是test分支
git fetch origin master:master  // 当前是test分支
git merge h5-backstage  // 合并到test分支
```

[![segmentfault](https://cdn.segmentfault.com/r-7b7553ca/static/logo-b.d865fc97.svg)](https://segmentfault.com/)

[首页](https://segmentfault.com/)[问答](https://segmentfault.com/questions)[专栏](https://segmentfault.com/blogs)[资讯](https://segmentfault.com/news)[课程](https://ke.sifou.com)[活动](https://segmentfault.com/events)

[发现](https://segmentfault.com/a/1190000004317077#)



![img](http://sponsor.segmentfault.com/lg.php?bannerid=0&campaignid=0&zoneid=2&loc=https%3A%2F%2Fsegmentfault.com%2Fa%2F1190000004317077&referer=https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3DL4Hkzfp7nd7S_iEVzwfE6evDNhyZOTBRr8jhDg0YH_LCClbObtB6WVX4q0NQFQ3CArLikZNX20Dy-Pr40Jfxya%26wd%3D%26eqid%3Dcc326865000f125e00000003604c5d8f&cb=b8be12dcd8)

[首页](https://segmentfault.com/)[专栏](https://segmentfault.com/blogs)[git](https://segmentfault.com/t/git/blogs)文章详情

# [Github使用方法及常见错误](https://segmentfault.com/a/1190000004317077)

[![img](https://avatar-static.segmentfault.com/320/979/3209793098-575cbb5965fd2_big64)**xiaoxiongmila**](https://segmentfault.com/u/xiaoxiongmila)发布于 2016-01-16 

![img](http://sponsor.segmentfault.com/lg.php?bannerid=0&campaignid=0&zoneid=25&loc=https%3A%2F%2Fsegmentfault.com%2Fa%2F1190000004317077&referer=https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3DL4Hkzfp7nd7S_iEVzwfE6evDNhyZOTBRr8jhDg0YH_LCClbObtB6WVX4q0NQFQ3CArLikZNX20Dy-Pr40Jfxya%26wd%3D%26eqid%3Dcc326865000f125e00000003604c5d8f&cb=d72ffd96fd)

## 第一步:当然是先安装

```
windows上安装git  http://msysgit.github.io/
```

配置你的username 和email

```
$ git config --global user.name "Yourname"
$ git config --global user.email "email@example.com"
```

创建版本库

```
$ mkdir learngit $ cd learngit $ pwd /Users/michael/learngit
```

## 第二步，通过git init命令把这个目录变成git可以管理的仓库

```
$ git init
```

> Initialized empty Git repository in /Users/michael/learngit/.git/

瞬间Git就把仓库建好了，而且告诉你是一个空的仓库（empty Git  repository），细心的读者可以发现当前目录下多了一个.git的目录，这个目录是Git来跟踪管理版本库的，没事千万不要手动修改这个目录里面的文件，不然改乱了，就把Git仓库给破坏了。
如果你没有看到.git目录，那是因为这个目录默认是隐藏的，用

> ls -ah

命令就可以看见。

现在我们编写一个readme.txt文件，内容如下：

```
Git is a version control system.
Git is free software.
```

一定要放到learngit目录下（子目录也行），因为这是一个Git仓库，放到其他地方Git再厉害也找不到这个文件。

和把大象放到冰箱需要3步相比，把一个文件放到Git仓库只需要两步。

## 第一步,用命令git add告诉Git，把文件添加到仓库

```
$ git add readme.txt
```

## 第二步，用命令git commit告诉Git，把文件提交到仓库:

```
$ git commit -m "wrote a readme file"
```

> [master (root-commit)cb926e7] wrote a readme file   1 file changed, 2
> insertions(+)   create mode 10064 readme.txt

git commit命令,-m后面输入的是本次提交的说明，可以输入任意内容，当然最好是有意义的，这样就能从历史记录里方便地找到改动记录

git commit命令执行成功后会告诉你，1个文件改动（我们新添加的readme.txt文件），插入了两行内容（readme.txt有两行内容）

为什么Git添加文件需要add，commit一共两步呢？因为commit可以一次提交很多文件，所以你可以多次add不同的文件，比如:

> $ git add file1.txt
>
> $ git add file2.txt file3.txt
>
> $ git commit -m "add 3 files"

我们已经成功地添加并提交了一个readme.txt文件，现在，是时候继续工作了，于是，我们继续修改readme.txt文件，改成如下内容：

## 远程仓库

第1步：创建SSH Key  在用户主目录下，看看有没有.ssh目录，如果有，再看看这个目录下有没有id_rsa和id_rsa.pub这两个文件，如果已经有了，可直接跳到下一步。如果没有，打开Shell（Windows下打开Git Bash），创建SSH Key：

```
$ ssh-keygen -t rsa -C "youremail@example.com"
```

第2步：登陆GitHub，打开“Account settings”，“SSH Keys”页面：
然后，点“Add SSH Key”，填上任意Title，在Key文本框里粘贴id_rsa.pub文件的内容：
点“Add Key”，你就应该看到已经添加的Key：

ps: id_rsa是私钥，不能泄露出去，id_rsa.pub是公钥，可以放心地告诉任何人。

目前，在GitHub上的这个learngit仓库还是空的，GitHub告诉我们，可以从这个仓库克隆出新的仓库，也可以把一个已有的本地仓库与之关联，然后，把本地仓库的内容推送到GitHub仓库。

现在，我们根据GitHub的提示，在本地的learngit仓库下运行命令：

```
$ git remote add origin git@github.com:michaelliao/learngit.git
```

添加后，远程库的名字就是origin，这是Git默认的叫法，也可以改成别的，但是origin这个名字一看就知道是远程库。

下一步，就可以把本地库的所有内容推送到远程库上

```
$ git push -u origin master
```

> Counting objects: 19, done. Delta compression using up to 4 threads.
> Compressing objects: 100% (19/19), done. Writing objects: 100%
> (19/19), 13.73 KiB, done. Total 23 (delta 6), reused 0 (delta 0) To
> git@github.com:michaelliao/learngit.git
>
> - [new branch] master -> master Branch master set up to track remote branch master from origin.

把本地库的内容推送到远程，用git push命令，实际上是把当前分支master推送到远程。

由于远程库是空的，我们第一次推送master分支时，加上了-u参数，Git不但会把本地的master分支内容推送的远程新的master分支，还会把本地的master分支和远程的master分支关联起来，在以后的推送或者拉取时就可以简化命令。

从现在起，只要本地作了提交，就可以通过命令：

```
$ git push origin master
```

把本地master分支的最新修改推送至GitHub，现在，你就拥有了真正的分布式版本库！

SSH警告

当你第一次使用Git的clone或者push命令连接GitHub时，会得到一个警告：

> The authenticity of host 'github.com (xx.xx.xx.xx)' can't be
>
> 1. RSA key fingerprint is xx.xx.xx.xx.xx. Are you sure you
> 2. to continue connecting (yes/no)?

这是因为Git使用SSH连接，而SSH连接在第一次验证GitHub服务器的Key时，需要你确认GitHub的Key的指纹信息是否真的来自GitHub的服务器，输入yes回车即可。

Git会输出一个警告，告诉你已经把GitHub的Key添加到本机的一个信任列表里了：

Warning: Permanently added 'github.com' (RSA) to the list of known hosts.
从远程库克隆:
要克隆一个仓库，首先必须知道仓库的地址，然后使用git clone命令克隆。

Git支持多种协议，包括https，但通过ssh支持的原生git协议速度最快。

如何参与一个开源项目呢？比如人气极高的bootstrap项目，这是一个非常强大的CSS框架，你可以访问它的项目主页`https://github.com/twbs/bootstrap`，点“Fork”就在自己的账号下克隆了一个bootstrap仓库，然后，从自己的账号下clone：

```
git clone git@github.com:michaelliao/bootstrap.git
```

一定要从自己的账号下clone仓库，这样你才能推送修改。如果从bootstrap的作者的仓库地址git@github.com:twbs/bootstrap.git克隆，因为没有权限，你将不能推送修改。

小结

在GitHub上，可以任意Fork开源仓库;
自己拥有Fork后的仓库的读写权限；
可以推送pull request给官方仓库来贡献代码。在安装Git一节中，我们已经配置了user.name和user.email，实际上，Git还有很多可配置项。比如，让Git显示颜色，会让命令输出看起来更醒目：
$ git config --global color.ui true
小结

忽略某些文件时，需要编写.gitignore；
.gitignore文件本身要放到版本库里，并且可以对.gitignore做版本管理！

## 常见错误总结

如果输入

> $ git remote add origin
> git@github.com:djqiang（github帐号名）/gitdemo（项目名）.git

提示出错信息：

```
fatal: remote origin already exists.
```

解决办法如下：

1、先输入`$ git remote rm origin`

2、再输入`$ git remote add origin git@github.com:djqiang/gitdemo.git` 就不会报错了！

3、如果输入`$ git remote rm origin` 还是报错的话，

> error: Could not remove config section 'remote.origin'

. 我们需要修改gitconfig文件的内容

4、找到你的github的安装路径，我的是

> C:UsersASUSAppDataLocalGitHubPortableGit_ca477551eeb4aea0e4ae9fcd3358bd96720bb5c8etc

5、找到一个名为gitconfig的文件，打开它把里面的`[remote "origin"]`那一行删掉就好了！

如果输入`$ ssh -T git@github.com`
出现错误提示：`Permission denied (publickey)`.因为新生成的key不能加入ssh就会导致连接不上github。

解决办法如下：

1、先输入`$ ssh-agent`，再输入`$ ssh-add ~/.ssh/id_key`，这样就可以了。

2、如果还是不行的话，输入`ssh-add ~/.ssh/id_key` 命令后出现报错

> Could not open a connection to your authentication agent

.解决方法是key用Git Gui的ssh工具生成，这样生成的时候key就直接保存在ssh中了，不需要再ssh-add命令加入了，其它的user，token等配置都用命令行来做。

3、最好检查一下在你复制id_rsa.pub文件的内容时有没有产生多余的空格或空行，有些编辑器会帮你添加这些的。

如果输入`$ git push origin master`

提示出错信息：

> error:failed to push som refs to .......

解决办法如下：

1、先输入`$ git pull origin master` //先把远程服务器github上面的文件拉下来

2、再输入

> $ git push origin master

3、如果出现报错

> fatal: Couldn't find remote ref master或者fatal: 'origin' does not
> appear to be a git repository以及fatal: Could not read from remote
> repository.

4、则需要重新输入`$ git remote add origingit@github.com:djqiang/gitdemo.git`

提示出错信息:

```
fatal: Unable to create '/path/my_proj/.git/index.lock': File exists.

If no other git process is currently running, this probably means a
git process crashed in this repository earlier. Make sure no other git
process is running and remove the file manually to continue.
```

解决方法如下:

```
rm -f ./.git/index.lock
```

## 使用git在本地创建一个项目的过程

> $ makdir ~/hello-world    //创建一个项目hello-world

$ cd ~/hello-world      > //打开这个项目
 $ git init            //初始化 
touchREADME touch README touchREADME git add README > //更新README文件 
$ git commit -m 'first commit'     //提交更新，并注释信息“first

> commit” $ git remote add origin git@github.com:defnngj/hello-world.git > //连接远程github项目

$ git push -u origin master     //将本地项目更新到github项目上去

gitconfig配置文件

Git有一个工具被称为git config，它允许你获得和设置配置变量；这些变量可以控制Git的外观和操作的各个方面。这些变量可以被存储在三个不同的位置：
1./etc/gitconfig 文件：包含了适用于系统所有用户和所有库的值。如果你传递参数选项’--system’ 给 git config，它将明确的读和写这个文件。
2.~/.gitconfig 文件 ：具体到你的用户。你可以通过传递--global 选项使Git 读或写这个特定的文件。
3.位于git目录的config文件 (也就是 .git/config) ：无论你当前在用的库是什么，特定指向该单一的库。每个级别重写前一个级别的值。因此，在.git/config中的值覆盖了在/etc/gitconfig中的同一个值。
在Windows系统中，Git在HOME目录中查找.gitconfig文件（对大多数人来说，位于C:DocumentsandSettingsHOME目录中查找.gitconfig文件（对大多数人来说，位于C:Documents and SettingsHOME目录中查找.gitconfig文件（对大多数人来说，位于C:DocumentsandSettingsUSER下）。它也会查找/etc/gitconfig，尽管它是相对于Msys 根目录的。这可能是你在Windows中运行安装程序时决定安装Git的任何地方。

> warning: LF will be replaced by CRLF

问题解决方法

windows中的换行符为 CRLF， 而在linux下的换行符为LF，所以在执行add . 时出现提示，解决办法：

```
$ rm -rf .git 
```

// 删除.git
`$ git config --global core.autocrlf false`  //禁用自动转换 （两个虚线）

然后重新执行：

```
$ git init
$ git add .
```

## 总结

当我们想要在gitub上的不同仓库推送代码的时候，先在gitub新建repository，在本地新建文件夹，又可以被称为work directory，cd directory，然后git init  为了防止 错误， 输入gitremoteaddorigingit@github.com:xiaoxiongmila/gitdemo.git就不会报错了！gitadddemogitcommit−m"demo说明"接着先输入 git remote add origin git@github.com:xiaoxiongmila/gitdemo.git 就不会报错了！git add demo   git commit -m "demo说明" 接着  先输入gitremoteaddorigingit@github.com:xiaoxiongmila/gitdemo.git就不会报错了！gitadddemogitcommit−m"demo说明"接着先输入 git pull origin master //先把远程服务器github上面的文件拉下来
 下一步，就可以把本地库的所有内容推送到远程库上 $ git push -u origin master
把本地库的内容推送到远程，用git push命令，实际上是把当前分支master推送到远程。

由于远程库是空的，我们第一次推送master分支时，加上了-u参数，Git不但会把本地的master分支内容推送的远程新的master分支，还会把本地的master分支和远程的master分支关联起来，在以后的推送或者拉取时就可以简化命令。

从现在起，只要本地作了提交，就可以通过命令：

$ git push origin master

第二次往相同的仓库里面添加文件，就直接cd directory git add directory git commit -m “文件说明” git push origin master就可以了，，不管你行不行，反正我是行了^-^
PS： 遇到错误一定不要放弃，，坚持就是胜利！~~



Git学习笔记 1，GitHub常用命令
轮子去哪儿了 2019-02-22 19:32:00 51 收藏
分类专栏： Git
版权
Git学习笔记 1，GitHub常用命令1

廖雪峰Git教程
莫烦Git教程
莫烦Git视频教程
---------------

    init
    
    > apt-get install git  # 安装
    > mkdir /home/yzn_git  # 给git创建一个工作文件夹
    > cd /home/yzn_git 
    > git init  # 创建版本库（init），产生一个隐藏文件夹.git
    > git config --global user.name "yzn"  # 设置用户名
    > git config --global user.email "yangzhaonan18@qq.com"  # 设置用户邮箱
    
    config
    
    > git config user.name  # 查看设置用户的用户名
    > git config user.email  # 查看设置用户的邮箱
    
    add 和 commit
    
    > touch readme.txt  # 创建文件
    > git add readme.txt  # 添加文件管理（add）
    > git commit -m "wrote a readme file"  # 提交改变(commit)
    
    > git add .  # add所有文件
    > git commit -m "wrote a readme file"  # 提交改变(commit)
    
    diff：三种状态unstaged->staged->master分支
    
    > git diff  # 对比没有add(unstaged)和已经commit(master)的
    > git diff HEAD  # 对比没有add(unstaged)和已经add(staged)的
    > git diff --cached  # 对比已经add(staged)和已经commit(master)的
    
    status 和 log
    
    > git status  # 查看文件的修改、删除、新建等状态
    > git log  # 查看commit日志
    > git log --oneline  # 在一行显示日志
    > git log --pretty=oneline  # 在一行显示日志
    > git log --oneline --graph  # 在一行显示日志

posted @ 2019-02-22 19:32 YangZhaonan 阅读(...) 评论(...) 编辑 收藏
————————————————
版权声明：本文为CSDN博主「轮子去哪儿了」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_42419002/article/details/88859256