# Linux 下yum，apt-get，wget详解及软件安装方式等

dpkg本身是一个底层的工具。上层的工具，如APT，被用于从远程获取软件包以及处理复杂的软件包关系。

**一般来说linux系统基本上分两大类：cat /etc/issue查看linux系统版本**
RedHat系列：Redhat、Centos、Fedora等
Debian系列：Debian、Ubuntu等

**RedHat 系列常见的安装包格式 ：**
1、rpm包,安装rpm包的命令是“rpm -参数”
2、包管理工具 **yum**
3、支持tar包

**Debian系列常见的安装包格式 ：**
1、deb包,安装deb包的命令是“dpkg -参数”
2、包管理工具 **apt-get**
3、支持tar包

**wget**不是安装方式，它是一种下载工具，类似于迅雷
通过HTTP、HTTPS、FTP三个最常见的TCP/IP协议下载，并可以使用HTTP代理，名字是World Wide Web”与“get”的结合。

如果要下载一个软件,可以直接运行：
**wget 下载地址**， 例如： **wget https://nodejs.org/dist/v10.9.0/node-v10.9.0-linux-x64.tar.xz**

如果当前ubuntu未安装wget，可按下列操作进行安装和检查是否安装成功：
sudo apt-get update 
sudo apt-get install wget 
wget --version 

\---------------------------------------------------------------------------------------------------------------------------------------------------------------

**一、Redhat安装yum**

```
在Linux Redhat 9.0使用YUM伺服器來管理rpm套件升級方法由於 Redhat 公司己經停止了對Linux Redhat 9.0的維護,所以我們這些使用者必須找到另一個方法去升級套件,這時使用YUM伺服器來管理rpm套件升級, 因為它可以避免套件間相依性而安裝失敗.
 
要連線YUM伺服器必須要先要裝下列程式:
yum-2.0.4-1.rh.fr.i386.rpm
 
 
此檔案可到 http://ayo.freshrpms.net/ 網站下載,此一個檔案,之後到"終端機"內打入su再輸你root的密碼,進入root後,再打入下列指令:
rpm -ivh yum-2.0.4-1.rh.fr.i386.rpm 
 
 
安裝完此程式後, 就可以使用下列指令來使用YUM伺服器來管理rpm套件升級:
yum update 升級你的RPM套件, 此指令等同於apt-get update 加上apt-get install 的功能.
yum install 安裝新的RPM套件.
yum clean 清除己經完成安裝而不必要的暫存程式.
yum remove 移除你的RPM套件.
```

**二、yum的使用(Rehat)**

```
1. Redhat的yum高级的包管理
 
1）.用YUM安装删除软件
装了系统添加删除软件是常事，yum同样可以胜任这一任务，只要软件是rpm安装的。
安装的命令是yum install xxx，yum会查询数据库，有无这一软件包，如果有，则检查其依赖冲突关系，如果没有依赖冲突，那么最好，下载安装;如果有，则会给出提示，询问是否要同时安装依赖，或删除冲突的包，你可以自己作出判断。
删除的命令是，yum remove xxx，同安装一样，yum也会查询数据库，给出解决依赖关系的提示。
 
2）.用YUM安装软件包
命令：yum install 
 
3）.用YUM删除软件包
命令：yum remove 
用YUM查询软件信息，我们常会碰到这样的情况，想要安装一个软件，只知道它和某方面有关，但又不能确切知道它的名字。这时yum的查询功能就起作用了。你可以用 yum search keyword这样的命令来进行搜索，比如我们要则安装一个Instant Messenger，但又不知到底有哪些，这时不妨用 yum search messenger这样的指令进行搜索，yum会搜索所有可用rpm的描述，列出所有描述中和messeger有关的rpm包，于是我们可能得到 gaim，kopete等等，并从中选择。有时我们还会碰到安装了一个包，但又不知道其用途，我们可以用yum info packagename这个指令来获取信息。
 
4）.使用YUM查找软件包
命令：yum search 
 
5）.列出所有可安装的软件包
命令：yum list 
 
6）.列出所有可更新的软件包
命令：yum list updates 
 
7）.列出所有已安装的软件包
命令：yum list installed 
 
8）.列出所有已安装但不在 Yum Repository 內的软件包
命令：yum list extras 
 
9）.列出所指定的软件包
命令：yum list 
```

**三、 Ubuntu** **软件Dpkg方式安装**

```
deb是debian linus的安装格式，跟red hat的rpm非常相似，最基本的安装命令是：dpkg -i file.deb或者直接双击此文件。
 
dpkg是Debian Package的简写，是为Debian 专门开发的套件管理系统，方便软件的安装、更新及移除。所有源自Debian的Linux发行版都使用dpkg，例如Ubuntu、Knoppix等。
 
 
以下是一些 Dpkg 的普通用法： 普通安装：dpkg -i package_name.deb
 
1、dpkg -i
    安装一个 Debian 软件包，如你手动下载的文件。
 
2、dpkg -c
   列出 的内容。
 
3、dpkg -I
   从 中提取包裹信息。
 
4、dpkg -r
   移除一个已安装的包裹。
 
5、dpkg -P
   完全清除一个已安装的包裹。和 remove 不同的是，remove 只是删掉数据和可执行文件，purge 另外还删除所有的配制文件。
 
6、dpkg -L
   列出 安装的所有文件清单。同时请看 dpkg -c 来检查一个 .deb 文件的内容。
 
7、dpkg -s
   显示已安装包裹的信息。同时请看 apt-cache 显示 Debian 存档中的包裹信息，以及 dpkg -I 来显示从一个 .deb 文件中提取的包裹信息。
 
8、dpkg-reconfigure
   重新配制一个已经安装的包裹，如果它使用的是 debconf (debconf 为包裹安装提供了一个统一的配制界面)。 
```

**四、apt-get 的使用(Ubuntu)**

```
1.Ubuntu中的高级包管理方法apt-get
 
除了apt的便捷以外，apt-get的一大好处是极大地减小了所谓依赖关系恶梦的发生几率(dependency hell)，即使是陷入了dependency hell，apt-get也提供了很好的援助手段，帮你逃出魔窟。
 
通常 apt-get 都和网上的压缩包一起出没，从互联网上下载或是安装。全世界有超过200个debian官方镜像，还有繁多的非官方软件包提供网站。你所使用的基于Debian的发布版不同，你所使用的软件仓库可能需要手工选择或是可以自动设置。你能从Debian官方网站得到完整的镜像列表。而很多非官方网站提供各种特殊用途的非官方软件包，当然，使用非官方软件包会有更多风险了。 
 
软件包都是为某一个基本的Debian发布版所准备的(从unstable 到stable)，并且划分到不同类别中(如 main contrib nonfree)，这个是依据 debian 自由软件纲领而划分的(也就是常说的dfsg)，因为美国限制加密软件出口，还有一个non-us类别。 
 
 
（1）普通安装：apt-get install softname1 softname2 …;
（2）修复安装：apt-get -f install softname1 softname2… ;(-f Atemp to correct broken dependencies)
（3）重新安装：apt-get –reinstall install softname1 softname2…;
 
 
2.常用的APT命令参数
apt-cache search package  搜索包 
apt-cache show package    获取包的相关信息，如说明、大小、版本等 
sudo apt-get install package  安装包 
sudo apt-get install package -- reinstall 重新安装包 
sudo apt-get -f install     修复安装"-f = --fix-missing" 
 
sudo apt-get remove package 删除包 
sudo apt-get remove package -- purge 删除包，包括删除配置文件等 
sudo apt-get update  更新源 
sudo apt-get upgrade 更新已安装的包 
sudo apt-get dist-upgrade 升级系统 
 
sudo apt-get dselect-upgrade 使用 dselect 升级 
apt-cache depends package 了解使用依赖 
apt-cache rdepends package 是查看该包被哪些包依赖 
sudo apt-get build-dep package 安装相关的编译环境 
apt-get source package 下载该包的源代码 
 
sudo apt-get clean && sudo apt-get autoclean 清理无用的包 
sudo apt-get check 检查是否有损坏的依赖
```

**五、apt-get软件安装后相关文件位置**

```
1.下载的软件存放位置
/var/cache/apt/archives
ubuntu 默认的PATH为
PATH=/home/brightman/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games
路径配置 /etc/profile
 
2.安装后软件默认位置
/usr/share
 
3.可执行文件位置 
/usr/bin
 
4.配置文件位置
/etc
 
5.lib文件位置
/usr/lib
```

**六、linux 下源码安装软件方式：**

```
源码安装（.tar、tar.gz、tar.bz2、tar.Z）
源码的安装一般由3个步骤组成：配置(configure)、编译(make)、安装(make install)
 
 
首先解压缩源码压缩包然后通过tar命令来完成
 
a．解xx.tar.gz：tar zxf xx.tar.gz 
b．解xx.tar.Z：tar zxf xx.tar.Z 
c．解xx.tgz：tar zxf xx.tgz 
d．解xx.bz2：bunzip2 xx.bz2 
e．解xx.tar：tar xf xx.tar
 
然后进入到解压出的目录中，建议先读一下README之类的说明文件，因为此时不同源代码包或者预编译包可能存在差异，然后建议使用ls -F –color或者ls -F命令（实际上我的只需要 l 命令即可）查看一下可执行文件，可执行文件会以*号的尾部标志。
 
 
依次执行
1) ./configure
备注; 
linux下configure配置参数：
https://mp.csdn.net/postedit/90769740
https://blog.csdn.net/hy1020659371/article/details/38320661
 
Configure是一个可执行脚本，它有很多选项，在待安装的源码路径下使用命令./configure –help输出详细的选项列表。
 
其中--prefix选项是配置安装的路径。
如果不配置 --prefix 选项，安装后：
可执行文件默认放在/usr /local/bin，
库文件默认放在/usr/local/lib，
配置文件默认放在/usr/local/etc，
其它的资源文件放在/usr /local/share，
./configure --prefix=xxx --host=yyy..
 
 
2) make
 
3) sudo make install
```



## dpkg 简介

“dpkg ”是“Debian Packager ”的简写。为 “Debian” 专门开发的套件管理系统，方便软件的安装、更新及移除。所有源自“Debian”的“Linux ”发行版都使用 “dpkg”，例如 “Ubuntu”、“Knoppix ”等。
dpkg是Debian软件包管理器的基础，它被伊恩·默多克创建于1993年。dpkg与RPM十分相似，同样被用于安装、卸载和供给.deb软件包相关的信息。
dpkg本身是一个底层的工具。上层的工具，如APT，被用于从远程获取软件包以及处理复杂的软件包关系。 “dpkg”是“Debian Package”的简写。

### 常用命令

1）安装软件
命令行：dpkg -i <.deb file name>
示例：dpkg -i avg71flm_r28-1_i386.deb
2）安装一个目录下面所有的软件包
命令行：dpkg -R
示例：dpkg -R /usr/local/src
3）释放软件包，但是不进行配置
命令行：dpkg –-unpack package_file 如果和-R一起使用，参数可以是一个目录
示例：dpkg –-unpack avg71flm_r28-1_i386.deb
4）重新配置和释放软件包
命令行：dpkg –configure package_file
如果和-a一起使用，将配置所有没有配置的软件包
示例：dpkg –configure avg71flm_r28-1_i386.deb
5）删除软件包（保留其配置信息）
命令行：dpkg -r
示例：dpkg -r avg71flm
6）替代软件包的信息
命令行：dpkg –update-avail 
7）合并软件包信息
dpkg –merge-avail 
8）从软件包里面读取软件的信息
命令行：dpkg -A package_file
9）删除一个包（包括配置信息）
命令行：dpkg -P
10）丢失所有的Uninstall的软件包信息
命令行：dpkg –forget-old-unavail
11）删除软件包的Avaliable信息
命令行：dpkg –clear-avail
12）查找只有部分安装的软件包信息
命令行：dpkg -C
13）比较同一个包的不同版本之间的差别
命令行：dpkg –compare-versions ver1 op ver2
14）显示帮助信息
命令行：dpkg –help
15）显示dpkg的Licence
命令行：dpkg –licence (or) dpkg –license
16）显示dpkg的版本号
命令行：dpkg --version
17）建立一个deb文件
命令行：dpkg -b directory [filename]
18）显示一个Deb文件的目录
命令行：dpkg -c filename
19）显示一个Deb的说明
命令行：dpkg -I filename [control-file]
20）搜索Deb包
命令行：dpkg -l package-name-pattern
示例：dpkg -I vim
21）显示所有已经安装的Deb包，同时显示版本号以及简短说明
命令行：dpkg -l
22）报告指定包的状态信息
命令行：dpkg -s package-name
示例：dpkg -s ssh
23）显示一个包安装到系统里面的文件目录信息
命令行：dpkg -L package-Name
示例：dpkg -L apache2
24）搜索指定包里面的文件（模糊查询）
命令行：dpkg -S filename-search-pattern
25）显示包的具体信息
命令行：dpkg -p package-name
示例：dpkg -p cacti

最后：
1、很多人抱怨用了Ubuntu或者Debian以后，不知道自己的软件给安装到什么地方了。其实可以用上面的dpkg -L命令来方便的查找。看来基础还是非常重要的，图形界面并不能够包办一切。
2、有的时候，用“新力得”下载完成以后，没有配置，系统会提示用“dpkg –configure -all”来配置，具体为什么也可以从上面看到。
3、现在Edgy里面可以看到Deb的信息。不过是在没有安装的时候（当然也可以重新打开那个包），可以看到Deb的文件路径。
4、如果想暂时删除程序以后再安装，第5项还是比较实用的，毕竟在Linux下面配置一个软件也并非容易。

## 普通APT用法

### 简介

Advanced Packaging Tool（apt）是Linux下的一款安装包管理工具，是一个客户/服务器系统。
最初只有.tar.gz的打包文件，用户必须编译每个他想在GNU/Linux上运行的软件。用户们普遍认为系统很有必要提供一种方法来管理这些安装在机器上的软件包，当Debian诞生时，这样一个管理工具也就应运而生，它被命名为dpkg。从而著名的“package”概念第一次出现在GNU/Linux系统中，稍后Red Hat才决定开发自己的“rpm”包管理系统。
很快一个新的问题难倒了GNU/Linux制作者，他们需要一个快速、实用、高效的方法来安装软件包，当软件包更新时，这个工具应该能自动管理关联文件和维护已有配置文件。Debian再次率先解决了这个问题，APT(Advanced Packaging Tool）作为dpkg的前端诞生了。APT后来还被Conectiva改造用来管理rpm，并被其它Linux发行版本采用为它们的软件包管理工具。
APT由几个名字以“apt-”打头的程序组成。apt-get、apt-cache 和apt-cdrom是处理软件包的命令行工具。
Linux命令—apt，也是其它用户前台程序的后端，如dselect 和aptitude。
作为操作的一部分，APT使用一个文件列出可获得软件包的镜像站点地址，这个文件就是/etc/apt/sources.list。

### 工作原理

APT是一个客户/服务器系统。在服务器上先复制所有DEB包（DEB是Debian软件包格式的文件扩展名），然后用APT的分析工具（genbasedir）根据每个DEB 包的包头（Header）信息对所有的DEB包进行分析，并将该分析结果记录在一个文件中，这个文件称为DEB 索引清单，APT服务器的DEB索引清单置于base文件夹内。一旦APT 服务器内的DEB有所变动，一定要使用genbasedir产生新的DEB索引清单。客户端在进行安装或升级时先要查询DEB索引清单，从而可以获知所有具有依赖关系的软件包，并一同下载到客户端以便安装。
当客户端需要安装、升级或删除某个软件包时，客户端计算机取得DEB索引清单压缩文件后，会将其解压置放于/var/state/apt/lists/，而客户端使用apt-get install或apt-get upgrade命令的时候，就会将这个文件夹内的数据和客户端计算机内的DEB数据库比对，知道哪些DEB已安装、未安装或是可以升级的。

### 常用命令

apt-get install 
下载 以及所有倚赖的包裹,同时进行包裹的安装或升级.如果某个包裹被设置了 hold (停止标志,就会被搁在一边(即不会被升级).更多 hold 细节请看下面.

apt-get remove [–purge] 
移除 以及任何倚赖这个包裹的其它包裹.
–purge 指明这个包裹应该被完全清除 (purged) ,更多信息请看 dpkg -P .

apt-get update
升级来自 Debian 镜像的包裹列表,如果你想安装当天的任何软件,至少每天运行一次,而且每次修改了
/etc/apt/sources.list 后,必须执行.

apt-get upgrade [-u]
升级所以已经安装的包裹为最新可用版本.不会安装新的或移除老的包裹.如果一个包改变了倚赖关系而需要安装一个新的包裹,那么它将不会被升级,而是标志为 hold .apt-get update 不会升级被标志为 hold 的包裹 (这个也就是 hold 的意思).请看下文如何手动设置包裹为 hold .我建议同时使用 ‘-u’ 选项,因为这样你就能看到哪些包裹将会被升级.

apt-get dist-upgrade [-u]
和 apt-get upgrade 类似,除了 dist-upgrade 会安装和移除包裹来满足倚赖关系.因此具有一定的危险性.

apt-cache search 
搜索满足 的包裹和描述.

apt-cache show 
显示 的完整的描述.

apt-cache showpkg 
显示 许多细节,以及和其它包裹的关系.

dselect
console-apt
aptitude
gnome-apt
APT 的几个图形前端(其中一些在使用前得先安装).这里 dselect 无疑是最强大的,也是最古老,最难驾驭.