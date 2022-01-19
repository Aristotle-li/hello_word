```python
'''python提取文件夹内指定类型文件'''
import os,shutil
import re
path = 'G:\琴趣科技乐谱\小组完成乐谱的副本'
targetDir = 'G:\输出'


def gci (path):
    """this is a statement"""
    parents = os.listdir(path)
    for parent in parents:
        child = os.path.join(path,parent)
        if os.path.isdir(child):
            gci(child)
        # print(child)
        else:
            if os.path.splitext(child)[1] == '.sib':
                fpath,fname=os.path.split(child)
                filename, extension = os.path.splitext(fname)
                str = re.sub("[0-9\+\\\!\%\[\]\,\。]", "", filename)
                mycopyfile(child,targetDir+"/"+str+".sib")
                print(str)



def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print('不是文件')
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print('复制完成')


gci(path)

```



```python
# -*- coding:utf-8 -*-
'''rename'''
import os

class ImageRename():
  def __init__(self):
  self.path = 'D:/xpu/paper/plate_data'

  def rename(self):
    filelist = os.listdir(self.path)
    total_num = len(filelist)

  i = 0

  for item in filelist:
    if item.endswith('.jpg'):
      src = os.path.join(os.path.abspath(self.path), item)
      dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')
      os.rename(src, dst)
      print 'converting %s to %s ...' % (src, dst)
      i = i + 1
  print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
  newname = ImageRename()
  newname.rename()
```

# python读写、创建文件、文件夹等等



得到当前工作目录，即当前Python脚本工作的目录路径:**os.getcwd()**

返回指定目录下的所有文件和目录名:**os.listdir()**

函数用来删除一个文件:**os.remove()**

删除多个目录：**os.removedirs（r“c：\python”）**

检验给出的路径是否是一个文件：**os.path.isfile()**

检验给出的路径是否是一个目录：**os.path.isdir()**

判断是否是绝对路径：**os.path.isabs()**

检验给出的路径是否真地存:**os.path.exists()**

返回一个路径的目录名和文件名:**os.path.split()** eg os.path.split('/home/swaroop/byte/code/poem.txt') 结果：('/home/swaroop/byte/code', 'poem.txt')

分离扩展名：**os.path.splitext()**

获取路径名：**os.path.dirname()**

获取文件名：**os.path.basename()**

运行shell命令:**os.system()**

读取和设置环境变量:**os.getenv() 与os.putenv()**

给出当前平台使用的行终止符:**os.linesep**  Windows使用'\r\n'，Linux使用'\n'而Mac使用'\r'

指示你正在使用的平台：**os.name** 对于Windows，它是'nt'，而对于Linux/Unix用户，它是'posix'

重命名：**os.rename（old， new）**

创建多级目录：**os.makedirs（r“c：\python\test”）**

创建单个目录：**os.mkdir（“test”）**

获取文件属性：**os.stat（file）**

修改文件权限与时间戳：**os.chmod（file）**

终止当前进程：**os.exit（）**

获取文件大小：**os.path.getsize（filename）**

**文件操作：**

**os.mknod("test.txt")** 创建空文件

**fp = open("test.txt",w)** 直接打开一个文件，如果文件不存在则创建文件

关于open 模式：

w   以写方式打开，

a   以追加模式打开 (从 EOF 开始, 必要时创建新文件)

r+   以读写模式打开

w+   以读写模式打开 (参见 w )

a+   以读写模式打开 (参见 a )

rb   以二进制读模式打开

wb   以二进制写模式打开 (参见 w )

ab   以二进制追加模式打开 (参见 a )

rb+  以二进制读写模式打开 (参见 r+ )

wb+  以二进制读写模式打开 (参见 w+ )

ab+  以二进制读写模式打开 (参见 a+ )

 

**fp.read([size])** #size为读取的长度，以byte为单位

**fp.readline([size])** #读一行，如果定义了size，有可能返回的只是一行的一部分

**fp.readlines([size])**  #把文件每一行作为一个list的一个成员，并返回这个list。其实它的内部是通过循环调用readline()来实现的。如果提供size参数，size是表示读取内容的总长，也就是说可能只读到文件的一部分。

**fp.write(str)** #把str写到文件中，write()并不会在str后加上一个换行符

**fp.writelines(seq)**  #把seq的内容全部写到文件中(多行一次性写入)。这个函数也只是忠实地写入，不会在每行后面加上任何东西。

**fp.close()**  #关闭文件。python会在一个文件不用后自动关闭文件，不过这一功能没有保证，最好还是养成自己关闭的习惯。 如果一个文件在关闭后还对其进行操作会产生ValueError

**fp.flush()**  #把缓冲区的内容写入硬盘

**fp.fileno()**  #返回一个长整型的”文件标签“

**fp.isatty()**  #文件是否是一个终端设备文件（unix系统中的）

**fp.tell()**#返回文件操作标记的当前位置，以文件的开头为原点

**fp.next()**   #返回下一行，并将文件操作标记位移到下一行。把一个file用于for … in file这样的语句时，就是调用next()函数来实现遍历的。

**fp.seek(offset[,whence])** #将文件打操作标记移到offset的位置。这个offset一般是相对于文件的开头来计算的，一般为正数。但如果提供了whence参数就不一定了，whence可以为0表示从头开始计算，1表示以当前位置为原点计算。2表示以文件末尾为原点进行计算。需要注意，如果文件以a或a+的模式打开，每次进行写操作时，文件操作标记会自动返回到文件末尾。

**fp.truncate([size])**  #把文件裁成规定的大小，默认的是裁到当前文件操作标记的位置。如果size比文件的大小还要大，依据系统的不同可能是不改变文件，也可能是用0把文件补到相应的大小，也可能是以一些随机的内容加上去。

 

**目录操作：**

**os.mkdir("file")**   创建目录

复制文件：

**shutil.copyfile("oldfile","newfile")** oldfile和newfile都只能是文件

**shutil.copy("oldfile","newfile")**  oldfile只能是文件夹，newfile可以是文件，也可以是目标目录

复制文件夹：

**shutil.copytree("olddir","newdir")** olddir和newdir都只能是目录，且newdir必须不存在

重命名文件（目录）

**os.rename("oldname","newname")** 文件或目录都是使用这条命令

移动文件（目录）

**shutil.move("oldpos","newpos")**  

删除文件

**os.remove("file")**

删除目录

**os.rmdir("dir")**只能删除空目录

**shutil.rmtree("dir")**  空目录、有内容的目录都可以删

转换目录

**os.chdir("path")**  换路径

 

Python读写文件

1.open

使用open打开文件后一定要记得调用文件对象的close()方法。比如可以用try/finally语句来确保最后能关闭文件。

 

![img](https://upload-images.jianshu.io/upload_images/13717038-90bf60b265052476.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

注：不能把open语句放在try块里，因为当打开文件出现异常时，文件对象file_object无法执行close()方法。

2.读文件

读文本文件

 

![img](https://upload-images.jianshu.io/upload_images/13717038-c27066f318fe1d3c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

读二进制文件

 

![img](https://upload-images.jianshu.io/upload_images/13717038-4e15aa203c3472dd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

读取所有内容

 

![img](https://upload-images.jianshu.io/upload_images/13717038-f0ab3125e17f7c9d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

读固定字节

 

![img](https://upload-images.jianshu.io/upload_images/13717038-79d3d019174c5b3d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

读每行

 

![img](https://upload-images.jianshu.io/upload_images/13717038-00d02d4b326b7f84.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如果文件是文本文件，还可以直接遍历文件对象获取每行：

 

![img](https://upload-images.jianshu.io/upload_images/13717038-d268a6d39ff983e0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

3.写文件

写文本文件

 

![img](https://upload-images.jianshu.io/upload_images/13717038-f2fdcad6d84d091c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

写二进制文件

 

![img](https://upload-images.jianshu.io/upload_images/13717038-33413e0d1cd49550.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

追加写文件

 

![img](https://upload-images.jianshu.io/upload_images/13717038-04ff0b729373afee.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

写数据

 

![img](https://upload-images.jianshu.io/upload_images/13717038-1ee6c399b33e9831.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

写入多行

 

![img](https://upload-images.jianshu.io/upload_images/13717038-a1ac071d49295db0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

注意，调用writelines写入多行在性能上会比使用write一次性写入要高。

在处理日志文件的时候，常常会遇到这样的情况：日志文件巨大，不可能一次性把整个文件读入到内存中进行处理，例如需要在一台物理内存为 2GB 的机器上处理一个 2GB 的日志文件，我们可能希望每次只处理其中 200MB 的内容。

在 Python 中，内置的 File 对象直接提供了一个 readlines(sizehint) 函数来完成这样的事情。以下面的代码为例：

 

![img](https://upload-images.jianshu.io/upload_images/13717038-10583c7b72699dbf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

每次调用 readlines(sizehint) 函数，会返回大约 200MB  的数据，而且所返回的必然都是完整的行数据，大多数情况下，返回的数据的字节数会稍微比 sizehint 指定的值大一点（除最后一次调用  readlines(sizehint) 函数的时候）。通常情况下，Python 会自动将用户指定的 sizehint  的值调整成内部缓存大小的整数倍。

file在python是一个特殊的类型，它用于在python程序中对外部的文件进行操作。在python中一切都是对象，file也不例外，file有file的方法和属性。下面先来看如何创建一个file对象：

file(name[, mode[, buffering]])

file()函数用于创建一个file对象，它有一个别名叫open()，可能更形象一些，它们是内置函数。来看看它的参数。它参数都是以字符串的形式传递的。name是文件的名字。

mode是打开的模式，可选的值为r w a U，分别代表读（默认） 写  添加支持各种换行符的模式。用w或a模式打开文件的话，如果文件不存在，那么就自动创建。此外，用w模式打开一个已经存在的文件时，原有文件的内容会被清空，因为一开始文件的操作的标记是在文件的开头的，这时候进行写操作，无疑会把原有的内容给抹掉。由于历史的原因，换行符在不同的系统中有不同模式，比如在 unix中是一个\n，而在windows中是‘\r\n’，用U模式打开文件，就是支持所有的换行模式，也就说‘\r’ '\n'  '\r\n'都可表示换行，会有一个tuple用来存贮这个文件中用到过的换行符。不过，虽说换行有多种模式，读到python中统一用\n代替。在模式字符的后面，还可以加上+ b t这两种标识，分别表示可以对文件同时进行读写操作和用二进制模式、文本模式（默认）打开文件。

buffering如果为0表示不进行缓冲;如果为1表示进行“行缓冲“;如果是一个大于1的数表示缓冲区的大小，应该是以字节为单位的。

file对象有自己的属性和方法。先来看看file的属性。

closed #标记文件是否已经关闭，由close()改写

encoding #文件编码

mode #打开模式

name #文件名

newlines #文件中用到的换行模式，是一个tuple

softspace #boolean型，一般为0，据说用于print

file的读写方法：

F.read([size]) #size为读取的长度，以byte为单位

F.readline([size])

\#读一行，如果定义了size，有可能返回的只是一行的一部分

F.readlines([size])

\#把文件每一行作为一个list的一个成员，并返回这个list。其实它的内部是通过循环调用readline()来实现的。如果提供size参数，size是表示读取内容的总长，也就是说可能只读到文件的一部分。

F.write(str)

\#把str写到文件中，write()并不会在str后加上一个换行符

F.writelines(seq)

\#把seq的内容全部写到文件中。这个函数也只是忠实地写入，不会在每行后面加上任何东西。

file的其他方法：

F.close()

\#关闭文件。python会在一个文件不用后自动关闭文件，不过这一功能没有保证，最好还是养成自己关闭的习惯。如果一个文件在关闭后还对其进行操作会产生ValueError

F.flush()

\#把缓冲区的内容写入硬盘

F.fileno()

\#返回一个长整型的”文件标签“

F.isatty()

\#文件是否是一个终端设备文件（unix系统中的）

F.tell()

\#返回文件操作标记的当前位置，以文件的开头为原点

F.next()

\#返回下一行，并将文件操作标记位移到下一行。把一个file用于for ... in file这样的语句时，就是调用next()函数来实现遍历的。

F.seek(offset[,whence])

\#将文件打操作标记移到offset的位置。这个offset一般是相对于文件的开头来计算的，一般为正数。但如果提供了whence参数就不一定了，whence可以为0表示从头开始计算，1表示以当前位置为原点计算。2表示以文件末尾为原点进行计算。需要注意，如果文件以a或a+的模式打开，每次进行写操作时，文件操作标记会自动返回到文件末尾。

F.truncate([size])

\#把文件裁成规定的大小，默认的是裁到当前文件操作标记的位置。如果size比文件的大小还要大，依据系统的不同可能是不改变文件，也可能是用0把文件补到相应的大小，也可能是以一些随机的内容加上去。

