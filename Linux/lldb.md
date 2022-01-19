#             LLDB 调试命令使用指南        

​                                                



#### 前言

LLDB Debugger (LLDB) 是一个开源、底层调试器(low level debugger)，具有 REPL  (Read-Eval-Print Loop，交互式解释器)、C++ 和 Python 插件，位于 Xcode 窗口底部控制台中，也有其他 IDE 加入了 LLDB 调试器，如 CLion，当然其也可以在 terminal  中使用。搜罗了一圈，发现大多数教程还是可视化的使用，本文就命令行使用 LLDB 调试程序做一个总结。

#### 调试前

需要安装 lldb 调试器，如果你安装了 llvm 编译环境，lldb 是其自带的工具，就不用安装了。

#### 开始或结束调试

> 特别注意，需调试的程序一定要加上 `-g` 参数进行编译，否则调试时无法看到源码。

lldb 有两种进入方式，本例使用方法一（假设可执行程序名为 `toy`，参数为 `example1`）

方式一，直接运行 `lldb toy example ` 进入调试环境

![img](https://img.hacpai.com/file/2019/04/lldb1-8f9dc6fb.png?imageView2/2/interlace/1/format/webp)

方式二，先运行 `lldb`，再通过 `file toy` 进入调试环境

![img](https://img.hacpai.com/file/2019/04/lldb2-49172df0.png?imageView2/2/interlace/1/format/webp)

除此之外，还可调试正在运行时的程序，此时需要找到此程序的 PID 或者进程名，再使用

> ```
> (lldb) process attach --pid 9939 
> ```

或者

> ```
> (lldb) process attach --name Safari
> ```

进入调试环境。

退出 lldb 环境，输入 `quit` 或者 `exit` 即可。

#### 查看代码

##### 用 `list` 查看

和 gdb 一样，使用 `list` 查看代码，这里有两个小技巧

- 不输入命令的时候直接按回车，就会执行上一次执行的命令。
- 一直 `list` 到底了之后再 `list` 就没有了，这时候怎么办？`list 1` 就回到第一行了。`l 13` 就是从第 13 行开始往下看 10 行。

##### 看其他文件的代码

如果程序编译的时候是由很多文件组成的，那么就可以使用 `list 文件名` 看其他文件的代码, 以后再执行 `list 3` 的时候，看的就是你前面设置的文件名的第三行。

##### 看某个函数的代码

直接输入某个函数的名字即可。

演示如下：

![img](https://img.hacpai.com/file/2019/04/lldb3-26dc2be4.png?imageView2/2/interlace/1/format/webp)

#### 设置断点

##### 根据文件名和行号下断点

```
(lldb) breakpoint set --file toy.cpp --line 10
```

##### 根据函数名下断点

```
# C函数
(lldb) breakpoint set --name main
# C++类方法
(lldb) breakpoint set --method foo
```

##### 根据某个函数调用语句下断点

```
# lldb有一个最小子串匹配算法，会知道应该在哪个函数那里下断点
breakpoint set -n "-[SKTGraphicView alignLeftEdges:]"
```

##### 小技巧

你可以通过设置命令的别名来简化上面的命令

```
# 比如下面的这条命令
(lldb) breakpoint set --file test.c --line 10

# 你就可以写这样的别名
(lldb) command alias bfl breakpoint set -f %1 -l %2

# 使用的时候就像这样就好了
(lldb) bfl test.c 10
```

##### 查看断点列表、启用/禁用断点、删除断点

```
# 查看断点列表
(lldb) breakpoint list
   
# 禁用断点
# 根据上面查看断点列表的时候的序号来操作断点
(lldb) breakpoint disable 2

# 启用断点
(lldb) breakpoint enable 2
   
# 删除断点
(lldb) breakpoint delete 1
```

演示如下：

![img](https://img.hacpai.com/file/2019/04/lldb4-d0b5b40a.png?imageView2/2/interlace/1/format/webp)

#### 调试环境操作

##### 启动

OK, 我们前面已经下好断点了，现在就要启动这个程序了！前面留一个断点是断在 main 函数。

```
# run命令就是启动程序
(lldb) run
```

##### 下一步、步入、步出、继续执行

```
# 下一步 (next 或 n)
(lldb) next

# 步入(step 或 s)
(lldb) step
  
# 步出(finish)
(lldb) finish

# 继续执行到下一个断点停, 后面没有断点的话就跑完了（continue 或 c）
(lldb) continue
```

##### 查看变量、跳帧查看变量

```
# 使用po或p，po一般用来输出指针指向的那个对象，p一般用来输出基础变量。普通数组两者都可用
(lldb) po result_array
 p
 # 查看所有帧(bt)
(lldb) bt
   
# 跳帧（frame select）
(lldb) frame select 1
  
# 查看当前帧中所有变量的值（frame variable）
 (lldb) frame variable
```

#### 修改变量或调用函数

使用 expression 命令可以在调试过程中修改变量的值，或者执行函数。

##### 修改变量值

```
(lldb) expression a
(int) $0 = 9
(lldb) frame variable a
(int) a = 9
(lldb) expression a=10
(int) $1 = 10
(lldb) frame variable a
(int) a = 10
(lldb) 
```

##### 调用函数

```
(lldb) expression printf("execute function %i",a)
(int) $2 = 19
execute function 10
```

对于执行结果都会自动保存，以备他用。

#### 线程控制

在加载进程开始调试后，当执行到设置断点时，可以使用 thread 命令控制代码的执行。

##### 线程继续执行

```
(lldb) thread continue 
Resuming thread 0x2c03 in process 46915 
Resuming process 46915 
(lldb)
```

##### 线程进入、单步执行或跳出

```
(lldb) thread step-in    
(lldb) thread step-over  
(lldb) thread step-out
```

##### 单步指令的执行

```
(lldb) thread step-inst  
(lldb) thread step-over-ins
```

##### 执行指定代码行数直到退出当前帧

```
(lldb) thread until 100
```

##### 查看线程列表，第一个线程为当前执行线程

```
(lldb) thread list
Process 46915 state is Stopped
* thread #1: tid = 0x2c03, 0x00007fff85cac76a, where = libSystem.B.dylib`__getdirentries64 + 10, stop reason = signal = SIGSTOP, queue = com.apple.main-thread
  thread #2: tid = 0x2e03, 0x0000
```

##### 查看当前线程栈

```
(lldb) thread backtrace
thread #1: tid = 0x2c03, stop reason = breakpoint 1.1, queue = com.apple.main-thread
 frame #0: 0x0000000100010d5b, where = Sketch`-[SKTGraphicView alignLeftEdges:] + 33 at /Projects/Sketch/SKTGraphicView.m:1405
 frame #1: 0x00007fff8602d152, where = AppKit`-[NSApplication sendAction:to:from:] + 95
 frame #2: 0
```

##### 查看所有线程的调用栈

```
(lldb) thread backtrace all
```

##### 设置当前线程

```
(lldb) thread select 2
```

部分参考：

[使用 LLDB 调试程序](https://link.ld246.com/forward?goto=https%3A%2F%2Fcasatwy.com%2Fshi-yong-lldbdiao-shi-cheng-xu.html)

[LLDB 常用命令](https://link.ld246.com/forward?goto=%5Bhttps%3A%2F%2Fblog.csdn.net%2Fu011374318%2Farticle%2Fdetails%2F79648178%5D(https%3A%2F%2Fblog.csdn.net%2Fu011374318%2Farticle%2Fdetails%2F79648178))

- ​    [     LLDB     ](https://ld246.com/tag/lldb)    
- ​    [     LLVM     ](https://ld246.com/tag/LLVM)    

##                             相关帖子                        

- ​                                    [                                                                              ](https://ld246.com/member/Hanseltu)                                    [Kaleidoscope 系列第五章：扩展语言—控制流](https://ld246.com/article/1570019695487)                                
- ​                                    [                                                                              ](https://ld246.com/member/Hanseltu)                                    [Kaleidoscope 系列第六章：扩展语言—用户自定义运算符](https://ld246.com/article/1570028565306)                                
- ​                                    [                                                                              ](https://ld246.com/member/Hanseltu)                                    [Kaleidoscope 系列第七章：扩展语言—可变变量](https://ld246.com/article/1570072984687)                                
- ​                                    [                                                                              ](https://ld246.com/member/Hanseltu)                                    [Kaleidoscope 系列第十章：总结和其他技巧](https://ld246.com/article/1570084835754)                                
- ​                                    [                                                                              ](https://ld246.com/member/Hanseltu)                                    [Kaleidoscope 系列第八章：编译为目标代码](https://ld246.com/article/1570096476155)                                
- ​                                    [                                                                              ](https://ld246.com/member/Hanseltu)                                    [Kaleidoscope 系列第九章：增加调试信息](https://ld246.com/article/1570114253333)                                
- ​                                    [                                                                              ](https://ld246.com/member/Hanseltu)                                    [KLEE 源码安装（Ubuntu 16.04 + LLVM 9）](https://ld246.com/article/1599015720782)                                

##                             随便看看                        

- ​                                    [                                                                              ](https://ld246.com/member/lijp)                                    [播放视频](https://ld246.com/article/1482814862194)                                
- ​                                    [                                                                              ](https://ld246.com/member/hanzanr123)                                    [sym 社区  搜索模块, 希望能更新个本地搜索](https://ld246.com/article/1536975303129)                                
- ​                                    [                                                                              ](https://ld246.com/member/nxh2000)                                    [不能同步](https://ld246.com/article/1615555384819)                                
- ​                                    [                                                                              ](https://ld246.com/member/88250)                                    [Java 开源博客——B3log Solo 0.5.5 正式版发布了！](https://ld246.com/article/1353748010708)                                
- ​                                    [                                                                              ](https://ld246.com/member/jinghong)                                    [你们过年回家吗？](https://ld246.com/article/1612228317112)                                
- ​                                    [                                                                              ](https://ld246.com/member/armstrong)                                    [波波之下楼记](https://ld246.com/article/1357387858622)                                
- ​                                    [                                                                              ](https://ld246.com/member/88250)                                    [关于 Solo 与社区同步的设置](https://ld246.com/article/1353772377257)                                

##            赞助商            [我要投放](https://ld246.com/article/1460083956075)    

​                                              

​                    回帖                

​                    [LLDB 调试命令使用指南](https://ld246.com/article/1556200452086)                



​                    

##                                     欢迎来到这里！                                

​                                    我们正在构建一个小众社区，大家在这里相互信任，以平等 • 自由 • 奔放的价值观进行分享交流。最终，希望大家能够找到与自己志同道合的伙伴，共同成长。                                

[注册](https://ld246.com/register)[关于](https://ld246.com/article/1440573175609)

请输入回帖内容                            ...                        

​                

​                

​    

​                [关于](https://ld246.com/article/1440573175609)                [API](https://ld246.com/article/1488603534762)                [数据统计](https://ld246.com/statistic)            

​                 © 2021 [链滴](https://ld246.com)            

​                [                                                                                    ](https://github.com/88250/symphony)                [                                                                                    ](https://weibo.com/b3log)                                    记录生活，连接点滴                            

​                    [滇ICP备14007358号-5](https://beian.miit.gov.cn) •                [Sym](https://b3log.org/sym)                v3.6.3            

当然还有其他的命令（和gdb命令通用）：

 

 

1. 命令            解释  
2. break NUM        在指定的行上设置断点。  
3. bt           显示所有的调用栈帧。该命令可用来显示函数的调用顺序。  
4. clear          删除设置在特定源文件、特定行上的断点。其用法为：clear FILENAME:NUM。  
5. continue        继续执行正在调试的程序。该命令用在程序由于处理信号或断点而导致停止运行时。  
6. display EXPR      每次程序停止后显示表达式的值。表达式由程序定义的变量组成。  
7. file FILE        装载指定的可执行文件进行调试。  
8. help NAME        显示指定命令的帮助信息。  
9. info break       显示当前断点清单，包括到达断点处的次数等。  
10. info files       显示被调试文件的详细信息。  
11. info func        显示所有的函数名称。  
12. info local       显示当函数中的局部变量信息。  
13. info prog        显示被调试程序的执行状态。  
14. info var        显示所有的全局和静态变量名称。  
15. kill          终止正被调试的程序。  
16. list          显示源代码段。  
17. make          在不退出 gdb 的情况下运行 make 工具。  
18. next          在不单步执行进入其他函数的情况下，向前执行一行源代码。  
19. print EXPR       显示表达式 EXPR 的值。   
20. print-object      打印一个对象  
21. print (int) name   打印一个类型  
22. print-object [artist description]  调用一个函数  
23. set artist = @"test"  设置变量值  
24. whatis         查看变理的数据类型  





使用lldb调试工具，结合 [解决EXC_BAD_ACCESS错误的一种方法--NSZombieEnabled](http://blog.csdn.net/likendsl/article/details/7566305)[ ](http://blog.csdn.net/likendsl/article/details/7566305)一起使用，实在是查找crash的一大利器啊，很是方便！

 

NSZombieEnabled 只能在Debug下用，发布时候，务必去掉。

 

lldb命令常用（备忘）

假如你准备在模拟器里面运行这个，你可以在“（lldb）”提示的后面输入下面的：

```
(lldb) po $eax
```

LLDB在xcode4.3或者之后的版本里面是默认的调试器。假如你正在使用老一点版本的xcode的话，你又GDB调试器。他们有一些基本的相同的命令，因此假如你的xcode使用的是“（gdb）”提示，而不是“（lldb）”提示的话，你也能够更随一起做，而没有问题。

“po”命令是“print object”（打印对象）的简写。“$eax”是cup的一个寄存器。在一个异常的情况下，这个寄存器将会包含一个异常对象的指针。注意：$eax只会在模拟器里面工作，假如你在设备上调试，你将需要使用”$r0″寄存器。

例如，假如你输入：

```
(lldb) po [$eax class]
```

你将会看像这样的东西：

```
(id) $2 = 0x01446e84 NSException
```

这些数字不重要，但是很明显的是你正在处理的NSException对象在这里。

你可以对这个对象调用任何方法。例如：

```
(lldb) po [$eax name]
```

这个将会输出这个异常的名字，在这里是NSInvalidArgumentException，并且：

```
(lldb) po [$eax reason]
```

这个将会输出错误消息：

```
(unsigned int) $4 = 114784400 Receiver () has no segue with identifier 'ModalSegue'
```

 

注意：当你仅仅使用了“po $eax”，这个命令将会对这个对象调用“description”方法和打印出来，在这个情况下，你也会得到错误的消息。

 

 

 

实用LLDB命令

命令名 用法 说明

|                         |                         |                                                              |
| ----------------------- | ----------------------- | ------------------------------------------------------------ |
| expr                    | expr 表达式             | 可以在调试时动态执行指定表达式，并将结果打印出来，很有用的命令。 |
| po                      | po 表达式               | 与expr类似，打印对象，会调用对象description方法。是*print-object*的简写 |
| print                   | print (type) 表达式     | 也是打印命令，需要指定类型。                                 |
| bt                      | bt [all]                | 打印调用堆栈，是*thread backtrace*的简写，加all可打印所有thread的堆栈。 |
| br l                    | br l                    | 是*breakpoint list*的简写                                    |
| process continue l      | process continue        | 简写：*c*                                                    |
| thread step-in l        | thread step-in l        | 简写：*s*                                                    |
| thread step-inst l      | thread step-inst l      | 简写：*si*                                                   |
| thread step-over l      | thread step-over l      | 简写：*n*                                                    |
| thread step-over-inst l | thread step-over-inst l | 简写：*ni*                                                   |
| thread step-out l       | thread step-out l       | 简写：*f*                                                    |
| thread list             | thread list             | 简写：*th l*                                                 |

 

 

#  

# **内存泄漏隐患提示**： Potential Leak of an object allocated on line …… **数据赋值隐患提示**： The left operand of …… is a garbage value; **对象引用隐患提示**： Reference-Counted object is used after it is released;

**对retain、copy、init、****release****、autorelease等在计数时的使用情况的详细讲解，推荐一下：**

http://www.cnblogs.com/andyque/archive/2011/08/08/2131236.html

调用autorelease这意味着，你可以在这个函数里面使用vari，但是，一旦下一次run  loop被调用的时候，它就会被发送release对象。然后引用计数改为0，那么内存也就被释放掉了。（关于autorelease到底是怎么工作的，我的理解是：每一个线程都有一个autoreleasePool的栈，里面放了很多autoreleasePool对象。当你向一个对象发送autorelease消息之后，就会把该对象加到当前栈顶的autoreleasePool中去。当当前runLoop结束的时候，就会把这个pool销毁，同时对它里面的所有的autorelease对象发送release消息。而autoreleasePool是在当前runLoop开始的时候创建的，并压入栈顶。那么什么是一个runLoop呢？一个UI事件，Timer call， delegate call， 都会是一个新的Runloop。）

 

 

 

 

# 当程序崩溃的时候怎么办,有如下两部分（英文版的）:

http://www.raywenderlich.com/10209/my-app-crashed-now-what-part-1

（中文的part-1）http://article.ityran.com/archives/1006

http://www.raywenderlich.com/10505/my-app-crashed-now-what-part-2

（中文的part-2）http://article.ityran.com/archives/1143

 

 

# 内存使用详细介绍：

http://www.cocoachina.com/bbs/simple/?t94017.html





# lldb 调试常用命令

## 设置目标程序

### 通过程序名调试

```
lldb /Applications/demo-app-ob-storyboard.app
(lldb) target create "/Applications/demo-app-ob-storyboard.app"
Current executable set to '/Applications/demo-app-ob-storyboard.app' (x86_64).
(lldb)
```

### 进入lldb之后设置程序名

```
➜  ~ lldb
(lldb) file /Applications/demo-app-ob-storyboard.app
Current executable set to '/Applications/demo-app-ob-storyboard.app' (x86_64).
(lldb)
```

### 调试可执行文件

```
➜  demo-app-hook git:(dark-mode) ✗ lldb -w Products/demo_app_hook.framework/demo_app_hook
(lldb) target create "Products/demo_app_hook.framework/demo_app_hook"
Current executable set to 'Products/demo_app_hook.framework/demo_app_hook' (x86_64).
(lldb)
```

## 断点

### 设置断点

#### 设置入口

```
(lldb) b -[NSObject init]
Breakpoint 1: where = libobjc.A.dylib`-[NSObject init], address = 0x000000000000a3a8
```

#### 给某个方法下断点

```
(lldb) breakpoint set --method Append
Breakpoint 4: 55 locations.
(lldb)
```

#### 指定类的某个方法

```
(lldb) breakpoint set -n "-[AppendText Append:]"
Breakpoint 6: where = demo-app-ob-storyboard`-[AppendText Append:] + 7 at AppendText.m:13:12, address = 0x0000000100001ac5
(lldb)
```

简写

```
br s -n "-[AppendText Append:]"
```

### 查看断点

```
(lldb) 
Current breakpoints:
1: name = '-[NSObject init]', locations = 1
  1.1: where = libobjc.A.dylib`-[NSObject init], address = libobjc.A.dylib[0x000000000000a3a8], unresolved, hit count = 0

2: name = 'delete', locations = 1
  2.1: where = CoreData`-[NSPersistentHistoryChangeRequestToken delete], address = CoreData[0x00000000002fcbd0], unresolved, hit count = 0

(lldb)
```

### 删除断点

```
(lldb) breakpoint delete
About to delete all breakpoints, do you want to do that?: [Y/n] y
All breakpoints removed. (2 breakpoints)
(lldb)
```

## 启动程序

### process launch

### run

### r

## 调试命令

这里和idea命令做对比

- 跳过断点（F9）
  - thread continue
- 步入（F7）
  - thread step-in
- 步出（F8）
  - thread stop-out
- 继续执行
  - c
- 查看变量
  - p 变量名
  - print 变量名
- 修改方法返回值
  - `thread return <value>`
- 查看线程列表
  - thread list

```
    lldb) thread list
   Process 45369 stopped
 thread #1: tid = 0x17b21f, 0x0000000100001ac5 demo-app-ob-storyboard`-[AppendText Append:](self=0x00000001002d9850, _cmd="Append:", name=@"sadf") at AppendText.m:13:12, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
  thread #9: tid = 0x17b26c, 0x00007fff679dc25a libsystem_kernel.dylib`mach_msg_trap + 10, name = 'com.apple.NSEventThread'
  thread #13: tid = 0x17bdf6, 0x00007fff679dd92e libsystem_kernel.dylib`__workq_kernreturn + 10
  thread #14: tid = 0x17c121, 0x00007fff679dd92e libsystem_kernel.dylib`__workq_kernreturn + 10
  thread #15: tid = 0x17c163, 0x00007fff679dd92e libsystem_kernel.dylib`__workq_kernreturn + 10
  thread #16: tid = 0x17c164, 0x00007fff679dd92e libsystem_kernel.dylib`__workq_kernreturn + 10
```

- 查看调用栈
  - thread backtrace

```
    (lldb) thread backtrace
    thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
   frame #0: 0x0000000100001ac5 demo-app-ob-storyboard`-[AppendText Append:](self=0x00000001002d9850, _cmd="Append:", name=@"sadf") at AppendText.m:13:12 [opt]
    frame #1: 0x000000010000195b demo-app-ob-storyboard`-[DetailController viewDidLoad](self=0x00000001002dc200, _cmd=<unavailable>) at DetailController.m:20:35 [opt]
    frame #2: 0x00007fff2d43817d AppKit`-[NSViewController _sendViewDidLoad] + 87
    frame #3: 0x00007fff2d492d0e AppKit`_noteLoadCompletionForObject + 643
    frame #4: 0x00007fff2d3994ad AppKit`-[NSIBObjectData nibInstantiateWithOwner:options:topLevelObjects:] + 1930
    frame #5: 0x00007fff2d41f93a AppKit`-[NSNib _instantiateNibWithExternalNameTable:options:] + 647
    frame #6: 0x00007fff2d41f5be AppKit`-[NSNib _instantiateWithOwner:options:topLevelObjects:] + 143
    frame #7: 0x00007fff2d41e92f AppKit`-[NSViewController loadView] + 345
    frame #8: 0x00007fff2d41e676 AppKit`-[NSViewController _loadViewIfRequired] + 72
    frame #9: 0x00007fff2d41e5f3 AppKit`-[NSViewController view] + 23
    frame #10: 0x00007fff2d6a29bf AppKit`+[NSWindow windowWithContentViewController:] + 41
    ......
(lldb) run
Process 45369 launched: '/Applications/demo-app-ob-storyboard.app/Contents/MacOS/demo-app-ob-storyboard' (x86_64)
2020-06-02 11:45:28.635549+0800 demo-app-ob-storyboard[45369:1552927] -------------hook method -------------
2020-06-02 11:45:28.914460+0800 demo-app-ob-storyboard[45369:1552927] [Nib Loading] Failed to connect (nameTest) outlet from (ViewController) to (NSTextFieldCell): missing setter or instance variable
demo-app-ob-storyboard was compiled with optimization - stepping may behave oddly; variables may not be available.
Process 45369 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
    frame #0: 0x0000000100001ac5 demo-app-ob-storyboard`-[AppendText Append:](self=0x0000000100300be0, _cmd="Append:", name=@"sadf") at AppendText.m:13:12 [opt]
   10
   11  	@implementation AppendText
   12  	- (NSString *)Append:(NSString *)name {
-> 13  	    return [NSString stringWithFormat:@"你好 %@", name];
   14  	}
   15  	@end
Target 0: (demo-app-ob-storyboard) stopped.
(lldb) p name
(NSTaggedPointerString *) $0 = 0x4efa213d7d9a3739 @"sadf"
(lldb) print name
(NSTaggedPointerString *) $1 = 0x4efa213d7d9a3739 @"sadf"
(lldb)
```

## 退出

```
exit
```