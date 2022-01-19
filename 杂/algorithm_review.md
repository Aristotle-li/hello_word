



# 笔记

## 数据结构

### 字符串

```python
#end_time = time.time()   #计算程序运行的时间
#复杂度
#from timeit import Timer
#字符串
s='Hello Python'.find('P')
print('Hello Python'.replace('o','eee'))#替换
print('hello'.count('e'))
print('hello'.__len__())
```

### 字典

```python
dict_a = {
    "name":"lishuo",
    "age": 25,
    "xingbie": "nan"
}
print(dict_a.items())
dict_a.pop('tel')   # 执行删除操作
print(dict_a.get('name1'))          #后面不写，不存在此key则返回nan，存在此key则返回value# 查找
dict_a['tel']=17690795591           #增加键值对
print(dict_a.items())
dict_a.pop('tel')   # 执行删除操作
#集合
girl_c=set(['marry','lily','xiaohua'])
girl_c.add('shop')  #增加项目
print(girl_c)
girl_c.remove('jerry') # 删减某项key
```

### time

```python
#list 操作测试
import timeit
def wtest1():
   n = 0
   for i in range(1010):
      n += i
   return n
def wtest2():
   return sum(range(1010))
def wtest3():
   return sum(x for x in range(1010))
if __name__ == '__main__':
   from timeit import Timer
   t1 = Timer("wtest1()", "from __main__ import wtest1")
   t2 = Timer("wtest2()", "from __main__ import wtest2")
   t3 = Timer("wtest3()", "from __main__ import wtest3")
   print(t1.timeit(10000))
   print(t2.timeit(10000))
   print(t3.timeit(10000))
   print(t1.repeat(3, 10000))
   print(t2.repeat(3, 10000))
   print(t3.repeat(3, 10000))
   t4 = timeit.timeit(stmt=wtest1, setup="from __main__ import wtest1", number=10000)
   t5 = timeit.timeit(stmt=wtest2, setup="from __main__ import wtest2", number=10000)
   t6 = timeit.timeit(stmt=wtest3, setup="from __main__ import wtest3", number=10000)
   print(t4)  # 0.05130029071325269
   print(t5)  # 0.015494466822610305
   print(t6)  # 0.05650903115721077
   print(timeit.repeat(stmt=wtest1, setup="from __main__ import wtest1",
                       number=10000))  # [0.05308853391023148, 0.04544335904366706, 0.05969025402337652]
   print(timeit.repeat(stmt=wtest2, setup="from __main__ import wtest2",
                       number=10000))  # [0.012824560678924846, 0.017111019558035345, 0.01429126826003152]
   print(timeit.repeat(stmt=wtest3, setup="from __main__ import wtest3",
                       number=10000))  # [0.07385010910706968, 0.06244617606430164, 0.06273494371932059]
```

## 时间复杂度



时间复杂度的几条基本计算规则
基本操作，即只有常数项，认为其时间复杂度为O(1)
顺序结构，时间复杂度按加法进行计算
循环结构，时间复杂度按乘法进行计算
判断一个算法的效率时，往往只需要关注操作数量的最高次项，其它次要项和常数项可以忽略
在没有特殊说明时，我们所分析的算法的时间复杂度都是指最坏时间复杂度

<img src="http://ww1.sinaimg.cn/large/006qX6GLgy1g8xlcfv744j30c90cd750.jpg" alt="list操作.png" style="zoom:80%;" />

<img src="http://ww1.sinaimg.cn/large/006qX6GLgy1g8xlcpq7zrj30dd06lglu.jpg" alt="dict操作.png" style="zoom:80%;" />

<img src="http://ww1.sinaimg.cn/large/006qX6GLgy1g8ypixm4vnj30qu0gcgm0.jpg" alt="undefined" style="zoom:50%;" />

## 堆区栈区



| 栈区（向下增长） | 高地址 |
| ---------------- | ------ |
| 堆区（向上增长） |        |
| 静态区（全局区） |        |
| 常量区           |        |
| 代码区           | 底地址 |

<img src="https://img-blog.csdnimg.cn/20191021161835642.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BhbmppYXBlbmdmbHk=,size_16,color_FFFFFF,t_70" alt="img" style="zoom:75%;" />

1、栈区：存放函数的参数值、局部变量等，由编译器自动分配和释放，通常在函数执行完后就释放了，其操作方式类似于数据结构中的栈。栈内存分配运算内置于CPU的指令集，效率很高，但是分配的内存量有限，比如iOS中栈区的大小是2M。

2、堆区：就是通过new、malloc、realloc分配的内存块，编译器不会负责它们的释放工作，需要用程序区释放。分配方式类似于数据结构中的链表。“内存泄漏”通常说的就是堆区。

3、静态区：全局变量和静态变量的存储是放在一块的，初始化的全局变量和静态变量在一块区域，未初始化的全局变量和未初始化的静态变量在相邻的另一块区域。程序结束后，由系统释放。

4、常量区：常量存储在这里，不允许修改。

5、代码区：顾名思义，存放代码。

## **栈**

栈可以由数组和链表实现:

栈是限定仅仅在表尾进行插入和删除操作的线性表，把允许插入和删除的一端称之为栈顶，另外一端称之为栈底。特点：后进先出，称之为后进先出线性表。

栈的应用：递归。

## **堆**

是一种经过排序的树形数据结构，每一个节点都有一个值，通常所说堆的数据结构是二叉树，堆的存取是随意的。所以堆在数据结构中通常可以被看做是一棵树的数组对象。而且堆需要满足一下两个性质：
（1）堆中某个节点的值总是不大于或不小于其父节点的值；
（2）堆总是一棵完全二叉树。（若设二叉树的深度为k，除第 k 层外，其它各层 (1～k-1) 的结点数都达到最大个数，第k 层所有的结点都**连续集中在最左边**，这就是完全二叉树。）

堆的应用：堆排序，快速找出最大值、最小值，简化时间复杂度，像这样支持**插入元素**和**寻找最大（小）值元素**的数据结构称之为**优先队列**。

## 二叉查找树

一维数组建立二叉树，其实就是建立一个二叉查找树

###### 性质：a.每一个树根要大于左子树小于右子树

###### 			 b.中序遍历后输出是从小到大的一个排序

##  timeit模块

timeit模块可以用来测试一小段Python代码的执行速度。

###### class timeit.Timer(stmt='pass', setup='pass', timer=)

Timer是测量小段代码执行速度的类。

stmt参数是要测试的代码语句（statment）；

setup参数是运行代码时需要的设置；

timer参数是一个定时器函数，与平台有关。

###### timeit.Timer.timeit(number=1000000)

Timer类中测试语句执行速度的对象方法。number参数是测试代码时的测试次数，默认为1000000次。方法返回执行代码的平均耗时，一个float类型的秒数。

## 广度优先搜索

​    广度优先搜索（也称宽度优先搜索，缩写BFS）是连通图的一种遍历算法这一算法也是很多重要的图的算法的原型。Dijkstra单源最短路径算法和Prim最小生成树算法都采用了和宽度优先搜索类似的思想。其别名又叫BFS，属于一种盲目搜寻法，目的是系统地展开并检查图中的所有节点，以找寻结果。换句话说，它并不考虑结果的可能位置，彻底地搜索整张图，直到找到结果为止。基本过程，BFS是从根节点开始，沿着树(图)的宽度遍历树(图)的节点。如果所有节点均被访问，则算法中止。

一般用队列数据结构来辅助实现BFS算法。

## 深度优先遍历

- 设树的根结点为D，左子树为L，右子树为R，且要求L一定在R之前，则有下面几种遍历方式：

   

  - 前序遍历，也叫先序遍历、也叫先根遍历，DLR

  - 中序遍历，也叫中根遍历，LDR

  - 后序遍历，也叫后根遍历，LRD

    

    

    # 堆排序Heap Sort

## 堆Heap

- 堆是一个完全二叉树，物理结构可以是顺序存储也可以是链式存储(

- 每个非叶子结点都要大于或者等于其左右孩子结点的值称为大顶堆

- 每个非叶子结点都要小于或者等于其左右孩子结点的值称为小顶堆

- 根结点一定是大顶堆中的最大值，一定是小顶堆中的最小值

- n个关键字序列List[1,2,3...N]称为堆(后面简写为L)，在例子中我们统一使用顺序存储结构)，

- 当L(i) <= L(2i + 1)且L(i) <= L(2i + 2),则称为 小根堆

  当L(i) >= L(2i + 1)且L(i) >= L(2i + 2),则称为 大根堆（本质就是小根堆的反转）

  

### 建堆原则

从一颗完全二叉树建堆有两个原则：（大顶堆）

1、父节点与孩子作比较，若小于孩子，则交换。若大于则继续向下比较

2、若发生交换，则要朝着根节点重复1，才能继续

### 小顶堆

- 完全二叉树的每个非叶子结点都要小于或者等于其左右孩子结点的值称为小顶堆

- 根结点一定是小顶堆中的最小值

  

### 大顶堆

- 完全二叉树的每个非叶子结点都要大于或者等于其左右孩子结点的值称为大顶堆

- 根结点一定是大顶堆中的最大值

  

## 优先队列

堆排序是用堆实现的一个算法，优先队列是用堆实现的数据结构，堆排序仅仅是排序，优先队列可以随时取队中最大值，插入等，手动实现也可以一定程度上实现删除

优先队列指的是元素入队和出队的顺序与时间无关，既不是先进先出，也不是先进后出，而是根据元素的重要性来决定的。

例如，操作系统的任务执行是优先队列。一些情况下，会有新的任务进入，并且之前任务的重要性也会改变或者之前的任务被完成出队。而这个出队、入队的过程利用堆结构，时间复杂度是`O(log2_n)`。

## 静态数据及动态数组的创建

  静态数据：  

​      int a[10]；

​      int a[]={1,2,3};

​      数组的长度必须为常量。

  动态数组：

​      int len;

​      int *a=new int [len];

​      delete a;

​      数组的大小可以为变量。

int  p[]=new  int[len];  编译器会说不能把int*型转化为int[]型，因为用new开辟了一段内存空间后会返回这段内存的首地址，所以  要把这个地址赋给一个指针，所以要用

int  *p=new  int[len];

## sizeof

对静态数组名进行sizeof运算时，结果是整个数组占用空间的大小；因此可以用sizeof(数组名)/sizeof(*数组名)来获取数组的长度。

int a[5]; 则sizeof(a)=20,sizeof(*a)=4.因为整个数组共占20字节，首个元素（int型）占4字节。

###### 但是：

静态数组作为函数参数时，在函数内对数组名进行sizeof运算，结果为4，因为此时数组名代表的指针即一个地址，占用4个字节的内存

## 通过函数返回一个数组的问题

  函数声明的静态数组不可能通过函数返回，因为生存期的问题，函数调用完其内部变量占用的内存就被释放了。如果想通过函数返回一个数组，可以在函数中用new动态创建该数组，然后返回其首地址。
其原因可以这样理解，因为[]静态数组是在栈中申请的，而函数中的局部变量也是在栈中的，而new动态数组是在堆中的分配的，所以函数返回后，栈中的东西被自动释放，而堆中的东西如果没有delete不会自动释放。

### new/delete and malloc/free

new/delete是操作符，malloc/free是标准库函数，前者可以重载，可以有构造和析构函数，可以返回某种类型对象的指针，后者返回void指针。

例如：

```c++
#include<iostream>
#include<algorithm>
using namespace std;
int *test(int *b) //b可以是静态数组的数组名，也可以是动态数组的首地址
{ 
int i=0;
for(int i=0;i<5;i++) //输出传入的数组各元素 
cout<<*(b+i)<<" "; 
cout<<endl; 
int *c=new int[5]; //动态创建一个数组  
//如果将绿部分换为int c[5];则主函数中调用test无法得到c数组  
for(i=0;i<5;i++)  //新数组的各项值等于传入的数组各项值加5  
 *(c+i)=*(b+i)+5; 
return c;     //返回新创建的动态数组的首地址
}
int main(){
int i = 0;
int *b=new int[5]; 
//创建动态数组b 
for(int i=0;i<5;i++)//赋值 
 *(b+i)=i;  //绿色部分也可以换为int b[5]={0,1,2,3,4};即也可以是静态数组
int *c=test(b);   //将b作为参数，调用test函数，返回值赋给c 
for(i=0;i<5;i++)  //输出test返回的数组的各项  
 cout<<*(c+i)<<" "; 
cout<<endl; 
return 0;
}
```



### 分治

#### 1、采用分而治之的思想，将一个凸多边形分成两个凸多边形，

然后再把每一个 凸多边形再分成两个凸多边形，直到化成三角形为止。因为凸多边形的任意一条 边必定属于某个三角形，所以我们以某一条边为基准，以这条边两个顶点为起点 P1 和终点 Pn，将凸多边形顶点依序标记 P1,P2...Pn，再在该多边形中找任意一 个不属于这两个点的顶点 Pk(2<=k<=n-1)，构成一个三角形，用这个三角形把一 个凸多边形分成两个。最后根据乘法定理:凸多边形分解的数量就等价于凸 k 边 形划分的方案数乘凸 n-k+1 边形划分方案数，即

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201217160559380.png" alt="image-20201217160559380" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201217160414855.png" alt="image-20201217160414855" style="zoom:50%;" />

#### 2、分治策略，寻找逆序数

归并排序加计数，每次归并排序计算两组数的所有逆序然后再merge，递归即可

看懂了！！！

•**循环不变量**：数组A[l, k] 包含了两个子数组L[1, *n*1] 和R[1, *n*2] 中最小的*k -* *l* + 1 个元素，且有序排列。L[i] 和R[j] 表示对应子数组中尚未被并入A 的最小元素，L和R本身有序。若L[i] > (R[j] * 3) 成立，那么对于L 中比L[i] 大的元素，这一关系仍然成立。A 中的跨越逆序对数量不受到L 与R 整体是否有序的影响。

•**初始化**：*k* = *l*，此时A 为空，循环不变量成立。

•**维护**：若L[i] < R[j] 成立，此时A[l, k] 包含了两个子数组中最小的*k – l* +1 个元素，将L[i] 并入数组，A[l, k + 1] 包含了两个子数组中最小的*k - l* + 2 个元素。新考察的L[i + 1]是目前L 中尚未被归入的元素中的最小元素。新考察的R[j + 1] 是目前R 中尚未被归入的元素中的最小元素。若L[i] >(R[j] * 3) 成立，既有L[i + 1] > L[i] 成立，则L[i + 1] > (R[j] * 3) 成立，对任意L[i]及其之后的元素这一关系均成立，共有*n*1 *-* *i* + 1 个逆序对。因此，循环不变量成立。

•**终止**：归并完成，*k* = *n*1 + *n*2，其包含了两个子数组的*n*1 + *n*2 及对应的所有元素。

每一计算得到的逆序对数量的和，再加上L和R本身的逆序对数量，就是整个数组中逆序对数量。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201217182428905.png" alt="image-20201217182428905" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201217182613440.png" alt="image-20201217182613440" style="zoom:50%;" />

#### 3、分治策略，寻找一颗完全二叉树的局部极小值，O(logn)复杂度

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201217191339607.png" alt="image-20201217191339607" style="zoom:50%;" />

```c++
//
Find_min(T)
	if min(probe(T.left()),probe(T.right()),probe(T.root()))==probe(T.root()):
	then 
    return T.root()
   else:
     if probe(T.left()) > probe(T.right())
        Find_min(T.right()):
      else:
        Find_min(T.left()):
       
```

##### 4、**问题描述：数组中一个或多个连续元素形成一个子序列，找出所有子序列的和的最大值**



<img src="/Users/lishuo/Library/Containers/com.tencent.qq/Data/Library/Caches/Images/43934FF5AE7E40FD034E38A6F0ABF7E2.jpg" alt="43934FF5AE7E40FD034E38A6F0ABF7E2" style="zoom:50%;" />

上面的操作结果就是返回了 中、左加中、右加中、左加右加中 的最大值 。最后再和 左、右取最大。

左 右 我已经从上次里面算了出来，所以每次只计算考虑中间

的数进来的情况

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201217215028723.png" alt="image-20201217215028723" style="zoom:50%;" />



#### 5、求有序数组中目标元素的起始位置与结束位置，要求时间复杂度为O(logn)

典型的二分查找问题。原问题可以分解为两个子问题求解，即寻找目标元素的起始位置与结束位置。令left=0，right=数组的长度N-1，pos=[-1,-1]。寻找起始位置时，取mid=int((left+right)/2)，判断mid处的数值与target的关系，若A[mid]<target，则将mid+1赋值给left，否则将mid-1赋值给right，若遇到mid处等于target时，同时更新pos[0]的取值。递归直至left>right。寻找结束位置时同理

复 杂 度：T(n) = T(n/2) + O(1) 

​       = O(logn)

l接下来证明在每次循环中，满足循环条件left<=right时，target的起始位置和始终在[left, right]内。假设第k次循环后，若left<=right，由于数组是有序的，求得中间元素A[mid]，比较A[mid]与target的大小，若A[mid]<target，则说明左半块数组不会出现target，于是令left=mid+1；否则说明右半块数组不会出现target第一个出现的位置，令right=mid-1，同时若出现A[mid]=target的情况，则对返回的pos进行更新，即记录起始位置为mid。从而使得查询的范围缩小一半。结束位置的证明同理。



#### 6、旋转数组的极值  O(logn)

我们先把第一个指针指向第0个元素，把第二个指针指向第4个元素，如图所示。位于两个指针中间（在数组的下标是2）的数字是5，它大于第一个指针指向的数字。因此中间数字5一定位于第一个递增字数组中，并且最小的数字一定位于它的后面。因此我们可以移动第一个指针让它指向数组的中间。

此时位于这两个指针中间的数字为1，它小于第二个指针指向的数字。因此这个中间数字为1一定位于第二个递增子数组中，并且最小的数字一定位于它的前面或者它自己就是最小的数字。因此我们可以移动第二个指针指向两个指针中间的元素即下标为3的元素。

此时两个指针的距离为1，表明第一个指针已经指向了第一个递增子数组的末尾，而第二个指针指向第二个递增子数组的开头。第二个子数组的第一个数字就是最小的数字，因此第二个指针指向的数字就是我们查找的结果。

![20150728171658381](/Users/lishuo/Library/Application Support/typora-user-images/20150728171658381.png)

```python

使用索引left、right分别指向数组首尾。 
求中点 mid = ( left + right ) / 2 
A[mid]＞A[mid+1]，丢弃后半段：right=mid 
A[mid+1]＞A[mid]，丢弃前半段：left=mid+1 
递归直至left==right 
时间复杂度为O(logN)
def loca_max(list):
    left = 0
    right = len(list)-1
    #
    while left != right:
        mid = int(left + ((right - left) >> 1))
        print(mid)
        if list[mid] > list[mid+1]:
            right = mid
        else:
            left = mid+1
    return left
list = [1,2,4,5,4,3,4,5,8,9,0,1,2,3,4,54,5,66]
print(loca_max(list))
```

### DP算法

##### 1、劫匪问题

两个相邻的房屋不能一晚同时被抢，且没有最大容量限制。现在假设我们已经得到子问题的最优解，当前考虑第一个决策即考虑最后一个房子抢还是不抢。如果抢的话，则原问题的解变成了最后一个房子的钱加上n-2个房子的钱；如果不抢的话，则原问题的解变成了最后n-1个房子的钱。这两种选择的最大值就是最终的解。可以得到递归式

​		**子 问 题：**前i个房子中抢劫犯有可能抢到的最大总金额

​		**最优子结构：**OPT(i)，每次决策是否要抢当前房屋

​		**递推关系式：**

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201217230006041.png" alt="image-20201217230006041" style="zoom:33%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201217225938688.png" alt="image-20201217225938688" style="zoom:50%;" />

​		房子排成一个圈：只要考虑（1~n-1）和（2~n）中的最优值就可得最优解

##### 2、最大可分子集

首先需要对题目给出的数组进行排序，这样的作用是我们从左到右遍历一 次，每次只看后面的数字能不能被前面的整除就行。

使用一个一维 DP，其含义是题目要求的数组，DP[i]的含义是，从 0~i 位 置满足题目的最长数组。先用 i 遍历每个数字，然后用 j 从后向前寻找能被 nums[i]整除的数字，这样如果判断能整除的时候，再判断 dp[i] < dp[j] + 1，即 对于以 i 索引结尾的最长的数组是否变长了。在变长的情况下，需要更新

dp[i]，同时使用 parent[i]更新 i 的前面能整除的数字下标。每一个parent里面放的是上一个最大整除的数字下标，这样就能回溯到所有最大可分子集

另外还要统计对于整个数、组最长的子数组长度。mx+1

知道了对于每个位置最长的子数组之后，我们也就知道了对于 0~n 区间内 最长的满足题目条件的数组，最后需要再次遍历，使用 parent 回溯。

因为是两层循环，时间复杂度是 O(N^2)，空间复杂度是 O(N).

**最优子结构：**OPT[i]**，表示包含第**i个元素的最大整除子集的长度

决策已存在的子集能否加入第**i**个元素

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201217230759059.png" alt="image-20201217230759059" style="zoom:50%;" />

```python
nums = [int(x) for x in input().split()]
nums.sort()
N = len(nums)
dp = [1]*N
parent =[0]*N
mx = 1
mx_index = -1
for i in range(N):
    for j in range(i-1,-1,-1):
        if nums[i] % nums[j] == 0 and dp[i] < dp[j] + 1:
            dp[i]=dp[j]+1  #存了每一个位置能整除的最多的数，不包括自己，所以最大子集 = max dp[i]+1
            parent[i]=j
            if dp[i]>mx:
                mx = dp[i]
                mx_index=i

res=list()
for k in range(mx):
    res.append(nums[mx_index])
    mx_index=parent[mx_index]    # parent里面保存的是回溯路径的索引
print(res[::-1])

```

##### 3、Given n, how many structurally unique BST’s 

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201217155458298.png" alt="image-20201217155458298" style="zoom:50%;" />

​		对于以1~n为节点值组成的二叉树中，1~n每个数都可以作为根节点的节点值，当以i作为根节点时，1~(i-1) 这些数位于根节点的左子树中，(i+1)~n这些数位于根节点的右子节点中,即当以i作为根节点时有2个子问题，而1~n每个数都可以作为根节点的节点值，所以总共有2n个子问题。

​		以(i+1)~n为节点组成二叉搜索树的数目等于以1~(n-i)为节点组成二叉搜索树的数目。

那么为什么是乘法呢，是因为左面每一种情况都对应右边所有情况，符合乘法公式。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201219204018611.png" alt="image-20201219204018611" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201219203433838.png" alt="image-20201219203433838" style="zoom:50%;" />

```java
//DP算法  这个好！！因为开了一个result数组，把结果都存了起来，避免了重复计算
public class Solution {  
        public int numTrees(int n) {  
            int[] result = new int[n+1]; 
            //或者这样vector<int> result(n+1,0);
            result[0] = 1;  
            result[1] = 1;  
            for(int i = 2; i <= n; i++)  
                for(int j = 1; j < =i; j++) {  
                    result[i] += result[j-1] * result[i-j];  
                }  
            return result[n];  
        }  
    }  

```

```java
//递归  太慢了，因为含有很多重复计算！！！
class Solution {
public:
    int numTrees(int n)
    {
        if(n<=1)    return 1;
        int sum=0;
        for(int i=1;i<=n;++i)
            sum+=numTrees(i-1)*numTrees(n-i);

        return sum;
    }
};
```

### 4、Master 定理

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201217170715662.png" alt="image-20201217170715662" style="zoom:50%;" />

 a 代表分成子问题的个数

b 分后子问题的规模

d 每个子问题的开销

#### DP: 给定一堆已知重量的石头，把石头分成两堆，使得两堆的重量之差最小，求两堆石头最小的重量差。

**空间优化：**

​    由于每次计算 dp[i][w]时，总是只与上一项 dp[i-1]有关，因此可以使用一维数组重复更新的方式来代替二维数组，即 递推关系式可以写作: 

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201219204646872.png" alt="image-20201219204646872" style="zoom:50%;" />

#### DP:有m种面值的硬币，有一个目标金额n， 请求出所有换零方式

假设当前考察钱i种硬币的凑出情况，如果不使用第种硬币，那么情况数取决于前i-1种硬币，此时的情况数为opt[j]=opt[j] 。如果使用第 i种硬币，且存在多个该面值硬币凑出金额的情形，那么情况数取决于，已经有该硬币参与下，加上这枚硬币恰能凑出j的情况数，即凑出j-coins[i]的结果opt[j]=opt[j-coins[i]]。综合两种决策，最终情况数为

opt[j] = opt[j] + opt[j-coins[i]]   

若 coins[i] > j ,那就不要加入了，对应opt[i] [j] = opt[i-1] [j] 

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201219200251371.png" alt="image-20201219200251371" style="zoom:50%;" />

1、最简单的case  对应一枚硬币，一种金额，能整除为1 ， 不能为0

2、要加入第 i 枚硬币时，没有 i 时所有组合 + 包含i  的情况下，再加一个 i 时恰能凑出j的情况数！

#### DP : 给定一个整数数组，返回它的某个非空子数组（连续元素）在执行一次可选的删除操作后，所能得到的最大连续子数组和。

联想：最大子序和的问题
$$
dp[i] = max  
{dp[i-1]+a[i],a[i]}
$$
**复杂度分析：** O(N)

最优子结构：加入第2维表示是否执行过1次删除

​     dp[i][0]表示前i个数中还没有执行过删除的、以第i个数结尾的最大连续子数组和

​     dp[i][1]表示前i个数中执行过1次删除的、以第i个数结尾的最大连续子数组和

```matlab
function maxSum(a){
dp[0][0] = 0;
dp[1][1] = int_min;
res = int_min:
for i = 1 to n do
	dp[i][0] = max(dp[i-1][0]+a[i]);\\等价于求最大连续子数组和
	dp[i][1] = max(dp[i-1][1]+a[i]);\\等价于删了前面的或者当前这个数
	res = max(maxSum,max(dp[i][0],dp[i][1]))
end for
return res; 
}
```

#### DP：给定一堆已知重量的石头，把石头分成两堆，使得两堆的重量之差最小，求两堆石头最小的重量差。

​     两堆石头的重量和越接近 sum/2(sum 表示所有石头的总重量和)，两堆的重量差越小。

问题转化为0-1背包问题，背包容量为sum/2，每件物品重量a[i] , 如何选取物品使得重量最大

子问题：从前i个石头中，选取重量和不超过容量w的石头，获得的重量越大越好。

最优子结构：dp[i] [w] 表示从前i 个石头中选取重量和不超过w 所能获得最大重量和，多步决策中，每一步考虑是否要选择第 i 个石头。

* 选择第 i 个石头，则背包内剩余重量限制变为w - a[i]
* 不选择 ，则背包内剩余重量仍为w

### Greedy 

#### 1、决定n个任务J1,J2,...,Jn的顺序。一台服务器和n台PC上能最快的处理完这些 任务。任务首先由服务器预处理，之后由PC处理。每个任务互不相关。一台服务器或一 台PC同时只能处理一个任务。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201220132310994.png" alt="image-20201220132310994" style="zoom:50%;" />

**正 确 性:**若最后一个任务A的PC处理时间不为最短，将其与之前的一个PC处理时 间更短的B排在A后。

#### 2、一群人乘船渡河，单个人的体重不会超过船的载重，每条船最多载两人并且 不能超出船的载重，最少使用多少条船才能让所有人过河。



算法：

​				先排序，让首尾结合坐船，同时移动指针，

​				若不行，则重的单独坐船，然后单独移动指针。

O(nlogn)，n为人数，排序复杂度。

##### 正确性：

​				1、先按照直观地贪心做法，对体重排序，先让最重的上船，然后尝试让剩下的里面最重的上船，若有加入，共乘一条船，若没有，则单独乘船。

​				2、证明： 直观地的贪心可以转化为此贪心

若有两条船，（p1，p3） （p2，p4），那么，其中p1>p2>p3>p4 （假如p1 p2 加起来太重，不能同乘）

可证明交换 p3 p4 后，使用的船数目一致但是复杂度降低了！！！

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201220144615947.png" alt="image-20201220144615947" style="zoom:50%;" />

#### 3、给定1到n的一个排列，求如何将其分割成k份后，使得每个分割里的最大值 的和最大。

算法：

将所有数排序后，得到其最大的k个值，使其处于独立的k个分割中，剩下的数随意分配。具体的分割方法是每个剩余的数的分配方法的乘积。

**复 杂 度:**O(nlogn)，n为数的个数。

正确性：若不按照上述分割，不妨设第 i 大的值为一个区域的最大值，且和第 j 大的值在一个区域，可将这个区域中第 i大的值两边的值分给其两边的区域（若这个区域在分割的一端，则分一边的区域即可），这个区域最大值不会变，其他区域的最大值至少不会变小。

#### 4、问题描述:N个男孩，第i个男孩所在队伍至少有a_i个人，求最多可组成的队 伍数。

#### 6、n个玩具建筑，每次操作可选择一个连续子区间使建筑高度+1，求最少操作 数使建筑从左到右单调递增。

```python
count = 0
def toy_building(A):
  for i in range(1,len(A)):
    k = max{A[i-1]-A[i],0}
    count = count + k 
    return count

    
	
```





#### LP

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20201220163725903.png" alt="image-20201220163725903" style="zoom:50%;" />













