



学生管理系统：

1、https://blog.csdn.net/qq_42780025/article/details/94453068

2、https://download.csdn.net/download/qcyfred/9737745

3、https://download.csdn.net/download/qq_44519484/12426815

pytorch-cpp

https://zhuanlan.zhihu.com/p/56090735

https://github.com/BIGBALLON/PyTorch-CPP



## C++程序设计

### 宏



![1](/Users/lishuo/Downloads/1.png)

> 宏定义没有数据类型仅仅是符号的替换，不建议用，使用内联函数。

![image-20210312143542082](/Users/lishuo/Library/Application Support/typora-user-images/image-20210312143542082.png)

### 命名空间



![image-20210312143801386](/Users/lishuo/Library/Application Support/typora-user-images/image-20210312143801386.png)



> 使用 #include<iostream.h>是调用c的库函数，是将标准库功能定义在全局空间里面，最新的c++使用#include<iostream> 是调用c++的库函数，没有定义全局命名空间， 必须使用namespace std

### 输入输出

![image-20210312145032343](/Users/lishuo/Library/Application Support/typora-user-images/image-20210312145032343.png)

> C 中是 printf  endl 代表回车换行

### 泛型和多态

通过重载和虚函数实现多态：

* 重载称为编译时的多态
* 虚函数称为运行时的多态

单引号和双引号不能混用：单引号>字符（中间不能有单引号和反斜杠），双引号>字符串。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210314131145492.png" alt="image-20210314131145492" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210314131557042.png" alt="image-20210314131557042" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210314131614961.png" alt="image-20210314131614961" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210314132859422.png" alt="image-20210314132859422" style="zoom:50%;" />

常变量：

* const int a=3; a=3不能改变
* (stringXXX == "yes"?"no":"yes")

> 前的表达式为true则取:前的值，否则:后的值
> stringXXX =="yes" 就取"no" 否则"yes"

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210314170758577.png" alt="image-20210314170758577" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210314171023367.png" alt="image-20210314171023367" style="zoom:50%;" />



条件运算符结合方式自右向左
<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210314172656224.png" alt="image-20210314172656224" style="zoom:50%;" />

![image-20210314174620285](/Users/lishuo/Library/Application Support/typora-user-images/image-20210314174620285.png)

![image-20210318143320479](/Users/lishuo/Library/Application Support/typora-user-images/image-20210318143320479.png)

![image-20210318143341343](/Users/lishuo/Library/Application Support/typora-user-images/image-20210318143341343.png)

![image-20210318143650647](/Users/lishuo/Library/Application Support/typora-user-images/image-20210318143650647.png)

![image-20210318144010928](/Users/lishuo/Library/Application Support/typora-user-images/image-20210318144010928.png)

![image-20210318144112906](/Users/lishuo/Library/Application Support/typora-user-images/image-20210318144112906.png)

![image-20210318144133470](/Users/lishuo/Library/Application Support/typora-user-images/image-20210318144133470.png)

![image-20210318144212798](/Users/lishuo/Library/Application Support/typora-user-images/image-20210318144212798.png)

![image-20210318144250230](/Users/lishuo/Library/Application Support/typora-user-images/image-20210318144250230.png)

![image-20210318144306782](/Users/lishuo/Library/Application Support/typora-user-images/image-20210318144306782.png)



## 字符串





![image-20210326150047130](/Users/lishuo/Library/Application Support/typora-user-images/image-20210326150047130.png)

> C++函数的定义不允许嵌套,但是允许递归调用，在未出现函数调用时，形参并不占内存的存储单元，只 有在函数开始调用时，形参才被分配内存单元。调用结 束后，形参所占用的内存单元被释放
>
> 字符串结束要有一个`\0`，当一个字符串含有n个字符时，用于存储该字符串的空 间至少为n+1个存储单元。

> 函数声明：如果使用用户自己定义的函数，而该函数与调用 它的函数(即主调函数)在同一个程序单位中且 位置在主调函数之后，则必须在调用此函数之前 对被调用的函数作声明。 函数声明也称函数原型，由于函数原型是一条语句，因此函数原型必须以分号结束，函数原型不必包含参数的名字，可只包含参数的类型。
>
> 返回类型 函数名(数据类型 参数名,数据类型 参数名....);

* 按值传递
* 地址传递
* 引用传递

## 嵌套



由前述可知，C++函数不能嵌套定义，即一个函数不能在另一个函数体中进行定义。

• 但在使用时，允许嵌套调用，即在调用一个函数的过 程中又调用另一个函数。 但是每次调用，保存前一个的现场，浪费内存，调用完成后恢复。

## 递归的实现

* 一个问题能否用递归实现，看其是否具有下面的 特点:

– 需有完成任务的递推公式。

– 结束递归的条件。

* 编写递归函数时，程序中必须有相应的语句:

– 一个递归调用语句。

– 测试结束语句。先测试(递归结束的条件)，

后递归调用。

> 是否利用递归编程要看实际问题，如果要节约内 存就用循环语句实现。若对内存要求并不高，可 以用递归编程。

## 存储区

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210326160434229.png" alt="image-20210326160434229" style="zoom:67%;" />

> * 在C++中建立和删除堆对象使用new和delete两个运算符。当堆对象不再使用时，应予以删除，回收其所占用的动
>
>   态内存。
>
> * 栈区由编译器自动分配并且释放，用来存放局部变量 、函数参数、函数返回值和临时变量等;
>
> * 堆区是程序空间中存在的一些空闲存储单元，这些空 闲存储单元组成堆，堆也称为自由存储单元，由程序 员申请分配和释放。





`为了在函数体内使用与局部变量同名的全局变 量，应在全局变量前使用作用域作用符“::”	`





![image-20210402142613572](/Users/lishuo/Library/Application Support/typora-user-images/image-20210402142613572.png)

![image-20210402143046296](/Users/lishuo/Library/Application Support/typora-user-images/image-20210402143046296.png)

函数声明用了默认值，定义就不用了

![image-20210402143501842](/Users/lishuo/Library/Application Support/typora-user-images/image-20210402143501842.png)

![image-20210402143546237](/Users/lishuo/Library/Application Support/typora-user-images/image-20210402143546237.png)

用static声明静态全局变量

• 在全局变量定义时前面加上static

– 有时程序设计中希望某些全局变量只限于本文 件引用，而不能被其他文件引用，这时可在定 义外部变量时加一个static声明。

• 使用静态全局变量的优点是:

– 当许多程序员分工协作开发项目时，为了避免使用了相同的全局变量而影响到程序的正确性，

可以定义成静态全局变量。

![image-20210416132853859](/Users/lishuo/Library/Application Support/typora-user-images/image-20210416132853859.png)

![image-20210416132942040](/Users/lishuo/Library/Application Support/typora-user-images/image-20210416132942040.png)

![image-20210416133015438](/Users/lishuo/Library/Application Support/typora-user-images/image-20210416133015438.png)

##  指针

一般认为:任何一个指针变量本身数据值的类型都是 unsigned long int，占4个字节。

![image-20210402153828566](/Users/lishuo/Library/Application Support/typora-user-images/image-20210402153828566.png)

![image-20210402155306545](/Users/lishuo/Library/Application Support/typora-user-images/image-20210402155306545.png)



![image-20210416164733942](/Users/lishuo/Library/Application Support/typora-user-images/image-20210416164733942.png)

1、p=a和p=&a[0]是一样的

2、它执行的运算不是两指针存储的地址值相减，而是按下列公式得出结果：((px)-(py))/数据长度px-py运算的结果值是两指针指向的地址位置之间的数据个数。所以，两指针相减的结果值不是地址量，而是一个整数。

3、数组名**a**是地址常量，不可实现其自身的改变，如**a++**非法; • 而数组指针**p**是地址变量，**p++**合法。

4、 两个指针的比较一般用于下列两种情况:

​	 一是比较两个指针所指向的对象在内存中的位置关系。 比如双指针快排

​	 二是判断指针是否为空指针。比如申请动态数组，申请完先判断是否申请成功。

### 数组指针：其实就是一个指向数组的指针

 int (*p)[5];

就等价于int a[5]; int *p=a

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210416171658814.png" alt="image-20210416171658814" style="zoom:33%;" />

### 指针数组：数据元素为指针的数组。

int *p[5];

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210416171719025.png" alt="image-20210416171719025" style="zoom:33%;" />



一般存放的是二维数组的行地址

```c++
#include<iostream>
using namespace std;
int main()
{   int x[2][3]={{1,2,3},{14,15,26}};
    int i,j;
    int *p[2]= {x[0],x[1]};                //����ָ�����鲢��ʼ��

    for(i=0;i<2;i++)
    {     for(j=0;j<3;j++)
               cout<<*(p[i]+j)<<'\t';     //����ָ���������Ԫ��ֵ
          cout<<endl;
    }

    return 0;
}
输出：
1		2		3
14	15	26
```

### 指针传递

* 函数**定义**时将：**形参的类型说明成指针**。void swap(int *x, int *y)

* 函数**调用**时：就需要**指定地址值形式的实参**。	swap(&a, &b);

这时的参数传递方式即为地址传递方式。

### 引用

* 定义：当“&”的前面有类型符时(如int &a)，它必然是对 引用的声明;如果前面无类型符(如cout<<&a),则是取 变量的地址

* 引用传递函数参数：

  – 引用具有传地址的作用，这一点非常类似于指针类型。 正是由于引用这个的特点，往往可用它来代替指针类 型的参数。

  – 这种方式不需要为参数开辟存储空间，避免了传递大 量数据带来的额外空间开销，从而提高程序的执行效 率。

* 好处：

  指针传递方式虽然可以使得形参的改变对相应的实参有效，但如果在函数中反复利用指针进行间接访问，会使程序容易产生错误且难以阅读。

  如果以引用作为参数，则既可以使得对形参的任何操作都能改变相应的实参的数据，又使函数调用显得方便、自然。

* 引用返回：

  函数类型 &函数名(形参列表);

  **C++**引入引用的目的为了方便函数间数据的传递。 引用的另一个主要用途是用于返回引用的函数，即 引用返回。

  引用返回的主要目的是为了 

  1、将该函数用在赋值运算符的左边。2、利用引用返回多个值

  用引用返回一个函数值的最大好处是，在内存中不产生被返回值的副本。

### 指针型函数

* **返回值**是指针类型时，这个函数就是指针型函数
* 用途：有时需要从被调函数返 回一批数据到主调函数中，这时可以通过指针型函数来解决。

int *fun(int a,int b)：

* 注意：不要将非静态局部地址用作函数的返回值，因为非静态局部变量作用域仅限于本函数体内

### 函数指针

*  函数指针就是指向函数的指针。
* int (*p)(int,int);或者int fun(int,int); p=fun
* 作用：编写一个process函数，在调用它的时候，每次实现不同的功能。



<指针变量>= new<类型名> (<值>)；

```c++
int *p, i ;
p = new int(8) ;
i = *p;
```

<指针变量> = new <类型名> [<下标表达式>];

```
int *p;
p = new int[10] ;
...... p[i] ...... ; 
...... *(p+i) ...... ;
```

<指针变量> = new <类型名> [<表达式1>] [<表达式2>];

```
int (*p)[4] ;
p = new int[3][4] ;
...... p[i][j] ...... ; 
...... *(*(p+i)+j) ...... ;
```

 delete <指针名>;

 delete [N]指针名;delete [ ] p ; 



## 面向对象

由于类是一种数据类型，系统不会为其分配存储空间

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210423142155502.png" alt="image-20210423142155502" style="zoom:67%;" />

> 一般情况下:
>
> – 类的数据成员都定义为private
>
> – 把访问数据的成员函数定义为public

###  内联函数

放在类体内定义的函数被默认为内联函数

放在类体外定义的函数是一般函数，需要在函数名前使用类名加以而限定，如果要定义为内联函数则需在前面加上关键字inline。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210423143421687.png" alt="image-20210423143421687" style="zoom:67%;" />

### 对象存储空间：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210423150814920.png" alt="image-20210423150814920" style="zoom:67%;" />

###  “.**”**和“::**”**的不同:

“.**”**用于对象与成员之间。

“::**”**用于类与其成员之间。

### 访问：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210423152125275.png" alt="image-20210423152125275" style="zoom:67%;" />

私有访问(private)：私有成员的类外:不能通过对象访问类的私有成员，要访问类体内的私有数据成员，必须通过设置相应的公有函数，只能通过对象的公有成员函数来获取



### 构造函数：

初始化是指：定义对象的同时为其赋初值。

C++通过构造函数， 可自动完成对对象的初始化任务

构造函数(Constructor)是一种特殊的成员函数， 它是用来完成在声明对象的同时，对对象中的数 据成员进行初始化

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210423153107336.png" alt="image-20210423153107336" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210423153429999.png" alt="image-20210423153429999" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210423155526993.png" alt="image-20210423155526993" style="zoom:67%;" />

### 用参数初始化表

对数据成员初始化：说明如果数据成员是数组，则应在函数体内对其赋值，不能用参数初始化表的方法。

类名::构造函数名([参数表])[:成员初始化表]

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210423155553911.png" alt="image-20210423155553911" style="zoom:67%;" />



### 拷贝构造函数：

已知a是Clock类型的一个已初始化过的对象，若希望新定义

一个Clock的对象b，并用a来初始化，即: Clock b(a) ，则需用到拷贝构造函数。

![image-20210507130344482](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507130344482.png)

![image-20210507130706675](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507130706675.png)

![image-20210507130719583](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507130719583.png)

* 普通构造函数在建立对象时被调用，拷贝构造函数在用已有对象复制新对象时被调 用。

* 拷贝构造函数被调用的三种情况

  1. 当用类的一个对象去初始化该类的另一个对象时 系统自动调用拷贝构造函数实现拷贝赋值。

  2. 若函数的形参为类对象，调用函数时，系统处理成用实参对象初始化形参对象，自动调用拷贝构造函数

     ![image-20210507131017007](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507131017007.png)

  3. 如果函数的返回值是类的对象，函数执行完成返回主 调函数时，将使用return语句中的对象初始化一个临时 无名对象，传递给主调函数，

     

![image-20210507131112245](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507131112245.png)

隐含的拷贝构造函数:如果没有为类声明拷贝初始化构造函数，则编译器自己生成一个隐含的拷贝构造函数

* 一般情况下使用编译器默认的拷贝构造函数就可以了，不需要自己写拷贝构造函数。
* 特殊情况下，如需在拷贝的同时修改数据成员的值、或需实现深拷贝等，则需自己编写

> 实际上，当类中有动态申请的数据空间时，必须定义拷贝构造函数，否则出错

![image-20210507131644769](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507131644769.png)

###  构造函数的重载:

![:](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507131806309.png)

### 析构函数:

![image-20210507131907546](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507131907546.png)

![image-20210507131959964](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507131959964.png)

![image-20210507132020292](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507132020292.png)

![image-20210507132224823](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507132224823.png)

### 前向引用声明:

可以声明一个类而不定义它。这个声明，有时候被称为前向声明(forward declaration)。

不完全类型只能用于定义指向该类型的指针及引用， 或者用于声明(而不是定义)使用该类型作为形参类型或 返回类型的函数

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210507132332583.png" alt="image-20210507132332583" style="zoom:50%;" />



### 对象数组:

自动调用匹配的构造函数完成数组内每个对 象的初始化工作。

![image-20210507132514526](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507132514526.png)

### 对象指针:

![image-20210507132622591](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507132622591.png)

![image-20210507132657017](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507132657017.png)

### this指针:

告诉这个函数，是哪一个对象在调用它：

就是当前正在调用的该成员函数的对象的起始地址。



![image-20210507132854648](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507132854648.png)

![image-20210507133050138](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507133050138.png)

![image-20210507133954898](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507133954898.png)

![image-20210507134005119](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507134005119.png)

## 数据的共享与保护：

* 静态成员解决同类不同对象之间数据的共享 

  static int a；静态局部变量

* 友元实现不同类和对象之间数据的共享

![image-20210507134243770](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507134243770.png)

### 静态成员：

* 静态成员提供一种同类对象对数据的共享机制。

  – 静态成员:类属性，存储在静态区

  – 非静态成员:对象属性，存储在动态栈区

* 静态成员分为:

   – 静态数据成员 ：实现数据和共享

  – 静态成员函数：可以使用类调用   类::fun

* 很少使用全局变量：任何一个函数中都可以改变 全局变量的值，这样全局变量的安全性就得不到保证，，破坏了类的封装性，也做不到信息隐藏

  ![image-20210507135053399](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507135053399.png)

* C++通过静态数据成员来解决这个问题

  静态数据成员是类的属性，这个属性不属于类的任何对象，但所有的对象都可以访问和使用它

![image-20210507135301038](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507135301038.png)

![image-20210507135350699](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507135350699.png)

### 初始化：

初始化成类成员，而不是对象成员决定了：



![image-20210507140243814](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507140243814.png)

![image-20210507140533886](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507140533886.png)

![image-20210507142404522](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507142404522.png)

![image-20210507144030495](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507144030495.png)

![image-20210507144119148](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507144119148.png)

## 类的友元：

友元不是该类的成员函数，但是可以访问该类的私有成员

> **友元函数近似于普通的函数，它不带有this指针，因此 必须将对象名或对象的引用作为友元函数的参数，这 样才能访问到对象的成员。**

![image-20210507144235925](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507144235925.png)

### 普通函数：

![image-20210507144550301](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507144550301.png)

![image-20210507145618244](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507145618244.png)

### 类的成员函数作为友元函数：

![image-20210507151021642](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507151021642.png)

###  友元类：

![image-20210507152219431](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507152219431.png)

## 数据的共享与保护;

![image-20210507153156653](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507153156653.png)

1、常对象：

* 常对象只能调用其常成员函数，常成员函数是常对象唯一的对外接口。

* 常对象的数据成员是常数据成员，但成员函数 如不加const声明，编译系统将其作为非const成员函数。

* 常成员函数：

  * 可以访问常对象中的数据成员，但不允许修改常对象中数据成员的值。
* 常量成员函数只能调用类的其它常量成员函数，不能调用类的非常量成员函数。
  * 非常量成员函数不但可以调用非常量成员函数，也可以调用常量成员函数。
  * 

## 继承：

当通过派生类对象调用同名的成员时，系统将自动调用派生类中新定义的成员，而不是从基类继承的同名成员，基类的该成员在派生类中就被隐藏。



![image-20210507160748804](/Users/lishuo/Library/Application Support/typora-user-images/image-20210507160748804.png)

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210514142722383.png" alt="image-20210514142722383" style="zoom:67%;" />

私有成员永远只能被本类的成员函数访问，或者是友元访问

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210514143951034.png" alt="image-20210514143951034" style="zoom:67%;" />

如果是私有继承，私有基类的公有和保护成员在派生 类中的访问属性相当于派生类的私有成员，即派生类 的成员函数可以访问它们，而在派生类外不能访问它 们。

需要访问基类的成员时，需要在该派生类中再重新定义 一个公有函数

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210514145247044.png" alt="image-20210514145247044" style="zoom:67%;" />

projected：

* 对于其派生类来说，它与public成员的性质相同， 可使派生类访问效率高，方便继承，实现代码重用，同时对其他非继承类又实现了数据隐藏

* 在类外和私有成员一样，不能被访问 



类中protected成员的优点是:

– 既可以在本类中实现数据的隐藏(在类外不可被直接访问)，又可以将其类内直接访问特性传递到派生类中(在派生类中可直接访问)。

1、类外不能访问	2、 可以定义派生类函数访问

– 但private成员只能实现本类中的数据隐藏，而不能将其类内直接访问特性传递到派生类中。

1、类外不能访问	2、要想访问只能用基类的成员函数

### 初始化：

基类的构造函数和析构函数是不能被继承，需要在派生类中重新定义：可通过调用基类的构造函数完成初始化。



![image-20210514151726984](/Users/lishuo/Library/Application Support/typora-user-images/image-20210514151726984.png)

### 单继承

* 派生类构造函数的调用顺序:

  (**1**)调用基类的构造函数。– 调用顺序按照基类被继承时说明的顺序。

  (**2**)派生类构造函数体中的内容。

* 派生类构造函数的定义中可以省略对基类构造函 数的调用，其条件是在基类根本没有定义构造函数，即只有默认的构造函数。

* 当基类的构造函数使用一个或多个参数时，则派 生类必须定义构造函数，提供将参数传递给基类构造函数途径。

![image-20210514152023472](/Users/lishuo/Library/Application Support/typora-user-images/image-20210514152023472.png)

![image-20210514152059785](/Users/lishuo/Library/Application Support/typora-user-images/image-20210514152059785.png)

![image-20210514152208882](/Users/lishuo/Library/Application Support/typora-user-images/image-20210514152208882.png)

![image-20210514152220821](/Users/lishuo/Library/Application Support/typora-user-images/image-20210514152220821.png)



依次传入参数：一共五个：基类在前面三个，派生类两个

![image-20210514152329269](/Users/lishuo/Library/Application Support/typora-user-images/image-20210514152329269.png)

### 子对象:

![image-20210514154005448](/Users/lishuo/Library/Application Support/typora-user-images/image-20210514154005448.png)

![image-20210514154137390](/Users/lishuo/Library/Application Support/typora-user-images/image-20210514154137390.png)

### 多级派生时的构造函数：

不需要列出每一层派生类的构造函数，只 需写出其直接基类的构造函数即可。

![image-20210514155030535](/Users/lishuo/Library/Application Support/typora-user-images/image-20210514155030535.png)

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210514155059902.png" alt="image-20210514155059902" style="zoom:50%;" />

### 多继承：

多继承是指派生类具有多个基类，派生类 与每个基类之间的关系仍可看作是一个继承。

![image-20210514155444378](/Users/lishuo/Library/Application Support/typora-user-images/image-20210514155444378.png)

在多继承的情况下，多个基类构造函数的调用次序是按基类在被继承时所声明的次序从左到右依次调用，与它们在派生类的构造函数实现中的初始化列表出现的次序无关。



![image-20210514155609654](/Users/lishuo/Library/Application Support/typora-user-images/image-20210514155609654.png)

![image-20210514155802850](/Users/lishuo/Library/Application Support/typora-user-images/image-20210514155802850.png)

>
>
>派生类构造函数执行顺序是:
>
>– 多继承的构造函数的调用顺序与单继承的相同，也是遵循 先祖先(基类)，再客人(成员对象)，后自己(派生类) 的原则。
>
>– 在多个基类之间则严格按照派生定义时从左至右的顺序来 排列先后。与派生类构造函数中所定义的成员初始化列表的各项顺序无关。

### 虚基类



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210521141226993.png" alt="image-20210521141226993" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210521141607012.png" alt="image-20210521141607012" style="zoom:67%;" />

![image-20210521142315892](/Users/lishuo/Library/Application Support/typora-user-images/image-20210521142315892.png)

![image-20210521142330115](/Users/lishuo/Library/Application Support/typora-user-images/image-20210521142330115.png)

### 基类和派生类的转换：

![image-20210521142713578](/Users/lishuo/Library/Application Support/typora-user-images/image-20210521142713578.png)

![image-20210521143203372](/Users/lishuo/Library/Application Support/typora-user-images/image-20210521143203372.png)

![image-20210521143254502](/Users/lishuo/Library/Application Support/typora-user-images/image-20210521143254502.png)





## 多态性

![image-20210521145057178](/Users/lishuo/Library/Application Support/typora-user-images/image-20210521145057178.png)

![image-20210521145118325](/Users/lishuo/Library/Application Support/typora-user-images/image-20210521145118325.png)



### 虚函数重载

![image-20210521160032150](/Users/lishuo/Library/Application Support/typora-user-images/image-20210521160032150.png)

![image-20210521160126870](/Users/lishuo/Library/Application Support/typora-user-images/image-20210521160126870.png)

![image-20210521160314953](/Users/lishuo/Library/Application Support/typora-user-images/image-20210521160314953.png)

1、基类成员函数：virtual

2、函数中一定要是基类的指针或者引用

3、调用对应类函数的成员函数，如果没有virtual就会全都调用基类同名成员函数了

![image-20210521161124935](/Users/lishuo/Library/Application Support/typora-user-images/image-20210521161124935.png)

![image-20210521161103826](/Users/lishuo/Library/Application Support/typora-user-images/image-20210521161103826.png)

出现这个问题的原因是没在基类成员函数前面加virtual



![image-20210528134519043](/Users/lishuo/Library/Application Support/typora-user-images/image-20210528134519043.png)

![image-20210528134545088](/Users/lishuo/Library/Application Support/typora-user-images/image-20210528134545088.png)

构造函数里面，虚函数不支持动态多态性：

也就是说构造函数调用类本身虚函数，加不加虚函数都视为不加。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210528135445970.png" alt="image-20210528135445970" style="zoom: 67%;" />

析构函数：

![image-20210528141215333](/Users/lishuo/Library/Application Support/typora-user-images/image-20210528141215333.png)

![image-20210528141232483](/Users/lishuo/Library/Application Support/typora-user-images/image-20210528141232483.png)

![image-20210528141142229](/Users/lishuo/Library/Application Support/typora-user-images/image-20210528141142229.png)

![image-20210528141122083](/Users/lishuo/Library/Application Support/typora-user-images/image-20210528141122083.png)



抽象类没有对象：

![image-20210528141818637](/Users/lishuo/Library/Application Support/typora-user-images/image-20210528141818637.png)

![image-20210528141850160](/Users/lishuo/Library/Application Support/typora-user-images/image-20210528141850160.png)





STL:

![image-20210606155634611](/Users/lishuo/Library/Application Support/typora-user-images/image-20210606155634611.png)

