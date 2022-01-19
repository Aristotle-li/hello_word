## self参数  - __ init__ ()方法  super(Net, self).__init__()是什么



相信大家在很多场合特别是写神经网络的代码的时候都看到过下面的这种代码：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
123456789101112131415161718192021222324252627
```

大家可能刚开始会有点疑惑的是下面的这三行代码是干什么用的，要看懂这三行代码需要了解三个东西：

- self参数
- __ init__ ()方法
- super(Net, self).**init**()
   接下来就为大家逐一讲解一下。

```python
 def __init__(self):
 super(Net, self).__init__()
 def forward(self, x):
123
```

##### self参数

self指的是实例Instance本身，在Python类中规定，函数的第一个参数是实例对象本身，并且约定俗成，把其名字写为self，也就是说，**类中的方法的第一个参数一定要是self，而且不能省略**。
 我觉得关于self有三点是很重要的：

1. self指的是实例本身，而不是类
2. self可以用this替代，但是不要这么去写
3. 类的方法中的self不可以省略

首先第一点self指的是实例本身，而不是类

```python
class Person():
    def eat(self):
        print(self)

Bob=Person()
Bob.eat()
print(Person)
1234567
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708164236725.png)
 看输出的结果我们可以看到，self指的是实例

第二点self可以用this替代，但是不要这么去写，其实我理解self就相当于Java中的this，我们试着换一下

```python
class Person():
    def eat(this):
        print(this)

Bob=Person()
Bob.eat()
print(Person)
1234567
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708164449587.png)
 是没有报错的，但是大家还是按照规范用self
 第三点类的方法中的self不可以省略，看下面的代码，pycharm自动提示需要参数self。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/2020070816460784.png)

##### __ init__ ()方法

在python中创建类后，通常会创建一个 __ init__ ()方法，这个方法会在创建类的实例的时候**自动执行**。 __ init__ ()方法**必须包含**一个self参数，而且要是**第一个参数**。

比如下面例子中的代码，我们在实例化Bob这个对象的时候， __ init__ ()方法就已经自动执行了，但是如果不是 __ init__ ()方法，比如说eat（）方法，那肯定就只有调用才执行

```python
class Person():
    def __init__(self):
        print("是一个人")
    def eat(self):
        print("要吃饭" )
Bob=Person()
123456
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708154749245.png)
 再比如说下面的代码，如果 __ init__ ()方法中还需要传入另一个参数name，**但是我们在创建Bob的实例的时候没有传入name，那么程序就会报错，** 说我们少了一个__ init__ ()方法的参数，因为__ init__ ()方法是会在创建实例的过程中自动执行的，这个时候发现没有name参数，肯定就报错了

```python
class Person():
    def __init__(self,name):
        print("是一个人")
        self.name=name
    def eat(self):
        print("%s要吃饭" %self.name)

Bob=Person()
Bob.eat()
123456789
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708155210568.png)
 传入了Bob之后就不会了，而且eat方法也可以使用name这个参数。

```python
class Person():
    def __init__(self,name):
        print("是一个人")
        self.name=name
    def eat(self):
        print("%s要吃饭" %self.name)

Bob=Person('Bob')
Bob.eat()
123456789
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020070815550722.png)
 这样我们其实就比较清晰的知道什么东西需要在__ init__  ()方法中定义了，就是希望有一些操作是在创建实例的时候就有的时候，比如说下面的这个代码，其实就应该把money这个量定义在__ init__  ()方法中，这样就不需要在执行eat（）方法后再执行qian（）方法。或者说我们写神经网络的代码的时候，一些网络结构的设置，也最好放在__  init__ ()方法中。

```python
class Person():
    def __init__(self,name):
        print("是一个人")
        self.name=name
    def eat(self,money):
        print("%s要吃饭" %self.name)
        self.money=money
    def qian(self):
        print("花了%s元" %self.money)

Bob=Person('Bob')
Bob.eat(12)
Bob.qian()
12345678910111213
```

##### super(Net, self).**init**()

python中的super(Net, self).**init**()是指首先找到Net的父类（比如是类NNet），然后把类Net的对象self转换为类NNet的对象，然后“被转换”的类NNet对象调用自己的init函数，其实简单理解就是子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()的那些东西。

回过头来看看我们的我们最上面的代码，Net类继承nn.Module，super(Net, self).**init**()就是对继承自父类nn.Module的属性进行初始化。而且是用nn.Module的初始化方法来初始化继承的属性。

```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
123456
```

也就是说，子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化。
 当然，如果初始化的逻辑与父类的不同，不使用父类的方法，自己重新初始化也是可以的。比如：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
 
class Person(object):
    def __init__(self,name,gender,age):
        self.name = name
        self.gender = gender
        self.age = age
 
class Student(Person):
    def __init__(self,name,gender,age,school,score):
        #super(Student,self).__init__(name,gender,age)
        self.name = name.upper()  
        self.gender = gender.upper()
        self.school = school
        self.score = score
 
s = Student('Alice','female',18,'Middle school',87)
print s.school
print s.name
```