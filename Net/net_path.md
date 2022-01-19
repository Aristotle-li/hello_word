## pytorch图像分类篇：1.卷积神经网络基础与补充



最近在b站发现了一个非常好的 **计算机视觉 + pytorch** 的教程，相见恨晚，能让初学者少走很多弯路。
 因此决定按着up给的教程路线：图像分类→目标检测→…一步步学习用pytorch实现深度学习在cv上的应用，并做笔记整理和总结。

up主教程给出了pytorch和tensorflow两个版本的实现，我暂时只记录pytorch版本的笔记。

参考内容来自：

- up主的b站链接：https://space.bilibili.com/18161609/channel/index
- up主将代码和ppt都放在了github：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
- up主的CSDN博客：https://blog.csdn.net/qq_37541097/article/details/103482003

## 卷积神经网络

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705095708139.png#pic_center)

------

## CNN正向传播——以LeNet举例

第一节课主要是通过LeNet网络讲解了CNN中的卷积层、池化层和全连接层的正向传播过程。（包含卷积层的神经网络都可以称为卷积神经网络），由于是基础，就不再赘述。

关于CNN基础可以参考[CNN基础](https://blog.csdn.net/m0_37867091/article/details/105462334)
 关于LeNet网络可以参考[LeNet详解](https://blog.csdn.net/qq_42570457/article/details/81460807)
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705092431751.png#pic_center)

这里推荐一个LeNet的可视化网页，有助于理解（需翻墙）https://www.cs.ryerson.ca/~aharley/vis/conv/

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705095517634.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3ODY3MDkx,size_16,color_FFFFFF,t_70#pic_center)

------

## 补充1——反向传播中误差的计算：softmax/sigmoid

之前我自己总结过[神经网络的反向传播过程](https://blog.csdn.net/m0_37867091/article/details/104742705)，即根据误差的反向传播来更新神经网络的权值。

一般是用 **交叉熵损失** （**Cross Entropy Loss**）来计算误差

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705104913885.png?#pic_center)
 需要注意的是，**softmax的所有输出概率和为1**，例如进行图像分类时，输入一张图像，是猫的概率为0.3，是狗的概率为0.6，是马的概率为0.1。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020070510400555.png?#pic_center)

而sigmoid的输出则没有概率和为1的要求。

------

## 补充2——权重的更新

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705105538396.png?#pic_center)
 实际应用中，用**优化器**（**optimazer**）来优化梯度的求解过程，即让网络得到更快的收敛：

- SGD优化器（Stochastic Gradient Descent 随机梯度下降）   
  - 缺点：1. 易受样本噪声影响；2. 可能陷入局部最优解
  - 改进：SGD+Momentum优化器
- Agagrad优化器（自适应学习率）   
  - 缺点：学习率下降的太快，可能还没收敛就停止训练
  - 改进：RMSProp优化器：控制下降速度
- Adam优化器（自适应学习率）

详情可参考[机器学习：各种优化器Optimizer的总结与比较](https://blog.csdn.net/weixin_40170902/article/details/80092628)
 附图：几种优化器下降的可视化比较
 ![在这里插入图片描述](https://img-blog.csdn.net/20180426130002689#pic_center)













<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210110152654576.png" alt="image-20210110152654576" style="zoom:50%;" />







<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210112153817103.png" alt="image-20210112153817103" style="zoom:50%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210112154829464.png" alt="image-20210112154829464" style="zoom:50%;" />

但在实际应用中，有时会出现N为非整数的情况（例如在alexnet，googlenet网络的第一层输出），再例如输入的矩阵 **H=W=5，**卷积核的**F=2，S=2，Padding=1**。经计算我们得到的**N =（5 - 2 + 2\*1）/ 2 +1 = 3.5** 此时在Pytorch中是如何处理呢，先直接告诉你结论：**在卷积过程中会直接将最后一行以及最后一列给忽略掉，以保证N为整数，此时N = （5 - 2 + 2\*1 - 1）/ 2 + 1 = 3**，接下来我们来看个简单的实例：

```python
import torch
import torch.nn as nn
im = torch.randn(1,1,5,5)
c = nn.Conv2d(1,1,kernel_size=2,stride=2,padding=1)
output = c(im)
print(im)
print(output)
print(list(c.parameters()))
```

## 架构的思路：

### model：

```python
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x
```

### train：

1、载入数据集，并进行预处理，

<img src="https://upload-images.jianshu.io/upload_images/11963149-06b0294043403bf3.png?imageMogr2/auto-orient/strip|imageView2/2/w/1078" alt="img" style="zoom:50%;" />

<img src="https://upload-images.jianshu.io/upload_images/11963149-9916634fa3634524.png?imageMogr2/auto-orient/strip|imageView2/2/w/1085" alt="img" style="zoom: 50%;" />

用到的函数有：

```python
Transforms: transforms.ToTensor() 、transform.Normalize

train_set = torchvision.dataset.---(root='./data',train = True ,download = False,transform = transform)
train_loader = torch.utils.data.DataLoader(train_set,batch_size = 50，shuffle = True , num_worker = 0)
 
```

2、定义一个迭代器 iter

```python
val_data_iter = iter(val_loader)
val_image, val_label = val_data_iter.next()
```

3、实例化模型、定义损失函数、定义优化器：

```python
net = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
```

开始训练：

```python
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        #  enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()    # 反向传播
        optimizer.step()   # 参数更新
        # print statistics
        running_loss += loss.item()
```

计算准确率：

```python
if step % 500 == 499:    # print every 500 mini-batches
    with torch.no_grad():
        outputs = net(val_image)  # [batch, 10]
        predict_y = torch.max(outputs, dim=1)[1]
        accuracy = (predict_y == val_label).sum().item() / val_label.size(0)   #

        print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, step + 1, running_loss / 500, accuracy))
        running_loss = 0.0
```

保存模型：

```python
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)
```

### test

实例化模型，导入模型参数

导入数据，预处理

进行预测



# Pytorch 函数学习

## iterm函数在nn中的用法，得到的是一个bool型的tensor

```python
accuracy = (predict_y == val_label).sum().item() / val_label.size(0) 
```

```python
# -*- coding: utf-8 -*-import torch

import numpy as np 

data1 = np.array([
    [1,2,3],
    [2,3,4]
])
data1_torch = torch.from_numpy(data1)

data2 = np.array([
    [1,2,3],
    [4,5,6]
])
data2_torch = torch.from_numpy(data2)

p = (data1_torch == data2_torch) 
print p
print type(p)

d1 = p.sum() 
print d1 
print type(d1)

d2 = d1.item() 
print d2 
print type(d2)
```



## torch.manual_seed(1)  

reproducible 可以复现的

在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。如果设置初始化，则每次初始化都是固定的。
if args.seed is not None:
　　random.seed(args.seed) #
　　torch.manual_seed(args.seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
　　 torch.cuda.manual_seed(args.seed) #为当前GPU设置随机种子；
　　 cudnn.deterministic = True

\#如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。

## numpy.squeeze(a,axis = None)

作用：从数组的形状中删除单维度条目，即把shape中为1的维度去掉

应用：在机器学习和深度学习中，通常算法的结果是可以表示向量的数组（即包含两对或以上的方括号形式[[]]），如果直接利用这个数组进行画图可能显示界面为空（见后面的示例）。我们可以利用squeeze（）函数将表示向量的数组转换为秩为1的数组，这样利用matplotlib库函数画图时，就可以正常的显示结果了。

用法：1）a表示输入的数组； 2）axis用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错； 3）axis的取值可为None 或 int 或 tuple of ints, 可选。若axis为空，则删除所有单维度的条目； 4）返回值：数组 5) 不会修改原数组；

## iter

```python
# 普通循环 for x in list
numbers = [1, 2, 3,]
for n in numbers:
  print(n) # 1,2,3

# for循环实际干的事情
# iter输入一个可迭代对象list，返回迭代器
# next方法取数据
my_iterator = iter(numbers)
next(my_iterator) # 1
next(my_iterator) # 2
next(my_iterator) # 3
next(my_iterator) # StopIteration exception

# 迭代器循环 for x in iterator
for i,n in enumerate(numbers):
  print(i,n) # 0,1 / 1,3 / 2,3

# 生成器循环 for x in generator
for i in range(3):
  print(i) # 0,1,2

三种循环模式常用函数
for x in container 方法:
list, deque, …
set, frozensets, …
dict, defaultdict, OrderedDict, Counter, …
tuple, namedtuple, …
str
for x in iterator 方法:
enumerate() # 加上list的index
sorted() # 排序list
reversed() # 倒序list
zip() # 合并list
for x in generator 方法：
range()
map()
filter()
reduce()
[x for x in list(...)]

```

python中循环的三种模式：

- for x in container 可迭代对象
- for x in iterator 迭代器
- for x in generator 生成器

pytorch中的数据加载模块 Dataloader，使用生成器来返回数据的索引，使用迭代器来返回需要的张量数据，可以在大量数据情况下，实现小批量循环迭代式的读取，避免了内存不足问题。

## Pytorch torchvision.utils.make_grid()用法

make_grid的作用是将若干幅图像拼成一幅图像。其中padding的作用就是子图像与子图像之间的pad有多宽。nrow是一行放入八个图片。

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210113091638681.png" alt="image-20210113091638681" style="zoom:50%;" />

## [pytorch的函数中的dilation参数的作用](https://www.cnblogs.com/wanghui-garcia/p/10775367.html)

如果我们设置的dilation=0的话，效果如图：

蓝色为输入，绿色为输出，可见卷积核为3*3的卷积核

<img src="https://img2018.cnblogs.com/blog/1446032/201904/1446032-20190426165827108-2055771203.png" alt="img" style="zoom:50%;" />

 

如果我们设置的是dilation=1，那么效果如图：

蓝色为输入，绿色为输出，卷积核仍为3*3，但是这里卷积核点与输入之间距离为1的值相乘来得到输出

<img src="https://img2018.cnblogs.com/blog/1446032/201904/1446032-20190426165728998-932168491.png" alt="img" style="zoom:50%;" />

 

好处：

这样单次计算时覆盖的面积（即感受域）由dilation=0时的3*3=9变为了dilation=1时的5*5=25

在增加了感受域的同时却没有增加计算量，保留了更多的细节信息，对图像还原的精度有明显的提升

## format函数

```python
name = "Sandy"
gender = "女"
age = 18
print("姓名：%s,性别：%s年龄：%d" % (name, gender, age))
print("姓名：{},性别：{}年龄：{}".format(name, gender, age))
# 有了数字编号可以反复调用
print("姓名：{0},性别：{1}年龄：{2}学生姓名：{0}".format(name, gender, age))
# 标识名称更容易读懂
print("姓名：{name},性别：{gender}年龄：{age}学生姓名：{name}".format(name=name, gender=gender, age=age))

print("姓名：{:10}".format(name))#默认左对齐
print("姓名：{:<10}".format(name))#标识左对齐
print("姓名：{:>10}".format(name))#右对齐
print("姓名：{:^10}".format(name))#中间对齐
print("{:.2f}".format(3.1415926))#保留2位有效数字
print("{:10.2f}".format(3.1415926))#保留2位有效数字默认右对齐
print("{:>10.2f}".format(3.1415926))#保留2位有效数字指明右对齐
print("{:<10.2f}".format(3.1415926))#保留2位有效数字指明左对齐
print("{:^10.2f}".format(3.1415926))#保留2位有效数字中间对齐

num01,num02=200,300
print("十六进制打印：{0:x}{1:x}".format(num01,num02))
print("八进制打印：{0:o}{1:o}".format(num01,num02))
print("二进制打印：{0:b}{1:b}".format(num01,num02))
print("{0:c}".format(76))#可以把编码转换为特定的字符，参考ASCll
print("{:e}".format(123456.77544))#默认小数点后面保留6位
print("{:0.2e}".format(123456.77544))#小数点后面保留2位
print("{:g}".format(123456.77544))#保留6位
print("{:g}".format(123456789.77544))#超过6位用科学计数法表示
print("{:%}".format(34))#默认小数点后面保留6位
print("{:0.2%}".format(34))
print("{:,}".format(1234567890))
```



## **assert是啥**

python内置的断言语句(assert statement)。语法如下：

```
assert <condition>
assert <condition>,<error message>
123
```

当`condition`为True时，顺序执行下一语句；当`condition`为False时，程序会终止并抛出`AssertionError`，如果assert语句有error message，则会在`AssertionError`后显示该信息。

**assert的使用场景**

- 检查参数的类型和取值。
  相比`raise Error()`,代码更简洁

- debug

  

## **pytorch之ImageFolder**

torchvision已经预先实现了常用的Dataset，包括前面使用过的CIFAR-10，以及ImageNet、COCO、MNIST、LSUN等数据集，可通过诸如torchvision.datasets.CIFAR10来调用。在这里介绍一个会经常使用到的Dataset——ImageFolder。

ImageFolder假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名，其构造函数如下：

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210113121232040.png" alt="image-20210113121232040" style="zoom:50%;" />

```python
ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
```

**它主要有四个参数：**

root：在root指定的路径下寻找图片

transform：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象

target_transform：对label的转换

loader：给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象

label是按照文件夹名顺序排序后存成字典，即{类名:类序号(从0开始)}，一般来说最好直接将文件夹命名为从0开始的数字，这样会和ImageFolder实际的label一致，如果不是这种命名规范，建议看看self.class_to_idx属性以了解label和文件夹名的映射关系。

图片结构如下所示：

![img](https://img.jbzj.com/file_images/article/201912/20200106173909.jpg)

```python
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder


dataset = ImageFolder('data/dogcat_2/')

# cat文件夹的图片对应label 0，dog对应1
print(dataset.class_to_idx)

# 所有图片的路径和对应的label
print(dataset.imgs)

# 没有任何的transform，所以返回的还是PIL Image对象
#print(dataset[0][1])# 第一维是第几张图，第二维为1返回label
#print(dataset[0][0]) # 为0返回图片数据
plt.imshow(dataset[0][0])
plt.axis('off')
plt.show()
```

**加上transform**

```python
normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
transform = T.Compose([
     T.RandomResizedCrop(224),
     T.RandomHorizontalFlip(),
     T.ToTensor(),
     normalize,
])
dataset = ImageFolder('data1/dogcat_2/', transform=transform)

# 深度学习中图片数据一般保存成CxHxW，即通道数x图片高x图片宽
#print(dataset[0][0].size())

to_img = T.ToPILImage()
# 0.2和0.4是标准差和均值的近似
a=to_img(dataset[0][0]*0.2+0.4)
plt.imshow(a)
plt.axis('off')
plt.show()
```

## json.dumps()

json.dumps将一个Python数据结构转换为JSON

```python
import json
data = {
    'name' : 'myname',
    'age' : 100,
}
json_str = json.dumps(data)
123456
```

#### json库的一些用法

| 方法         | 作用                                         |
| ------------ | -------------------------------------------- |
| json.dumps() | 将python对象编码成Json字符串                 |
| json.loads() | 将Json字符串解码成python对象                 |
| json.dump()  | 将python中的对象转化成json储存到文件中       |
| json.load()  | 将文件中的json的格式转化成python对象提取出来 |

#### json.dump()和json.dumps()的区别

json.dumps() 是把python对象转换成json对象的一个过程，生成的是字符串。
json.dump() 是把python对象转换成json对象生成一个fp的文件流，和文件相关。

#### json参数

```python
json.dumps(obj, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, encoding="utf-8", default=None, sort_keys=False, **kw)
```

- obj:转化成json的对象。
- sort_keys =True:是告诉编码器按照字典排序(a到z)输出。如果是字典类型的python对象，就把关键字按照字典排序。
- **indent**:参数根据数据格式缩进显示，读起来更加清晰。
- separators:是分隔符的意思，参数意思分别为不同dict项之间的分隔符和dict项内key和value之间的分隔符，把：和，后面的空格都除去了。

```python
import json

x = {'name':'你猜','age':19,'city':'四川'}
#用dumps将python编码成json字符串
y = json.dumps(x)
print(y)
i = json.dumps(x,separators=(',',':')，ensure_ascii=False)  
# 默认输出ASIC码，改成False可以输出中文)
print(i)
# 输出结果
{"name": "\u4f60\u731c", "age": 19, "city": "\u56db\u5ddd"}
{"name":"你猜","age":19,"city":"四川"}
```

- skipkeys：默认值是False，如果dict的keys内的数据不是python的基本类型(str,unicode,int,long,float,bool,None)，设置为False时，就会报TypeError的错误。此时设置成True，则会跳过这类key 。
- ensure_ascii=True：默认输出ASCLL码，如果把这个该成False,就可以输出中文。
- check_circular：如果check_circular为false，则跳过对容器类型的循环引用检查，循环引用将导致溢出错误(或更糟的情况)。
- allow_nan：如果allow_nan为假，则ValueError将序列化超出范围的浮点值(nan、inf、-inf)，严格遵守JSON规范，而不是使用JavaScript等价值(nan、Infinity、-Infinity)。
- default：default(obj)是一个函数，它应该返回一个可序列化的obj版本或引发类型错误。默认值只会引发类型错误。

**从结构上看，所有的数据（data）最终都可以分解成三种类型**：

> 第一种类型是**标量**（scalar），也就是一个单独的字符串（string）或数字（numbers），比如"北京"这个单独的词。
>
> 第二种类型是**序列**（sequence），也就是若干个相关的数据按照一定顺序并列在一起，又叫做数组（array）或列表（List），比如"北京，上海"。
>
> 第三种类型是**映射**（mapping），也就是一个名/值对（Name/value），即数据有一个名称，还有一个与之相对应的值，这又称作散列（hash）或字典（dictionary），比如"首都：北京"。

## with as

```python
with open('output.txt', 'w') as f:    
f.write('Hi there!') 
```

> The above with statement will automatically close the file after the nested block of code.  

#### 使用with as的原理是：

with 语句包裹起来的代码块，在执行语句体之前会调用上下文管理器的 __enter__() 方法，执行完语句体之后会执行 __exit__() 方法。

file对象就是一个context manager！，但是，使用with as还是有可能抛出异常，打开文件操作正确的写法是：

```python
try:
    with open( "a.txt" ) as f :
        do something
except xxxError:
    do something about exception
```

## perf_counter 进度条实例：

```python
t1 = time.perf_counter()   # 计时开始的时间
   # print train process
rate = (step + 1) / len(train_loader)
a = "*" * int(rate * 50)
b = "." * int((1 - rate) * 50)
print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
print()
print(time.perf_counter()-t1)  # 输出训练的时间
# \r用来在每次输出完成后，将光标移至行首，这样保证进度条始终在同一行输出，即在一行不断刷新的效果；{:^3.0f}，输出格式为居中，占3位，小数点后0位，浮点型数，对应输出的数为c；{}，对应输出的数为a；{}，对应输出的数为b；{:.2f}，输出有两位小数的浮点数，对应输出的数为dur；end=''，用来保证不换行，不加这句默认换行。
```



## Python-非关键字参数和关键字参数(*args **kw)



Python的函数具有非常灵活的参数形态，既可以实现简单的调用，又可以传入非常复杂的参数。

可变参数和关键字参数的语法：

- *args是可变参数，args接收的是一个tuple；
- **kw是关键字参数，kw接收的是一个dict。

使用*args和**kw是Python的习惯写法，当然也可以用其他参数名，但最好使用习惯用法。

### 一、可变参数*args

```python
定义：可变参数就是传入的参数个数是可变的，可以是0个，1个，2个，……很多个。
作用：就是可以一次给函数传很多的参数
特征：*args
```

我们以数学题为例子，给定一组数字a，b…z，请计算sum = a * a + b * b + …+z * z

要定义出这个函数，我们必须确定输入的参数。由于参数个数不确定，我们首先想到可以把a,b,…,z作为一个list或tuple传进来，这样，函数可以定义如下：

```python
def cout(numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
```

定义可变参数和定义一个list或tuple参数相比，仅仅在参数前面加了一个*号。在函数内部，参数numbers接收到的是一个tuple，因此，函数代码完全不变。但是，调用该函数时，可以传入任意个参数，包括0个参数：

```
>>> cout([1, 2, 3])
14
>>> cout((1, 3, 5, 7))
84
1234
```

如果利用可变参数，调用函数的方式可以简化成这样：

```
>>> cout(1, 2, 3)
14
>>> cout(1, 3, 5, 7)
84
1234
```

所以，我们把函数的参数改为可变参数：

```python
def cout(*numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
```

定义可变参数和定义一个list或tuple参数相比，仅仅在参数前面加了一个*号。在函数内部，参数numbers接收到的是一个tuple，因此，函数代码完全不变。但是，调用该函数时，可以传入任意个参数，包括0个参数：

```python
>>> cout(1, 2)
5
>>> cout()
0
如果已经有一个list或者tuple，要调用一个可变参数怎么办？可以这样做：
>>> nums = [1, 2, 3]
>>> cout(nums[0], nums[1], nums[2])
14
这种写法当然是可行的，问题是太繁琐，所以Python允许你在list或tuple前面加一个*号，把list或tuple的元素变成可变参数传进去：
>>> nums = [1, 2, 3]
>>> calc(*nums)
```

*nums表示把nums这个list的所有元素作为可变参数传进去。这种写法相当有用，而且很常见。

### 二、关键字参数**kw

```python
定义：关键字参数允许你传入0个或任意个含参数名的参数，这些关键字参数在函数内部自动组装为一个dict。在调用函数时，可以只传入必选参数。
作用：扩展函数的功能
特征：**kw
123
```

**请看示例：**

```python
def person(name, age, **kw):
    print('name:', name, 'age:', age, 'other:', kw)
12
```

函数person除了必选参数name和age外，还接受关键字参数kw。在调用该函数时，可以只传入必选参数：
\>>> person(‘Michael’, 30)
name: Michael age: 30 other: {}

```python
也可以传入任意个数的关键字参数：
>>> person('Bob', 35, city='Beijing')
name: Bob age: 35 other: {'city': 'Beijing'}

>>> person('Adam', 45, gender='M', job='Engineer')
name: Adam age: 45 other: {'gender': 'M', 'job': 'Engineer'}
```

关键字参数有什么用？它可以扩展函数的功能。比如，在person函数里，我们保证能接收到name和age这两个参数，但是，如果调用者愿意提供更多的参数，我们也能收到。试想你正在做一个用户注册的功能，除了用户名和年龄是必填项外，其他都是可选项，利用关键字参数来定义这个函数就能满足注册的需求。

和可变参数类似，也可以先组装出一个dict，然后，把该dict转换为关键字参数传进去：

```python
>>> extra = {'city': 'Beijing', 'job': 'Engineer'}
>>> person('Jack', 24, city=extra['city'], job=extra['job'])
name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}

当然，上面复杂的调用可以用简化的写法：
>>> extra = {'city': 'Beijing', 'job': 'Engineer'}
>>> person('Jack', 24, **extra)
name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}
```

**extra表示把extra这个dict的所有key-value用关键字参数传入到函数的**kw参数，kw将获得一个dict，注意kw获得的dict是extra的一份拷贝，对kw的改动不会影响到函数外的extra。



## super(Net, self).__init__()

```python
class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
    # 定义槽函数
    def hello(self):
        self.textEdit.setText("hello world")
```

针对super(mywindow, self).__init__()理解不了。

* `子类定义与父类的同名函数，覆盖父类`

* 查询了一下资料瞬间明白了。

```python
python中的super(Net, self).__init__()
首先找到Net的父类（比如是类NNet），然后把类Net的对象self转换为类NNet的对象，然后“被转换”的类NNet对象调用自己的init函数
```

## join

```
print(' '.join(stuff))   # 连接字符串数组。将字符串、元组、列表中的元素以指定的字符(分隔符)连接生成一个新的字符串
```

## zip

``` python
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]

 #将zip 封装成函数
>>> x
[1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> group_adjacent = lambda a, k: zip(*[a[i::k] for i in range(k)])
>>> group_adjacent(x,3)
[(1, 2, 3), (4, 5, 6), (7, 8, 9)]
>>> group_adjacent(x,2)
[(1, 2), (3, 4), (5, 6), (7, 8)]
>>> group_adjacent(x,1)
[(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]


```

## With

```python
with open('/etc/passwd') as f:
    for line in f:
        print(line)
```

with context_expr() as var:

　　　　doSomething()

1. 当with语句执行时，便执行上下文表达式（context_expr）(一般为某个方法)来获得一个上下文管理器对象，上下文管理器的职责是提供一个上下文对象，用于在with语句块中处理细节：
2. 一旦获得了上下文对象，就会调用它的__enter__()方法，将完成with语句块执行前的所有准备工作，如果with语句后面跟了as语句，则用__enter__()方法的返回值来赋值；
3. 当with语句块结束时，无论是正常结束，还是由于异常，都会调用上下文对象的__exit__()方法，__exit__()方法有3个参数，如果with语句正常结束，三个参数全部都是  None；如果发生异常，三个参数的值分别等于调用sys.exc_info()函数返回的三个值：类型（异常类）、值（异常实例）和跟踪记录（traceback），相应的跟踪记录对象。
4. 因为上下文管理器主要作用于共享资源，__enter__()和__exit__()方法基本是完成的是分配和释放资源的低层次工作，比如：数据库连接、锁分配、信号量加/减、状态管理、文件打开/关闭、异常处理等。

## a[::-1]

这里的s表示步进，缺省为1.（-1时即翻转读取）
所以a[i:j:1]相当于a[i:j]
当s<0时，i缺省时，默认为-1. j缺省时，默认为-len(a)-1
所以a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍。所以你看到一个倒序的东东。

```
print(a[::-1])     取从后向前（相反）的元素
结果：[ 5 4 3 2 1 ]
 
print(a[2::-1])     取从下标为2的元素**翻转读取**
结果：[ 3 2 1 ]
```



## NN中特殊层的反向传播



对于mean pooling，真的是好简单：假设pooling的窗大小是2x2, 在forward的时候啊，就是在前面卷积完的输出上依次不重合的取2x2的窗平均，得到一个值就是当前mean pooling之后的值。backward的时候，把一个值分成四等分放到前面2x2的格子里面就好了。如下

forward: [1 3; 2 2] -> [2]
backward: [2] -> [0.5 0.5; 0.5 0.5]

max pooling就稍微复杂一点，forward的时候你只需要把2x2窗子里面那个最大的拿走就好了，backward的时候你要把当前的值放到之前那个最大的位置，其他的三个位置都弄成0。如下

forward: [1 3; 2 2] -> 3
backward: [3] -> [0 3; 0 0]

 **Pooling池化操作的反向梯度传播**

 CNN网络中另外一个不可导的环节就是Pooling池化操作，因为Pooling操作使得feature map的尺寸变化，假如做2×2的池化，假设那么第l+1层的feature map有16个梯度，那么第l层就会有64个梯度，这使得梯度无法对位的进行传播下去。其实解决这个问题的思想也很简单，就是把1个像素的梯度传递给4个像素，但是 **需要保证传递的loss（或者梯度）总和不变** 。根据这条原则，mean pooling和max pooling的反向传播也是不同的。

 **1、mean pooling**

 mean pooling的前向传播就是把一个patch中的值求取平均来做pooling，那么反向传播的过程也就是把某个元素的梯度等分为n份分配给前一层，这样就保证池化前后的梯度（残差）之和保持不变，还是比较理解的，图示如下 ：

  ![img](https://img-blog.csdn.net/20170615205352655?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMjExOTAwODE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


 mean pooling比较容易让人理解错的地方就是会简单的认为直接把梯度复制N遍之后直接反向传播回去，但是这样会造成loss之和变为原来的N倍，网络是会产生梯度爆炸的。

 **2、max pooling**

 max pooling也要满足梯度之和不变的原则 ，max pooling的前向传播是把patch中最大的值传递给后一层，而其他像素的值直接被舍弃掉。那么反向传播也就是 把梯度直接传给前一层某一个像素，而其他像素不接受梯度，也就是为0 。所以max pooling操作和mean pooling操作不同点在于需要记录下池化操作时到底哪个像素的值是最大，也就是max id ，这个变量就是记录最大值所在位置的，因为在反向传播中要用到，那么假设前向传播和反向传播的过程就如下图所示 ：

 ![img](https://img-blog.csdn.net/20170615211413093?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMjExOTAwODE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 

 max pooling也要满足梯度之和不变的原则，max pooling的前向传播是把patch中最大的值传递给后一层，而其他像素的值直接被舍弃掉。那么反向传播也就是把梯度直接传给前一层某一个像素，而其他像素不接受梯度，也就是为0。所以max pooling操作和mean pooling操作不同点在于需要记录下池化操作时到底哪个像素的值是最大，也就是max id，这个可以看caffe源码的pooling_layer.cpp，下面是caffe框架max pooling部分的源码


```python
// If max pooling, we will initialize the vector index part.
if (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX && top.size() == 1)
{
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,pooled_width_);
  }
```

![image-20210330172049804](/Users/lishuo/Library/Application Support/typora-user-images/image-20210330172049804.png)

![image-20210330172220960](/Users/lishuo/Library/Application Support/typora-user-images/image-20210330172220960.png)





### torch.max()

```
output = torch.max(input, dim)
```

# 1. torch.max(input, dim) 函数

```
output = torch.max(input, dim)
```

> 输入
>
> - `input`是softmax函数输出的一个`tensor`
> - `dim`是max函数索引的维度`0/1`，`0`是每列的最大值，`1`是每行的最大值

> 输出
>
> - 函数会返回两个`tensor`，第一个`tensor`是每行的最大值；第二个`tensor`是每行最大值的索引。

在多分类任务中我们并不需要知道各类别的预测概率，所以返回值的第一个`tensor`对分类任务没有帮助，而第二个`tensor`包含了预测最大概率的索引，所以在实际使用中我们仅获取第二个`tensor`即可。

下面通过一个实例可以更容易理解这个函数的用法。



```
import torch
a = torch.tensor([[1,5,62,54], [2,6,2,6], [2,65,2,6]])
print(a)
```

输出：

```python
tensor([[ 1,  5, 62, 54],
        [ 2,  6,  2,  6],
        [ 2, 65,  2,  6]])
```

索引每行的最大值：

```python
torch.max(a, 1)
```

输出：

```
torch.return_types.max(
values=tensor([62,  6, 65]),
indices=tensor([2, 3, 1]))
```

在计算准确率时第一个tensor `values`是不需要的，所以我们只需提取第二个tensor，并将tensor格式的数据转换成array格式。

```
torch.max(a, 1)[1].numpy()
```

输出：

```cpp
array([2, 3, 1], dtype=int64)
```

这样，我们就可以与标签值进行比对，计算模型预测准确率。

*注：在有的地方我们会看到`torch.max(a, 1).data.numpy()`的写法，这是因为在早期的pytorch的版本中，variable变量和tenosr是不一样的数据格式，variable可以进行反向传播，tensor不可以，需要将variable转变成tensor再转变成numpy。现在的版本已经将variable和tenosr合并，所以只用`torch.max(a,1).numpy()`就可以了。

### 2.准确率的计算

```
pred_y = torch.max(predict, 1)[1].numpy()
label_y = torch.max(label, 1)[1].data.numpy()
accuracy = (pred_y == label_y).sum() / len(label_y)
```

`predict` - softmax函数输出
 `label` - 样本标签，这里假设它是one-hot编码