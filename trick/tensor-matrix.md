## pytorch中对于tensor的一些骚操作

### 1. (a[:, None] * b[None, :]).view(-1)

这是在计算anchor的k值出现的操作，其中a为一维向量[a1, a2, ..., an]，b也为一维向量[b1, b2, ...,  bm]。a[:, None]目的是增加一个新维度shape从[n] -> [n, 1]，同理b[None, :]的shape从[m]  -> [1, m]。接着两个矩阵相乘shape从[n, 1] * [1, m] -> [n,  m] ,最后通过view（-1）展开成一维[n, m] ->  [nm]。通俗的说，假设面积尺度有n个，高宽比例因子有m个，那么就能够组合成n * m个不同的矩形框（anchor）。

![img](https://img-blog.csdnimg.cn/20191212102638556.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70)

 

### 2. (a.view(-1, 1, 4) + b.view(1, -1, 4)).reshape(-1, 4)

这是将所有anchor绘制到原图上的出现的操作，a为二维向量shape为[n, 4]，b也为二维向量shape为[m,  4]，view和reshape的功能是类似的，a.view(-1, 1, 4)操作后shape[m, 4] -> [m, 1,  4]，b.view(1, -1, 4)操作后shape[n, 4] -> [1, n,  4]，接下来的相加就是一波骚操作了，按常理来讲两个维度不同的矩阵是不能相加的，但torch的tensor是可以的，a.view(-1, 1,  4)+b.view(1, -1, 4)后的shape是[m, n, 4]，如下图所示（我个人理解的相加过程），假设shape代表维度(x, y, z)，m=2, n=3, 首先我们对a进行view后得到我们图中的Tensor(a) shape[2, 1,  4]可以理解x方向两个单位，y方向一个单位，z方向4个单位。接着我们对b进行view得到我们图中的Tensor(b) shape[1, 3,  4]可以理解x方向一个单位，y方向三个单位，z方向4个单位。为了Tensor(a)和Tensor(b)能够相加，Tensor(a)在y方向复制了m=3次，Tensor(b)在x方向复制了n=2次，这样就得到了相同维度的两个Tensor shape[2, 3, 4]，这样就可以愉快的相加了。最后在reshape一下得到tensor的shape为[m*n, 4]。

![img](https://img-blog.csdnimg.cn/20191212205442182.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/20191212155809514.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70)

 

## 记一次np.zeros()的使用经验

```python
data_size =5
data_label = np.zeros(2 * data_size)
data_label[0:data_size] = 1
print(data_label)
print(np.shape(data_label))

```

结果：

![image-20210327173933675](/Users/lishuo/Library/Application Support/typora-user-images/image-20210327173933675.png)

```python
data_size =5
data_label = np.zeros((2 * data_size,1))  #  这里是两个括号
data_label[0:data_size,:] = 1   # 而这里是对0-5行，所有列赋值为1
print(data_label)
print(np.shape(data_label))
'''''''''''''''''''
data_size =3
data_label = np.zeros((2 * data_size,3))   #生成6*3的矩阵，同样需要两个括号
data_label[0:data_size,2] = 1    #  这里是对第三列前三行赋值1.
print(data_label)
print(np.shape(data_label))
```

![image-20210327174457353](/Users/lishuo/Library/Application Support/typora-user-images/image-20210327174457353.png)![image-20210327174531193](/Users/lishuo/Library/Application Support/typora-user-images/image-20210327174531193.png)
$$
\mu _1=\left[ \begin{matrix}
	1&		0\\
	0&		1\\
\end{matrix} \right] 
\\
\sigma _1=\left[ \begin{matrix}
	0.6&		0\\
	0&		0.8\\
\end{matrix} \right] 
\\
\mu _1=\left[ \begin{matrix}
	-1&		0\\
	0&		-1\\
\end{matrix} \right] 
\\
\sigma _1=\left[ \begin{matrix}
	0.3&		0\\
	0&		0.5\\
\end{matrix} \right] 
$$


## python 列表，数组，矩阵之间转换

```python

# -*- coding: utf-8 -*-
from numpy import *

a1 =[[1,2,3],[4,5,6]] #列表
print('a1 :',a1)
#('a1 :', [[1, 2, 3], [4, 5, 6]])

a2 = array(a1)   #列表 -----> 数组
print('a2 :',a2)
#('a2 :', array([[1, 2, 3],[4, 5, 6]]))

a3 = mat(a1)      #列表 ----> 矩阵
print('a3 :',a3)
#('a3 :', matrix([[1, 2, 3],[4, 5, 6]]))

a4 = a3.tolist()   #矩阵 ---> 列表
print('a4 :',a4)
#('a4 :', [[1, 2, 3], [4, 5, 6]])

print(a1 == a4)
#True

a5 = a2.tolist()   #数组 ---> 列表

print('a5 :',a5)
#('a5 :', [[1, 2, 3], [4, 5, 6]])
print(a5 == a1)
#True

a6 = mat(a2)   #数组 ---> 矩阵
print('a6 :',a6)
#('a6 :', matrix([[1, 2, 3],[4, 5, 6]]))

print(a6 == a3)
#[[ True  True  True][ True  True  True]]

a7 = array(a3)  #矩阵 ---> 数组
print('a7 :',a7)
#('a7 :', array([[1, 2, 3],[4, 5, 6]]))
print(a7 == a2)
#[[ True  True  True][ True  True  True]]

###################################################################
a1 =[1,2,3,4,5,6] #列表
print('a1 :',a1)
#('a1 :', [1, 2, 3, 4, 5, 6])

a2 = array(a1)   #列表 -----> 数组
print('a2 :',a2)
#('a2 :', array([1, 2, 3, 4, 5, 6]))

a3 = mat(a1)      #列表 ----> 矩阵
print('a3 :',a3)
#('a3 :', matrix([[1, 2, 3, 4, 5, 6]]))

a4 = a3.tolist()   #矩阵 ---> 列表
print('a4 :',a4)
#('a4 :', [[1, 2, 3, 4, 5, 6]])   #注意！！有不同
print(a1 == a4)
#False

a8 = a3.tolist()[0]   #矩阵 ---> 列表
print('a8 :',a8)
#('a8 :', [1, 2, 3, 4, 5, 6])  #注意！！有不同
print(a1 == a8)
#True

a5 = a2.tolist()   #数组 ---> 列表
print('a5 :',a5)
#('a5 :', [1, 2, 3, 4, 5, 6])
print(a5 == a1)
#True

a6 = mat(a2)   #数组 ---> 矩阵
print('a6 :',a6)
#('a6 :', matrix([[1, 2, 3, 4, 5, 6]]))

print(a6 == a3)
#[[ True  True  True  True  True  True]]

a7 = array(a3)  #矩阵 ---> 数组
print('a7 :',a7)
#('a7 :', array([[1, 2, 3, 4, 5, 6]]))
print(a7 == a2)
#[[ True  True  True  True  True  True]]

```

