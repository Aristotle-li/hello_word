* 平时我们使用最多的就是一，二维和三维矩阵，以前我容易将其跟立体几何联系起来。后来发现这样是非常错误的，因为再高一点的维度就不能想象了。
* 所以，按照矩阵的形式，从外向内，逐层分解才能掌握好矩阵。多维矩阵按括号的层级，从外向内，一次是第1，2，3，...维
* 在numpy中，维度可以看作是嵌套列表的数量。（ In numpy, the dimension can be seen as the number of nested lists. ）

## python中numpy的stack、vstack、hstack、concatenate、



在python的numpy库中有一个函数np.stack(), 看过一些博文后觉得别人写的太复杂，然后自己有了一些理解之后做了一些比较简单的解释



## np.stack

首先stack函数用于堆叠数组，其调用方式如下所示：

np.stack(arrays,axis=0)

其中arrays即需要进行堆叠的数组，axis是堆叠时使用的轴，比如：

arrays = [[1,2,3,4], [5,6,7,8]]

这是一个二维数组，axis=0表示的是第一维，也即是arrays[0] = [1,2,3,4]或者arrays[1] = [5,6,7,8]

axis=i时，代表在堆叠时首先选取第i维进行“打包”



具体例子：

当执行np.stack(arrays, axis=0)时，取出第一维的1、2、3、4，打包，[1, 2, 3, 4]，其余的类似，然后结果如下：

```python
>>> arrays = [[1,2,3,4], [5,6,7,8]]
>>> arrays=np.array(arrays)
>>> np.stack(arrays,axis=0)

array([[1, 2, 3, 4],
       [5, 6, 7, 8]])
```

当执行np.stack(arrays, axis=1)时，先对arrays中的第二维进行“打包”，也即是将1、5打包成[1, 5]，其余的类似，结果如下：

```python
>>> np.stack(arrays, axis=1)
array([[1, 5],
       [2, 6],
       [3, 7],
       [4, 8]])
```

有这个“打包”的概念后，对于三维的数组堆叠也不难理解了，例如:

a = np.array([[1,2,3,4], [5,6,7,8]])

arrays = np.asarray([a, a , a])

```python
>>> arrays

array([[[1, 2, 3, 4],
        [5, 6, 7, 8]],

       [[1, 2, 3, 4],
       [5, 6, 7, 8]],

       [[1, 2, 3, 4],
       [5, 6, 7, 8]]])
```

执行np.stack(arrays, axis=0)，也就是对第一维进行打包，结果如下：

```python
>>> np.stack(arrays, axis=0)

array([[[1, 2, 3, 4],
       [5, 6, 7, 8]],

       [[1, 2, 3, 4],
        [5, 6, 7, 8]],

       [[1, 2, 3, 4],
      [5, 6, 7, 8]]])
```

执行np.stack(arrays, axis=1)，也就是对第二维进行打包，取出第二维的元素[1,2,3,4]、[1,2,3,4]、[1,2,3,4]，打包，[[1,2,3,4],[1,2,3,4],[1,2,3,4]]，对其余的也做类似处理，结果如下：

```python
>>> np.stack(arrays, axis=1)
array([[[1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4]],

       [[5, 6, 7, 8],
        [5, 6, 7, 8],
        [5, 6, 7, 8]]])
```

执行np.stack(arrays, axis=2),与之前类似，取出第三维元素1、1、1，打包[1,1,1],结果如下：

```python
>>> np.stack(arrays, axis=2)

array([[[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]],
  
       [[5, 5, 5],
        [6, 6, 6],
        [7, 7, 7],
        [8, 8, 8]]])
```



总结而言，也就是arrays是你要进行堆叠的数组，axis控制你要将arrays中哪个维度组合起来（也就是文中的“打包”）。



## np.concatenate



np.concatenate((a1,a2,a3,...), axis=0)，这个函数就是按照特定方向轴进行拼接，默认是第一维，在numpy官网上的示例如下：

```python
>>> a = np.array([[1, 2], [3, 4]])

>>> b = np.array([[5, 6]])

>>> np.concatenate((a, b), axis=0)

array([[1, 2],
       [3, 4],
       [5, 6]])

>>> np.concatenate((a, b.T), axis=1)


array([[1, 2, 5],
       [3, 4, 6]])
```

当axis=0时，将b的元素加到a的尾部，这里比较难以理解的是第二个np.concatenate((a, b.T), axis=1)，其实也类似，b.T的shape为（1，2）,axis=1,则在a的第二维加上b的每个元素，所以这里axis=i时, 输入参数（a1,a2,a3...）除了第i维，其余维度的shape应该一致，例如：

```python
>>> a = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])

>>> b = np.array([[[1,2,3],[4,5,6]]])

>>> np.concatenate((a, b), axis=0)

array([[[1, 2, 3],
        [4, 5, 6]],

       [[1, 2, 3],
        [4, 5, 6]],

       [[1, 2, 3],
        [4, 5, 6]]])
```

这里a的shape为（2，2，3），b的shape为（1，2，3），axis=0则要求a,b在其他两维的形状是一致的，如果直接在其他维度进行concatenate操作则会报错（因为axis=1时，a和b在第一维的长度不一致）：

```python
>>> np.concatenate((a, b), axis=1)

Traceback (most recent call last):

  File "<stdin>", line 1, in <module>

ValueError: all the input array dimensions except for the concatenation axis must match exactly

>>> np.concatenate((a, b), axis=2)

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: all the input array dimensions except for the concatenation axis must match exactly
```

下面一个例子能够说明：

```python
>>> c=np.array([[[5,6,7],[7,8,9]],[[4,5,6],[5,6,7]]])

>>> c
array([[[5, 6, 7],
        [7, 8, 9]],

       [[4, 5, 6],
       [5, 6, 7]]])
>>> c.shape

(2, 2, 3)
>>> np.concatenate((a, c), axis=1)

array([[[1, 2, 3],
        [4, 5, 6],
        [5, 6, 7],
        [7, 8, 9]],

       [[1, 2, 3],
        [4, 5, 6],
        [4, 5, 6],
        [5, 6, 7]]])

>>> np.concatenate((a, c), axis=2)


array([[[1, 2, 3, 5, 6, 7],
        [4, 5, 6, 7, 8, 9]],

       [[1, 2, 3, 4, 5, 6],
        [4, 5, 6, 5, 6, 7]]])
```





## np.hstack

np.hstack(tup), 按照列的方向堆叠， tup可以是元组，列表，或者numpy数组, 其实也就是axis=1,即

np.hstack(tup) = np.concatenate(tup, axis=1)

按照上面对concatenate的理解则下面的示例很好理解

```python
>>> a = np.array((1,2,3))

>>> b = np.array((2,3,4))
>>> np.hstack((a,b))
array([1, 2, 3, 2, 3, 4])

>>> a = np.array([[1],[2],[3]])

>>> b = np.array([[2],[3],[4]])

>>> np.hstack((a,b))

array([[1, 2],
       [2, 3],
       [3, 4]])
```



## np.vstack

np.vstack(tup), 按照行的方向堆叠， tup可以是元组，列表，或者numpy数组, 理解起来与上相同

np.vstack(tup) = np.concatenate(tup, axis=0)

```python
>>> a = np.array([1, 2, 3])
>>> b = np.array([2, 3, 4])
>>> np.vstack((a,b))
array([[1, 2, 3],
       [2, 3, 4]])
>>> a = np.array([[1], [2], [3]])
>>> b = np.array([[2], [3], [4]])
>>> np.vstack((a,b))
array([[1],
       [2],
       [3],
       [2],
       [3],
       [4]])
```

对于第二段代码，a的第一维元素分别时[1],[2],[3]，所以堆叠时将b的对应元素直接加入 





## np.dstack

np.dstack(tup), 按照第三维方向堆叠，也即是

np.dstack(tup) = np.concatenate(tup, axis=2), 这里较好理解，所以直接放官网的示例

```python
>>> a = np.array((1,2,3))
>>> b = np.array((2,3,4))
>>> np.dstack((a,b))
array([[[1, 2],
        [2, 3],
        [3, 4]]])
>>> a = np.array([[1],[2],[3]])
>>> b = np.array([[2],[3],[4]])
>>> np.dstack((a,b))
array([[[1, 2]],
       [[2, 3]],
       [[3, 4]]])
```



## np.column_stack和np.row_stack

np.column_stack函数将一维的数组堆叠为二维数组，方向为列

np.row_stack函数将一维的数组堆叠为二维数组，方向为行

其实如果对前面的内容理解之后这两个算是比较简单的了

```python
>>> a = np.array((1,2,3))
>>> b = np.array((2,3,4))
>>> np.column_stack((a,b))
array([[1, 2],
       [2, 3],
       [3, 4]])
>>> np.row_stack([np.array([1, 2, 3]), np.array([4, 5, 6])])
array([[1, 2, 3],
       [4, 5, 6]])
```



## 总结

其实也就是两种操作，stack和concatenate，其中stack是首先找到axis轴的元素，然后对该轴的元素进行组合，然后形成新的数组，而concatenate则是在axis轴进行拓展，将a1,a2,a3...按照axis指定的轴进行增加操作...