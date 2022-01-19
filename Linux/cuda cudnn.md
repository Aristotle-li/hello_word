# CUDA与cuDNN

#### **1****、什么是CUDA** 

​    CUDA(ComputeUnified Device Architecture)，是显卡厂商NVIDIA推出的运算平台。 CUDA是一种由NVIDIA推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。

------

#### **2****、什么是CUDNN** 

​    NVIDIA cuDNN是用于深度神经网络的GPU加速库。它强调性能、易用性和低内存开销。NVIDIA cuDNN可以集成到更高级别的机器学习框架中，如谷歌的Tensorflow、加州大学伯克利分校的流行caffe软件。简单的**插入式设计**可以让开发人员专注于设计和实现神经网络模型，而不是简单调整性能，同时还可以在GPU上实现高性能现代并行计算。

------

#### **3、CUDA与CUDNN的关系** 

   CUDA看作是一个工作台，上面配有很多工具，如锤子、螺丝刀等。cuDNN是基于CUDA的深度学习GPU加速库，有了它才能在GPU上完成深度学习的计算。它就相当于工作的工具，比如它就是个扳手。但是CUDA这个工作台买来的时候，并没有送扳手。想要在CUDA上运行深度神经网络，就要安装cuDNN，就像你想要拧个螺帽就要把扳手买回来。这样才能使GPU进行深度神经网络的工作，工作速度相较CPU快很多。

------

#### **4、CUDNN不会对CUDA造成影响**

官方Linux安装指南表述：

![img](https://upload-images.jianshu.io/upload_images/13499843-c058a1346b429ec6.png?imageMogr2/auto-orient/strip|imageView2/2/w/1080)

​    从官方安装指南可以看出，只要把cuDNN文件复制到CUDA的对应文件夹里就可以，即是所谓插入式设计，把cuDNN数据库添加CUDA里，cuDNN是CUDA的扩展计算库，不会对CUDA造成其他影响。

cuDNN的安装文件有两个文件夹，共五个文件，如下

![img](https://upload-images.jianshu.io/upload_images/13499843-6e94bd8cf8f682b6.png?imageMogr2/auto-orient/strip|imageView2/2/w/649)

cudnn.h是调用加速库的文件，*.os是

CUDA平台里对应文件夹的文件，如下

![img](https://upload-images.jianshu.io/upload_images/13499843-2e0242572ee31040.png?imageMogr2/auto-orient/strip|imageView2/2/w/528)

可以看到，CUDA已有的文件与cuDNN没有相同的文件，复制CUDNN的文件后，CUDA里的文件并不会被覆盖，CUDA其他文件并不会受影响。

------

#### **5、Linux下CUDNN的安装** 

在服务器上共安装了三个不同版本的CUDA，并不知道哪个能正常调用，所以需要安装三个不同版本的cuDNN。

cuDNN的文件已经放入服务器我的文件夹下

linu命令如下：如果不行，就全部去掉sudo。

cp 是复制，chmod是给与文件可读权限，使这个文件可以读取，rm 是删除文件

（1）

sudo cp /public/home/qliang/lyr/ysl/cudnn9.1/cuda/include/cudnn.h /usr/local/cuda-9.1/include

sudo cp /public/home/qliang/lyr/ysl/cudnn9.1/cuda/include/libcudnn* /usr/local/cuda-9.1/lib64

sudo chmod a+r /usr/local/cuda-9.1/include/cudnn.h

sudo chmod a+r /usr/local/cuda-9.1/lib64/libcudnn*

（2）

sudo cp /public/home/qliang/lyr/ysl/cudnn9.1/cuda/include/cudnn.h /public/software/cuda-9.1/include

sudo cp /public/home/qliang/lyr/ysl/cudnn9.1/cuda/lib64/libcudnn* /public/software/cuda-9.1/lib64

sudo chmod a+r /public/software/cuda-9.1/include/cudnn.h

sudo chmod a+r [/public/software/cuda-9.1](https://www.jianshu.com/p/622f47f94784)lib64/libcudnn*

（3）

sudo cp /public/home/qliang/lyr/ysl/cudnn10/cuda/include/cudnn.h/ usr/local/cuda-9.1/include

sudo cp [/public/home/qliang/lyr/ysl/cudnn10/cuda/lib64/libcudnn*](https://www.jianshu.com/p/622f47f94784) /usr/local/cuda-9.1/lib64

sudo chmod a+r /public/software/cuda-10.0/include/cudnn.h

sudo chmod a+r [/public/software/cuda-10.0/lib64/libcudnn*](https://www.jianshu.com/p/622f47f94784)

------

#### **7、卸载cuDNN** 

因为是插入式设计，cuDNN的卸载也非常简单，只需要把相关文件删除就可以了。指令如下：

rm –rf /usr/local/cuda-9.1/include/cudnn.h

rm –rf /usr/local/cuda-9.1/lib64/libcudnn*

rm –rf /public/software/cuda-9.1/include/cudnn.h

rm –rf /public/software/cuda-9.1/lib64/libcudnn*

rm –rf /public/software/cuda-10.0/include/cudnn.h

rm –rf /public/software/cuda-10.0/lib64/libcudnn*