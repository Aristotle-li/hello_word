![image-20210122154436328](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122154436328.png)

![image-20210122154712228](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122154712228.png)

这里RCNN是不同的！

RCNN将每一个候选框输入深度网络，得到特征图

Fast RCNN

1、将全图输入深度网络，得特征图，将ss算法生成的候选框投影到特征图获得相应的特征矩阵

2、将每个特征矩阵通过ROI pooling 层都缩放到7*7大小的特征图，接着将特征图展平，通过一系列全连接层得到预测结果。

3、不用单独训练SVM分类器和回归器，在一个深度网络里面都集成了。

* > 疑问：全图输入网络后，原本对应位置类别的物体已经被卷的面目全非了！！！那么将ss算法生成的框投影到对应位置后，是如何保留这些信息的！！！！
  >
  > 难以理解！！！
  >
  > `答：任意的原始图像中的输入是可以映射到特征图中的，卷积只会改变空间分辨率，不改变比例和位置。`<img src="https://img-blog.csdnimg.cn/20181224222704781.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhY2tlcl9sb25n,size_16,color_FFFFFF,t_70" alt="img" style="zoom:50%;" />
  >
  > 映射很简单，等比例缩放即可，实现时考虑好padding，stride等操作。
  >
  > 那这跟roi pooling有什么关系呢？

![image-20210122155652267](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122155652267.png)
![image-20210122160023622](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122160023622.png)

> IoU大于0.5规定是正样本、IoU小于0.5规定是负样本

## **sppnet**

**不从源头而是在最后一环的特征上做处理**，将任意尺度(大于4*4)大小的特征，进行3种pooling，串接起来得到固定的21维，从而避免了固定尺寸输入的约束，如下。

<img src="https://img-blog.csdnimg.cn/20181224222651896.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhY2tlcl9sb25n,size_16,color_FFFFFF,t_70" alt="img" style="zoom:50%;" />

当然你用的时候，不必要局限于4*4，2*2，1*1。

以前为了满足全连接层固定大小输入，需要在输入图进行缩放，然后提取特征。现在既然已经可以在特征空间进行缩放了，**那么就不必要求输入一样大了**。

那这跟roi pooling有什么关系呢？

ROI pooling是一个简化的spp池化，不需要这么复杂，**直接一次分块pooling**就行了，**在经过了从原始空间到特征空间的映射之后，设定好一个pooled_w，一个pooled_h**，就是将W*H的输入pooling为pooled_w*pooled_h的特征图，然后放入全连接层。 它完成的下面的输入输出的变换。

![img](https://img-blog.csdnimg.cn/20181224222720725.png)

每个特征矩阵通过ROI pooling 层都缩放到7*7大小的特征图，好处是不用限制输入图像的尺寸



![image-20210122160423822](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122160423822.png)

![image-20210122160602341](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122160602341.png)

![image-20210122160654704](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122160654704.png)

![image-20210122161145411](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122161145411.png)


$$
分类损失:L_{cls}(p,u)=-logp_u \ 就是交叉熵损失，因为u作为真实的标签，one-hot编码是：[0,0,0...1,..0],只有在对应的位置才有效
\\
预测softmax概率为[0.1,0.1,...0.7,0],那么loss就等于-log（0.7）
$$
![image-20210122161501347](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122161501347.png)

![image-20210122162319530](/Users/lishuo/Library/Application Support/typora-user-images/image-20210122162319530.png)

![image-20210125133514228](/Users/lishuo/Library/Application Support/typora-user-images/image-20210125133514228.png)

