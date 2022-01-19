![image-20220103195736998](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20220103195736998.png)

## 训练过程

**（1）先验框匹配** 
在训练过程中，**首先要确定训练图片中的ground truth（真实目标）与哪个先验框来进行匹配，与之匹配的先验框所对应的边界框将负责预测它**。



在Yolo中，ground  truth的中心落在哪个单元格，该单元格中与其IOU最大的边界框负责预测它。



但是在SSD中却完全不一样，

SSD的先验框与ground  truth的匹配原则主要有**两点。**

**首先**，对于图片中每个ground  truth**，找到与其IOU最大的先验框，**该先验框与其匹配，这样，可以保证每个ground truth一定与某个先验框匹配。通常称与ground  truth匹配的先验框为正样本（其实应该是先验框对应的预测box，不过由于是一一对应的就这样称呼了），反之，若一个先验框没有与任何ground  truth进行匹配，那么该先验框只能与背景匹配，就是负样本。

一个图片中ground truth是非常少的，  而先验框却很多，如果仅按第一个原则匹配，很多先验框会是负样本，正负样本极其不平衡，所以需要第二个原则。



**第二个原则是：**对于剩余的未匹配先验框，若某个ground truth的 ![[公式]](https://www.zhihu.com/equation?tex=\text{IOU}) 大于某个阈值（一般是0.5），那么该先验框也与这个ground truth进行匹配。这意味着某个ground  truth可能与多个先验框匹配，这是可以的。

但是反过来却不可以，因为一个先验框只能匹配一个ground truth，**如果多个ground  truth与某个先验框 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BIOU%7D) 大于阈值，那么先验框只与IOU最大的那个ground truth进行匹配。**



第二个原则一定在第一个原则之后进行，仔细考虑一下这种情况，如果某个ground truth所对应最大 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BIOU%7D) 小于阈值，并且所匹配的先验框却与另外一个ground truth的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BIOU%7D) 大于阈值，那么该先验框应该匹配谁，答案应该是前者，首先要确保某个ground truth一定有一个先验框与之匹配。但是，这种情况我觉得基本上是不存在的。由于先验框很多，某个ground truth的最大 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BIOU%7D) 肯定大于阈值，所以可能只实施第二个原则既可以了，这里的[TensorFlow版本](https://github.com/xiaohu2015/SSD-Tensorflow/blob/master/nets/ssd_common.py)就是只实施了第二个原则，但是这里的[Pytorch](https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py)两个原则都实施了。图8为一个匹配示意图，其中绿色的GT是ground truth，红色为先验框，FP表示负样本，TP表示正样本。