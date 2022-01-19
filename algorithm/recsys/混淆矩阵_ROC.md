```
>>> from sklearn.metrics import confusion_matrix
>>> y_true = [2, 0, 2, 2, 0, 1]
>>> y_pred = [0, 0, 2, 2, 0, 2]
>>> confusion_matrix(y_true, y_pred)
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])

>>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
>>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
>>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])
```



```
Thus in binary classification, the count of true negatives is
:math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
:math:`C_{1,1}` and false positives is :math:`C_{0,1}`.
```

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220192827439.png" alt="image-20211220192827439" style="zoom: 67%;" />

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220193242677.png" alt="image-20211220193242677" style="zoom: 67%;" />

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220194613224.png" alt="image-20211220194613224" style="zoom:67%;" />

对于LR等预测类别为概率的分类器，依然用上述例子，假设预测结果如下：

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-d4865f1e8d675f21cbbe656eb546547d_720w.jpg)![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-d4865f1e8d675f21cbbe656eb546547d_720w-20211220194645557.jpg)

这时，需要设置阈值来得到混淆矩阵，不同的阈值会影响得到的TPRate，FPRate，如果阈值取0.5，小于0.5的为0，否则为1，那么我们就得到了与之前一样的混淆矩阵。其他的阈值就不再啰嗦了。依次使用所有预测值作为阈值，得到一系列TPRate，FPRate，描点，求面积，即可得到AUC。

最后说说AUC的优势，AUC的计算方法同时考虑了分类器对于正例和负例的分类能力，在样本不平衡的情况下，依然能够对分类器作出合理的评价。

例如在反欺诈场景，设欺诈类样本为正例，正例占比很少（假设0.1%），如果使用准确率评估，把所有的样本预测为负例，便可以获得**99.9%的准确率**。

但是如果使用AUC，把所有样本预测为负例，TPRate和FPRate同时为0（没有Positive），与(0,0) (1,1)连接，得出**AUC仅为0.5**，成功规避了样本不均匀带来的问题。

