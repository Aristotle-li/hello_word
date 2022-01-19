 XGBoost的全程为eXtreme Gradient Boosting，即极端梯度提升树。要深入理解整个XGBoost模型系统，建议还是要认真研读陈天奇的 ***XGBoost: A Scalable Tree Boosting System\*** 论文

 :损失函数需要二阶可导，相比GBDT利用到了二阶导数信息。





<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220204347458.png" alt="image-20211220204347458" style="zoom:67%;" />



  然后我们使用**二阶泰勒公式**

<img src="https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20211220204139015.png" alt="image-20211220204139015" style="zoom:67%;" />

 其中，gi 为损失函数的一阶导，hi 为损失函数的二阶导，需要注意的是这里是对 $\hat{y_i}^{(t-1)}$求导。XGBoost相较于GBDT而言用到了二阶导数信息，所以如果要自定义损失函数，首要的要求是其二阶可导。





[XGBoot ](https://zhuanlan.zhihu.com/p/87885678?utm_source=wechat_session&utm_medium=social&utm_oi=660108897164201984)

