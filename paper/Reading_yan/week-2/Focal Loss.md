> 题目：Focal Loss for Dense Object Detection
>
> 来源： ICCV 2017
>
> 作者：Tsung-Yi Lin Priya Goyal Ross Girshick Kaiming He Piotr Dolla ́r



### Motivation：

检测分为one-stage和two-stage。two-stage首先检测出粗糙的候选框，通过启发式采样然后在这些粗糙的候选框中精确回归候选框的坐标以及长宽，实现了可控的样本均衡。One-stage为了减少检测出粗糙候选框的步骤，为每一个位置都设置一个anchor。然后回归这些anchor的坐标。One-stage的效果明显差于two-stage。作者发现原因在于one-stage的方法样本极度不均衡（这个是因为easy examples虽然单个样本loss很小，但是数目巨大，主导了梯度下降的方向）。而two-stage由于先做了初步筛选，样本不均衡程度得到改善。


### idea：



通过降低容易分类样本的权重，  Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. 

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210426154107175.png" alt="image-20210426154107175" style="zoom:67%;" />

<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210426154144212.png" alt="image-20210426154144212" style="zoom:67%;" />

调制因子减少了简单例子中的损耗贡献，并且扩展了例子接收低损耗的范围

例如，与$γ = 2$，分类为$p_t=0.9$，loss比CE低100倍，这反过来又相对加强了纠正错误分类的loss（$p_t<0.5$，loss大约比CE低4倍）



<img src="/Users/lishuo/Library/Application Support/typora-user-images/image-20210426154216257.png" alt="image-20210426154216257" style="zoom:67%;" />

### 以后改进方向：



### 词句：

1、RetinaNet is able to match the speed of pre- vious one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors.

2、we identify class imbalance during training as the main obstacle impeding one-stage detector from achieving state-of-the-art accuracy and propose a new loss function that eliminates this barrier.

impeding ... from ... 阻碍...达到...

例句：The public benefit of a functioning health system far outweighs any harm in impeding sellers from maximising their profits.

一个正常运转的医疗系统所带来的公共利益，远远超过了阻碍销售商实现利润最大化所带来的危害。

3、yielding large gains in accuracy and ushering in the modern era of object detection 从而在精度上获得巨大提高，并开创了目标检测的现代时代

4、Region Proposal Networks (RPN) integrated proposal generation with the second-stage classifier into a single convolution network

​	integrate theory with practice 理论联系实际

5、incur a loss with non-trivial magnitude.. 非常巨大

6、in the presence of   在......的存在下





