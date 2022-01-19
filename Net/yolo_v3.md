yolo_v3 spp

 ![image-20220107214256727](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20220107214256727.png)



GIoU:

![image-20220109160550291](/Users/lishuo/Library/Application Support/typora-user-images/image-20220109160550291.png)

当两个框水平，GIoU退化到IoU

DIoU:解决了这个问题，收敛速度快

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimage-20220107222407035.png)

![ ](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20220107222407035.png)

CIoU：

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimgimage-20220107214256727.png)



![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimgimgimage-20220107223152767.png)

![image-20220107223152767](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgimage-20220107223152767.png)





Focal loss：容易受噪音干扰

p越大说明越容易分，属于易分样本，通过$（1-p_t）^{\gamma}$ 

来降低权重，$\alpha$ 控制正负样本的权重，$\gamma$    控制易分样本权重 。





![image-20220109160706309](/Users/lishuo/Library/Application Support/typora-user-images/image-20220109160706309.png)





FCOS、CornerNet、Center-ness、





个人愚见：这几种方法的核心在于训练的策略：



样本与gt的对应，即为如何选取精确合适的样本与真实值进行对应。anchor的加入在一定程度上降低了回归问题的难度，为分类提供了选则样本的途径。但是这种选取方法也会造成定位模糊以及背景特征干扰，而且，简单的基于IoU的样本选取策略甚至会造成同一个或同一类特征对应于冲突的label。

准确的讲是同一类



赞同。其实样本选择本身就要遵从真实准确的反映gt的原则。
个人觉得：FCN与后面的几种方法稍微有点不同，因为它在一定程度上添加了人类对gt的解释：part，这个信息参与了分类。
问一下题主：有关于分类和定位这两个任务在multi-task里面是如何协同的相关研究或文章吗？



分类和定位协同的好像没见过，知道一个实例分割htc是mask分支和box分支协同的

 (作者) 2019-10-17

Anchor的设定，就是为了让优化空间变得合理，只是这个目的，可以有其他方式。很合理的。。

今年刚出的FairMOT有点这种感觉，先确定目标位置，再通过位置去取对应位置的特征用于分类，可以看下





