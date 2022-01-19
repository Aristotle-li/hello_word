首发于[深度学习【目标追踪】](https://www.zhihu.com/column/c_1177216807848529920)

<img src="https://pic1.zhimg.com/v2-01baa2cec9daa53f061647bd71fa6c2e_1440w.jpg?source=172ae18b" alt="【MOT】详解SORT与卡尔曼滤波算法" style="zoom:25%;" />

# 【MOT】详解SORT与卡尔曼滤波算法





## 1 前言

SORT（SIMPLE ONLINE AND REALTIME TRACKING），从名字上看，就感觉该方法应当很简单。

作者论文提到

> Despite only using a rudimentary combination of familiar techniques such as the **Kalman Filter** and **Hungarian algorithm** for the tracking components, this approach achieves an accuracy comparable to state-of-the-art online trackers

该算法提出于2016年，作者仅灵活地使用了**卡尔曼滤波算法**和**匈牙利算法**的结合，就实现了当时SOTA的表现效果。结果非常amazing呀！

有关卡尔曼滤波算法和匈牙利算法，网上均有相关的博客进行教程，我们这篇文章先推荐几篇不错的教程，后面根据自己的理解，并结合SORT的论文和代码进行详解。

卡尔曼滤波：

英文：

[http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/#mathybits](http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/#mathybits)

[www.bzarg.com](http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/#mathybits)

中文：

[论智：图说卡尔曼滤波，一份通俗易懂的教程](https://zhuanlan.zhihu.com/p/39912633)[zhuanlan.zhihu.com![图标](https://pic4.zhimg.com/v2-954bea3147c72502022920b95819621b_ipico.jpg)](https://zhuanlan.zhihu.com/p/39912633)

匈牙利算法：

[ZihaoZhao：带你入门多目标跟踪（三）匈牙利算法&KM算法](https://zhuanlan.zhihu.com/p/62981901)[zhuanlan.zhihu.com![图标](https://zhstatic.zhihu.com/assets/zhihu/editor/zhihu-card-default.svg)](https://zhuanlan.zhihu.com/p/62981901)

上面这些文章，都是我入门SORT或者DeepSORT时细读的，个人觉得还不错。

接下来就简单聊聊**卡尔曼滤波**这个硬骨头，以及他是怎么被用到SORT中的。

## 2 卡尔曼滤波

在阅读我对卡尔曼滤波的理解前，希望你已经看过上面发过的链接了。

如下图所示（图来源于[这儿](http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/#mathybits)）。由此可见，卡尔曼滤波主要有两个过程，一个叫做（1）**预测**，另一个叫做（2）**更新**。

![img](https://pic3.zhimg.com/80/v2-6c733bbc642bc568bdaf73924bf0804a_1440w.jpg)图 1 卡尔曼滤波的双过程

何为**预测**，何为**更新**呢？

这里举个例子，一个机器小车在路上走（路不一定平哈），假设他k-1时候的状态是 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bx%7D_%7Bk-1%7D) ,这个状态不妨包含两个参数

- （1）当前位置参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bp%7D_%7Bk-1%7D) 
- （2）当前速度参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bv%7D_%7Bk-1%7D) 

我们在后台可以看到这个小车的一些**运行参数**，比如（速度、加速度、转角等）。那么我们根据这些**运行参数**来推理**时刻k**机器小车的状态是可行的。比如这里假设机器小车是均速直线运动的，那么预测的

- ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bp%7D_%7Bk%7D%3D%5Chat%7Bp%7D_%7Bk-1%7D%2B%5Chat%7Bv%7D_%7Bk-1%7D.t) 
- ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bv%7D_%7Bk%7D%3D%5Chat%7Bv%7D_%7Bk-1%7D) 

那么这个预测的状态就是

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bx%7D_%7Bk%7D%3D%28%7B+%5Chat%7Bp%7D_%7Bk%7D%2C%5Chat%7Bv%7D_%7Bk%7D%7D%29) 

由于状态信息中，每个变量间可能存在一定的**相关性**，所以存在一个描述**相关性**的**协方差矩阵** ![[公式]](https://www.zhihu.com/equation?tex=P_%7Bk-1%7D) 。当然了，这个**协方差矩阵**也是会随着**状态的预测**而进行**调整**的。具体的调整规则，在[前面链接](https://zhuanlan.zhihu.com/p/39912633)中作者已经探讨过了。以上的过程被称为**预测**（图上的predict）。

那么到这里肯定有人会问，我们知道了运行过程的一些参数，不就能够根据**前面的状态**来估计后**后面的状态**了？

答案是：可以估计一个**大概的范围**，但是不能做到精确估计，因为可能会有误差，比如车轮打滑等情况发生，观测的速度可能不变，但是**实际走的路程**要比**预测的短。**

那么怎么可以减少预测的误差呢？有没有其他可以借鉴的信息呢？

我们假设机器小车上装了其他的传感器，比如**GPS定位传感器**，但是我们知道GPS是存在误差的。那么我们从GPS上获得的小车状态 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BZ%7D_%7Bk%7D) 也是具有**误差范围**的。上图中，该**误差的范围**用协方差 ![[公式]](https://www.zhihu.com/equation?tex=R_%7Bk%7D) 表示。虽然该GPS提供的信息存在误差，但是他可以**辅助**我们进一步地判断小车地**状态区间**，这个过程被称为**更新**（图中的update）。

所以预测和更新的过程，就是**信息融合**的过程，目的就是融合所有已知的信息，使得状态的预测更加准确，该过程中具体算法实现，就是大名鼎鼎的**卡尔曼滤波算法**。

所以卡尔曼滤波会经常和“**多传感器信息融合**”等词眼一起出来。

上面只是我简单的个人总结，如果想要详细了解卡尔曼滤波，建议看完我上面发的链接。

## 3 SORT中的卡尔曼滤波

SORT中将卡尔曼滤波器用于检测框运动的预测，那么描述一个检测框需要以下**四个状态**，即

- （1）检测框中心的横坐标
- （2）检测框中心的纵坐标
- （3）检测框的大小（论文中叫做scale或者area）
- （4）长宽比

以上四个状态可以描述一个检测框的基本信息，但是不能完全描述一个状态的**运动状态信息**，所以需要引入上述的状态的**变化量信息**（可以看作变化速度）来进行运动状态信息的描述。

由于SORT假设一个物体在不同帧中检测框的**长宽比不变**，是个**常数**，所以变化量只考虑上面的（1）（2）（3），不考虑（4），即

- （1）检测框中心的横坐标的变化速度
- （2）检测框中心的纵坐标的变化速度
- （3）检测框的大小（论文中叫做scale或者area）的变化速度

所以SORT中共使用了**7个参数**，用来描述检测框的状态。论文中是这么说的

![img](https://pic1.zhimg.com/80/v2-8ba3aad188f6a82e884a6364084af6d0_1440w.jpg)

大致的意思就是将**帧间的位移**假设为**线性匀速模型**，所以每个目标的**状态**综合了上述提到的7个信状态值。

但是代码中，仍然用到了8个状态值（因为代码是DeepSORT中的SORT,其实主要思路是一致的）。

不妨看一下代码，该代码定义了个KalmanFilter类，总共分为6部分：

- （1）类初始化__init__
- （2）**初始化**状态（mean)与状态协方差(covariance)的函数initiate
- （3）**预测阶段**函数predict
- （4）**分布转换**函数project
- （**5) 更新阶段**函数update
- （6) 计算**状态分布**和**测量（检测框）**之间距离函数gating_distance

（1）类初始化__init__

该部分定义了一些基础参数，便于后面函数的调用。

```python
class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
```

（2）初始化状态（mean)与状态协方差(covariance)的初始化函数initiate

```python
    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance
```

（3）预测阶段函数predict

```python
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance
```

这段实现的内容对应[上面链接](https://zhuanlan.zhihu.com/p/39912633)中的公式

![img](https://pic4.zhimg.com/80/v2-d14ea0ff0371aaa61defcf15d6e72b33_1440w.jpg)

（4）分布转换函数project

```python
    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov
```

这段实现对应[上面链接](https://zhuanlan.zhihu.com/p/39912633)中的公式

![img](https://pic3.zhimg.com/80/v2-fded6cb744c6b8489e4138b68be33472_1440w.jpg)

（5）更新阶段函数update

```python
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
```

这段实现对应[上面链接](https://zhuanlan.zhihu.com/p/39912633)中的公式

![img](https://pic3.zhimg.com/80/v2-1a0b1c56432b0cf2a09ce08a5fe23ab6_1440w.jpg)

（6) 计算状态分布和测量（检测框)之间距离函数gating_distance

```python
    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
```

结合上面的代码和解析（包括链接），很容易弄清楚卡尔曼滤波的具体实现过程。

有关匈牙利算法在SORT中的使用。个人认为，这篇文章已经讲的够详细了，本文我就不再赘述！

[ZihaoZhao：带你入门多目标跟踪（三）匈牙利算法&KM算法](https://zhuanlan.zhihu.com/p/62981901)[zhuanlan.zhihu.com![图标](https://zhstatic.zhihu.com/assets/zhihu/editor/zhihu-card-default.svg)](https://zhuanlan.zhihu.com/p/62981901)

## 4 总结

SORT是我入门MOT的第一步，其中最让人难以理解的，就是当中的卡尔曼滤波算法，结合一些大佬的文章和对代码的解析，总算对卡尔曼滤波算法入门了。也希望我的总结能够对大家的理解有所帮助，谢谢！

有关SORT的升级版DeepSORT的解析，也已经更新了，欢迎阅读！

[周威：【MOT】详解DeepSORT多目标追踪模型](https://zhuanlan.zhihu.com/p/202993073)

[zhuanlan.zhihu.com![图标](https://pic1.zhimg.com/v2-8fc5be249a0f07bf0b33a846006e3ba4_180x120.jpg)](https://zhuanlan.zhihu.com/p/202993073)



编辑于 2020-08-29

[目标跟踪(target tracking)](https://www.zhihu.com/topic/20206159)

[深度学习（Deep Learning）](https://www.zhihu.com/topic/19813032)

### 文章被以下专栏收录

- [![深度学习【目标追踪】](https://pic1.zhimg.com/v2-47c0cdf6fd34149843f60eb841f54a63_xs.jpg?source=172ae18b)](https://www.zhihu.com/column/c_1177216807848529920)

- ## [深度学习【目标追踪】](https://www.zhihu.com/column/c_1177216807848529920)

- 深度学习单目标追踪（SOT)和多目标追踪(MOT)

- [![目标跟踪算法](https://pic1.zhimg.com/v2-04c5190e22d43b1397e1efb46a8ab998_xs.jpg?source=172ae18b)](https://www.zhihu.com/column/visual-tracking)

- ## [目标跟踪算法](https://www.zhihu.com/column/visual-tracking)

- Visual Tracking Algorithm Introduction.

### 推荐阅读



[![卡尔曼滤波算法思想理解 Kalman filter 第一篇](https://pic4.zhimg.com/v2-26216817770c295209d82cf5aaba05de_250x0.jpg?source=172ae18b)卡尔曼滤波算法思想理解 Kalman filter 第一篇史蒂芬方](https://zhuanlan.zhihu.com/p/129370341)[视觉跟踪(Visual Tracking)论文解读之相关滤波(Correlation Filter)篇（1）数学基础第一次写博文，诚惶诚恐。 视觉跟踪这几年发展很快。乘着这段时间在研究这个方面的机会，想把这方向的一些阅读体会写下。知乎有很多大神对论文解读，可大多数都没有补充论文的细节部分，而…rockking](https://zhuanlan.zhihu.com/p/38253462)[![Kalman滤波在MOT中的应用(三)——实践篇](https://pic4.zhimg.com/v2-94dbec08bbcf71cff7dd392bd3a744c2_250x0.jpg?source=172ae18b)Kalman滤波在MOT中的应用(三)——实践篇黄飘发表于ISE-M...](https://zhuanlan.zhihu.com/p/110163693)[视觉跟踪(Visual Tracking)论文解读之相关滤波(Correlation Filter)篇（2）C-COT视觉跟踪这几年发展很快。乘着这段时间在研究这个方面的机会，想把这方向的一些阅读体会写下。知乎有很多大神对论文解读，可大多数都没有补充论文的细节部分，而是原论文公式的重复。我正好…rockking](https://zhuanlan.zhihu.com/p/38343134)

## 3 条评论

写下你的评论...



- [![星辰大海](https://pic4.zhimg.com/v2-ee6b0fa005ae9dfbb75e7d5cdfd9d1e8_s.jpg?source=06d4cd63)](https://www.zhihu.com/people/tang-chen-bin)[星辰大海](https://www.zhihu.com/people/tang-chen-bin)01-27

  为什么那两个std项都只用了mean或者measurement中的h（也就是[3]）？我没看懂这个。

[![cjhfhb](https://pic4.zhimg.com/da8e974dc_s.jpg?source=06d4cd63)](https://www.zhihu.com/people/cjhfhb)[cjhfhb](https://www.zhihu.com/people/cjhfhb)[](https://www.zhihu.com/xen/market/vip-privileges)

回复[星辰大海](https://www.zhihu.com/people/tang-chen-bin)05-19

从物理现实来说，好像这几个状态变量的var只跟目标高度相关，高度越小的时候，说明目标越小或者在图像中越远，这些状态变量的var越小，这符合实际。和在图像中的x,y无关，和长宽比也没有什么关系

[![ENIAC](https://pic1.zhimg.com/v2-29b022d5cc77747a53268722a5607416_s.jpg?source=06d4cd63)](https://www.zhihu.com/people/liu-zhu-chen)[ENIAC](https://www.zhihu.com/people/liu-zhu-chen)05-24

mean 和 var是针对单个目标state统计得到的，还是对当前帧下所有目标统计得到的？一个Kalman filter预测一次是为单个目标预测，还是为多个目标同时预测？



# 【MOT】详解DeepSORT多目标追踪模型

[![周威](https://pic4.zhimg.com/v2-e127be459ba552a5697994cd5a80ee2d_xs.jpg?source=172ae18b)](https://www.zhihu.com/people/zhou-wei-37-26)

[周威](https://www.zhihu.com/people/zhou-wei-37-26)[](https://www.zhihu.com/question/48510028)



东南大学 工学博士在读

## 1. 前言

DeepSORT是目前非常常见的多目标追踪算法（虽说性能一般，但是速度还挺可观，也比较简单），网络上有很多基于不同检测器（YOLO V3/YOLO V4/CenterNet）的DeepSORT实战。

链接分别如下：

（1）YOLO V3+DeepSORT

[mikel-brostrom/Yolov3_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov3_DeepSort_Pytorch)

[github.com![图标](https://pic2.zhimg.com/v2-6a4b3a81f9717fd2505f1dabd2a71d8d_ipico.jpg)](https://github.com/mikel-brostrom/Yolov3_DeepSort_Pytorch)

（2）YOLO V4+DeepSORT

[https://github.com/theAIGuysCode/yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort)[github.com](https://github.com/theAIGuysCode/yolov4-deepsort)

（3）CenterNet +DeepSORT

[https://github.com/kimyoon-young/centerNet-deep-sort](https://github.com/kimyoon-young/centerNet-deep-sort)[github.com](https://github.com/kimyoon-young/centerNet-deep-sort)

那么如此通用的一个追踪器（DeepSORT），可以被接在任何一个检测器上（不仅仅是上面提到的）。

我之前做过一个基于YOLOV4+DeepSORT的交通流参数提取，部分效果图如下

![img](https://pic2.zhimg.com/80/v2-f15278107ebc79bd7578b2d2c75ed489_1440w.jpg)

上一篇文章中，我们对SORT，特别是SORT中的**卡尔曼滤波算法**进行了详细解析。链接如下

[周威：【MOT】详解SORT与卡尔曼滤波算法](https://zhuanlan.zhihu.com/p/196622890)[zhuanlan.zhihu.com![图标](https://pic3.zhimg.com/v2-01baa2cec9daa53f061647bd71fa6c2e_120x160.jpg)](https://zhuanlan.zhihu.com/p/196622890)

## 2 DeepSORT中的Deep

光看名字就知道了DeepSORT是SORT算法中的**升级版本**。那这个Deep的意思，很显然就是该算法中使用到了Deep Learning网络。

那么相比于SORT算法，DeepSORT到底做了哪部分的改进呢？这里我们简单了解下SORT算法的缺陷。

SORT算法利用**卡尔曼滤波算法**预测**检测框在下一帧的状态**，将**该状态**与**下一帧的检测结果**进行匹配，实现车辆的追踪。

那么这样的话，一旦物体受到遮挡或者其他原因没有被检测到，卡尔曼滤波预测的状态信息将无法和检测结果进行匹配，该追踪片段将会提前结束。

遮挡结束后，车辆**检测**可能又将被**继续执行**，那么SORT只能分配给该物体一个**新的ID编号**，代表一个**新的追踪片段**的开始。所以SORT的缺点是

> *受遮挡等情况影响较大，会有大量的**ID切换***

那么如何解决SORT算法出现过多的ID切换呢？毕竟是online tracking，不能利用全局的视频帧的检测框数据，想要缓解拥堵造成的ID切换需要利用到**前面已经检测到**的物体的**外观特征**（假设之前被检测的物体的外观特征都被**保存**下来了），那么当物体收到遮挡后到遮挡结束，我们能够利用之前保存的外观特征**分配该物体受遮挡前的ID编号**，降低ID切换。

当然DeepSORT就是这么做的，论文中提到

> We overcome this issue by replacing the **association metric** with a **more informed metric** that combines **motion and appearance information**.In particular,we apply a **convolutional neural network** (CNN) that has been trained to **discriminate pedestrians** on a large-scale person re-identification dataset.

很显然，DeepSORT中采用了**一个简单（运算量不大）**的CNN来提取被检测物体（检测框物体中）的**外观特征**（低维向量表示），在每次（每帧）检测+追踪后，进行一次物体外观特征的提取并保存。

后面每执行一步时，都要执行一次**当前帧被检测物体外观特征**与**之前存储的外观特征**的**相似度计算，**这个相似度将作为一个重要的判别依据（不是唯一的，因为作者说是将**运动特征**与**外观特征**结合作为判别依据，这个运动特征就是SORT中卡尔曼滤波做的事）。

那么这个小型的CNN网络长什么样子呢？论文中给出了结构表，如下

![img](https://pic2.zhimg.com/80/v2-877cb4b8f17d9c9f3f84dfa9f6482689_1440w.jpg)

那么这个网络最后输出的是一个128维的向量。有关残差网络和上图表中残差模块的结构就不多说，很简单。

值得关注的是，由于DeepSORT主要被用来做行人追踪的，那么输入的大小为128（高）x 64（宽）的矩形框。如果你需要做其他物体追踪，可能要把网络模型的输入进行修改。

实现该网络结构的代码如下：

```python
class Net(nn.Module):
    def __init__(self, num_classes=751 ,reid=False):
        super(Net,self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64,64,2,False)
        # 32 64 32
        self.layer2 = make_layers(64,128,2,True)
        # 64 32 16
        self.layer3 = make_layers(128,256,2,True)
        # 128 16 8
        self.layer4 = make_layers(256,512,2,True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8,4),1)
        # 256 1 1 
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x
```

可以看出，网络的输出为**分类数**，并不是128。那么后面想要提取图像的特征，只需要在使用过程中，将最后一层classifier忽略即可（128维向量为**输入到classifier前的特征图**）。

## 3  DeepSORT中的卡尔曼滤波

我们在SORT深度解析时提到过，SORT中的卡尔曼滤波算法使用的状态是一个7维的向量。即

![img](https://pic1.zhimg.com/80/v2-32343d459362d5f028a71dd86cc9208c_1440w.jpg)

在DeepSORT中，使用的状态是一个8维的向量

![img](https://pic4.zhimg.com/80/v2-05e41804e334ff38d1f36d7840d0c2fb_1440w.png)

相较于SORT中的状态，多了一个**长宽比（aspect ratio）**的**变化率**。这是合情合理的，毕竟像SORT中假设物体检测框的长宽比是固定的，实际过程中，随着镜头移动或者物体与相机的相对运动，物体的长宽比也是会发生变化的。

同时，DeepSORT对追踪的初始化、新生与消失进行了设定。

- **初始化**：如果一个检测**没有**和之前**记录**的track相**关联**，那么从该检测开始，初始化一个新的目标（并不是新生）
- **新生**：如果一个目标被初始化后，且在**前三帧**中均被**正常的捕捉**和**关联成功**，那么该物体产生一个新的track，否则将被删除。
- **消失**：如果超过了设定的**最大保存时间**（原文中叫做predefined maximum age）没有被**关联**到的话，那么说明这个物体离开了视频画面，该物体的信息（记录的外观特征和行为特征）将会被删除。

具体有关卡尔曼滤波算法，建议您看看之前我在SORT分析时给出的链接和我的分析。

[周威：【MOT】详解SORT与卡尔曼滤波算法](https://zhuanlan.zhihu.com/p/196622890)[zhuanlan.zhihu.com![图标](https://pic3.zhimg.com/v2-01baa2cec9daa53f061647bd71fa6c2e_120x160.jpg)](https://zhuanlan.zhihu.com/p/196622890)

## 4. DeepSORT中的分配问题

惯例中（类似SORT），解决分配问题使用的是**匈牙利算法**（仅使用**运动特征**计算**代价矩阵**），该算法解决了由滤波算法预测的位置与检测出来的位置间的匹配。DeepSORT中，作者结合了**外观特征**（由小型CNN提取的128维向量）和**运动特征**（卡尔曼滤波预测的结果）来计算**代价矩阵**，从而根据该代价矩阵使用匈牙利算法进行目标的匹配。

1. **运动（motion）特征**

作者使用了马氏距离，用来衡量预测到的卡尔曼滤波状态和新获得的测量值（检测框）之间的距离。

![img](https://pic4.zhimg.com/80/v2-b876d68eecf31e442dcc75696ebb7ef7_1440w.png)公式1

上述公式中 ![[公式]](https://www.zhihu.com/equation?tex=%28y_%7Bi%7D%2CS_%7Bi%7D%29) 表示**第i个追踪分布**（卡尔曼滤波分布）在测量空间上的投影， ![[公式]](https://www.zhihu.com/equation?tex=y_%7Bi%7D) 为均值， ![[公式]](https://www.zhihu.com/equation?tex=S_%7Bi%7D) 为协方差。因为要和**测量值（检测框）**进行距离测算，所以必须转到**同一空间**分布中才能进行。

马氏距离通过测量**卡尔曼滤波器的追踪位置均值**（mean track location）之间的**标准差**与**检测框**来计算状态估计间的**不确定性，**即 ![[公式]](https://www.zhihu.com/equation?tex=d%5E%7B1%7D%28i%2Cj%29) 为第i个追踪分布和第j个检测框之间的马氏距离（不确定度）。

值得注意的是，这里的两个符号含义分别为

- i：追踪的序号
- j：检测框的序号

i，j的含义将在后面的解析中仍然出现。

使用对**马氏距离**设定一定的阈值，可以**排除那些没有关联**的目标。文章中给出的阈值是

> the Mahalanobis distance at a 95% confidence interval computed from the inverse  χ2 distribution.

就是倒卡方分布计算出来的95%置信区间作为阈值。

有关马氏距离的实现，定义在**Tracker类**中可以获得，代码如下：

```python
    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
```

显然当目标运动过程中的**不确定度比较低（马氏距离小**）的时候（也就是满足卡尔曼滤波算法的假设，即所有物体的运动具有一定规律，且没有什么遮挡），那么基于motion特征的方法，即上面提到的方法（可是视为改进的SORT）自然有效。

但是实际情况哪有那么理想，所有仅靠motion特征是不行的了，需要用到appearance特征来弥补不足。

**2.  外观（appearance）特征**

前面我们提到了外观特征提取网络——小型的残差网络。该网络接受reshape的检测框（大小为128x64，针对行人的）内物体作为输入，返回1**28维度的向量**表示。

对于每个检测框（编号为j）内物体 ![[公式]](https://www.zhihu.com/equation?tex=d_%7Bj%7D) ，其128维度的向量设为 ![[公式]](https://www.zhihu.com/equation?tex=r_%7Bj%7D) ，该向量的模长为1，即 ![[公式]](https://www.zhihu.com/equation?tex=%7C%7Cr_%7Bj%7D%7C%7C%3D1) 。这个应该是经过了一个softmax层的原因。

接着作者对每个**目标k**创建了一个gallery，该gallery用来**存储**该目标在不同帧中的**外观特征**（128维向量），论文中用 ![[公式]](https://www.zhihu.com/equation?tex=R_%7Bk%7D) 表示。

注意，这里的**k**的含义是**追踪的目标k**，也就是**object-in-track的序号**。为了区分i和k，我画了个示意图，如下。

<img src="https://pic2.zhimg.com/80/v2-4d9116b43e4885f20435554afec5632d_1440w.jpg" alt="img" style="zoom:25%;" />

作者原论文是这么提到的

![img](https://pic1.zhimg.com/80/v2-2ac7761ff187755322d94c51297a98d8_1440w.jpg)

这里的 ![[公式]](https://www.zhihu.com/equation?tex=R_%7Bk%7D%3D%5C%7B+r_%7Bk%7D%5E%7B%28i%29%7D%5C%7D_%7Bk%3D1%7D%5E%7B%28L_%7Bk%7D%29%7D) 就是gallery，作者限定了 ![[公式]](https://www.zhihu.com/equation?tex=L_%7Bk%7D) 的大小，它最大不超过100，即最多只能存储**目标k**当前时刻**前100帧**中的**目标外观特征**。这里的**i**表示的就是前面提到的**追踪（track）的序号**。

接着在某一时刻，作者获得出检测框（编号为j）的外观特征，记作 ![[公式]](https://www.zhihu.com/equation?tex=r_%7Bj%7D) 。然后求解所有**已知的gallery中的外观特征**与获得的检测框（编号为j）的**外观特征**的**最小余弦距离**。即

![img](https://pic2.zhimg.com/80/v2-a9046dc25cee37ef449d7847aabbefe9_1440w.png)公式2

接着作者对最小余弦距离设定了阈值，来区分关联是否合理，如下

![img](https://pic1.zhimg.com/80/v2-c2307d40e9fec51f60251006a84292ec_1440w.jpg)

**3. 运动（motion）特征与外观（appearance）特征的融合**

motion特征和appearance特征是**相辅相成**的。在DeepSORT中，motion特征（由马氏距离计算获得）**提供了物体定位的可能信息**，这在短期预测中非常有效。

appearance特征（由余弦距离计算获得）可以在目标被**长期遮挡**后，**恢复目标的ID编号**，减少ID切换次数。

为了结合两个特征，作者做了一个简单的**加权运算**。也就是

![img](https://pic4.zhimg.com/80/v2-a3fb10801ee81ea223559d59a469d9ab_1440w.png)公式3

 这里的 ![[公式]](https://www.zhihu.com/equation?tex=d%5E%7B1%7D%28i%2Cj%29) 为马氏距离， ![[公式]](https://www.zhihu.com/equation?tex=d%5E%7B2%7D%28i%2Cj%29) 为余弦距离。 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) 为权重系数。所以当 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda%3D1) 时，那么就是**改进版的SORT**, ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda%3D0) 时，仅仅依靠外观特征进行匹配也是可以进行追踪的。

最后，作者设定了如何判断关联是否匹配的判别总阈值，作者提到

> where we call an **association admissible** if it is **within the gating region of both metrics**

![img](https://pic1.zhimg.com/80/v2-89bc90ca8b342b17f57846f060c3b918_1440w.jpg)公式4

作者将上面提到的两个阈值（分别为马氏距离和余弦距离的阈值）综合到了一起，**联合判断**该某一关联（**association**）是否合理可行的（**admissible）**。

## 4  更多匹配的细节

论文中作者提到了Matching Cascade，该算法流程的伪代码如下：

![img](https://pic4.zhimg.com/80/v2-1d792df5131ec8229f59689db539f62f_1440w.jpg)

**输入**：该算法接受三个输入，分别为

- 追踪的索引集合 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctau) ,![[公式]](https://www.zhihu.com/equation?tex=i%5Cin%5B1%2CN%5D) ，i在前面已经讲过了。
- 当前帧检测框索引的集合 ![[公式]](https://www.zhihu.com/equation?tex=D) , ![[公式]](https://www.zhihu.com/equation?tex=j%5Cin%5B1%2CM%5D) ，j在前面已经讲过了。
- 最大保留时长（Maximum age) ![[公式]](https://www.zhihu.com/equation?tex=A_%7Bmax%7D) 

**步骤1**：根据上面图名为公式3的公式，计算**联合代价矩阵**；

**步骤2**：根据上面图名为公式4的公式，计算gate**矩阵**；

**步骤3**：初始化**匹配列表** ![[公式]](https://www.zhihu.com/equation?tex=M) ，为空

**步骤4**：初始化**非匹配列表** ![[公式]](https://www.zhihu.com/equation?tex=U) ，将 ![[公式]](https://www.zhihu.com/equation?tex=D) 赋予

**步骤5**：循环

- 按照给定的age选择track
- （使用匈牙利算法）计算最小代价匹配时的i,j
- 将满足合适条件的i，j赋值给**匹配列表** ![[公式]](https://www.zhihu.com/equation?tex=M) ，保存
- 重新更新**非匹配列表** ![[公式]](https://www.zhihu.com/equation?tex=U) 

**步骤6**：循环结束，匹配完成

**返回**：**匹配列表** ![[公式]](https://www.zhihu.com/equation?tex=M)和**非匹配列表**![[公式]](https://www.zhihu.com/equation?tex=U)

代码实现如下：

```python
def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections
```

和上面的伪代码一一对应，很清晰。

级联匹配Cascade Match结束后，作者提到

> In a final matching stage, we run **intersection over union association** as proposed in the original SORT algorithm [12] **on the set of unconfirmed and unmatched tracks of age n = 1**.This helps to to account for sudden appearance changes, e.g., due to partial occlusion with static scene geometry, and to increase robustness  against erroneous initialization.

也就是对**刚初始化**的目标等无法确认（匹配）的追踪，因为没**有之前的运动信息和外观信息**，这里我们采用**IOU匹配关联**进行追踪！代码实现如下：

```python
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
```