# 目标检测——选择性搜索（selective search）

**该文翻译整理自：[selective search for object detection（c++ / python）](https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/)**

 

**一、目标检测 VS 目标识别**

目标识别（objec  recognition）是指明一幅输入图像中包含那类目标。其输入为一幅图像，输出是该图像中的目标属于哪个类别（class  probability）。而目标检测（object  detection）除了要告诉输入图像中包含了哪类目前外，还要框出该目标的具体位置（bounding boxes）。

在目标检测时，为了定位到目标的具体位置，通常会把图像分成许多子块（sub-regions /  patches），然后把子块作为输入，送到目标识别的模型中。分子块的最直接方法叫滑动窗口法（sliding window  approach）。滑动窗口的方法就是按照子块的大小在整幅图像上穷举所有子图像块。这种方法产生的数据量想想都头大。和滑动窗口法相对的是另外一类基于区域（region proposal）的方法。selective search就是其中之一！

**二、selective search算法流程**

![img](https://img-blog.csdn.net/20171206151215524?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3VveXVuZmVpMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

step0：生成区域集R，具体参见论文[《Efficient Graph-Based Image Segmentation》](http://blog.csdn.net/guoyunfei20/article/details/78727972)

step1：计算区域集R里每个相邻区域的相似度S={s1,s2,…} 
 step2：找出相似度最高的两个区域，将其合并为新集，添加进R 
 step3：从S中移除所有与step2中有关的子集 
 step4：计算新集与所有子集的相似度 
 step5：跳至step2，直至S为空

 

**三、相似度计算**

论文考虑了颜色、纹理、尺寸和空间交叠这4个参数。

3.1、颜色相似度（color similarity）
 将色彩空间转为HSV，每个通道下以bins=25计算直方图，这样每个区域的颜色直方图有25*3=75个区间。 对直方图除以区域尺寸做归一化后使用下式计算相似度：

![img](https://img-blog.csdn.net/20171206151946328?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3VveXVuZmVpMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

3.2、纹理相似度（texture similarity）

 论文采用方差为1的高斯分布在8个方向做梯度统计，然后将统计结果（尺寸与区域大小一致）以bins=10计算直方图。直方图区间数为8*3*10=240（使用RGB色彩空间）。

![img](https://www.learnopencv.com/wp-content/ql-cache/quicklatex.com-169d419080f56b69f9645cd13ee5b0ac_l3.png)

其中，![img](https://www.learnopencv.com/wp-content/ql-cache/quicklatex.com-a9283008bc26743f78b7ac5644fa42d7_l3.png)是直方图中第![img](https://www.learnopencv.com/wp-content/ql-cache/quicklatex.com-19a1201960f1c720275a7fd8ab39ea27_l3.png)个bin的值。

3.3、尺寸相似度（size similarity）

![img](https://www.learnopencv.com/wp-content/ql-cache/quicklatex.com-ed6bd32a9661aa84228d1ca1c75f5d29_l3.png)

保证合并操作的尺度较为均匀，避免一个大区域陆续“吃掉”其他小区域。

例：设有区域a-b-c-d-e-f-g-h。较好的合并方式是：ab-cd-ef-gh -> abcd-efgh ->  abcdefgh。 不好的合并方法是：ab-c-d-e-f-g-h ->abcd-e-f-g-h ->abcdef-gh -> abcdefgh。

3.4、交叠相似度（shape compatibility measure）

![img](https://www.learnopencv.com/wp-content/ql-cache/quicklatex.com-9a3fdf638488b3c77915b9b83bf2f3e1_l3.png)

![img](https://img-blog.csdn.net/20171206153459399?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3VveXVuZmVpMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

3.5、最终的相似度

![img](https://www.learnopencv.com/wp-content/ql-cache/quicklatex.com-67a3c5c3f45a9407ee513056c759f095_l3.png)

 

**四、OpenCV 3.3 实现了selective search**

在OpenCV的contrib模块中实现了selective search算法。类定义为：

 

```
cv::ximgproc::segmentation::SelectiveSearchSegmentation
```

 

 

举例：

```cpp
#include "opencv2/ximgproc/segmentation.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <ctime>
 
using namespace cv;
using namespace cv::ximgproc::segmentation;
 
static void help() {
    std::cout << std::endl <<
    "Usage:" << std::endl <<
    "./ssearch input_image (f|q)" << std::endl <<
    "f=fast, q=quality" << std::endl <<
    "Use l to display less rects, m to display more rects, q to quit" << std::endl;
}
 
 
int main(int argc, char** argv) {
    // If image path and f/q is not passed as command
    // line arguments, quit and display help message
    if (argc < 3) {
        help();
        return -1;
    }
 
    // speed-up using multithreads
    // void cv::setUseOptimized(bool onoff), Enables or disables the optimized code.
    setUseOptimized(true);
    setNumThreads(4);
 
    // read image
    Mat im = imread(argv[1]);
    // resize image
    int newHeight = 200;
    int newWidth = im.cols*newHeight/im.rows;
    resize(im, im, Size(newWidth, newHeight));
 
    // create Selective Search Segmentation Object using default parameters
    Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
    // set input image on which we will run segmentation
    ss->setBaseImage(im);
 
    // Switch to fast but low recall Selective Search method
    if (argv[2][0] == 'f') {
        ss->switchToSelectiveSearchFast();
    }
    // Switch to high recall but slow Selective Search method
    else if (argv[2][0] == 'q') {
        ss->switchToSelectiveSearchQuality();
    } 
    // if argument is neither f nor q print help message
    else {
        help();
        return -2;
    }
 
    // run selective search segmentation on input image
    std::vector<Rect> rects;
    ss->process(rects);
    std::cout << "Total Number of Region Proposals: " << rects.size() << std::endl;
 
    // number of region proposals to show
    int numShowRects = 100;
    // increment to increase/decrease total number of reason proposals to be shown
    int increment = 50;
 
    while(1) {
        // create a copy of original image
        Mat imOut = im.clone();
 
        // itereate over all the region proposals
        for(int i = 0; i < rects.size(); i++) {
            if (i < numShowRects) {
                rectangle(imOut, rects[i], Scalar(0, 255, 0));
            }
            else {
                break;
            }
        }
 
        // show output
        imshow("Output", imOut);
 
        // record key press
        int k = waitKey();
 
        // m is pressed
        if (k == 109) {
            // increase total number of rectangles to show by increment
            numShowRects += increment;
        }
        // l is pressed
        else if (k == 108 && numShowRects > increment) {
            // decrease total number of rectangles to show by increment
            numShowRects -= increment;
        }
        // q is pressed
        else if (k == 113) {
            break;
        }
    }
    return 0;
}
```


  python代码

```python
#!/usr/bin/env python
'''
Usage:
    ./ssearch.py input_image (f|q)
    f=fast, q=quality
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
'''

import sys
import cv2

if __name__ == '__main__':
    # If image path and f/q is not passed as command
    # line arguments, quit and display help message
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);

    # read image
    im = cv2.imread(sys.argv[1])
    # resize image
    newHeight = 200
    newWidth = int(im.shape[1]*200/im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))    

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    # Switch to fast but low recall Selective Search method
    if (sys.argv[2] == 'f'):
        ss.switchToSelectiveSearchFast()

    # Switch to high recall but slow Selective Search method
    elif (sys.argv[2] == 'q'):
        ss.switchToSelectiveSearchQuality()
    # if argument is neither f nor q print help message
    else:
        print(__doc__)
        sys.exit(1)

    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))
    
    # number of region proposals to show
    numShowRects = 100
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 50

    while True:
        # create a copy of original image
        imOut = im.copy()

        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        # show output
        cv2.imshow("Output", imOut)

        # record key press
        k = cv2.waitKey(0) & 0xFF

        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
        # q is pressed
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()
```

上边代码git地址：https://code.csdn.net/guoyunfei20/selective_search_opencv_demo.git（运行需要安装OpenCV3.0以上 + contrib）





# 基于图的图像分割（Graph-Based Image Segmentation）

**一、介绍**

基于图的图像分割（Graph-Based Image Segmentation），论文《Efficient Graph-Based  Image Segmentation》，P. Felzenszwalb, D. Huttenlocher，International  Journal of Computer Vision, Vol. 59, No. 2, September 2004

论文下载和论文提供的C++代码在[这里](http://cs.brown.edu/~pff/segment/)。
 

Graph-Based Segmentation是经典的图像分割算法，其作者Felzenszwalb也是提出DPM（Deformable Parts Model）算法的大牛。

Graph-Based  Segmentation算法是基于图的贪心聚类算法，实现简单，速度比较快，精度也还行。不过，目前直接用它做分割的应该比较少，很多算法用它作垫脚石，比如Object Propose的开山之作《Segmentation as Selective Search for Object  Recognition》就用它来产生过分割（over segmentation）。
 

**二、图的基本概念**

因为该算法是将图像用加权图抽象化表示，所以补充图的一些基本概念。

1、图

是由顶点集V（vertices）和边集E（edges）组成，表示为***G=(V, E)***，顶点***v∈V***，在论文即为单个的像素点，连接一对顶点的边***(vi, vj)***具有权重***w(vi, vj)***，本文中的意义为顶点之间的不相似度（dissimilarity），所用的是无向图。

2、树

特殊的图，图中任意两个顶点，都有路径相连接，但是没有回路。如下图中加粗的边所连接而成的图。如果看成一团乱连的珠子，只保留树中的珠子和连线，那么随便选个珠子，都能把这棵树中所有的珠子都提起来。

如果顶点i和h这条边也保留下来，那么顶点h,i,c,f,g就构成了一个回路。
 

![img](https://img-blog.csdn.net/20171206104959304?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3VveXVuZmVpMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
 

3、最小生成树（minimum spanning tree）

特殊的树，给定需要连接的顶点，选择边权之和最小的树。
 

论文中，初始化时每一个像素点都是一个顶点，然后逐渐合并得到一个区域，确切地说是连接这个区域中的像素点的一个MST。如下图，棕色圆圈为顶点，线段为边，合并棕色顶点所生成的MST，对应的就是一个分割区域。分割后的结果其实就是森林。
 

![img](https://img-blog.csdn.net/20171206105413116?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3VveXVuZmVpMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
 


 

**三、相似性**

既然是聚类算法，那应该依据何种规则判定何时该合二为一，何时该继续划清界限呢？对于孤立的两个像素点，所不同的是灰度值，自然就用灰度的距离来衡量两点的相似性，本文中是使用RGB的距离，即
 

![img](http://images.cnitblog.com/blog/460184/201407/212139044793634.png)
 

当然也可以用perceptually uniform的Luv或者Lab色彩空间，对于灰度图像就只能使用亮度值了，此外，还可以先使用纹理特征滤波，再计算距离，比如先做Census Transform再计算Hamming distance距离。
 

**四、全局阈值 >> 自适应阈值，区域的类内差异、类间差异**

上面提到应该用亮度值之差来衡量两个像素点之间的差异性。对于两个区域（子图）或者一个区域和一个像素点的相似性，最简单的方法即只考虑连接二者的边的不相似度。如下图，已经形成了棕色和绿色两个区域，现在通过紫色边来判断这两个区域是否合并。那么我们就可以设定一个阈值，当两个像素之间的差异（即不相似度）小于该值时，合二为一。迭代合并，最终就会合并成一个个区域，效果类似于区域生长：星星之火，可以燎原。
 

![img](https://img-blog.csdn.net/20171206110045764?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3VveXVuZmVpMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
 

举例说明：

![img](https://img-blog.csdn.net/20171206111022215?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3VveXVuZmVpMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
 

对于上右图，显然应该聚成上左图所示的3类：高频区h,斜坡区s,平坦区p。

如果我们设置一个全局阈值，那么如果h区要合并成一块的话，那么该阈值要选很大，但是那样就会把p和s区域也包含进来，分割结果太粗。如果以p为参考，那么阈值应该选特别小的值，那样的话p区是会合并成一块，但是h区就会合并成特别特别多的小块，如同一面支离破碎的镜子，分割结果太细。显然，全局阈值并不合适，那么自然就得用自适应阈值。对于p区该阈值要特别小，s区稍大，h区巨大。

先来两个定义，原文依据这两个附加信息来得到自适应阈值。
 

一个区域内的类内差异***Int(C)***：

![img](http://images.cnitblog.com/blog/460184/201407/212139059799375.png)
 

可以近似理解为一个区域内部最大的亮度差异值，定义是MST中不相似度最大的一条边。
 

俩个区域的类间差异***Diff(C1, C2)***：

![img](http://images.cnitblog.com/blog/460184/201407/212139078857059.png)
 

即连接两个区域所有边中，不相似度最小的边的不相似度，也就是两个区域最相似的地方的不相似度。
 

直观的判断，当：

![img](http://images.cnitblog.com/blog/460184/201407/212139082135961.png)
 

时，两个区域应当合并！

**五、算法步骤**

1、计算每一个像素点与其8邻域或4邻域的不相似度。

![img](https://img-blog.csdn.net/20171206112900548?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3VveXVuZmVpMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
 

如上图，实线为只计算4领域，加上虚线就是计算8邻域，由于是无向图，按照从左到右，从上到下的顺序计算的话，只需要计算右图中灰色的线即可。
 

2、将边按照不相似度non-decreasing排列（从小到大）排序得到***e1, e2, ..., en***。

3、选择***ei***

4、对当前选择的边**ej**（vi和vj不属于一个区域）进行合并判断。设其所连接的顶点为***(vi, vj)***，

if 不相似度小于二者内部不相似度：

5、更新阈值以及类标号

else：

6、如果i < n，则按照排好的顺序，选择下一条边转到Step 4，否则结束。

**
** 

**六、论文提供的代码**

打开本博文最开始的连接，进入论文网站，下载C++代码。下载后，make编译程序。命令行运行格式：



```cpp
/********************************************
sigma  对原图像进行高斯滤波去噪
k      控制合并后的区域的数量
min:   后处理参数，分割后会有很多小区域，当区域像素点的个数小于min时，选择与其差异最小的区域合并
input  输入图像（PPM格式）
output 输出图像（PPM格式）
sigma: Used to smooth the input image before segmenting it.
k:     Value for the threshold function.
min:   Minimum component size enforced by post-processing.
input: Input image.
output:Output image.
Typical parameters are sigma = 0.5, k = 500, min = 20.
Larger values for k result in larger components in the result.
*/
./segment sigma k min input output
```


**七、OpenCV3.3 cv::ximgproc::segmentation::GraphSegmentation类**





```cpp
/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc/segmentation.hpp
```





- # 1 前言

在目标检测时，为了定位到目标的具体位置，通常会把图像分成许多子块（sub-regions / patches），然后把子块作为输入，送到目标识别的模型中。selective search就是一种选择子块的启发式方法。

![img](https://img-blog.csdn.net/20170621153048269?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMjgxMzI1OTE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

------

------

- # 2 Selective Search算法

主要思路：输入一张图片，首先通过图像分割的方法（如大名鼎鼎的felzenszwalb算法）获得很多小的区域，然后对这些小的区域不断进行合并，一直到无法合并为止。此时这些原始的小区域和合并得到的区域的就是我们得到的bounding box.

![img](https://img-blog.csdn.net/2018031720132095?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L1NtYWxsX011bmljaA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![img](https://img-blog.csdn.net/20170621171633599?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMjgxMzI1OTE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

算法分为如下几个大步：

\1. 生成原始的区域集R（利用felzenszwalb算法）

\2. 计算区域集R里每个相邻区域的相似度S={s1,s2,…} 

\3. 找出相似度最高的两个区域，将其合并为新集，添加进R 

\4. 从S中移除所有与第3步中有关的子集 

\5. 计算新集与所有子集的相似度 

6.跳至第三步，不断循环，合并，直至S为空（到不能再合并时为止）

 

------

 

- # 3 Python源码分析

Github上有一个选择性搜索的简单实现 --[ selectivesearch](https://github.com/AlpacaDB/selectivesearch) ，可以帮助大家理解

 

```python
import skimage.data
import selectivesearch
 
img = skimage.data.astronaut()
img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
regions[:10]
=>
[{'labels': [0.0], 'rect': (0, 0, 15, 24), 'size': 260},
 {'labels': [1.0], 'rect': (13, 0, 1, 12), 'size': 23},
 {'labels': [2.0], 'rect': (0, 15, 15, 11), 'size': 30},
 {'labels': [3.0], 'rect': (15, 14, 0, 0), 'size': 1},
 {'labels': [4.0], 'rect': (0, 0, 61, 153), 'size': 4927},
 {'labels': [5.0], 'rect': (0, 12, 61, 142), 'size': 177},
 {'labels': [6.0], 'rect': (7, 54, 6, 17), 'size': 8},
 {'labels': [7.0], 'rect': (28, 50, 18, 32), 'size': 22},
 {'labels': [8.0], 'rect': (2, 99, 7, 24), 'size': 24},
 {'labels': [9.0], 'rect': (14, 118, 79, 117), 'size': 4008}]
```

1. 用户生成原始区域集的函数，其中用到了felzenszwalb图像分割算法。每一个区域都有一个编号，将编号并入图片中，方便后面的操作

```python
def _generate_segments(im_orig, scale, sigma, min_size):
    """
        segment smallest regions by the algorithm of Felzenswalb and
        Huttenlocher
    """
 
    # open the Image
    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,
        min_size=min_size)
 
    # merge mask channel to the image as a 4th channel
    im_orig = numpy.append(
        im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask
 
    return im_orig
```

2. 计算两个区域的相似度
    论文中考虑了四种相似度 -- 颜色，纹理，尺寸，以及交叠。

其中颜色和纹理相似度，通过获取两个区域的直方图的交集，来判断相似度。

最后的相似度是四种相似度的加和。

```python
def _sim_colour(r1, r2):
    """
        calculate the sum of histogram intersection of colour
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])
 
 
def _sim_texture(r1, r2):
    """
        calculate the sum of histogram intersection of texture
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])
 
 
def _sim_size(r1, r2, imsize):
    """
        calculate the size similarity over the image
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize
 
 
def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize
 
 
def _calc_sim(r1, r2, imsize):
    return (_sim_colour(r1, r2) + _sim_texture(r1, r2)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))
```

3. 用于计算颜色和纹理的直方图的函数

颜色直方图：将色彩空间转为HSV，每个通道下以bins=25计算直方图，这样每个区域的颜色直方图有25*3=75个区间。 对直方图除以区域尺寸做归一化后使用下式计算相似度：

![img](https://img-blog.csdn.net/20171206151946328?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3VveXVuZmVpMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

纹理相似度：论文采用方差为1的高斯分布在8个方向做梯度统计，然后将统计结果（尺寸与区域大小一致）以bins=10计算直方图。直方图区间数为8*3*10=240（使用RGB色彩空间）。这里是用了LBP（local binary pattern）获取纹理特征，建立直方图，其余相同

![img](https://www.learnopencv.com/wp-content/ql-cache/quicklatex.com-169d419080f56b69f9645cd13ee5b0ac_l3.png)

其中，![img](https://www.learnopencv.com/wp-content/ql-cache/quicklatex.com-a9283008bc26743f78b7ac5644fa42d7_l3.png)是直方图中第![img](https://www.learnopencv.com/wp-content/ql-cache/quicklatex.com-19a1201960f1c720275a7fd8ab39ea27_l3.png)个bin的值。

```python
def _calc_colour_hist(img):
    """
        calculate colour histogram for each region
        the size of output histogram will be BINS * COLOUR_CHANNELS(3)
        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
        extract HSV
    """
 
    BINS = 25
    hist = numpy.array([])
 
    for colour_channel in (0, 1, 2):
 
        # extracting one colour channel
        c = img[:, colour_channel]
 
        # calculate histogram for each colour and join to the result
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(c, BINS, (0.0, 255.0))[0]])
 
    # L1 normalize
    hist = hist / len(img)
 
    return hist
 
 
def _calc_texture_gradient(img):
    """
        calculate texture gradient for entire image
        The original SelectiveSearch algorithm proposed Gaussian derivative
        for 8 orientations, but we use LBP instead.
        output will be [height(*)][width(*)]
    """
    ret = numpy.zeros((img.shape[0], img.shape[1], img.shape[2]))
 
    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)
 
    return ret
 
 
def _calc_texture_hist(img):
    """
        calculate texture histogram for each region
        calculate the histogram of gradient for each colours
        the size of output histogram will be
            BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """
    BINS = 10
 
    hist = numpy.array([])
 
    for colour_channel in (0, 1, 2):
 
        # mask by the colour channel
        fd = img[:, colour_channel]
 
        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(fd, BINS, (0.0, 1.0))[0]])
 
    # L1 Normalize
    hist = hist / len(img)
 
    return hist
 
```

4. 提取区域的尺寸，颜色和纹理特征

```python
def _extract_regions(img):
 
    R = {}
 
    # get hsv image
    hsv = skimage.color.rgb2hsv(img[:, :, :3])
 
    # pass 1: count pixel positions
    for y, i in enumerate(img):
 
        for x, (r, g, b, l) in enumerate(i):
 
            # initialize a new region
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}
 
            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y
 
    # pass 2: calculate texture gradient
    tex_grad = _calc_texture_gradient(img)
 
    # pass 3: calculate colour histogram of each region
    for k, v in list(R.items()):
 
        # colour histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)
 
        # texture histogram
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])
 
    return R

```

5. 找邻居 -- 通过计算每个区域与其余的所有区域是否有相交，来判断是不是邻居

```python
def _extract_neighbours(regions):
 
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False
 
    R = list(regions.items())
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))
 
    return neighbours
```

\6. 合并两个区域的函数

```python
def _merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt
```

7. 主函数 -- Selective Search

scale：图像分割的集群程度。值越大，意味集群程度越高，分割的越少，获得子区域越大。默认为1

signa: 图像分割前，会先对原图像进行高斯滤波去噪，sigma即为高斯核的大小。默认为0.8

min_size : 最小的区域像素点个数。当小于此值时，图像分割的计算就停止，默认为20

每次选出相似度最高的一组区域（如编号为100和120的区域），进行合并，得到新的区域（如编号为300）。然后计算新的区域300与区域100的所有邻居和区域120的所有邻居的相似度，加入区域集S。不断循环，知道S为空，此时最后只剩下一个区域，而且它的像素数会非常大，接近原始图片的像素数，因此无法继续合并。最后退出程序。

```python
def selective_search(
        im_orig, scale=1.0, sigma=0.8, min_size=50):
    '''Selective Search
    Parameters
    ----------
        im_orig : ndarray
            Input image
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int
            Minimum component size for felzenszwalb segmentation.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''
    assert im_orig.shape[2] == 3, "3ch image is expected"
 
    # load image and get smallest regions
    # region label is stored in the 4th value of each pixel [r,g,b,(region)]
    img = _generate_segments(im_orig, scale, sigma, min_size)
 
    if img is None:
        return None, {}
 
    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img)
 
    # extract neighbouring information
    neighbours = _extract_neighbours(R)
 
    # calculate initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = _calc_sim(ar, br, imsize)
 
    # hierarchal search
    while S != {}:
 
        # get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]
 
        # merge corresponding regions
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])
 
        # mark similarities for regions to be removed
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)
 
        # remove old similarities of related regions
        for k in key_to_delete:
            del S[k]
 
        # calculate similarity set with the new region
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)
 
    regions = []
    for k, r in list(R.items()):
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })
 
    return img, regions
```