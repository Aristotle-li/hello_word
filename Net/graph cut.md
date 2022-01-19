图像分割经典算法--《图割》（Graph Cut、Grab Cut-----python实现）

置顶 我的她像朵花 2018-11-12 10:51:21   27126   已收藏 142
分类专栏： 图像分割 经典算法 文章标签： graph cut grab cut python实现 图像分割
版权
1. 算法介绍

Graph Cut（图形切割）应用于计算机视觉领域用来有效的解决各种低级计算机视觉问题，例如图像平滑（image smoothing）、立体应对问题（stereo correspondence problem）、图像分割（image segmentation）等等。此类方法把图像分割问题与图的最小割（min cut）问题相关联，在计算机视觉的很多类似的问题中，最小能量（minimum energy）方案对应解决方案的最大后验估计（maximum posterior estimate）。
友情链接：图像分割经典算法（最小割最大流）
使用Graph Cut的方法可以精确的解决”二进制问题“（binary problem），例如对二进制图像进行去噪。可以用两个以上不同的标签（例如立体对应（stereo correspondence）或灰度图像的去噪）标记像素的问题不能精确解决，但产生的解决方案通常能够接近全局最优的效果。

2. GraphCut

GraphCut利用最小割最大流算法进行图像的分割，可以将图像分割为前景和背景。使用该算法时需要在前景和背景处各画几笔作为输入，算法将建立各个像素点与前景背景相似度的赋权图，并通过求解最小切割区分前景和背景。算法效果图如下：
原始图片

标注图片

分割后的图片

相关论文及python实现代码：

> PAPER:
> 1.Fast approximate energy minimization via graph cuts
> 2.Graph based algorithms for scene reconstruction from two or more views
> 3.What energy functions can be minimized via graph cuts?
> 4.Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D Images
> CODE
> https://github.com/cm-jsw/GraphCut
> 参考项目：https://github.com/NathanZabriskie/GraphCut

3.GrabCut

GrabCut的详细解释参考博客：
图像分割之（三）从Graph Cut到Grab Cut
图像分割之（四）OpenCV的GrabCut函数使用和源码解读

GrabCut
GrabCut是对其的改进版，是迭代的Graph Cut。OpenCV中的GrabCut算法是依据《“GrabCut” - Interactive Foreground Extraction using Iterated Graph Cuts》这篇文章来实现的。该算法利用了图像中的纹理（颜色）信息和边界（反差）信息，只要少量的用户交互操作即可得到比较好的分割结果。

GrabCut优点
（1）只需要在目标外面画一个框，把目标框住，它就可以实现良好的分割效果；

（2）增加额外的用户交互（由用户指定一些像素属于目标），对实现的效果进行优化以得到更好的效果；

（3）它的Border Matting技术会使目标分割边界更加自然和完美。


GrabCut同时存在这一些缺点：1.如果背景比较复杂或者背景和目标相似度很大，那分割的效果不太好；2.由于时迭代的GraphCut，所以速度较慢。

GrabCut和GraphCut的不同点
（1）GraphCut的目标和背景的模型是灰度直方图，GrabCut取代为RGB三通道的混合高斯模型GMM；
（2）GraphCut的能量最小化（分割）是一次达到的，而GrabCut取代为一个不断进行分割估计和模型参数学习的交互迭代过程；
（3）GraphCut需要用户指定目标和背景的一些种子点，但是GrabCut只需要提供背景区域的像素集就可以了。也就是说你只需要框选目标，那么在方框外的像素全部当成背景，这时候就可以对GMM进行建模和完成良好的分割了。即GrabCut允许不完全的标注（incomplete labelling）。

Python实现
代码：

```python
		#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.
This application shows interactive image segmentation using grabcut algorithm.
USAGE :
    python grabcut.py <filename>
README FIRST:
    Two windows will show up, one for input and one for output.
    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.
Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground
Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''

import numpy as np
import cv2
import sys

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness

def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (ix,iy,abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (ix,iy,abs(ix-x),abs(iy-y))
        rect_or_mask = 0
        print " Now press the key 'n' a few times until no further change \n"

    # draw touchup curves

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print "first draw rectangle \n"
        else:
            drawing = True
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

# print documentation
print __doc__

# Loading images
if len(sys.argv) == 2:
    filename = sys.argv[1] # for drawing purposes
else:
    print "No input image given, so loading default image, far.jpg \n"
    print "Correct Usage : python grabcut.py <filename> \n"
    filename = 'far.jpg'

img = cv2.imread(filename)
img2 = img.copy()                               # a copy of original image
mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
output = np.zeros(img.shape,np.uint8)           # output image to be shown

# input and output windows
cv2.namedWindow('output')
cv2.namedWindow('input')
cv2.setMouseCallback('input',onmouse)
cv2.moveWindow('input',img.shape[1]+10,90)

print " Instructions : \n"
print " Draw a rectangle around the object using right mouse button \n"

while(1):

    cv2.imshow('output',output)
    cv2.imshow('input',img)
    k = 0xFF & cv2.waitKey(1)

    # key bindings
    if k == 27:         # esc to exit
        break
    elif k == ord('0'): # BG drawing
        print " mark background regions with left mouse button \n"
        value = DRAW_BG
    elif k == ord('1'): # FG drawing
        print " mark foreground regions with left mouse button \n"
        value = DRAW_FG
    elif k == ord('2'): # PR_BG drawing
        value = DRAW_PR_BG
    elif k == ord('3'): # PR_FG drawing
        value = DRAW_PR_FG
    elif k == ord('s'): # save image
        bar = np.zeros((img.shape[0],5,3),np.uint8)
        res = np.hstack((img2,bar,img,bar,output))
        cv2.imwrite('grabcut_output.png',output)
        cv2.imwrite('grabcut_output_combined.png',res)
        print " Result saved as image \n"
    elif k == ord('r'): # reset everything
        print "resetting \n"
        rect = (0,0,1,1)
        drawing = False
        rectangle = False
        rect_or_mask = 100
        rect_over = False
        value = DRAW_FG
        img = img2.copy()
        mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
        output = np.zeros(img.shape,np.uint8)           # output image to be shown
    elif k == ord('n'): # segment the image
        print """ For finer touchups, mark foreground and background after pressing keys 0-3
        and again press 'n' \n"""
        if (rect_or_mask == 0):         # grabcut with rect
            bgdmodel = np.zeros((1,65),np.float64)
            fgdmodel = np.zeros((1,65),np.float64)
            cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
            rect_or_mask = 1
        elif rect_or_mask == 1:         # grabcut with mask
            bgdmodel = np.zeros((1,65),np.float64)
            fgdmodel = np.zeros((1,65),np.float64)
            cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
    output = cv2.bitwise_and(img2,img2,mask=mask2)

cv2.destroyAllWindows()


```

实现效果：
原始图片、处理过程、处理后的照片分别如下：

![image-20210325154203123](/Users/lishuo/Library/Application Support/typora-user-images/image-20210325154203123.png)

GitHub代码：

https://github.com/downingstreet/GrabCut
https://github.com/Orcuslc/GrabCuthttps://github.com/Orcuslc/GrabCut
Grab Cut论文：
“GrabCut”: interactive foreground extraction using iterated graph cuts

4.参考

基于GraphCuts图割算法的图像分割----OpenCV代码与实现
Graph cut入门学习
图像分割之（三）从Graph Cut到Grab Cut
图像分割之（四）OpenCV的GrabCut函数使用和源码解读
https://github.com/NathanZabriskie/GraphCut
https://github.com/downingstreet/GrabCut
https://github.com/Orcuslc/GrabCuthttps://github.com/Orcuslc/GrabCut

## 图像分割之最小割与最大流算法



**关键字**：图像处理, 最小割, 最大流, 图像分割



------

## 1. 题外话

图的最小割和最大流问题是图论中比较经典的问题，也是各大算法竞赛中经常出现的问题。图像分割中”Graph Cut”、”Grab Cut”等方法都有使用到最小割算法。网上资料介绍了Graph cut和Grab cut中图的构建方法，但对最小割的求解一笔带过。所以萌生了写一篇介绍图的最小割和最大流的博客。

## 2. 关于最小割（min-cut）

假设大家对图论知识已经有一定的了解。如图1所示，是一个有向带权图，共有4个顶点和5条边。每条边上的箭头代表了边的方向，每条边上的数字代表了边的权重。

`G = < V, E >`是图论中对图的表示方式，其中V表示顶点(vertex)所构成的集合，E表示边(edge)所构成的集合。顶点V的集合和边的集合E构成了图G(graph)。

![img](https://imlogm.github.io/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/mincut-maxflow/1.jpg)



图1

那什么是最小割呢？

以图1为例，图1中顶点s表示源点(source)，顶点t表示终点(terminal)，从源点s到终点t共有3条路径：

- s -> a -> t
- s -> b -> t
- s -> a -> b-> t

现在要求剪短图中的某几条边，使得不存在从s到t的路径，并且保证所减的边的权重和最小。相信大家能很快想到解答：剪掉边”s -> a”和边”b -> t”。

剪完以后的图如图2所示。此时，图中已不存在从s到t的路径，且所修剪的边的权重和为：2 + 3 = 5，为所有修剪方式中权重和最小的。

我们把这样的修剪称为最小割。

![img](https://imlogm.github.io/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/mincut-maxflow/2.jpg)



图2

## 3. 关于最大流（max-flow）

什么是最大流呢？

继续以图1为例，假如顶点s源源不断有水流出，边的权重代表该边允许通过的最大水流量，请问顶点t流入的水流量最大是多少？

从顶点s到顶点t的3条路径着手分析，从源点s到终点t共有3条路径：

- s -> a -> t：流量被边”s -> a”限制，最大流量为2
- s -> b -> t：流量被边”b -> t”限制，最大流量为3
- s -> a -> b-> t：边”s -> a”的流量已经被其他路径占满，没有流量

所以，顶点t能够流入的最大水流量为：2 + 3 = 5。

这就是最大流问题。所以，图1的最大流为：2 + 3 = 5。

细心的你可能已经发现：图1的最小割和最大流都为5。是的，经过数学证明可以知道，图的最小割问题可以转换为最大流问题。所以，算法上在处理最小割问题时，往往先转换为最大流问题。

那如何凭直觉解释最小割和最大流存在的这种关系呢？借用[Jecvy博客](https://jecvay.com/2014/11/what-is-min-cut.html)的一句话：1.最大流不可能大于最小割，因为最大流所有的水流都一定经过最小割那些割边，流过的水流怎么可能比水管容量还大呢？ 2.最大流不可能小于最小割，如果小，那么说明水管容量没有物尽其用，可以继续加大水流。

## 4. 最大流的求解

现在我们已经知道了最小割和最大流之间的关系，也理解了最小割问题往往先转换为最大流问题进行求解。那么，如何有效求解最大流问题呢？

一种是Ford-Fulkerson算法，参见博客：[装逼之二 最小割与最大流（mincut & maxflow）-carrotsss](https://blog.csdn.net/a519781181/article/details/51908303)

另一种是Yuri Boykov的优化算法，参见博客：[CV | Max Flow / Min Cut 最大流最小割算法学习-iLOVEJohnny的博客](https://blog.csdn.net/ilovejohnny/article/details/52016742)

事实上，我并不打算自己再把最大流算法讲解一遍，因为最大流算法很容易在搜索引擎上搜索到。我真正要讲的是下面的部分，关于如何把最大流的结果转换到最小割。

## 5. 如何把最大流的结果转换为最小割

网上介绍最小割和最大流往往介绍完最大流的求解方法后就不继续讲解了，我上面贴出的两篇博客都存在这个问题。这样大家肯定会有个疑惑：如何把最大流的结果转换为最小割。

我以上面贴出的Ford-Fulkerson算法的博客的结果为例讲解下如何转换。如图3是最大流求解算法的最终结果。边上的数字“@/#”表示这条边最大流量为#，在最大流求解算法中该边所使用的流量为@。比如边“13/15”表示该边最大能容纳的流量为15，但在最大流求解算法中使用到的流量为13。

![img](https://imlogm.github.io/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/mincut-maxflow/3.png)



图3

我们把流量已经占满的所有边去掉，得到图4：

![img](https://imlogm.github.io/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/mincut-maxflow/4.png)



图4

此时，在图4中，以顶点s为起点进行图遍历（遍历的方法可以选择BFS广度优先遍历）。遍历结束后，可以得到遍历经过的所有点：S、B、C、D。

有些没学过图遍历的小伙伴可以这样理解：从顶点s出发，最多能经过哪些点。那么，很显然，最多能经过S、B、C、D这几个点。只不过人脑回答这个问题比较简单，但计算机需要特定的算法来解答，也就是上一段所说的”图遍历算法”。

这样，把S、B、C、D构成一个子图（图4中紫色部分），其他的点A、E、F、t构成另一个子图（图4中黄色部分）。连接两个子图的边有两种情况：

- 已被占满的前向边：s -> A， B -> E， D -> t
- 没有流量的反向边：A -> B， E -> D

其中“已被占满的前向边”就是我们要求的最小割。对于图4来说，就是”s -> A”、”B -> E”、”D -> t”这3条边。

## 6. 如何将最小割方法应用到图像分割中

写这篇有关最小割的博客，其实是为了给下一篇博客做铺垫。下一篇博客将介绍Graph cut、Grab cut等算法是如何利用最小割来实现图像分割的。

[ 图像处理](https://imlogm.github.io/tags/图像处理/) [ 最小割](https://imlogm.github.io/tags/最小割/) [ 最大流](https://imlogm.github.io/tags/最大流/) [ 图像分割](https://imlogm.github.io/tags/图像分割/)

[奥卡姆剃刀和没有免费的午餐定理](https://imlogm.github.io/机器学习/occam-razor-NFL/)



