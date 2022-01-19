# 人工智能-OpenCV+Python实现人脸识别（视频人脸检测）

[![人工智能研究所](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-f2a130a5f29879178c8c741a7ef4c613_xs.jpg)](https://www.zhihu.com/people/powers415)

[人工智能研究所](https://www.zhihu.com/people/powers415)

上期文章我们分享了opencv识别图片中的人脸，[OpenCV图片人脸检测](https://www.toutiao.com/i6688649730194407944/?group_id=6688649730194407944)，本期我们分享一下如何从视频中检测到人脸

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-ab30d646e732df2c7c8345ad2bda1470_720w.jpg)


**视频人脸检测**
OpenCV打开摄像头特别简单，只需要如下一句代码
capture = cv2.VideoCapture(0) # 打开摄像头
打开摄像头后，我们使用如下一句代码，来获取视频中的图片（每帧图片）
ret, frame = capture.read() # 读取
有了图片我们就可以按照图片的识别方式来检测人脸了
有了以上的2句代码，再加上上期的图片识别，就可以从视频中检测人脸了
**完整代码：**

```text
import cv2
capture = cv2.VideoCapture(0) # 打开摄像头
face = cv2.CascadeClassifier(r'D:\Program Files (x86)\Anaconda3\pkgs\
libopencv-3.4.1-h875b8b8_3\Library\etc\haarcascades\
haarcascade_frontalface_alt.xml') # 导入人脸模型
cv2.namedWindow('摄像头') # 获取摄像头画面
while True:
ret, frame = capture.read() # 读取视频图片
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # 灰度
faces = face.detectMultiScale(gray,1.1,3,0,(100,100))
for (x, y, w, h) in faces: # 5个参数，一个参数图片 ，2 坐标原点，3 识别大小，4，颜色5，线宽
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('摄像头', frame) # 显示
if cv2.waitKey(5) & 0xFF == ord('q'):
break
capture.release() # 释放资源
cv2.destroyAllWindows() # 关闭窗口
opencv中人脸检测使用的是 detectMultiScale函数，小编使用手机播放一段视频，截取了几张人脸检测的图片
detectMultiScale(
const Mat& image,
CV_OUT vector<Rect>& objects,
double scaleFactor = 1.1,
int minNeighbors = 3,
int flags = 0,
Size minSize = Size(),
Size maxSize = Size()
);
```

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-6d67126d61bd72a89dc0dc0aac0727d3_720w.jpg)

识别视频中的人脸
函数介绍：
**参数1：**image--待检测图片，一般为灰度图像加快检测速度；
**参数2：**objects--被检测物体的矩形框向量组；
**参数3：**scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
**参数4：**minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。
如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
这种设定值一般用在用户自定义对检测结果的组合程序上；
**参数5：**flags--要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为
CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，
因此这些区域通常不会是人脸所在区域；
**参数6、7：**minSize和maxSize用来限制得到的目标区域的范围。

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-3fd92353615d9464f8774a5dbfdcced3_720w.jpg)

识别视频中的人脸
OpenCV作为对象检测的第三方库，其强大之处在于对象的检测，Dlib出现后，由于在人脸检测方面的准确度，得到了大家了认可，下期我们分享一下，如何使用Dlib来进行人脸的检测

![img](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-f530629b0fa735c2d95865fc06d27076_720w.jpg)

使用dlib检测的68个特征点





发布于 2020-06-06 13:35

[OpenCV](https://www.zhihu.com/topic/19587715)

[人脸识别](https://www.zhihu.com/topic/19559196)

[人工智能](https://www.zhihu.com/topic/19551275)

### 推荐阅读



[OpenCV人脸识别之二：模型训练欢迎访问http://www.leadai.org 作者：刘潇龙在该系列第一篇 《OpenCV人脸识别之一：数据收集和预处理》文章中，已经下载了ORL人脸数据库，并且为了识别自己的人脸写了一个拍照程序自拍。之…人工智能L...发表于人工智能L...](https://zhuanlan.zhihu.com/p/31290321)[dlib 使用OpenCV，Python和深度学习进行人脸识别 源代码dlib 使用OpenCV，Python和深度学习进行人脸识别 源代码请看原文 链接 https://hotdog29.com/?p=595 在 2019年7月7日 上张贴 由 hotdog发表回复 dlib 在今天的博客文章中，您将学习如何使用…热狗](https://zhuanlan.zhihu.com/p/73189852)[使用python3.7和opencv4.1来实现人脸识别和人脸特征比对以及模型训练OpenCV4.1已经发布将近一年了，其人脸识别速度和性能有了一定的提高，这里我们使用opencv来做一个实时活体面部识别的demo 首先安装一些依赖的库 pip install opencv-python pip install ope…刘悦的技术...发表于刘悦的技术...](https://zhuanlan.zhihu.com/p/127039696)[![人脸识别学习笔记二：进阶篇](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-5c41eb97df0139b90ca62fc6693cb378_250x0.jpg)人脸识别学习笔记二：进阶篇前倩欠潜钱发表于人脸识别](https://zhuanlan.zhihu.com/p/158947937)

## 还没有评论

评论区功能升级中