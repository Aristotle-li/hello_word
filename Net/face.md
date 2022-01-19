# 小白教程：树莓派3B+onnxruntime+scrfd+flask实现公网人脸检测系统

**前情提要；**最近出了一个新的人脸检测框架scrfd，scrfd的论文在5月10日挂在了阿凯上，感兴趣的同学们可以去看一看

[https://arxiv.org/abs/2105.04714arxiv.org/abs/2105.04714](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2105.04714)

新出的scrfd旨在实现模型效能的极致均衡：（_Extensive experiments conducted on WIDER FACE demonstrate the state-of-the-art efficiency-accuracy trade-off for the proposed \\scrfd family across a wide range of compute regimes_，论文如是说）

对比实验似乎也在说明这一点，最小的scrfd-0.5GF检测速度要比之前提出Retinaface-mb0.25快两倍，检测性能也实现全方位的突破：

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-f4548de61bb8d7ba2a417121a94af3f5_b.jpg)  

但具体效果如何，还是需要down下源码run一遍，才能得以验证。

**1、**inter卡跑库测试效果
------------------

实际效果如何，还是需要通过git clone下来测试才行，拉取官方代码；

    git clone https://github.com/deepinsight/insightface.git

将scrfd的源码单独提出来，其他删除即可，不影响跑库

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-a7a470834a2d98b98e40bcf6a779a833_b.jpg)  

确保文件齐全，安装环境所需库（具体写在requirements文件夹中），在Inter Core i5-4210M处理器上进行测试，测试结果如下：

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-d56fe6dae065db77a30348cf52d99eb8_b.jpg)  

以下几点补充：

*   测试的单帧Infer Time=推理一分钟/该分钟内处理的帧数
*   实际生产中其实并不需要这么高的input\_size（主测2-3m内的检测性能）
*   本次测试使用的是转化后的onnx模型（实在是不想调用其他杂七杂八的库）
*   以上测试均为CPU下进行，想测GPU的请自行安装onnxruntime-gpu

不同input\_size的耗时如下（以500m模型为例）：

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-f5078ad41520305315df99e1ee03af02_b.jpg)  

使用500m模型在input\_size=320\*320的条件下推理高密集人脸图片：

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-8516057c30b26681fa2f0f6988f4de69_b.jpg)  

2、动态尺寸onnx模型提取
--------------

*   **安装mmcv-full**

具体安装方法可参考mm的官网:

[](https://link.zhihu.com/?target=https%3A//github.com/open-mmlab/mmcv)

    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html

补充踩过的几个坑

*   博主的cuda=10.2，pytorch=1.6.0，根据官网给的命令行装，一般不会出错
*   实测window下编译的mmcv-full会缺少库，且1.3.4以上版本window安装不了
*   一定要安装最新的mmcv-full，不然会缺少一些函数接口，博主的版本是1.3.5
*   不要贪方便安装mmcv的精简版本，缺少库

  

*   **安装和编译mmdet**

    pip install -r requirements/build.txt
    pip install -v -e .# or "python setup.py develop"

完成后查看下list有没有这三个库：

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-67b0e1ff60383cb130b5fc7e4a0fe027_b.png)  

如果有，就可以开始提取onnx模型，官方给出的baseline如下：

[](https://link.zhihu.com/?target=https%3A//github.com/deepinsight/insightface/tree/master/detection/scrfd%23pretrained-models)

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-7207dfa8e77a73c84fe3ba4ee25423b8_b.jpg)  

看你所需的模型，点击download即可下载，使用以下命令导出onnx模型：

    python tools/scrfd2onnx.py configs/scrfd/scrfd_500m.py weights/scrfd_500m.pth --shape -1 -1 --input-img face.jpg

其中

*   configs里面存放模型参数信息，请与model配对好
*   进行动态尺寸的提取时请将shape指定为-1,
*   检验图片随意，建议包含多张人脸的图片
*   提取后的onnx可以用onnxsim刷一遍
*   包含关键点的bnkps模型不支持动态尺寸输入，使用固定尺寸进行提取

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-32a81ae30428d3b0416eb744562c363d_b.jpg)  

3、flask web实现推流
---------------

这里有两种公网推流的方式：

①使用云服务器作为中介服务器

②购买内网穿刺所用的域名

两种方式都尝试过

第一种使用的是那种9.9块包月的学生云资源，带宽和处理算力远达不到模型所需的吞吐率要求，这样造成了严重延迟，体验感贼差（还是软妹币的问题。。。。）

第二种通过软件进行内网穿刺，具体网上的资料有很多（可参考引用【2】，讲得很详细），优点在于省时省力还能白嫖

无论是哪种，思想无非就是借用公网IP来进行不同局域网下的互通

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-e9b23df695d155cfe42f976d19d489cd_b.jpg)  

进行内网穿刺后，就可以使用flask将检测后的帧推到公网上，代码如下：

    from flask import Flask, render_template, Response
    import argparse
    from tools.scrfd import *
    import datetime
    import cv2
    
    class VideoCamera(object):
        def __init__(self):
            # 通过opencv获取实时视频流
            self.video = cv2.VideoCapture(0)
        def __del__(self):
            self.video.release()
        def get_frame(self):
            success, image = self.video.read()
            # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
            # ret, jpeg = cv2.imencode('.jpg', image)
            # return jpeg.tobytes()
            return image
    
    app = Flask(__name__)
    
    @app.route('/')  # 主页
    def index():
        # 具体格式保存在index.html文件中
        return render_template('index.html')
    
    def scrfd(camera):
        detector = SCRFD(model_file='onnx/scrfd_500m_bnkps_shape160x160.onnx')
        while True:
            frame = camera.get_frame()
            # cv2.imshow('fourcc', frame)
            # img = cv2.imread(frame)
    
            for _ in range(1):
                ta = datetime.datetime.now()
                bboxes, kpss = detector.detect(frame, 0.5, input_size = (160, 160))
                tb = datetime.datetime.now()
                print('all cost:', (tb - ta).total_seconds() * 1000)
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                x1, y1, x2, y2, score = bbox.astype(np.int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if kpss is not None:
                    kps = kpss[i]
                    for kp in kps:
                        kp = kp.astype(np.int)
                        cv2.circle(frame, tuple(kp), 1, (0, 0, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    @app.route('/video_feed')  # 这个地址返回视频流响应
    def video_feed():
        if model == 'scrfd':
            return Response(scrfd(VideoCamera()),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Object Detection using YOLO-Fastest in OPENCV')
        parser.add_argument('--model', type=str, default='scrfd')
        args = parser.parse_args()
        model = args.model
        app.run(host='0.0.0.0', debug=True, port=1938)

运行命令：

    python flask_api.py

将检测结果推送到网页上，此时根据网段号[w400985k15.wicp.vip:54551](https://link.zhihu.com/?target=http%3A//w400985k15.wicp.vip%3A54551/)登录网页进行查看（[w400985k15.wicp.vip](https://link.zhihu.com/?target=http%3A//w400985k15.wicp.vip%3A54551/)为赠送的域名，54551为分配的端口，原host指定为本机ip，映射后自动跳转到公网域名，port也会跟着改变），此时的树莓派和电脑属于不同的局域网下：

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-dd5390fdcc39da02aeb46c8a2298e402_b.jpg)  

不仅如此，还可以用手机进行查看，不管在多远的距离都可以。

*   实测使用flask会比本地推理延迟大约0.5s左右，后期可以采用fastapi或者node.js进行尝试
*   建议使用树莓派4B进行体验，3B板子较老，有些性能跟不上
*   建议超频，参考[How to Overclock Raspberry Pi 4 to 2.0 GHz - CNX Software](https://link.zhihu.com/?target=https%3A//www.cnx-software.com/2019/07/26/how-to-overclock-raspberry-pi-4/)y-pi-4/

代码与所有onnx模型放在此连接上

[](https://link.zhihu.com/?target=https%3A//github.com/pengtougu/onnx-scrfd-flask)

最后，放上我喜欢听的一首歌

![](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/imgv2-23662c840b6638fbb9d66662e2f43ec8_b.jpg)  

参考：

【1】[nihui：详细记录insightface的SCRFD人脸检测ncnn实现](https://zhuanlan.zhihu.com/p/372332267)

【2】[树莓派远程监控 - web异地监控](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_41923091/article/details/103962704%3Futm_source%3Dapp%26app_version%3D4.7.1%26code%3Dapp_1562916241%26uLinkId%3Dusr1mkqgl919blen)
> 作者：pogg
> 链接：undefined