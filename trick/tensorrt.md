#                                                              Tensorrt踩坑日记 | python、pytorch 转 onnx 推理加速                

​                                                                                                                                              [                         计算机视觉联盟                      ](javascript:void(0);)                                                              *2022-01-03 18:01*                

  

 

点上方***\*计算机视觉联盟\****获取更多干货

仅作学术分享，不代表本公众号立场，侵权联系删除

转载于：作者 | makcooo

来源 | https://blog.csdn.net/qq_44756223/article/details/107727863 

编辑 | 极市平台

**985人工智能博士笔记推荐**

**[周志华《机器学习》手推笔记正式开源！附pdf下载链接，Github2500星！](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247486763&idx=1&sn=da878ccafba79932be8eeca955697477&chksm=fcb71194cbc0988259f32b0b7a77fe7830c477c9c9a4fcb00f2e366c97d776574c2bc949e993&scene=21#wechat_redirect)**

简单说明一下pytorch转onnx的意义。在pytorch训练出一个深度学习模型后，需要在TensorRT或者openvino部署，这时需要先把Pytorch模型转换到onnx模型之后再做其它转换。因此，在使用pytorch训练深度学习模型完成后，在TensorRT或者openvino或者opencv和onnxruntime部署时，pytorch模型转onnx这一步是必不可少的。本文介绍Python、pytorch转换onnx的过程中遇到的坑。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/gYUsOT36vfqqflS85IR0iajBbFb54LNic9laugmgQiarwjLy0xib1h2ibjiaL76qichg19AcFjIWYrshIR7ezGxsnGm1g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## 配置

Ubuntu 16.04
python 3.6
onnx 1.6
pytorch 1.5
pycuda 2019.1.2
torchvision 0.1.8

建议详读,先安装好环境:https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#import_onnx_python)

## 步骤

### 1.将pytorch模型转换成onnx模型

这边用的是Darknet生成的pytoch模型

```
import torch
from torch.autograd import Variable
import onnx


input_name = ['input']
output_name = ['output']
input = Variable(torch.randn(1, 3, 544, 544)).cuda()
model = x.model.cuda()#x.model为我生成的模型

# model = torch.load('', map_location="cuda:0")
torch.onnx.export(model, input, 'model.onnx', input_names=input_name, output_names=output_name, verbose=True)
```

其中

```
#model = x.model.cuda()
#若是不添加cuda()
model = x.model
```

出现报错

```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```

### 2.检查模型

```
model = onnx.load("model.onnx")
onnx.checker.check_model(model)
print("==> Passed")
```

### 3.测试onnx模型使用tensorrt推理前后对比

```
import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import os
import time
from PIL import Image
import cv2
import torchvision

filename = '000000.jpg'
max_batch_size = 1
onnx_model_path = 'yolo.onnx'

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


def get_img_np_nchw(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (1920, 1080))
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])
    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return img_np_nchw

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", \
               fp16_mode=False, int8_mode=False, save_engine=False,
               ):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(EXPLICIT_BATCH) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 30  # Your workspace size
            builder.max_batch_size = max_batch_size
            # pdb.set_trace()
            builder.fp16_mode = fp16_mode  # Default: False
            builder.int8_mode = int8_mode  # Default: False
            if int8_mode:
                # To be updated
                raise NotImplementedError

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())

                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    print("===========Parsing fail!!!!=================")
                else :
                    print('Completed parsing of ONNX file')

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs



img_np_nchw = get_img_np_nchw(filename)
img_np_nchw = img_np_nchw.astype(dtype=np.float32)

# These two modes are dependent on hardwares
fp16_mode = False
int8_mode = False
trt_engine_path = './model_fp16_{}_int8_{}.trt'.format(fp16_mode, int8_mode)
# Build an engine
engine = get_engine(max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode)
# Create the context for this engine
context = engine.create_execution_context()
# Allocate buffers for input and output
inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

# Do inference
shape_of_output = (max_batch_size, 1000)
# Load data to the buffer
inputs[0].host = img_np_nchw.reshape(-1)

# inputs[1].host = ... for multiple input
t1 = time.time()
trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
t2 = time.time()
feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)

print('TensorRT ok')
#将model改为自己的模型,此处为pytoch的resnet50,需联网下载
model = torchvision.models.resnet50(pretrained=True).cuda()
resnet_model = model.eval()

input_for_torch = torch.from_numpy(img_np_nchw).cuda()
t3 = time.time()
feat_2= resnet_model(input_for_torch)
t4 = time.time()
feat_2 = feat_2.cpu().data.numpy()
print('Pytorch ok!')


mse = np.mean((feat - feat_2)**2)
print("Inference time with the TensorRT engine: {}".format(t2-t1))
print("Inference time with the PyTorch model: {}".format(t4-t3))
print('MSE Error = {}'.format(mse))

print('All completed!')
```

报错：

```
In node -1 (importModel): INVALID_VALUE: Assertion failed: !_importer_ctx.network()->hasImplicitBatchDimension() && "This version of the ONNX parser only supports TensorRT INetworkDefinitions with an explicit batch dimension. Please ensure the network was created using the EXPLICIT_BATCH NetworkDefinitionCreationFlag."
```

解决：

```
    def build_engine(max_batch_size, save_engine):
 
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(EXPLICIT_BATCH) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
```

报错:

```
Traceback (most recent call last):
  line 126, in <listcomp>
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
pycuda._driver.LogicError: cuMemcpyHtoDAsync failed: invalid argument
```

解决:

```
def get_img_np_nchw(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (1920, 1080))
 
```

输入的检测图像尺寸需要resize成model的input的size
改为

```
def get_img_np_nchw(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (544,544))
```

报错

```
line 139, in postprocess_the_outputs
    h_outputs = h_outputs.reshape(*shape_of_output)
ValueError: cannot reshape array of size 5780 into shape (1,1000)
```

解决:

```
#shape_of_output = (max_batch_size, 1000)
#修改成自己模型ouput的大小
shape_of_output = (1,20,17,17)
```

**-------------------**

***END***

***--------------------***

我是[王博Kings](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzU2NTc5MjIwOA==&action=getalbum&album_id=1339435380461961218#wechat_redirect)，985AI博士，华为云专家、CSDN博客专家（人工智能领域优质作者）。单个AI开源项目现在已经获得了2100+标星。现在在做AI相关内容，欢迎一起交流学习、生活各方面的问题，一起加油进步！



我们微信交流群涵盖以下方向（但并不局限于以下内容）：人工智能，计算机视觉，自然语言处理，目标检测，语义分割，自动驾驶，GAN，强化学习，SLAM，人脸检测，最新算法，最新论文，OpenCV，TensorFlow，PyTorch，开源框架，学习方法...



这是我的私人微信，位置有限，一起进步！

**![图片](https://mmbiz.qpic.cn/mmbiz_jpg/uGWQRhqDh3LibMvPS1ib9QVEU0Mz2EEEnJ3N7yd66V5cQD5HP9I4ZamsK0YaV1MZltnAeVpnCU3AXy4f5Xib04hFw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)**

**王博的公众号，欢迎关注，干货多多
**

  

 

**手推笔记：**

[思维导图](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247486763&idx=1&sn=da878ccafba79932be8eeca955697477&chksm=fcb71194cbc0988259f32b0b7a77fe7830c477c9c9a4fcb00f2e366c97d776574c2bc949e993&scene=21#wechat_redirect) | [“模型评估与选择” | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247486235&idx=1&sn=60774f6e71af3c1df5fcdaf37407c1ce&chksm=fcb717a4cbc09eb2915aa8504be858f681e5cc4ea95cc091335f021b30b566f2f16e03ae0739&scene=21#wechat_redirect)[“线性模型” | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247486763&idx=1&sn=da878ccafba79932be8eeca955697477&chksm=fcb71194cbc0988259f32b0b7a77fe7830c477c9c9a4fcb00f2e366c97d776574c2bc949e993&scene=21#wechat_redirect)[“决策树” | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247486809&idx=2&sn=1db1d3e29268334b78faf1b981030974&chksm=fcb711e6cbc098f0db247fdf9739a4e82aaada8231d5b3c6c79b1c19e79d81e3216a7e7476d3&scene=21#wechat_redirect)[“神经网络” | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247486859&idx=1&sn=9ee1c93b8ca69d71ea1db16e022f2691&chksm=fcb71134cbc09822e382eaeb834f9ee66f78ea78636494cd22bcb0493b4e2ae7ef57fe99fe58&scene=21#wechat_redirect)[支持向量机（上） | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247486891&idx=1&sn=48f35f7bcaa7dd559288231c3ac07e30&chksm=fcb71114cbc09802e6d99fe3cd5a41d6da03b1e9eb5442b01b84d9c9edcd49cad0bb50a1309c&scene=21#wechat_redirect)[支持向量机（下） | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247487004&idx=1&sn=b5f13ba410cee5730d1e3784b7fc4c9b&chksm=fcb712a3cbc09bb5aa3e4f16254cd52a72a3e5393370bc5c1ca78682161c74ad81443e01135d&scene=21#wechat_redirect)[贝叶斯分类（上） | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247487301&idx=1&sn=a5220ac4604bfdb15ee2eb52357a19c1&chksm=fcb713facbc09aecde8ec575fe96d7ef2b406b4bb377c63e197cdeaf04cba14e76b8f75c30e4&scene=21#wechat_redirect)[贝叶斯分类（下） | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247487435&idx=2&sn=d3d9a9ecd93f6defe865da36aede1777&chksm=fcb71374cbc09a62446ef8f03e22dcd2b1092169ee0a472a7d3f0e1d6209300943b03d389e76&scene=21#wechat_redirect)[集成学习（上） | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247488226&idx=1&sn=6694a44c5bc8d8207e8348cfbb141297&chksm=fcb70e5dcbc0874b355226a7ef07f0d2e315b13df95c3df2812a70b4eb6ab2d68184e1c3cc86&scene=21#wechat_redirect)[集成学习（下） | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247488404&idx=2&sn=36b6c46669b1c5f1c0922f3c476b2d85&chksm=fcb70f2bcbc0863d86fdf807a48cd919d75556299c4b911602a052cd92d31cc57fbae1600fe2&scene=21#wechat_redirect)[聚类 | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247488468&idx=2&sn=7071a2a562afa337e7b46b490aa44090&chksm=fcb70f6bcbc0867dcd4cd5a16a7b5ce3b88bb10e9af6165797ffe10f7cca8b9d3a0eb365774f&scene=21#wechat_redirect)[降维与度量学习 | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247488759&idx=2&sn=91be2e59ada4107851289c68a64bffb3&chksm=fcb70848cbc0815e03947731557b16897112f37c9b67cd436013a7cf70738cada21925d37ab4&scene=21#wechat_redirect)[稀疏学习 | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247490386&idx=2&sn=321179886eb4f6a2c6bd55aed3578097&chksm=fcb707edcbc08efb406cec43da6540df1e7c99ed88f02ebd2c313ad3549d082679a10cdf6165&scene=21#wechat_redirect)[计算学习理论 | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247494603&idx=3&sn=1d0ca7f2eef4ace9941ce2f3ff130dbb&chksm=fcb4f774cbc37e62a1425aca8d97e593e30bd09a7b3a8f4b3d0d9ac32877979944af03030717&scene=21#wechat_redirect)[半监督学习 | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247495199&idx=3&sn=4e56ec540c4fa222d9e57804b554ea70&chksm=fcb4f2a0cbc37bb60193b8a653016a0d5ea933f1b9c69e39d1b06861813c78fc7c574b527454&scene=21#wechat_redirect)[概率图模型 ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247503912&idx=1&sn=7cd6fee3a4fd042be5ae8f54d750052b&chksm=fcb4cc97cbc345818ec45020b51297150f8b2f1f211f8d74523529f02e14a47af812a5827443&scene=21#wechat_redirect)| [规则学习](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247508471&idx=1&sn=9d6a781d8015ca651f231543977b99f0&chksm=fcb4bd48cbc3345e5169a622d44d6cb1ac4ff9a31970b4cf7a132e7d766f5ae90164a1d2b662&scene=21#wechat_redirect)



**增长见识：**

[博士毕业去高校难度大吗？](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247512691&idx=1&sn=5a9a03b656da97558fd537df2161a07e&chksm=fcb4aecccbc327da843ff574076da8701708dbd39a957ad41d6b2b9d4ac2d6e64584af2c7426&scene=21#wechat_redirect) | [研读论文有哪些经验之谈？ | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247511199&idx=1&sn=dcc96018356c7aa1f07f594ea2db53ab&chksm=fcb4b020cbc3393667267919d45479df44e03b8fccaf5ea1ecffadc4fd664e8505dc0db723ec&scene=21#wechat_redirect)[聊聊跳槽这件事儿 | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247510579&idx=1&sn=8b7f6facff81a9e55f653bad1c6b016f&chksm=fcb4b68ccbc33f9a2ffcfe67d87fafdd025566c76d9ad4caf9951d9ac084dae803f52ab70048&scene=21#wechat_redirect)[聊聊互联网工资收入的组成 | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247510383&idx=1&sn=f659ab4c9b0973276be95c9bab89d5b1&chksm=fcb4b5d0cbc33cc6a9771ce2d6cd88e57b3def32c9d0e3ca8b09e413855a71f422794971fe4f&scene=21#wechat_redirect)[机器学习硕士、博士如何自救？ | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247510244&idx=1&sn=5714c36390b4a7307252c4d02cfaecf0&chksm=fcb4b45bcbc33d4de651e9b095c49a760eb4b909a1a7c0553343bd2bb3bee93b34285bbfacd3&scene=21#wechat_redirect)[聊聊Top2计算机博士2021年就业选择 | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247505138&idx=1&sn=ef6102d36f7aa7191982c6b6eddb9872&chksm=fcb4c84dcbc3415be45e48c522029fb90f1c395278708f38c6d9b68d2aadaa0afff68b3ff607&scene=21#wechat_redirect)[非科班出身怎么转行计算机？ | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247498233&idx=2&sn=7663c29445a7f07f839424656ef28cb5&chksm=fcb4e546cbc36c50b7711c59b77d01048cc81302e4a4a9f4fdbf83715360b14c5110f873b485&scene=21#wechat_redirect) [有哪些相见恨晚的科研经验？ | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247497362&idx=2&sn=5b3436738550a7e4d62c2bb98befe3df&chksm=fcb4ea2dcbc3633b228e5712ecfc9d302ff56b7d96e83232f1c7ffb0a8bd2ffc908b5003b9d9&scene=21#wechat_redirect)[经验 | 计算机专业科班出身如何提高自己编程能力？ | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247498631&idx=2&sn=7a91e5850ec4885a4d7028b925e6f2d2&chksm=fcb4e738cbc36e2e8ed92fcc6acbf5024ee73d4f5b2e05bcd2c31333fad86d6d4638b06fef6f&scene=21#wechat_redirect)[博士如何高效率阅读文献](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247489104&idx=1&sn=189ff572dc14ee137e83e7e4e7a4cc9f&chksm=fcb70aefcbc083f9b351b1b29c56d70cc1748181583490c3769b77a495e74747b8f7491ee39a&scene=21#wechat_redirect) | [有哪些越早知道越好的人生经验？](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247487741&idx=2&sn=40bbb83c4ecda43fcc11fc3eb3a7130f&chksm=fcb70c42cbc085545c4ccfd51563cc59e38134147803c0c59e0aa3827385f060ccc75b3df888&scene=21#wechat_redirect) | 



**其他学习笔记：**

[PyTorch张量Tensor](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247504061&idx=1&sn=207d2bd4895e0b4e303c3d9d63511c72&chksm=fcb4cc02cbc3451449159333c651688e7d71c7a744a4e4e0d914fb14adb5ad4f7e122eff7e18&scene=21#wechat_redirect) | [卷积神经网络CNN的架构 | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247503173&idx=1&sn=e92e9adb903e7a87d0b64ceff982020c&chksm=fcb4d1facbc358ec09986cd8b32ce222c3394c46e2f0bd60c282d63ffeea897e7211984e6fca&scene=21#wechat_redirect)[深度学习语义分割 | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247502893&idx=1&sn=e5720b57b0804dca182552ba9d6c1eb3&chksm=fcb4d092cbc359841d6f5a9e3b838e3894d34b35d74a1627fa9bf37e27c019ca9d9af84c362d&scene=21#wechat_redirect)[深入理解Transformer | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247502134&idx=1&sn=ed4a262db0be219feb017c08f2185779&chksm=fcb4d589cbc35c9f0156ac454dccb82a501f4b8129adbbcf94a3ca83d3af129f4ebadf173e75&scene=21#wechat_redirect)[Scaled-YOLOv4！ | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247500056&idx=2&sn=95a04371dc0b49e8c862b03a88edbb8c&chksm=fcb4dda7cbc354b14c4cd4f63eadf02680112ee21015fe372cf3055f9b577b7dd989843eda1f&scene=21#wechat_redirect)[PyTorch安装及入门 | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247497349&idx=3&sn=b4b1ea0af5d42c817bc776cab0a8ac28&chksm=fcb4ea3acbc3632cd61c21d40df0c8cd9e84603fcbe551ceea721e4d8c43959a3c74e929a0fe&scene=21#wechat_redirect)[PyTorch神经网络箱 | ](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247497036&idx=2&sn=552069978cf317b9581b4c9a62a003e2&chksm=fcb4e9f3cbc360e51de7a7ceb3a6adde123e488d98a3baf4204270311ef609fa5510682ff29b&scene=21#wechat_redirect)[Numpy基础](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247496211&idx=2&sn=c635752b36dde7cfd0f7f3cd0d884d7f&chksm=fcb4eeaccbc367bae3db5e89fd9745a90ed2db415e237ecd65e1cd6d600e06a46cef47820010&scene=21#wechat_redirect) | [10篇图像分类](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247494797&idx=2&sn=f5f12374d71a429e73ca344ae6f50e81&chksm=fcb4f032cbc3792425ed6e33476ce13c7967afb40664a7be59fe52a50dc04018275edcc096c9&scene=21#wechat_redirect) | [CVPR 2020目标检测](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247493878&idx=2&sn=fa475321b32840ab7c5090214593b90d&chksm=fcb4f449cbc37d5f386922c7aa2f2d49cb6c0e8e2de8ca21b3e2b3507e483db04069876b219d&scene=21#wechat_redirect) | [神经网络的可视化解释](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247492033&idx=2&sn=5cd05be07df36d0bee8808c17c35ad7a&chksm=fcb4fd7ecbc37468e5da66fa148be83eac3885ecf6255c8c649126e58eb33ab7c458b8d994ea&scene=21#wechat_redirect) | [YOLOv4全文解读与翻译总结](http://mp.weixin.qq.com/s?__biz=MzU2NTc5MjIwOA==&mid=2247489140&idx=1&sn=7ec16d43b182dd56b2d975bb3cd05adf&chksm=fcb70acbcbc083ddbab8be612388135e4cc5ebee00b301755404c66f3da51f7ebd6a17f89eba&scene=21#wechat_redirect) | 

![图片](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/img640.gif)

**点分享**

![图片](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/img640-20220110130854684.gif)

**点收藏**

![图片](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/img640-20220110130854763.gif)

**点点赞**

![图片](https://xiaoguciu.oss-cn-beijing.aliyuncs.com/img640-20220110130854825.gif)

**点在看**

​                                                                                   

​                     喜欢此内容的人还喜欢               

​             Python的十大特性                        

​                        

​             AI前线           

​                        

​           

不看的原因

- 内容质量低
- 不看此公众号

​             Python 实现循环的最快方式（for、while 等速度对比）                        

​                        

​             一行玩python           

​                        

​           

不看的原因

- 内容质量低
- 不看此公众号

​             Python 200个标准库汇总                        

​                        

​             小白学视觉           

​                        

​           

不看的原因

- 内容质量低
- 不看此公众号

![img](https://mp.weixin.qq.com/mp/qrcode?scene=10000004&size=102&__biz=MzU2NTc5MjIwOA==&mid=2247526504&idx=2&sn=07092a590281773e9a7ede3a62d938b4&send_time=)

微信扫一扫
关注该公众号