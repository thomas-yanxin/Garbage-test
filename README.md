# Garbage-test
基于PaddleX的垃圾分类识别


@[TOC](基于PaddleX的智能垃圾分类识别)
# 项目介绍
## 垃圾分类
&emsp; 垃圾分类是对垃圾收集处置传统方式的改革，是对垃圾进行有效处置的一种科学管理方法。人们面对日益增长的垃圾产量和环境状况恶化的局面，如何通过垃圾分类管理，最大限度地实现垃圾资源利用，减少垃圾处置的数量，改善生存环境状态，是当前世界各国共同关注的迫切问题。
&emsp;2019年6月25日，固体废物污染环境防治法修订草案初次提请全国人大常委会审议。草案对“生活垃圾污染环境的防治”进行了专章规定。
&emsp;2019年9月，为深入贯彻落实习近平总书记关于垃圾分类工作的重要指示精神，推动全国公共机构做好生活垃圾分类工作，发挥率先示范作用，国家机关事务管理局印发通知，公布《公共机构生活垃圾分类工作评价参考标准》，并就进一步推进有关工作提出要求。 
&emsp;2019年12月6日，垃圾分类入选“2019年中国媒体十大流行语”。
## 分类意义
&emsp;**减少土地侵蚀**
&emsp;生活垃圾中有些物质不易降解，使土地受到严重侵蚀。垃圾分类，去掉可以回收的、不易降解的物质，减少垃圾数量达60%以上。
&emsp;**减少污染**
&emsp;中国的垃圾处理多采用卫生填埋甚至简易填埋的方式，占用上万亩土地；并且虫蝇乱飞，污水四溢，臭气熏天，严重污染环境。
&emsp;土壤中的废塑料会导致农作物减产；抛弃的废塑料被动物误食，导致动物死亡的事故时有发生。因此回收利用可以减少危害。
&emsp;**变废为宝**
&emsp;垃圾中的其他物质可以转化为资源，如食品、草木和织物可以堆肥，生产有机肥料；垃圾焚烧可以发电、供热或制冷；砖瓦、灰土可以加工成建材等等。如果能充分挖掘回收生活垃圾中蕴含的资源潜力，仅北京每年就可获得11亿元的经济效益。可见，消费环节产生的垃圾如果及时进行分类，回收再利用是解决垃圾问题的最好途径。
## 垃圾种类
#### 可回收物
&emsp;可回收物主要包括废纸、塑料、玻璃、金属和布料五大类。

 1. 废纸：主要包括报纸、期刊、图书、各种包装纸等。但是，要注意纸巾和厕所纸由于水溶性太强不可回收。
 2. 塑料：各种塑料袋、塑料泡沫、塑料包装（快递包装纸是其他垃圾/干垃圾）、一次性塑料餐盒餐具、硬塑料、塑料牙刷、塑料杯子、矿泉水瓶等。
玻璃：主要包括各种玻璃瓶、碎玻璃片、暖瓶等。（镜子是其他垃圾/干垃圾）
 3. 金属物：主要包括易拉罐、罐头盒等。
 4. 布料：主要包括废弃衣服、桌布、洗脸巾、书包、鞋等。

这些垃圾通过综合处理回收利用，可以减少污染，节省资源。如每回收1吨废纸可造好纸850公斤，节省木材300公斤，比等量生产减少污染74%；每回收1吨塑料饮料瓶可获得0.7吨二级原料；每回收1吨废钢铁可炼好钢0.9吨，比用矿石冶炼节约成本47%，减少空气污染75%，减少97%的水污染和固体废物。

#### 其他垃圾
&emsp;其他垃圾（上海称干垃圾）包括除上述几类垃圾之外的砖瓦陶瓷、渣土、卫生间废纸、纸巾等难以回收的废弃物及尘土、食品袋（盒）。

 1. 卫生纸：厕纸、卫生纸遇水即溶，不算可回收的“纸张”，类似的还有烟盒等。
 2. 果壳：在垃圾分类中，“果壳瓜皮”的标识就是花生壳，的确属于厨余垃圾。家里用剩的废弃食用油，也归类在“厨余垃圾”。
 3. 尘土：在垃圾分类中，尘土属于“其它垃圾”，但残枝落叶属于“厨余垃圾”，包括家里开败的鲜花等。

&emsp;采取卫生填埋可有效减少对地下水、地表水、土壤及空气的污染。


#### 厨余垃圾
&emsp;厨余垃圾（上海称湿垃圾）包括剩菜剩饭、骨头、菜根菜叶、果皮等食品类废物。
&emsp;经生物技术就地处理堆肥，每吨可生产0.6~0.7吨有机肥料。

#### 有害垃圾

&emsp;有害垃圾含有对人体健康有害的重金属、有毒的物质或者对环境造成现实危害或者潜在危害的废弃物。包括电池、荧光灯管、灯泡、水银温度计、油漆桶、部分家电、过期药品及其容器、过期化妆品等。
&emsp;这些垃圾一般使用单独回收或填埋处理。
# PaddleX简介
## Paddlepaddle飞桨
&emsp; 飞桨（PaddPaddle）以百度多年的深度学习技术和业务应用为基础，是中国首个开源开放、技术领先、功能完备的产业级深度学习平台，包括飞桨开源平台和飞桨企业版。飞桨开源平台包含核心框架、基础模型库、端到端开发套件与工具组件，持续开源核心能力，为产业、学术、科研创新提供基础底座。
## PaddleX简介
&emsp;  PaddleX作为飞桨全流程开发套件，以低代码的形式支持开发者快速实现产业实际项目落地。集成飞桨智能视觉领域图像分类、目标检测、语义分割、实例分割任务能力，将深度学习开发全流程从数据准备、模型训练与优化到多端部署端到端打通，并提供统一任务API接口.
&emsp; PaddleX 经过质检、安防、巡检、遥感、零售、医疗等十多个行业实际应用场景验证，沉淀产业实际经验，并提供丰富的案例实践教程，全程助力开发者产业实践落地。
### 产品模块说明
#### 数据准备
&emsp; 兼容ImageNet、VOC、COCO等常用数据协议，同时与Labelme、精灵标注助手、EasyData智能数据服务平台等无缝衔接，全方位助力开发者更快完成数据准备工作。

#### 数据预处理及增强
&emsp; 提供极简的图像预处理和增强方法--Transforms，适配imgaug图像增强库，支持上百种数据增强策略，是开发者快速缓解小样本数据训练的问题。

#### 模型训练
&emsp; 集成PaddleClas, PaddleDetection, PaddleSeg视觉开发套件，提供大量精选的、经过产业实践的高质量预训练模型，使开发者更快实现工业级模型效果。

#### 模型调优
&emsp; 内置模型可解释性模块、VisualDL可视化分析工具。使开发者可以更直观的理解模型的特征提取区域、训练过程参数变化，从而快速优化模型。

#### 多端安全部署
&emsp; 内置PaddleSlim模型压缩工具和模型加密部署模块，与飞桨原生预测库Paddle Inference及高性能端侧推理引擎Paddle Lite 无缝打通，使开发者快速实现模型的多端、高性能、安全部署。

# 项目实现

## 代码过程
本项目代码实现均基于以下环境：

Python版本：python 3.7
框架版本：PaddlePadd1.8.4
硬件信息： CPU  4
&emsp; &emsp; &emsp; &emsp;&ensp;RAM  32GB
&emsp; &emsp; &emsp; &emsp;&ensp;GPU    v100
&emsp; &emsp; &emsp; &emsp;&ensp;显存    16GB
&emsp; &emsp; &emsp; &emsp;&ensp;磁盘    100GB
运行环境：百度AIstudio平台
### 模型训练

#### 1、安装PaddleX
```python
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```
#### 2、数据集加载
&emsp; 本项目采用华为云人工智能大赛·垃圾分类挑战杯提供的数据集，并将其转换为ImageNet格式，使其适应于PaddleX的API接口。
数据集共分为四个大类，分别为可回收物、厨余垃圾、其他垃圾以及有害垃圾，每大类分类中又有细分，共分四十小类。
数据集总共含20736张图片，其中
&emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; 训练集（Train Set）:&emsp; 14802张图片；
&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; 测试集（Test Set）:&emsp;&ensp;  2956张图片；
&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; 验证集(Validation Set):&ensp; 2978张图片。
数据集如下为示例结构：
```html
garbage/ # 垃圾分类数据集根目录
|--0/ # 当前文件夹所有图片属于1类别
|  |--img_1.jpg
|  |--img_2.jpg
|  |--...
|  |--...
|
|--...
|
|--39/ # 当前文件夹所有图片属于39类别
|  |--img_19151.jpg
|  |--img_19152.jpg
|  |--...
|  |--...
```
**labels.txt**用于列出所有类别，类别对应行号表示模型训练过程中类别的id(行号从0开始计数);
**train_list.txt**列出用于训练时的图片集合，与其对应的类别id;
**val_list.txt**列出用于验证时的图片集成，与其对应的类别id，格式与train_list.txt一致.

###### PaddleX数据集加载


```python
import matplotlib 
matplotlib.use('Agg')
import os 
os.environ['CUDA_VISIBLE_DEVICES']='0'
import paddlex as pdx
import imghdr

import paddle.fluid as fluid
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#定义train_dataset数据集用于加载图像分类数据集
#pdx.datasets.ImageNet表示读取ImageNet格式的数据集

train_dataset=pdx.datasets.ImageNet(
    data_dir='garbage_data',
    file_list='garbage_data/train.txt',
    label_list='garbage_data/labels.txt',
    transforms=train_transforms,
    shuffle=True
)

eval_dataset = pdx.datasets.ImageNet(
    data_dir='garbage_data',
    file_list='garbage_data/validate.txt',
    label_list='garbage_data/labels.txt',
    transforms=eval_transforms
)
```


#### 3、定义训练/验证图像处理流程transforms

因为训练时加入了数据增强操作，因此在训练和验证过程中，模型的数据处理流程需要分别进行定义。如下所示，代码在train_transforms中加入了RandomCrop和RandomHorizontalFlip两种数据增强方式。

```python
from paddlex.cls import transforms
train_transforms=transforms.Compose([transforms.RandomCrop(crop_size=224),
transforms.RandomHorizontalFlip(),
transforms.RandomDistort(brightness_range=0.9, brightness_prob=0.5, contrast_range=0.9, contrast_prob=0.5, saturation_range=0.9, saturation_prob=0.5, hue_range=18, hue_prob=0.5)，
transforms.Normalize()
])

eval_transforms=transforms.Compose([transforms.ResizeByShort(short_size=256),
transforms.CenterCrop(crop_size=224),
transforms.Normalize()])
```

**RandomCrop**：
对图像进行随机剪裁，模型训练时的数据增强操作。

```python
paddlex.cls.transforms.RandomCrop(crop_size=224, lower_scale=0.08, lower_ratio=3. / 4, upper_ratio=4. / 3)
```

 1. 根据lower_scale、lower_ratio、upper_ratio计算随机剪裁的高、宽。
 2. 根据随机剪裁的高、宽随机选取剪裁的起始点。 
 3.  剪裁图像。 
 4. 调整剪裁后的图像的大小到crop_size*crop_size。

**参数**
 - crop_size (int): 随机裁剪后重新调整的目标边长。默认为224。 
 - lower_scale (float):裁剪面积相对原面积比例的最小限制。默认为0.08
 - lower_ratio (float): 宽变换比例的最小限制。默认为3. / 4。
 -  upper_ratio (float): 宽变换比例的最大限制。默认为4. / 3。

**RandomHorizontalFlip**：
以一定的概率对图像进行随机水平翻转，模型训练时的数据增强操作。

```python
paddlex.cls.transforms.RandomHorizontalFlip(prob=0.5)
```
**参数**
- prob (float): 随机水平翻转的概率。默认为0.5。

**RandomDistort**：
以一定的概率对图像进行随机像素内容变换，模型训练时的数据增强操作。

```python
paddlex.cls.transforms.RandomDistort(brightness_range=0.9, brightness_prob=0.5, contrast_range=0.9, contrast_prob=0.5, saturation_range=0.9, saturation_prob=0.5, hue_range=18, hue_prob=0.5)
```

 1. 对变换的操作顺序进行随机化操作。
 2. 按照1中的顺序以一定的概率对图像在范围[-range, range]内进行随机像素内容变换。
 3. 【注意】该数据增强必须在数据增强Normalize之前使用。
**参数**
- brightness_range (float): 明亮度因子的范围。默认为0.9。
- brightness_prob (float): 随机调整明亮度的概率。默认为0.5。
 - contrast_range (float): 对比度因子的范围。默认为0.9。
- contrast_prob (float): 随机调整对比度的概率。默认为0.5。
- saturation_range (float): 饱和度因子的范围。默认为0.9。
- saturation_prob (float): 随机调整饱和度的概率。默认为0.5。
- hue_range (int): 色调因子的范围。默认为18。
- hue_prob (float): 随机调整色调的概率。默认为0.5。
#### 4、使用ResNet50_vd_ssld模型开始训练

```python
num_classes=len(train_dataset.labels)
model=pdx.cls.ResNet50_vd_ssld(num_classes=num_classes)

model.train(num_epochs=300,
train_dataset=train_dataset,
train_batch_size=32,
eval_dataset=eval_dataset,
lr_decay_epochs=[4,6,8],
save_interval_epochs=1,
learning_rate=0.001,
save_dir='output/ResNet50_vd_ssld',
use_vdl=True)
```
#### 5、训练过程使用VisualDL查看训练指标变化
VisualDL是飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等。可帮助用户更清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型优化。

VisualDL提供丰富的可视化功能，支持标量、图结构、数据样本可视化、直方图、PR曲线及高维数据降维呈现等诸多功能，同时VisualDL提供可视化结果保存服务，通过VDL.service生成链接，保存并分享可视化结果。

在本项目中，通过使用VisualDL得到训练指标变化图表如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201128073307617.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01lZmlzaGVz,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201128073307621.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01lZmlzaGVz,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201128073643369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01lZmlzaGVz,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201128073306474.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01lZmlzaGVz,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201128073305637.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01lZmlzaGVz,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201128073543948.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01lZmlzaGVz,size_16,color_FFFFFF,t_70#pic_center)
从其在验证集中的结果来看，其精度可达99.26%，具有较好的识别性能。

#### 6、加载训练保存的模型预测
模型在训练过程中，会每间隔一定轮数保存一次模型，在验证集上评估效果最好的一轮会保存在save_dir目录下的best_model文件夹。通过如下方式可加载模型，进行预测。

```python
#加载最优训练模型
import paddlex as pdx
model=pdx.load_model('output/mobilenetv2/best_model')

#预测图片为从测试集中随机取出的图片
#输出结果
image_name='garbage_data/6/img_2421.jpg'
result=model.predict(image_name)
print('Predict Result:',result)
```
由于考虑到本项目的实际应用性，需要从摄像头获取图片的方式进行预测，故调用OpenCV库来达此目的：

```python
import cv2
import paddlex as pdx
import json
import time

cap = cv2.VideoCapture(0)  # 打开摄像头

while (1):
    # get a frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
    # show a frame
    cv2.imshow("capture", frame)  # 生成摄像头窗口

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q 就截图保存并退出
        cv2.imwrite("test.jpg", frame)  # 保存路径
        
        break
        
cap.release()

model=pdx.load_model('output/ResNet50_vd_ssld/best_model')

image_name='test.jpg'

result=model.predict(image_name)

filepath='output/garbage_classify_rule.json'
f_obj=open(filepath)

print('Predict Result:',result)

number=result[0]['category']
score=result[0]['score']
score="%.2f%%"%(score*100)
content=json.load(f_obj)[number][:4]
#输出识别结果    
print(f'The identification result is {content}')
#输出识别概率，给用户置信度
print(f'The identification credibility is {score}')
```
为提高用户的使用感，结合百度大脑语音合成API，调用playsound库实现实时语音播报：
```python
from urllib.request import urlopen

from urllib.request import Request

from urllib.error import URLError

from urllib.parse import urlencode

from urllib.parse import quote_plus


class Speech_synthesis():
    '''百度语音合成'''

    def __init__(self):
        # 发音人选择, 0为普通女声，1为普通男生，3为情感合成-度逍遥，4为情感合成-度丫丫，默认为普通女声
        self.PER = 3
        # 语速，取值0-15，默认为5中语速
        self.SPD = 4
        # 音调，取值0-15，默认为5中语调
        self.PIT = 3
        # 音量，取值0-9，默认为5中音量
        self.VOL = 6
        # 下载的文件格式, 3：mp3(default) 4： pcm-16k 5： pcm-8k 6. wav
        self.AUE = 6

        self.TTS_URL = "http://tsn.baidu.com/text2audio"

  

    def key(self):
        #获取token秘钥
        body = {
            "grant_type"    : "client_credentials",
            "client_id"     : "百度AI大脑",
            "client_secret" : "百度AI大脑"
        }
        url  = "https://aip.baidubce.com/oauth/2.0/token?"
        r = requests.post(url,data=body,verify=True,timeout=2)
        respond = json.loads(r.text)
        return  respond["access_token"]

    '''
    语音合成主函数
    '''
    def main(self,enobj):
        try:
            tex = quote_plus(enobj)  # 此处re_text需要两次urlencode
            params = {'tok': self.key(), 'tex': tex, 'per': self.PER, 'spd': self.SPD,
                    'pit': self.PIT, 'vol': self.VOL, 'aue': self.AUE, 'cuid': "123456PYTHON",'lan': 'zh', 'ctp': 1}  # lan ctp 固定参数
            data = urlencode(params)
            req = Request(self.TTS_URL, data.encode('utf-8'))
            try:
                f = urlopen(req,timeout=4)
                result_str = f.read()

                with open('garbage.wav', 'wb') as of:
                    of.write(result_str)

            except Exception as bug:

                return {'state': False,'data':'','msg':'可能是网络超时。'}

        except:
            return {'state': False,'data':'','msg':'可能是网络超时。'}

if __name__ == '__main__':

    print(Speech_synthesis().main(garbage))
    
    time.sleep(1)

    os.remove('/home/thomas/python/garbage.wav')
```
# 成果展示


![马克杯](https://img-blog.csdnimg.cn/20201128145312418.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01lZmlzaGVz,size_16,color_FFFFFF,t_70)


![口唇膏](https://img-blog.csdnimg.cn/20201128145311788.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01lZmlzaGVz,size_16,color_FFFFFF,t_70)![干电池](https://img-blog.csdnimg.cn/20201128145311622.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01lZmlzaGVz,size_16,color_FFFFFF,t_70)
![旧衣服](https://img-blog.csdnimg.cn/20201128145309905.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01lZmlzaGVz,size_16,color_FFFFFF,t_70)



# 未来展望

 1. 结合Android开发，将其做成一个简易的APP，部署在移动端，实现粗糙状态下的应用落地；
 2. 将其部署在垃圾桶上，作为垃圾分类的检测系统，检验人们是否合格进行垃圾分类；
 3. 利用行人步态分析，结合PaddleDetection，在行人有概率向垃圾桶方向行驶时自动判断并检索垃圾，且进行快速识别，然后自动打开垃圾桶使垃圾入内。

# 参考链接

 1. PaddlePaddle 的github地址：https://github.com/PaddlePaddle
 2. PaddlePaddle官网：https://www.paddlepaddle.org.cn/
 3. PaddleX 的GitHub地址：https://github.com/PaddlePaddle/PaddleX
 4. VisualDL的GitHub地址：https://github.com/PaddlePaddle/VisualDL
 5. 垃圾分类数据集地址：https://aistudio.baidu.com/aistudio/datasetdetail/43904
