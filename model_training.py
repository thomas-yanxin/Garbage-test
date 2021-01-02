import matplotlib 
matplotlib.use('Agg')
import os 
os.environ['CUDA_VISIBLE_DEVICES']='0'
import paddlex as pdx


import paddle.fluid as fluid
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#定义训练/验证图像处理流程transforms
from paddlex.cls import transforms
train_transforms=transforms.Compose([transforms.RandomCrop(crop_size=224),
transforms.RandomHorizontalFlip(),

transforms.RandomDistort(brightness_range=0.9, brightness_prob=0.5, contrast_range=0.9, contrast_prob=0.5, saturation_range=0.9, saturation_prob=0.5, hue_range=18, hue_prob=0.5),
transforms.Normalize()
])

eval_transforms=transforms.Compose([transforms.ResizeByShort(short_size=256),
transforms.CenterCrop(crop_size=224),
transforms.Normalize()])

# 定义dataset加载图像分类数据集
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
# 使用ResNet50_vd_ssld模型开始训练

num_classes=len(train_dataset.labels)

model=pdx.cls.ResNet50_vd_ssld(num_classes=num_classes)

model.train(num_epochs=300,
        train_dataset=train_dataset,
        train_batch_size=128,
        eval_dataset=eval_dataset,
        lr_decay_epochs=[80,100,150],
        save_interval_epochs=5,
        learning_rate=0.002,
        save_dir='output/ResNet50_vd_ssld',
        use_vdl=True
)

