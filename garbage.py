import warnings
warnings.filterwarnings('ignore')
import os
import time
import random
import imghdr
import paddle
import paddle.fluid as fluid
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import cpu_count

print("It is begining~")
crop_size=300
resize_size=300
is_color=3
USE_GPU=False
BUF_SIZE=64 
BATCH_SIZE=8
garbage_path='/home/thomas/python/paddle/PaddleDetection/dataset'

garbage_sub_file_path=['/home/thomas/python/paddle/PaddleDetection/dataset/garbage_data/cardboard','/home/thomas/python/paddle/PaddleDetection/dataset/garbage_data/glass','/home/thomas/python/paddle/PaddleDetection/dataset/garbage_data/metal','/home/thomas/python/paddle/PaddleDetection/dataset/garbage_data/paper','/home/thomas/python/paddle/PaddleDetection/dataset/garbage_data/plastic','/home/thomas/python/paddle/PaddleDetection/dataset/garbage_data/trash']


# # 生成图像列表

# with open('train_img_list','w') as f_train:
#     with open('test_img_list','w') as f_test:
#         test_img_num,train_img_num=0,0
#         for label, path in enumerate(garbage_sub_file_path):
#             paths = os.listdir(path)  # 获取img_path路径下的所有子文件
#             for i,img in enumerate(paths):
#                 try:
#                     img_path = path + '/' + img         # 合成每张图片的路径
#                     img_type = imghdr.what(img_path)           # 获取文件类型
#                     if (img_type=='jpeg') | (img_type=='png'):  # 要求图片格式
#                         img_arr = np.array(Image.open(img_path))
#                         if len(img_arr.shape) != 2:            # 不是灰度图
#                             if i%10 == 0:
#                                 test_img_num += 1
#                                 f_test.write(img_path + '\t' + str(label) + '\n')
#                             else:
#                                 train_img_num += 1
#                                 f_train.write(img_path + '\t' + str(label) + '\n')
#                 except:
#                     pass
                    
# print(f'The train imgs\' number is {train_img_num}\nThe test imgs\' number is {test_img_num}')

# 定义数据读取器
def train_mapper(sample): # 映射器
    img, label = sample
    img = paddle.dataset.image.load_image(file=img, is_color=is_color) # 加载file路径下彩色或者灰色图像
    img = paddle.dataset.image.simple_transform(im=img,
                            resize_size=resize_size, crop_size=crop_size,
                            is_color=is_color, is_train=True) # 简单的图像变换
    img = img.astype('float32') / 255.0
    # print(label,end="")
    return img,label
        
def train_r(train_list_path):
    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)
    return paddle.reader.xmap_readers(train_mapper,reader,cpu_count(),256) # 数据映射

def test_mapper(sample): # sample估计就是reader返回的img，label
    img, label = sample
    img = paddle.dataset.image.load_image(file=img, is_color=is_color)
    img = paddle.dataset.image.simple_transform(im=img,
                            resize_size=resize_size, crop_size=crop_size,
                            is_color=is_color, is_train=False)
    img = img.astype('float32') / 255.0
    # print(label,end="")
    return img, label
    
def test_r(test_list_path):
    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)
    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), 256)

# # 定义卷积神经网络
# def cnn(ipt):
#     conv1 = fluid.layers.conv2d(input=ipt,
#                                 filter_size=3,
#                                 num_filters=64,
#                                 stride=1,
#                                 padding=1,
#                                 act='relu',
#                                 name='conv1')
                                
#     pool1 = fluid.layers.pool2d(input=conv1,
#                                 pool_size=2,
#                                 pool_stride=2,
#                                 pool_type='max',
#                                 name='pool1')
                                
#     bn1 = fluid.layers.batch_norm(input=pool1, name='bn1')
    
#     conv2 = fluid.layers.conv2d(input=bn1,
#                                 filter_size=3,
#                                 num_filters=64,
#                                 stride=1,
#                                 padding=1,
#                                 act='relu',
#                                 name='conv2')
                                
#     pool2 = fluid.layers.pool2d(input=conv2,
#                                 pool_size=2,
#                                 pool_stride=2,
#                                 pool_type='max',
#                                 name='pool2')
                                
#     bn2 = fluid.layers.batch_norm(input=pool2, name='bn2')

#     conv3 = fluid.layers.conv2d(input=bn2,
#                                 filter_size=3,
#                                 num_filters=32,
#                                 stride=1,
#                                 padding=1,
#                                 act='relu',
#                                 name='conv3')
    
#     pool3 = fluid.layers.pool2d(input=conv3,
#                                 pool_size=2,
#                                 pool_stride=2,
#                                 pool_type='max',
#                                 name='pool3')

#     bn3 = fluid.layers.batch_norm(input=pool3, name='bn3')
    
#     fc1 = fluid.layers.fc(input=bn2, name='fc1', size=1024, act='relu')
    
#     fc2 = fluid.layers.fc(input=fc1, name='fc2', size=512,  act='relu')

#     fc3 = fluid.layers.fc(input=fc2, name='fc3', size=128,  act='relu')
    
#     fc4 = fluid.layers.fc(input=fc3, name='fc4', size=6,   act='softmax')
    
#     return fc4

# #获取网络
# c = 3 if is_color else 1
# image=fluid.layers.data(name='image',shape=[c,crop_size,crop_size],dtype='float32')
# net=cnn(image)

# #定义损失函数
# label=fluid.layers.data(name='label',shape=[1],dtype='int64')
# cost=fluid.layers.cross_entropy(input=net,label=label)
# avg_cost = fluid.layers.mean(cost)
# acc=fluid.layers.accuracy(input=net,label=label)

#克隆测试程序
test_program=fluid.default_main_program().clone(for_test=True)

# #定义优化方法
# optimizer=fluid.optimizer.AdadeltaOptimizer(learning_rate=0.0001)
# opt = optimizer.minimize(avg_cost)


# 创建执行器
place = fluid.CUDAPlace(0) if USE_GPU else fluid.CPUPlace()
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())

#生成图片数据reader
# train_reader=paddle.batch(reader=paddle.reader.shuffle(reader=train_r('./train_img_list'),buf_size=BUF_SIZE),batch_size=BATCH_SIZE)
# test_reader=paddle.batch(reader=paddle.reader.shuffle(reader=test_r('./test_img_list'),buf_size=BUF_SIZE),batch_size=BATCH_SIZE)

# #定义输入数据维度
# feeder=fluid.DataFeeder(place=place,feed_list=[image,label])

#训练数据
training_costs,testing_costs=[],[]
training_accs,testing_accs=[],[]

# PASS = 5 # 训练次数

# start = time.time()

# for pass_id in range(1, PASS+1):
#     train_costs, train_accs = [], []
#     for batch_id, data in enumerate(train_reader()):
#         train_cost, train_acc = exe.run(
#                             program=fluid.default_main_program(),
#                             feed=feeder.feed(data),
#                             fetch_list=[avg_cost, acc])
#         train_costs.append(train_cost)
#         train_accs.append(train_acc)
#         training_costs.append(train_cost)
#         training_accs.append(train_acc)
#     train_cost = sum(train_costs) / len(train_costs)
#     train_accs = sum(train_accs) / len(train_accs)
        
#     test_costs, test_accs = [], []  # 为了计算测试数据的平均值
#     for batch_id, data in enumerate(test_reader()):
#         test_cost, test_acc = exe.run(
#                             program=test_program,
#                             feed=feeder.feed(data),
#                             fetch_list=[avg_cost, acc])
#         test_costs.append(test_cost)
#         test_accs.append(test_acc)
#         testing_costs.append(test_cost)
#         testing_accs.append(test_acc)
#     test_cost = sum(test_costs) / len(test_costs)
#     test_acc = sum(test_accs) / len(test_accs) 

#     print('Pass:%d  \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f'%
#             (pass_id,train_cost,train_acc,test_cost,test_acc))

# end = time.time()

# print(f"总用时 {end-start}s ")

# # training_costs = [x[0] for x in training_costs]
# # training_accs  = [x[0] for x in training_accs]
# # testing_costs  = [x[0] for x in testing_costs]
# # testing_accs  = [x[0] for x in testing_accs]

# # plt.figure(figsize=(16,8))
# # plt.subplot(2,2,1)
# # plt.title('train_cost', fontsize=15)
# # sns.lineplot(x=list(range(1,1+len(training_costs))),y=training_costs)
# # plt.subplot(2,2,2)
# # plt.title('train_acc', fontsize=15)
# # sns.lineplot(x=list(range(1,1+len(training_accs))),y=training_accs,color='r')
# # plt.subplot(2,2,3)
# # plt.title('test_cost', fontsize=15)
# # sns.lineplot(x=list(range(1,1+len(testing_costs))),y=testing_costs,color='c')
# # plt.subplot(2,2,4)
# # plt.title('test_acc', fontsize=15)
# # sns.lineplot(x=list(range(1,1+len(testing_accs))),y=testing_accs,color='k')

# 保存模型
model_save_dir = '/home/thomas/python/paddle/PaddleDetection/model/garbage_2.model'


# # 保存模型参数到指定路径
# fluid.io.save_inference_model(model_save_dir,
#                               ['image'],    # 推理模型需要feed的数据
#                               [net],    # 保存推理结果的变量
#                               exe)      # 保存推理模型的执行器

# 加载模型并预测

# 创建预测用的执行器

infer_exe = fluid.Executor(place)

# 指定作用域

inference_scope = fluid.core.Scope()


with fluid.scope_guard(scope=inference_scope):
    # load model
    [inference_program, 
     feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(model_save_dir,infer_exe)
     
     # 获取测试数据
    infer_reader = paddle.batch(reader=paddle.reader.shuffle(
                                        reader=test_r('./test_img_list'),
                                        buf_size=BUF_SIZE),
                                 batch_size=500)
    test_data = next(infer_reader())
    test_x = np.array([data[0] for data in test_data]).astype('float32') # 提取图片数据
    test_y = np.array([data[1] for data in test_data]).astype('float32') # 提取图片标签
    
    result = infer_exe.run(program=inference_program,
                           feed={feed_target_names[0]:test_x},
                           fetch_list=fetch_targets)


infer_label = np.array([np.argmax(x) for x in result[0]])

print(infer_label)

print(test_y.astype('int32'))

print(f"测试准确率：{sum(test_y == infer_label)/len(infer_label)}")

print("~~~~OK~~~~")



