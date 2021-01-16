# !/usr/bin/python3
# -*- coding: utf-8 -*-

from urllib.request import urlopen

from urllib.request import Request

from urllib.error import URLError

from urllib.parse import urlencode

from urllib.parse import quote_plus


import os

import json,requests

import cv2

import paddlex as pdx

import time

from playsound import playsound
import numpy as np


def garbage_test():
    cap = cv2.VideoCapture(0)  # 打开摄像头
    
    # #形态学操作需要使用,获取一个3x3的卷积核
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # #创建混合高斯模型用于背景建模
    
    # fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
    
    img_back=cv2.imread('/home/thomas/下载/QQ图片20201211233654.png')

    while (1):
        # get a frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
        # # fgmask= fgbg.apply(frame)

        # rows, cols, channels = frame.shape


        # lower_color = np.array([95, 95, 95])
        # # lower_color = np.array([254, 254, 254])
        # # upper_color = np.array([255, 255, 255])
        # upper_color = np.array([150, 165, 165])
        # # 创建掩图
        # fgmask = cv2.inRange(frame, lower_color, upper_color)
        # # cv2.imshow('Mask', fgmask)
    
        # # 腐蚀膨胀
        # erode = cv2.erode(fgmask, None, iterations=1)
        # # cv2.imshow('erode', erode)
        # dilate = cv2.dilate(erode, None, iterations=1)
        # # cv2.imshow('dilate', dilate)
    
        # rows, cols = dilate.shape

        # img_back=img_back[0:rows,0:cols]

        # # print(img_back)

        # # #根据掩图和原图进行抠图

        # img2_fg = cv2.bitwise_and(img_back, img_back, mask=dilate)
        # Mask_inv = cv2.bitwise_not(dilate)

        # img3_fg = cv2.bitwise_and(frame, frame, mask=Mask_inv)
        # finalImg=img2_fg+img3_fg
        # cv2.imshow('res', finalImg)

        # show a frame
        
        cv2.imshow("capture", frame)  # 生成摄像头窗口

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q 就截图保存并退出dd
            
            # cv2.imwrite("test.jpg", finalImg)  # 保存路径
            cv2.imwrite("test.jpg", frame)  # 保存路径
            
            break
            
    cap.release()

    model=pdx.load_model('/home/thomas/python/paddle/PaddleDetection/output/ResNet50_vd_ssld/best_model')

    image_name='/home/thomas/python/test.jpg'

    result=model.predict(image_name)

    return result

result=garbage_test()

filepath='/home/thomas/python/paddle/PaddleDetection/output/garbage_classify_rule.json'

f_obj=open(filepath)

number=result[0]['category']

score=result[0]['score']

score="%.2f%%"%(score*100)

content=json.load(f_obj)[number]


print(f'The identification result is {content}')

print(f'The identification credibility is {score}')


# # # lime可解释性可视化
# # pdx.interpret.lime(
# #         image_name, 
# #         model,
# #         save_dir='./')


# class Speech_synthesis():
#     '''百度语音合成'''

#     def __init__(self):
#         # 发音人选择, 0为普通女声，1为普通男生，3为情感合成-度逍遥，4为情感合成-度丫丫，默认为普通女声
#         self.PER = 3
#         # 语速，取值0-15，默认为5中语速
#         self.SPD = 4
#         # 音调，取值0-15，默认为5中语调
#         self.PIT = 3
#         # 音量，取值0-9，默认为5中音量
#         self.VOL = 6
#         # 下载的文件格式, 3：mp3(default) 4： pcm-16k 5： pcm-8k 6. wav
#         self.AUE = 6

#         self.TTS_URL = "http://tsn.baidu.com/text2audio"

#     def key(self):
#         #获取token秘钥
#         body = {
#             "grant_type"    : "client_credentials",
#             "client_id"     : "*******************************",
#             "client_secret" : "*******************************"
#         }
#         url  = "https://aip.baidubce.com/oauth/2.0/token?"
#         r = requests.post(url,data=body,verify=True,timeout=2)
#         respond = json.loads(r.text)
#         return  respond["access_token"]

#     '''
#     语音合成主函数
#     '''
#     def main(self,enobj):
#         try:
#             tex = quote_plus(enobj)  # 此处re_text需要两次urlencode
#             params = {'tok': self.key(), 'tex': tex, 'per': self.PER, 'spd': self.SPD,
#                     'pit': self.PIT, 'vol': self.VOL, 'aue': self.AUE, 'cuid': "123456PYTHON",'lan': 'zh', 'ctp': 1}  # lan ctp 固定参数
#             data = urlencode(params)
#             req = Request(self.TTS_URL, data.encode('utf-8'))
#             try:
#                 f = urlopen(req,timeout=4)
#                 result_str = f.read()

#                 with open('garbage.wav', 'wb') as of:
#                     of.write(result_str)

#             except Exception as bug:

#                 return {'state': False,'data':'','msg':'可能是网络超时。'}

#         except:
#             return {'state': False,'data':'','msg':'可能是网络超时。'}

# if __name__ == '__main__':

#     print(Speech_synthesis().main(garbage_test))
    
#     time.sleep(1)

#     os.remove('/home/thomas/python/garbage.wav')


