# -*- coding: utf-8 -*-
# 人脸检测

import dlib
from imageio import imread
import glob


# 获取正面脸部检测器
detector = dlib.get_frontal_face_detector()
# 获取显示图片的window
win = dlib.image_window()
# 批量获取图片
paths = glob.glob('faces/*.jpg')

for path in paths:
    img = imread(path)
    # 1 表示将图片放大一倍，便于检测到更多人脸
    dets = detector(img, 1)
    print('检测到了 %d 个人脸' % len(dets))
    for i, d in enumerate(dets):
        print('- %d：Left %d Top %d Right %d Bottom %d' % (i, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    # 做一个覆盖物
    win.add_overlay(dets)
    # 等待敲回车
    dlib.hit_enter_to_continue()

path = 'faces/2007_007763.jpg'
img = imread(path)
# -1 表示人脸检测的判定阈值
# scores 为每个检测结果的得分，idx 为人脸检测器的类型
dets, scores, idx = detector.run(img, 1, -1)
for i, d in enumerate(dets):
    print('%d：score %f, face_type %f' % (i, scores[i], idx[i]))
win.clear_overlay()
win.set_image(img)
win.add_overlay(dets)
dlib.hit_enter_to_continue()