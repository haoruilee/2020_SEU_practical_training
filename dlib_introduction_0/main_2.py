# -*- coding: utf-8 -*-
#关键点检测

import dlib
from imageio import imread
import glob

detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()
paths = glob.glob('faces/*.jpg')

for path in paths:
    img = imread(path)
    win.clear_overlay()
    win.set_image(img)

    # 1 表示将图片放大一倍，便于检测到更多人脸
    dets = detector(img, 1)
    print('检测到了 %d 个人脸' % len(dets))
    for i, d in enumerate(dets):
        print('- %d: Left %d Top %d Right %d Bottom %d' % (i, d.left(), d.top(), d.right(), d.bottom()))
        # 等到人脸关键点
        shape = predictor(img, d)
        # 第 0 个点和第 1 个点的坐标
        print('Part 0: {}, Part 1: {}'.format(shape.part(0), shape.part(1)))
        win.add_overlay(shape)

    win.add_overlay(dets)
    dlib.hit_enter_to_continue()