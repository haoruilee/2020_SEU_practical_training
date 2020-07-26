# -*- coding: utf-8 -*-
# 人脸聚类

import dlib
from imageio import imread
import glob
import os
from collections import Counter

detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
paths = glob.glob('faces/*.jpg')

vectors = []
images = []
for path in paths:
    img = imread(path)
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        shape = predictor(img, d)
        face_vector = facerec.compute_face_descriptor(img, shape)
        vectors.append(face_vector)
        images.append((img, shape))

# 聚类函数
labels = dlib.chinese_whispers_clustering(vectors, 0.5)
num_classes = len(set(labels))
print('共聚为 %d 类' % num_classes)
biggest_class = Counter(labels).most_common(1)
print(biggest_class)

output_dir = 'most_common'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
face_id = 1
for i in range(len(images)):
    if labels[i] == biggest_class[0][0]:
        img, shape = images[i]
        # 把人脸切出来
        dlib.save_face_chip(img, shape, output_dir + '/face_%d' % face_id, size=150, padding=0.25)
        face_id += 1
