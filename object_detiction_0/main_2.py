# -*- coding: utf-8 -*-
# 基于main_1.py 进行修改为调用摄像头

import numpy as np
import tensorflow as tf
# plt ，image不要了

from utils import label_map_util
from utils import visualization_utils as vis_util

import cv2
# 0就代表笔记本上的摄像头
cap = cv2.VideoCapture(0)

PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
PATH_TO_LABELS = 'ssd_mobilenet_v1_coco_2017_11_17/mscoco_label_map.pbtxt'
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        od_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # 写个死循环，不断从摄像头中读图片
        while True:
            ret, image_np = cap.read()
            # 默认opencv获取的是图像的通道时BGR，但是我们模型接收的图片的通道时RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                                                               category_index, use_normalized_coordinates=True,
                                                               line_thickness=8)

            # cv2 提供的功能，object detection为标题 ，在把输出转会BGR
            cv2.imshow('object detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            # 用位运算符，就是如果按q键，就会把摄像头给释放，并且把所有的窗口摧毁
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

