# -*- coding: utf-8 -*-
# 处理视频

import numpy as np
import tensorflow as tf

from utils import label_map_util
from utils import visualization_utils as vis_util

import cv2
# 读取视频
cap = cv2.VideoCapture(r"./111.mp4")
# 从视频中读一帧，为什么只读一帧呢？读一帧以后就知道这个视频的宽和高了
ret, image_np = cap.read()
# 然后就知道要写的视频的宽和高了
# cap是摄像头，out是写的视频，cv2.CAP_PROP_FPS获取帧频，image_np.shape[1]代表高度，后面代表宽度
out = cv2.VideoWriter('bigface_output.mov', -1, cap.get(cv2.CAP_PROP_FPS), (image_np.shape[1], image_np.shape[0]))

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
        # 判断条件改为了视频是否读完
        while cap.isOpened():
            # 获取一帧
            ret, image_np = cap.read()
            if len((np.array(image_np)).shape) == 0:
                break

            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                                                               category_index, use_normalized_coordinates=True,
                                                               line_thickness=8)
            # 写到新的视频文件中
            out.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

# 把视频流关掉
cap.release()
out.release()
cv2.destroyAllWindows()