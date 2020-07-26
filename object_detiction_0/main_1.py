# -*- coding: utf-8 -*-
# 目标检测狗

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# 读取图片
from PIL import Image

# 辅助工具包
from utils import label_map_util
from utils import visualization_utils as vis_util

# 常量
# 预训练好的模型
PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

# 一个对照表
PATH_TO_LABELS = 'ssd_mobilenet_v1_coco_2017_11_17/mscoco_label_map.pbtxt'
NUM_CLASSES = 90


# 加载模型
# 加载一个模型就是加载一个计算图
detection_graph = tf.Graph()
# 定义一个局部的命名空间，把图设置为默认图
with detection_graph.as_default():
    # 图的定义
    od_graph_def = tf.GraphDef()
    # 以二进制的格式将文件读进来
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        # 从文件中把图的定义加载进来（包括参数和权重）
        od_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(od_graph_def, name='')

# 加载分类标签数据
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# 把类别名字提出来
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
# 把类别索引获取到
category_index = label_map_util.create_category_index(categories)

# 定义一个辅助函数
# 把一个图片加载成一个numpy数组，然后就可以丢给框架了
def load_image_into_numpy_array(image):
    # 获取图片的宽度和高度（第一位是宽度第二位是高度）
    (im_width, im_height) = image.size
    # 但是在numpy中第二位是宽度，第一位是高度，需要reshap一下，彩色的，所以是3通道
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


TEST_IMAGE_PATHS = ['test_images/image1.jpg', 'test_images/image2.jpg', 'test_images/image3大脸.jpg']

# 刚才已经加载了模型了，把它作为现在的默认图
with detection_graph.as_default():
    # 定义了一个tf是回话，把图传进去
    with tf.Session(graph=detection_graph) as sess:
        # 把图的一些tensor拿出来，
        # 图片的输入
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # anchor box的位置
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # 每个矩形框的得分（置信度）
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # 对应的类别是什么东西
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # 一共识别出来了多少个物体
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # 对测试图片中的每一个
        for image_path in TEST_IMAGE_PATHS:
            # 打开
            image = Image.open(image_path)
            # 加载位numpy数组
            image_np = load_image_into_numpy_array(image)
            # 扩展一个维度，为什么扩展，在前面再加一维，第一维是图片的个数
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # run什么，我要获取四样东西，模型的四个输出，要喂哪些数据进去
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # 这个工具类就是把结果加到原来的image_np中去
            vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                                                               category_index, use_normalized_coordinates=True,
                                                               line_thickness=8)
            # 画个底图
            plt.figure(figsize=[12, 8])
            # 把图画出来
            plt.imshow(image_np)
            plt.show()
