# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


# 整理两个映射文件，得到从类别编号到类别名的映射关系

# 字符串到英文
uid_to_human = {}
for line in tf.gfile.GFile('imagenet_synset_to_human_label_map.txt').readlines():
    items = line.strip().split('\t')
    uid_to_human[items[0]] = items[1]

# 从整数到字符串
node_id_to_uid = {}
for line in tf.gfile.GFile('imagenet_2012_challenge_label_map_proto.pbtxt').readlines():
    if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
    if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1].strip('\n').strip('\"')
        node_id_to_uid[target_class] = target_class_string

# id到名称
node_id_to_name = {}
for key, value in node_id_to_uid.items():
    node_id_to_name[key] = uid_to_human[value]


# 加载模型
def create_graph():
    # 因为是以pb文件格式存的，所以读取需要用ff读取，
    # 得到 f 为句柄
    with tf.gfile.FastGFile('classify_image_graph_def.pb', 'rb') as f:
        # 获得一个图的定义
        graph_def = tf.GraphDef()
        # 从句柄中把图的结构和参数读出来
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


# 定义一个分类图片的函数
# image是图片的路径 top_k是返回概率最大的几个类别
def classify_image(image, top_k=1):
    # 通过api将数据读进来
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # 创建图，这个时候图也有了
    create_graph()

    with tf.Session() as sess:
        # 'softmax:0': A tensor containing the normalized prediction across 1000 labels
        # 最后一层的前一层
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048 float description of the image
        # 输入的tensor
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG encoding of the image
        # 1000个类别的概率分布
        # sess.graph先拿到图，根据名字拿到tensor
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        # 我们要拿到softmax_tensor这个tensor的值，什么时候的值呢？喂给她的数据为image_data 的时候的结果
        predictions = sess.run(softmax_tensor, feed_dict={'DecodeJpeg/contents:0': image_data})
        # np.squeeze把一个tensor里面维度为1的那一维删掉（1，1，2，3）->(2,3)
        predictions = np.squeeze(predictions)

        # 按概率进行排序（排序之后取后k个，就是概率最大的那几个）
        top_k = predictions.argsort()[-top_k:]
        # 对于里面的每一个把名字取出来，
        for node_id in top_k:
            human_string = node_id_to_name[node_id]
            # 把分数取出来
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))


classify_image('test1.jpg')
