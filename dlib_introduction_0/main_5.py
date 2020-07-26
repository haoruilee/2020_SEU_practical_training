# -*- coding: utf-8 -*-
# 目标追踪

import dlib
from imageio import imread
import glob

# 追踪器
tracker = dlib.correlation_tracker()
win = dlib.image_window()
paths = sorted(glob.glob('video_frames/*.jpg'))

for i, path in enumerate(paths):
    img = imread(path)
    # 第一帧，指定一个区域
    if i == 0:
        tracker.start_track(img, dlib.rectangle(74, 67, 112, 153))
    # 后续帧，自动追踪
    else:
        tracker.update(img)

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(tracker.get_position())
    dlib.hit_enter_to_continue()
