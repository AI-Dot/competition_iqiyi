#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import fnmatch

import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2

detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 ,
                         accurate_landmark = False)

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def take_face(path):
    head, tail = os.path.split(path)
    tail = tail + '_face'
    dest = tail
    makedir(dest)

    for file in find_files(path, '*.jpg'):
        print(file)
        path_component = os.path.normpath(file).split(os.path.sep)

        for i, c in enumerate(path_component):
            if c == '..':
                path_component[i] = ''
        try:
            path_component.remove('')
        except:
            pass
        path_component[0] = dest

        basename = os.path.basename(file)
        jpg_dest = os.path.join(*path_component)
        makedir(jpg_dest)

        # command = 'ffmpeg -loglevel quiet -i {0} -vf fps=1 {1}/{2}_%03d.jpg'.format(
        #     file, jpg_dest, basename)

        img = cv2.imread(file)
        results = detector.detect_face(img)
        if results is not None:
            total_boxes = results[0]
            points = results[1]

            # extract aligned face chips
            chips = detector.extract_image_chips(img, points, 256, 0.37)
            for i, chip in enumerate(chips):
                cv2.imwrite('{0}/{1}_{2}.jpg'.format(jpg_dest, basename, i), chip)


def makedir(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)


#take_face('./IQIYI_VID_DATA_Part1_out_jpg')
take_face('./test_data_jpg_1')