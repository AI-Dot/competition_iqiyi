#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import fnmatch
import os

import cv2
import numpy as np

from insightface import face_model

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/home/lyk/machine_learning/competition/iqiyi/insightface/models/model-r50-am-lfw/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()
model = face_model.FaceModel(args)


def test():
    img = cv2.imread('/home/lyk/machine_learning/competition/iqiyi/take_face/test_data_jpg_face/IQIYI_VID_DATA_Part1/IQIYI_VID_TRAIN/IQIYI_VID_TRAIN_0000001.mp4/IQIYI_VID_TRAIN_0000001.mp4_003.jpg/IQIYI_VID_TRAIN_0000001.mp4_003.jpg_0.jpg')
    img = model.get_input(img)
    f1 = model.get_feature(img)
    #print(f1[0:10])

    img = cv2.imread('/home/lyk/machine_learning/competition/iqiyi/take_face/test_data_jpg_face/IQIYI_VID_DATA_Part1/IQIYI_VID_TRAIN/IQIYI_VID_TRAIN_0000001.mp4/IQIYI_VID_TRAIN_0000001.mp4_002.jpg/IQIYI_VID_TRAIN_0000001.mp4_002.jpg_0.jpg')
    img = model.get_input(img)
    f2 = model.get_feature(img)
    #dist = np.sum(np.square(f1-f2))
    #print(dist)
    sim = np.dot(f1, f2.T)
    print(sim)


def find_folder(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if fnmatch.fnmatch(dir, pattern):
                dirname = os.path.join(root, dir)
                yield dirname


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def makedir(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)


def to_feature_npz(path):
    head, tail = os.path.split(path)
    tail = tail + '_feature'
    dest = tail
    makedir(dest)

    for folder in find_folder(path, '*.mp4'):
        features = []
        for file in find_files(folder, '*.jpg'):
            #print(file)
            img = cv2.imread(file)
            try:
                img = model.get_input(img)
                feature = model.get_feature(img)
                features.append(feature)
            except Exception as e:
                #print(file)
                #print(e.message)
                pass

        if len(features) > 0:
            path_component = os.path.normpath(folder).split(os.path.sep)
            for i, c in enumerate(path_component):
                if c == '..':
                    path_component[i] = ''
            try:
                path_component.remove('')
            except:
                pass
            path_component[0] = dest

            features = np.vstack(tuple(features))
            #print(features.shape)
            dest_path = os.path.join(*(path_component[:-1]))
            makedir(dest_path)
            dest_file = os.path.join(*path_component)
            np.savez_compressed(dest_file, features)
        else:
            print('***** no feature \n{0}'.format(folder))



#test()
#to_feature_npz('test_data_jpg_face')
to_feature_npz('IQIYI_VID_DATA_Part1_out_jpg_face')



