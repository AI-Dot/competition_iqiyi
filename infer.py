#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os
import numpy as np

TRAIN_DICT_DATA_PATH =\
    './IQIYI_VID_DATA_Part1_out_jpg_face_feature/IQIYI_VID_DATA_Part1/train.txt'
VALID_DICT_DATA_PATH =\
    './IQIYI_VID_DATA_Part1_out_jpg_face_feature/IQIYI_VID_DATA_Part1/val.txt'
TRAIN_NPZ_PATH =\
    'IQIYI_VID_DATA_Part1_out_jpg_face_feature/IQIYI_VID_DATA_Part1/IQIYI_VID_TRAIN/*.npz'
VALID_NPZ_PATH =\
    'IQIYI_VID_DATA_Part1_out_jpg_face_feature/IQIYI_VID_DATA_Part1/IQIYI_VID_VAL/*.npz'

def build_train_dict():
    lines = [line.rstrip('\n') for line in open(TRAIN_DICT_DATA_PATH)]
    filename_to_id = {}
    for line in lines:
        file, id = line.split()
        filename_to_id[file] = id
    return filename_to_id


def build_valid_dict():
    lines = [line.rstrip('\n') for line in open(VALID_DICT_DATA_PATH)]
    filename_to_id = {}
    for line in lines:
        l = line.split()
        id = l[0]
        for filename in l[1:]:
            filename_to_id[filename] = id

    return filename_to_id


def build_train_feature_matrix():
    row_index_to_filename = {}
    npz_list = []
    row_index = 0
    for npz in glob.glob(TRAIN_NPZ_PATH):
        head, tail = os.path.split(npz)
        mp4_filename, npz_ext = os.path.splitext(tail)
        #print(mp4_filename)

        features = np.load(npz)
        features = features.f.arr_0
        npz_list.append(features)
        for index in range(features.shape[0]):
            row_index_to_filename[row_index] = mp4_filename
            row_index += 1

    feature_matrix = np.concatenate(tuple(npz_list), axis=0)
    print(feature_matrix.shape)
    return feature_matrix, row_index_to_filename


def infer():
    train_dict = build_train_dict()
    valid_dict = build_valid_dict()
    feature_matrix, row_index_to_filename = build_train_feature_matrix()
    feature_matrix = feature_matrix.T

    total = 0
    hit = 0

    misclassified = {}

    for npz in glob.glob(VALID_NPZ_PATH):
        head, tail = os.path.split(npz)
        mp4_filename, npz_ext = os.path.splitext(tail)

        features = np.load(npz)
        features = features.f.arr_0

        sim = np.dot(features, feature_matrix)
        sim = np.mean(sim, axis=0)
        index = np.argmax(sim)

        predict_id = train_dict[row_index_to_filename[index]]
        if mp4_filename in valid_dict :
            ground_truth = valid_dict[mp4_filename]
        else :
            continue
        if predict_id == ground_truth:
            hit += 1
        else :
            misclassified[mp4_filename] = (ground_truth, predict_id)
        total += 1

    print('hit: {0}, total: {1}, precision: {2}'.format(hit, total, hit / total))
    print('miss predict: ')
    for key, val in misclassified.iteritems():
        print('{0} {1} {2}'.format(key, val[0], val[1]))


if __name__ == "__main__":
    #build_train_dict()
    #build_train_feature_matrix()
    infer()