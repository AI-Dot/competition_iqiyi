#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_SET_ROOT = '../dataset'

TRAIN_PART1_LABEL_PATH =\
    '../dataset/IQIYI_VID_DATA_Part1/train.txt'
VALID_PART1_LABEL_PATH =\
    '../dataset/IQIYI_VID_DATA_Part1/val.txt'
TRAIN_PART1_DATA_PATH =\
    '../dataset/IQIYI_VID_DATA_Part1/IQIYI_VID_TRAIN'
VALID_PART1_DATA_PATH =\
    '../dataset/IQIYI_VID_DATA_Part1/IQIYI_VID_VAL'

TRAIN_PART2_LABEL_PATH =\
    '../dataset/IQIYI_VID_DATA_Part2/train.txt'
VALID_PART2_LABEL_PATH =\
    '../dataset/IQIYI_VID_DATA_Part2/val.txt'
TRAIN_PART2_DATA_PATH =\
    '../dataset/IQIYI_VID_DATA_Part2/IQIYI_VID_TRAIN'
VALID_PART2_DATA_PATH =\
    '../dataset/IQIYI_VID_DATA_Part2/IQIYI_VID_VAL'

TRAIN_PART3_LABEL_PATH =\
    '../dataset/IQIYI_VID_DATA_Part3/train.txt'
VALID_PART3_LABEL_PATH =\
    '../dataset/IQIYI_VID_DATA_Part3/val.txt'
TRAIN_PART3_DATA_PATH =\
    '../dataset/IQIYI_VID_DATA_Part3/IQIYI_VID_TRAIN'
VALID_PART3_DATA_PATH =\
    '../dataset/IQIYI_VID_DATA_Part3/IQIYI_VID_VAL'

def build_train_dict():
    filename_to_id = {}
    id_to_filenames = {}

    for file in [TRAIN_PART1_LABEL_PATH, TRAIN_PART2_LABEL_PATH,
                 TRAIN_PART3_LABEL_PATH]:
        lines = [line.rstrip('\n') for line in open(file)]
        for line in lines:
            file, id = line.split()
            filename_to_id[file] = id

            if id in id_to_filenames :
                id_to_filenames[id].append(file)
            else :
                id_to_filenames[id] = [file]
    return filename_to_id, id_to_filenames


def build_valid_dict():
    filename_to_id = {}
    id_to_filenames = {}

    for file in [VALID_PART1_LABEL_PATH, VALID_PART2_LABEL_PATH,
                 VALID_PART3_LABEL_PATH]:
        lines = [line.rstrip('\n') for line in open(file)]
        for line in lines:
            l = line.split()
            id = l[0]
            for filename in l[1:]:
                filename_to_id[filename] = id

                if id in id_to_filenames:
                    id_to_filenames[id].append(file)
                else:
                    id_to_filenames[id] = [file]

    return filename_to_id, id_to_filenames


if __name__ == "__main__" :
    _, train_id_to_filenames = build_train_dict()
    to_pandas = {}
    for k, v in train_id_to_filenames.iteritems():
        to_pandas[k] = len(v)

    dataframe = pandas.DataFrame.from_dict(to_pandas, orient='index',
                                           columns=['file_count'])
    dataframe = dataframe.sort_values(by=['file_count'], ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    dataframe.plot(ax=ax, color=['b'])
    fig.savefig('./imgs/train.png')


    _, val_id_to_filenames = build_valid_dict()
    to_pandas = {}
    for k, v in val_id_to_filenames.iteritems():
        to_pandas[k] = len(v)

    dataframe = pandas.DataFrame.from_dict(to_pandas, orient='index',
                                           columns=['file_count'])
    dataframe = dataframe.sort_values(by=['file_count'], ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    dataframe.plot(ax=ax, color=['g'])
    fig.savefig('./imgs/val.png')



