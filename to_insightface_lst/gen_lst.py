#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from analysis import statistical_info
from utils import file_folder

TRAIN_PART1_FACE_PATH =\
    './IQIYI_VID_DATA_Part1_out_jpg_face/IQIYI_VID_DATA_Part1/IQIYI_VID_TRAIN'
VAL_PART1_FACE_PATH =\
    './IQIYI_VID_DATA_Part1_out_jpg_face/IQIYI_VID_DATA_Part1/IQIYI_VID_VAL'


def build_lst(label_path, face_image_path):
    filename_to_id, id_to_filename = statistical_info.build_train_dict(
        [label_path])

    result = []

    # modify to 0 based id
    for k, v in id_to_filename.iteritems():
        id = int(k) - 1
        for file in v:
            image_path = os.path.join(face_image_path, file)
            for path in file_folder.find_files(image_path, '*.jpg'):
                s = '1\t{0}\t{1}\t0\t0\t0\t0\t'.format(os.path.abspath(path), id)
                result.append(s)

    return result

if __name__ == '__main__' :
    with open('iqiyi.lst', 'w') as f:
        r = build_lst(statistical_info.TRAIN_PART1_LABEL_PATH, TRAIN_PART1_FACE_PATH)
        f.write("\n".join(r))