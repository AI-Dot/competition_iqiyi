#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from infer import *

def build_train_id_to_filenames():
    lines = [line.rstrip('\n') for line in open(TRAIN_DICT_DATA_PATH)]
    id_to_filenames = {}
    for line in lines:
        file, id = line.split()
        if not id in id_to_filenames:
            id_to_filenames[id] = []
        id_to_filenames[id].append(file)
    return id_to_filenames


def build_train_feature_matrix_mean():
    filename_to_id = build_train_dict()
    row_index_to_id = {}
    npz_id_to_features_list = {}
    row_index = 0
    for npz in glob.glob(TRAIN_NPZ_PATH):
        head, tail = os.path.split(npz)
        mp4_filename, npz_ext = os.path.splitext(tail)
        # print(mp4_filename)

        features = np.load(npz)
        features = features.f.arr_0
        id = filename_to_id[mp4_filename]
        if not id in npz_id_to_features_list:
            npz_id_to_features_list[id] = []
        npz_id_to_features_list[id].append(features)

    feature_matrix = []
    row_index = 0
    row_index_to_id = {}
    for k, v in npz_id_to_features_list.iteritems():
        features = np.concatenate(tuple(v), axis=0)
        feature = np.mean(features, axis=0)
        feature_matrix.append([feature])
        row_index_to_id[row_index] = (k, len(v))
        row_index += 1

    # for debug
    for i in range(1, 575):
        if not str(i) in npz_id_to_features_list:
            print("** id has no features: ", i)
    print("id to features list:", len(npz_id_to_features_list))

    return np.concatenate(tuple(feature_matrix), axis=0), row_index_to_id


def calc_mean_average_precision():
    feature_matrix, row_index_to_id = build_train_feature_matrix_mean()
    print(len(row_index_to_id))
    feature_matrix = feature_matrix.T

    row_to_filename = []
    sim_list = []
    for npz in glob.glob(VALID_NPZ_PATH):
        head, tail = os.path.split(npz)
        mp4_filename, npz_ext = os.path.splitext(tail)

        row_to_filename.append(mp4_filename)

        features = np.load(npz)
        features = features.f.arr_0

        # sim[i] == class_i cos similarity
        sim = np.mean(np.dot(features, feature_matrix), axis=0)
        sim_list.append([sim])

    # sim_matrix[i, j] == file_i cos similarity to class_j
    sim_matrix = np.concatenate(tuple(sim_list), axis=0)
    print("sim matrix shape: ", sim_matrix.shape)
    print("sim matrix max:{0}, min:{1}".format(sim_matrix.max(), sim_matrix.min()))

    # calc class_i average precision
    filename_to_id = build_valid_dict()
    class_to_ap = {}
    for i in range(sim_matrix.shape[1]):
        class_id, gt_count = row_index_to_id[i]
        if gt_count > 100:
            gt_count = 100
        confidence = sim_matrix[:, i]

        # sort descending order
        arg_sort = np.argsort(-confidence)

        position = 1
        hit_count = 1
        precision = []
        for index in arg_sort:
            filename = row_to_filename[index]
            id = filename_to_id[filename]
            if id == class_id:
                precision.append(hit_count / position)
                hit_count += 1
            position += 1
            if position > gt_count:
                class_to_ap[class_id] = np.sum(np.array(precision)) / gt_count
                #print(class_to_ap[class_id])

    map = 0.0
    for ap in class_to_ap.values():
        map += ap
    map = map / len(class_to_ap)
    return map


if __name__ == "__main__":
    print(calc_mean_average_precision())

