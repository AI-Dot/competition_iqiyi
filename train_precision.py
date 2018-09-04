#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

TRAIN_DICT_DATA_PATH =\
    '/DATA/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part2_out_jpg_face_feature/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part2_out_jpg_face/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part2_out_jpg/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part2/train.txt'
TRAIN_NPZ_PATH =\
    '/DATA/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part2_out_jpg_face_feature/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part2_out_jpg_face/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part2_out_jpg/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part2/IQIYI_VID_TRAIN/*.npz'

#TRAIN_DICT_DATA_PATH =\
#    'IQIYI_VID_DATA_Part1/IQIYI_VID_DATA_Part1_out_jpg_face_feature/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part1/train.txt'
#TRAIN_NPZ_PATH =\
#    'IQIYI_VID_DATA_Part1/IQIYI_VID_DATA_Part1_out_jpg_face_feature/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part1/IQIYI_VID_TRAIN/*.npz'



def build_train_dict():
    lines = [line.rstrip('\n') for line in open(TRAIN_DICT_DATA_PATH)]
    filename_to_id = {}
    for line in lines:
        file, id = line.split()
        filename_to_id[file] = id
    return filename_to_id

def build_train_feature_matrix(network):
    row_index_to_filename = {}
    npz_list = []
    row_index = 0
    for npz in glob.glob(TRAIN_NPZ_PATH):
        head, tail = os.path.split(npz)
        mp4_filename, npz_ext = os.path.splitext(tail)
        #print(mp4_filename)

        features = np.load(npz)
        features = features.f.arr_0
        features = network.predict(features)
        npz_list.append(features)
        for index in range(features.shape[0]):
            row_index_to_filename[row_index] = mp4_filename
            row_index += 1

    feature_matrix = np.concatenate(tuple(npz_list), axis=0)
    print(feature_matrix.shape)
    return feature_matrix, row_index_to_filename

def build_train_feature_matrix_mean(network):
    filename_to_id = build_train_dict()
    npz_id_to_features_list = {}
    for npz in glob.glob(TRAIN_NPZ_PATH):
        head, tail = os.path.split(npz)
        mp4_filename, npz_ext = os.path.splitext(tail)
        # print(mp4_filename)

        features = np.load(npz)
        features = features.f.arr_0
        features = network.predict(features)
        class_id = filename_to_id[mp4_filename]
        if class_id not in npz_id_to_features_list:
            npz_id_to_features_list[class_id] = []
        npz_id_to_features_list[class_id].append(features)

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

def infer_2(network):
    train_dict = build_train_dict()
    feature_matrix, row_index_to_id = build_train_feature_matrix_mean(network)
    feature_matrix = feature_matrix.T

    total = 0
    hit = 0

    print(TRAIN_NPZ_PATH)

    for npz in tqdm(glob.glob(TRAIN_NPZ_PATH)):
        head, tail = os.path.split(npz)
        mp4_filename, npz_ext = os.path.splitext(tail)
        # print(mp4_filename)
        features = np.load(npz)
        features = features.f.arr_0
        features = network.predict(features)
        assert (len(features.shape) == 2)

        sim = np.dot(features, feature_matrix)
        sim = np.sum(sim, axis=0)
        index = np.argmax(sim)

        predict_id = row_index_to_id[index][0]
        ground_truth = train_dict[mp4_filename]
        if predict_id == ground_truth:
            hit += 1
        total += 1

    print(
        'infer2, hit: {0}, total: {1}, precision: {2}'.format(hit, total, hit / total))

def infer(network):
    train_dict = build_train_dict()
    feature_matrix, row_index_to_filename = build_train_feature_matrix(network)
    feature_matrix = feature_matrix.T

    total = 0
    hit = 0

    for npz in tqdm(glob.glob(TRAIN_NPZ_PATH)):
        head, tail = os.path.split(npz)
        mp4_filename, npz_ext = os.path.splitext(tail)

        features = np.load(npz)
        features = features.f.arr_0
        features = network.predict(features)
        sim = np.dot(features, feature_matrix)
        sim = np.sum(sim, axis=0)
        index = np.argmax(sim)

        predict_id = train_dict[row_index_to_filename[index]]
        ground_truth = train_dict[mp4_filename]
        if predict_id == ground_truth:
            hit += 1
        total += 1

    print('infer, hit: {0}, total: {1}, precision: {2}'.format(hit, total, hit / total))

def get_fc_embNet(embeddings, w_init):
    x = tf.layers.dense(inputs=embeddings, units=2048, activation=tf.nn.relu, kernel_initializer=w_init)
    x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu, kernel_initializer=w_init)
    return x 

class Net:
    def __init__(self, model_path):
        self.embeddings = tf.placeholder(name='emb_inputs', shape=[None, 512], dtype=tf.float32)
        w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
        self.output = get_fc_embNet(self.embeddings, w_init_method)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()

        self.saver.restore(self.sess, model_path)

    def predict(self, embedding):
        return self.sess.run(self.output, feed_dict={self.embeddings: embedding})


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    MODEL_PATH = '/home/huxuhua/InsightFace_TF/output/ckpt/InsightFace_iter_110000.ckpt'
    network = Net(MODEL_PATH)

    infer(network)
    infer_2(network)
