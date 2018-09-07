#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from concurrent.futures import ProcessPoolExecutor
import os
import fnmatch
import multiprocessing
import functools

import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def find_files(directory, pattern):
    result = []

    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                result.append(filename)

    return result


def find_dirs(directory, pattern):
    result = {}

    for root, dirs, files in os.walk(directory):
        for basename in dirs:
            if fnmatch.fnmatch(basename, pattern):
                dir_name = os.path.join(root, basename)
                result[basename] = dir_name

    return result


def makedir(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)


def take_face(path):
    print('take_face ', path)
    head, tail = os.path.split(path)
    tail = tail + '_face'
    dest = tail
    makedir(dest)

    file_list = find_files(path, '*.jpg')

    tasks = []
    for i, c in enumerate(chunks(file_list, 20000)):
        tasks.append((i, c))


    with ProcessPoolExecutor(int(multiprocessing.cpu_count() / 2)) as executor:
        executor.map(functools.partial(face_to_file, dest), tasks)


def check_take_face(path):
    print('check_take_face ', path)
    head, tail = os.path.split(path)
    tail = tail + '_face'
    dest = tail
    makedir(dest)

    not_complete = check_result(path, dest)

    file_list = []
    for d in not_complete:
        file_list += find_files(d, '*.jpg')

    tasks = []
    for i, c in enumerate(chunks(file_list, 20000)):
        tasks.append((i, c))

    with ProcessPoolExecutor(int(multiprocessing.cpu_count() / 2)) as executor:
        executor.map(functools.partial(face_to_file, dest), tasks)


def face_to_file(dest, task):

    # at least have three gpu, uses second and third gpu
    detector = MtcnnDetector(model_folder='model',
                             ctx=mx.cpu(0),
                             #ctx=mx.gpu(int(task[0] / 2) + 1),
                             num_worker=4,
                             accurate_landmark=False)

    for file in task[1]:
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
                cv2.imwrite('{0}/{1}_{2}.jpg'.format(jpg_dest, basename, i),
                            chip)
        else:
            print('no face in ', file)


def check_result(src_path, dest_path):
    src_dirs = find_dirs(src_path, '*.mp4')
    dest_dirs = find_dirs(dest_path, '*.mp4')

    diff = list(set(src_dirs.keys()) - set(dest_dirs.keys()))

    result = []
    for d in diff:
        result.append(src_dirs[d])

    return result



if __name__ == '__main__':
    #take_face('./IQIYI_VID_DATA_Part1_out_jpg')

    take_face('./IQIYI_VID_DATA_Part2_out_jpg')
    #check_take_face('./IQIYI_VID_DATA_Part2_out_jpg')

    #take_face('./IQIYI_VID_DATA_Part3_out_jpg')
    #check_take_face('./IQIYI_VID_DATA_Part3_out_jpg')

    #take_face('./test_data_jpg_1')
