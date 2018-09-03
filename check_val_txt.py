#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob, os

def check_label_txt(label_txt, data_path):
    lines = [line.rstrip('\n') for line in open(label_txt)]
    filename_to_id = {}
    for line in lines:
        l = line.split()
        id = l[0]
        for filename in l[1:]:
            filename_to_id[filename] = id

    print("label length: ", len(filename_to_id))

    for mp4 in glob.glob(data_path):
        head, tail = os.path.split(mp4)
        if tail not in filename_to_id:
            print("label dose not contain: ", mp4)
        else :
            del filename_to_id[tail]

    for key, val in filename_to_id.iteritems():
        print("file not found: ", key, val)


if __name__ == "__main__":
    print("ckeck part1 val...")
    check_label_txt("../dataset/IQIYI_VID_DATA_Part1/val.txt",
                    "../dataset/IQIYI_VID_DATA_Part1/IQIYI_VID_VAL/*.mp4")

    print("ckeck part2 val...")
    check_label_txt("../dataset/IQIYI_VID_DATA_Part2/val.txt",
                    "../dataset/IQIYI_VID_DATA_Part2/IQIYI_VID_VAL/*.mp4")

    print("ckeck part3 val...")
    check_label_txt("../dataset/IQIYI_VID_DATA_Part3/val.txt",
                    "../dataset/IQIYI_VID_DATA_Part3/IQIYI_VID_VAL/*.mp4")


