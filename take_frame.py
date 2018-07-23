#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import subprocess
import os

def take_frame(path):
    head, tail = os.path.split(path)
    tail = tail + '_out_jpg'
    dest = tail
    makedir(dest)

    for file in glob.iglob(os.path.join(path, '*', '*.mp4')):
        path_component = os.path.normpath(file).split(os.path.sep)

        for i, c in enumerate(path_component):
            if c == '..':
                path_component[i] = ''
        path_component.remove('')
        path_component[0] = dest

        basename = os.path.basename(file)
        jpg_dest = os.path.join(*path_component)
        makedir(jpg_dest)

        command = 'ffmpeg -loglevel quiet -i {0} -vf fps=1 {1}/{2}_%03d.jpg'.format(
            file, jpg_dest, basename)

        try:
            subprocess.check_output(command, shell=True)
        except Exception as e:
            print(e.message)
            print('error {0}'.format(command))


def makedir(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)


#take_frame('test_data')
take_frame('../dataset/IQIYI_VID_DATA_Part2')
