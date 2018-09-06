import re
import os
import glob
import fnmatch
from collections import defaultdict

DATASET_DIR = '/Users/home/Desktop/iqiyi_report/IQIYI_VID_DATA_Part1_out_jpg_face'
TRAIN_DICT_DATA_PATH = os.path.join(DATASET_DIR,'train.txt')
TRAIN_DIR = os.path.join(DATASET_DIR, 'IQIYI_VID_TRAIN')
VAL_DIR = os.path.join(DATASET_DIR, 'IQIYI_VID_VAL')

error_report_md = open('report.md','w')
misclassified_txt = open('misclassified.txt', 'r')

def build_train_dict():
    lines = [line.rstrip('\n') for line in open(TRAIN_DICT_DATA_PATH)]
    filename_to_id = defaultdict(list)
    for line in lines:
        file, id = line.split()
        filename_to_id[id].append(file)
    return filename_to_id

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def draw_mp4_images(mp4_name):
    mp4_name_dir = os.path.join(VAL_DIR, mp4_name)
#    print(mp4_name_dir)
    for file in find_files(mp4_name_dir, '*.jpg'):
        error_report_md.write('![]({})'.format(file))

    error_report_md.write('\n')


def draw_gt_id_images(gt_id, train_dict):

    for mp4_name in train_dict[gt_id]:

        mp4_name_dir = os.path.join(TRAIN_DIR, mp4_name)

        for file in find_files(mp4_name_dir, '*.jpg'):
            error_report_md.write('![]({})'.format(file))
           
    error_report_md.write('\n')


def draw_predict_id_images(predict_id, train_dict):

    for mp4_name in train_dict[predict_id]:

        mp4_name_dir = os.path.join(TRAIN_DIR, mp4_name)

        for file in find_files(mp4_name_dir, '*.jpg'):
            error_report_md.write('![]({})'.format(file))
           
    error_report_md.write('\n') 


def write_into_markdown(mp4_name, gt_id, predict_id):
    # title error mp4 file
    error_report_md.write('# {}\n'.format(mp4_name))
    draw_mp4_images(mp4_name) 

    #build train dict
    train_dict = build_train_dict()

    # title gt_id
    error_report_md.write('## GT:{}\n'.format(gt_id))
    draw_gt_id_images(gt_id, train_dict)

    # title predict_id 
    error_report_md.write('## GUESS:{}\n'.format(predict_id))
    draw_predict_id_images(predict_id, train_dict)

    error_report_md.write('\n')

for line in misclassified_txt.readlines():
    if line == '\n':
        continue

    mp4_name, gt_id, predict_id  = line.rstrip('\n').split(' ')
    print(mp4_name)
    print(gt_id)
    print(predict_id)

    write_into_markdown(mp4_name, gt_id, predict_id)

    

    
