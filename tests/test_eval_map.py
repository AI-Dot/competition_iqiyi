#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import eval_map


# prepare test data
                # (id, [confidence], gt_count, expected ap)
test_data_1 = [('1', [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, 0.1], 6, 1.0),
               ('2', [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.8, 0.9], 4, 1.0)]

test_data_2 = [('1', [1.0, 0.9, 0.15, 0.16, 0.17, 0.55, 0.4, 0.3, 0.2, 0.1], 6,
    # sort by confidence
    #                 [1.0, 0.9, 0.55, 0.4, 0.3, 0.2, 0.17, 0.16, 0.15, 0.1]
    # arg sort result [ 0,   1,   5,    6,   7,   8,   4,    3,    2,    9]
    # ap:
                (1/1+2/2 +3/3) / 6),
               ('2', [0.0, 0.1, 0.85, 0.84, 0.83, 0.45, 0.6, 0.7, 0.8, 0.9], 4,
    # sort by confidence
    #                 [0.9, 0.85, 0.84, 0.83, 0.8, 0.7, 0.6, 0.45, 0.1, 0.0]
    # arg sort result [ 9,   2,   3,    4,   8,   7,   6,    5,    1,    0]
    # ap:
                (1/1) / 4)
                ]


def index_to_class_id(index):
    mapping = {}
    mapping[0] = '1'
    mapping[1] = '1'
    mapping[2] = '1'
    mapping[3] = '1'
    mapping[4] = '1'
    mapping[5] = '1'

    mapping[6] = '2'
    mapping[7] = '2'
    mapping[8] = '2'
    mapping[9] = '2'

    return mapping[index]


def test_map_1():
    ap_1 = eval_map.calc_ap(test_data_1[0][0], np.array(test_data_1[0][1]),
                            test_data_1[0][2], index_to_class_id)
    assert(abs(ap_1 - test_data_1[0][3]) < 0.00001)

    ap_2 = eval_map.calc_ap(test_data_1[1][0], np.array(test_data_1[1][1]),
                            test_data_1[1][2], index_to_class_id)
    assert (abs(ap_2 - test_data_1[1][3]) < 0.00001)

    assert(abs((ap_1 + ap_2) / 2 - (test_data_1[0][3] + test_data_1[1][3]) / 2) < 0.00001)
    print((ap_1 + ap_2) / 2)


def test_map_2():
    ap_1 = eval_map.calc_ap(test_data_2[0][0], np.array(test_data_2[0][1]),
                            test_data_2[0][2], index_to_class_id)
    assert (abs(ap_1 - test_data_2[0][3]) < 0.00001)

    ap_2 = eval_map.calc_ap(test_data_2[1][0], np.array(test_data_2[1][1]),
                            test_data_2[1][2], index_to_class_id)
    assert (abs(ap_2 - test_data_2[1][3]) < 0.00001)

    assert (abs((ap_1 + ap_2) / 2 - (
                test_data_2[0][3] + test_data_2[1][3]) / 2) < 0.00001)
    print((ap_1 + ap_2) / 2)

if __name__ == "__main__":
    test_map_1()
    test_map_2()