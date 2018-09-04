import tensorflow as tf
import numpy as np
import argparse
import glob
import os
from losses.face_losses import arcface_loss, cosineface_losses
from tensorflow.core.protobuf import config_pb2
import time
from data.eval_data_reader import load_bin
from verification import ver_test


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--net_depth', default=100, help='resnet depth, default is 50')
    parser.add_argument('--epoch', default=100000, help='epoch to train the network')
    parser.add_argument('--batch_size', default=4096, help='batch size to train network')
    parser.add_argument('--lr_steps', default=[40000, 60000, 80000], help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--num_output', default=5000, help='the image size')
    parser.add_argument('--tfrecords_file_path', default='./datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=10000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=20, help='intervals to save ckpt file')
    args = parser.parse_args()
    return args

def get_fc_embNet(embeddings, w_init, dropout_rate):
    x = tf.layers.dense(inputs=embeddings, units=2048, activation=tf.nn.relu, kernel_initializer=w_init)
    x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu, kernel_initializer=w_init)
    return x 

PATH_DIR='/DATA/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part2_out_jpg_face_feature/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part2_out_jpg_face/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part2_out_jpg/disk1/huxuhua/IQIYI/IQIYI_VID_DATA_Part2'
TRAIN_NPZ_PATH = os.path.join(PATH_DIR, 'IQIYI_VID_TRAIN/*.npz') 
TRAIN_DICT_DATA_PATH =\
     os.path.join(PATH_DIR, 'train.txt')

def build_train_dict():
    lines = [line.rstrip('\n') for line in open(TRAIN_DICT_DATA_PATH)]
    filename_to_id = {}
    for line in lines:
        file, id = line.split()
        filename_to_id[file] = id

    return filename_to_id

def build_train_feature_matrix():
    row_index_to_filename = {}
    npz_list = []
    row_index = 0
    print(TRAIN_NPZ_PATH) 
    for npz in glob.glob(TRAIN_NPZ_PATH):
        head, tail = os.path.split(npz)
        mp4_filename, npz_ext = os.path.splitext(tail)
        # print(mp4_filename)
        features = np.load(npz)
        features = features.f.arr_0
        npz_list.append(features)
        for index in range(features.shape[0]):
            row_index_to_filename[row_index] = mp4_filename
            row_index += 1


    feature_matrix = np.concatenate(tuple(npz_list), axis=0)
    print(feature_matrix.shape)

    return feature_matrix, row_index_to_filename

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 1. define global parameters
    args = get_parser()
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    embeddings = tf.placeholder(name='emb_inputs', shape=[None, 512], dtype=tf.float32)
    labels = tf.placeholder(name='emb_labels', shape=[None, ], dtype=tf.int64)
    # trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)
    # 2 prepare train datasets and test datasets by using tensorflow dataset api
    # 2.1 train datasets
    # the image is substracted 127.5 and multiplied 1/128.
    # random flip left right

    features, row_index_to_filename = build_train_feature_matrix()
    filename_to_id = build_train_dict()
    label_values = []
    for row_index in range(features.shape[0]):
        label_values.append(filename_to_id[row_index_to_filename[row_index]])
    
    label_values = np.array(label_values)

#    print(features.shape)
#    print(labels.shape)
#    labels_first3 = labels[:3].copy()
#    features_first3 = features[:3].copy()
#    print('features_first3 before shuffle:{}'.format(features_first3))
#    print('labels_first3 before shuffle:{}'.format(labels_first3))
#    indices = np.arange(3)
#    print('indices before shuffle: {}'.format(indices))
#    np.random.shuffle(indices)
#    print('indices after shuffle: {}'.format(indices))
#
#    print('features_first3 after shuffle:{}'.format(features_first3[indices]))
#    print('labels_first3 after shuffle:{}'.format(labels_first3[indices]))
#   
#    exit() # debug point 1



    # 3. define network, loss, optimize method, learning rate schedule, summary writer, saver
    # 3.1 inference phase
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    fc_net = get_fc_embNet(embeddings, w_init=w_init_method, dropout_rate=dropout_rate)

    # 3.2 get cosineface_losses
    logit = cosineface_losses(embedding=fc_net, labels=labels, w_init=w_init_method, out_num=args.num_output)
    # test net  because of batch normal layer

    # 3.3 define the cross entropy
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))

    # 3.5 total losses
    total_loss = inference_loss
    # 3.6 define the learning rate schedule
    p = int(512.0/args.batch_size)
    lr_steps = [p*val for val in args.lr_steps]
    print(lr_steps)
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001], name='lr_schedule')
    # 3.7 define the optimize method
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)
    # 3.8 get train op
    grads = opt.compute_gradients(total_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)
    # train_op = opt.minimize(total_loss, global_step=global_step)
    # 3.9 define the inference accuracy used during validate or test
    pred = tf.nn.softmax(logit)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))
    # 3.10 define sess
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    # 3.11 summary writer
    summary = tf.summary.FileWriter(args.summary_path, sess.graph)
    summaries = []
    # # 3.11.1 add grad histogram op
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # 3.11.2 add trainabel variable gradients
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    # 3.11.3 add loss summary
    summaries.append(tf.summary.scalar('inference_loss', inference_loss))
    # 3.11.4 add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    summary_op = tf.summary.merge(summaries)
    # 3.12 saver
    saver = tf.train.Saver(max_to_keep=args.saver_maxkeep)
    # 3.13 init all variables
    sess.run(tf.global_variables_initializer())

    # restore_saver = tf.train.Saver()
    # restore_saver.restore(sess, '/home/aurora/workspaces2018/InsightFace_TF/output/ckpt/InsightFace_iter_1110000.ckpt')
    # 4 begin iteration
    if not os.path.exists(args.log_file_path):
        os.makedirs(args.log_file_path)
    log_file_path = args.log_file_path + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
    log_file = open(log_file_path, 'w')
    # 4 begin iteration
    count = 0
    total_accuracy = {}
    
#    exit() # debug point 2

    for i in range(args.epoch):
        indices = np.arange(features.shape[0])
        np.random.shuffle(indices)
        features = features[indices]
        label_values = label_values[indices]
        num_of_features = features.shape[0]
        num_of_batches = np.ceil( num_of_features / args.batch_size).astype(np.int32)
        size_of_last_batch = int(num_of_features % args.batch_size)
        one_of_last_batch = 1 if size_of_last_batch > 0 else 0

        for j in range(num_of_batches + one_of_last_batch):
            features_train = None
            labels_train = None
            start = j*args.batch_size

            if j < num_of_batches:
                end = (j+1)*args.batch_size
                features_train = features[start:end].copy()
                labels_train = label_values[start:end].copy()

            else:
                features_train = features[start:].copy()
                labels_train =  label_values[start:].copy()

            assert not features_train is None
            assert not labels_train is None

            feed_dict = {embeddings: features_train, labels: labels_train}
            # feed_dict.update(net.all_drop)
            start_time = time.time()
            _, total_loss_val, inference_loss_val,  _, acc_val = \
                   sess.run([train_op, total_loss, inference_loss, inc_op, acc],
                             feed_dict=feed_dict,
                             options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            end_time = time.time()
            pre_sec = args.batch_size/(end_time - start_time)

            # print training information
            if count > 0 and count % args.show_info_interval == 0:
                  print('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f,' 
                        ' training accuracy is %.6f, time %.3f samples/sec' %
                        (i, count, total_loss_val, inference_loss_val, acc_val, pre_sec))
            count += 1

            # save summary
            if count > 0 and count % args.summary_interval == 0:
                feed_dict = {embeddings: features_train, labels: labels_train}
                # feed_dict.update(net.all_drop)
                summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                summary.add_summary(summary_op_val, count)

            # save ckpt files
            if count > 0 and count % args.ckpt_interval == 0:
                filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                filename = os.path.join(args.ckpt_path, filename)
                saver.save(sess, filename)

    log_file.close()
    log_file.write('\n')
