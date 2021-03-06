# coding: utf-8
"""
@author Liuchen
2019
"""

import time
import os
import shutil

import tools
import random
import math
import numpy as np
import tensorflow as tf
from sklearn import metrics
import dnn_model as dm
import data_prepare as dp

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger('main')


def loop_dataset(model, params, g_list, sess, batch_size=50, train=True):
    """
    在数据集上作一个epoch迭代，训练、验证或测试模型，返回(平均损失, 平均正确率, AUC)

    model: 分类器
    params: 模型参数
    g_list: 图列表
    batch_size: mini-batch 大小
    return: (平均损失, 平均正确率)
    """
    total_loss = []   # 所有batch的损失之和和精度之和
    total_iters = math.ceil(len(g_list) / batch_size)
    total_labels = []  # 所有数据标签
    total_predicts = []   # 所有数据的模型输出
    if train:
        random.shuffle(g_list)
    for pos in range(total_iters):
        batch_graphs = g_list[pos * batch_size: (pos + 1) * batch_size]

        ajacent, features, batch_label, batch_label_fs, batch_label_bn, batch_label_pn, batch_label_ar, \
        dgree_inv, graph_indexes = dp.batching(batch_graphs, params)
        total_labels += batch_label

        feed_dict = {
            model.ajacent: ajacent,
            model.features: features,
            model.labels: batch_label,
            model.labels_fs: batch_label_fs,
            model.labels_bn: batch_label_bn,
            model.labels_pn: batch_label_pn,
            model.labels_ar: batch_label_ar,
            model.dgree_inv: dgree_inv,
            model.graph_indexes: graph_indexes,
        }

        if train:
            feed_dict[model.keep_prob] = params.keep_prob
            feed_dict[model.learning_rate] = params.learning_rate
            to_run = [model.predicts, model.loss, model.accuracy, model.optimizer]
            predicts, loss, acc, _ = sess.run(to_run, feed_dict=feed_dict)
            result = sess.run(merged, feed_dict=feed_dict)
            summary_writer.add_summary(result, i)
        else:
            feed_dict[model.keep_prob] = 1
            to_run = [model.predicts, model.loss, model.accuracy]
            timeb1 = time.time()
            predicts, loss, acc = sess.run(to_run, feed_dict=feed_dict)
            timee1 = time.time()
            time_delay1 = timee1 - timeb1
            logger.info((f"time_delay-{time_delay1}"))

        total_predicts.extend(predicts)  # detach用于安全获取数据

        total_loss.append(np.array([loss, acc]) * len(batch_graphs))  # 当前batch的损失之和、精度之和

    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / len(g_list)

    total_labels = np.argmax(total_labels, 1)
    fpr, tpr, _ = metrics.roc_curve(total_labels, total_predicts, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    loss_acc_auc = np.concatenate((avg_loss, [auc]))

    return loss_acc_auc  # (平均损失, 平均正确率, auc)


# ================== step0: 参数准备 =================

params = tools.Parameters()
params.set("k", 44)          # 不能小于conv1d_kernel_size[1]*2，否则没法做第2个1d卷积
params.set("conv1d_channels", [16, 32])
params.set("conv1d_kernel_size", [0, 10])
params.set("dense_dim", 128)
params.set("gcnn_dims", [64, 32, 32, 1])
params.set("learning_rate", 0.0001)
params.set("keep_prob", 0.99)
params.set("loss_algorithm", 'loss_mean')
params.set("mmoe_expert_num", 5)
params.set("mtl_mode", 'mmoe')

epoch_num = 200
batch_size = 50
# ================== step1: 数据准备 =================

train_set, test_set, param = dp.load_data('ALARM_', 0, 1)
logger.info(f"训练数据量 {len(train_set)}，测试数据量 {len(test_set)}")

params.extend(param)
print(params)

model = dm.DGCNN(params)
model.build()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    shutil.rmtree(f'./log/{params.mtl_mode}/{params.loss_algorithm}/batch_size_{batch_size}')
    os.mkdir(f'./log/{params.mtl_mode}/{params.loss_algorithm}/batch_size_{batch_size}')
    summary_writer = tf.summary.FileWriter(f'./log/{params.mtl_mode}/{params.loss_algorithm}/batch_size_{batch_size}', sess.graph)


    # train
    for i in range(epoch_num):
        loss, acc, auc = loop_dataset(model, params, train_set, sess, batch_size)
        logger.info(f"train epoch {i:<3}/{epoch_num:<3} -- loss-{loss:<9.6} -- acc-{acc:<9.6} -- auc-{auc:<9.6}")
    # test
    timeb = time.time()
    loss, acc, auc = loop_dataset(model, params, test_set, sess, train=False)
    timee = time.time()
    time_delay = timee - timeb
    logger.info(f"TEST {'>'*14} -- loss-{loss:<9.6} -- acc-{acc:<9.6} -- auc-{auc:<9.6} -- time_delay-{time_delay}")
