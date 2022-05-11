# coding: utf-8
"""
@author Liuchen
2019
"""
import math

import tensorflow as tf
import tools
import numpy as np
import logging

logger = logging.getLogger('main.dnn_model')


class DGCNN:
    def __init__(self, hyper_params=None):
        '''
        超参数输入方法1：以HyperParams对象的方式输入参数

        参数：
        class_num       分类类别数量
        embed_size      词向量维度
        lstm_sizes      RNN隐层维度，可有多层RNN
        vocab_size      词典大小
        embed_matrix    词向量矩阵
        fc_size         全连接层大小
        max_sent_len    最大句长
        isBiRNN         * 是否使用双向RNN
        refine          * 词向量是否refine
        '''
        tf.reset_default_graph()  # 清空计算图，否则连续多次执行会出错
        if hyper_params and not isinstance(hyper_params, tools.Parameters):
            raise Exception(f'hyper_params must be an object of {type(tools.Parameters)} --- by LIC')

        # 默认参数
        default_params = {
            'gcnn_dims': (32, 32, 32, 1)
        }
        hyper_params.default(default_params)
        self.hypers = hyper_params  # 所有超参数都保存在其中


    def weight_matrix(self, shape, name=None):
        """
        权值矩阵及初始化
        Glorot & Bengio (AISTATS 2010) init.
        """
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    def set_public(self):
        '''
        placeholder 参数
        '''
        with tf.name_scope("place_hoders"):
            self.learning_rate = tf.placeholder_with_default(0.01, shape=(), name='learning_rate')  # 学习率
            self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')  # dropout keep probability
            # self.l2reg = tf.placeholder_with_default(0.0, shape=(), name='L2reg')         # L2正则化参数
            self.ajacent = tf.sparse_placeholder(tf.float32, name="batch_adjacent")  # 邻接矩阵
            self.features = tf.placeholder(tf.float32, (None, self.hypers.feature_dim),
                                           name="batch_node_features")  # 节点特征
            self.dgree_inv = tf.sparse_placeholder(tf.float32, name="batch_degree_inv")  # 节点度矩阵的逆
            self.graph_indexes = tf.placeholder(tf.int32, (None, 2), name="batch_indecis")  # batch中每个网络节点特征的始未位置
            self.labels = tf.placeholder(tf.int32, [None, self.hypers.class_num], name='labels')  # 网络标签
            self.labels_fs = tf.placeholder(tf.int32, [None, self.hypers.fault_source_num], name='labels_fs')  # 故障网元
            self.labels_bn = tf.placeholder(tf.int32, [None, self.hypers.board_number_num], name='labels_bn')  # 故障单板
            self.labels_pn = tf.placeholder(tf.int32, [None, self.hypers.port_number_num], name='labels_pn')  # 故障端口
            self.labels_ar = tf.placeholder(tf.int32, [None, self.hypers.alarm_root_num], name='labels_ar')  # 根告警

    def gcnn_layer(self, input_Z, in_dim, out_dim, layer_id):
        """
        一个DGCNN层
        """
        with tf.name_scope(f"gcnn_layer_{layer_id}"):
            W = self.weight_matrix(shape=(in_dim, out_dim), name=f"dgcnn_W_{layer_id}")
            tf.summary.histogram(f'gcn_layer_{layer_id}/weights', W)
            AZ = tf.sparse_tensor_dense_matmul(self.ajacent, input_Z)  # AZ
            AZ = tf.add(AZ, input_Z)  # AZ+Z = (A+I)Z
            AZW = tf.matmul(AZ, W)  # (A+I)ZW
            DAZW = tf.sparse_tensor_dense_matmul(self.dgree_inv, AZW)  # D^-1AZW

        return tf.nn.tanh(DAZW)  # tanh 激活

    def gcnn_layers(self):
        """
        多个gcnn层
        """
        with tf.name_scope("gcnn_layers"):
            Z1_h = []
            in_dim = self.hypers.feature_dim
            Z = self.features
            for i, dim in enumerate(self.hypers.gcnn_dims):  # 多个GCNN层
                out_dim = dim
                Z = self.gcnn_layer(Z, in_dim, out_dim, i)
                in_dim = out_dim
                Z1_h.append(Z)
            Z1_h = tf.concat(Z1_h, 1)  # 拼接每个层的Z
        return Z1_h

    def sortpooling_layer(self, gcnn_out):
        def sort_a_graph(index_span):
            indices = tf.range(index_span[0], index_span[1])  # 获取单个图的节点特征索引
            graph_feature = tf.gather(gcnn_out, indices)  # 获取单个图的全部节点特征

            graph_size = index_span[1] - index_span[0]
            k = tf.cond(self.hypers.k > graph_size, lambda: graph_size, lambda: self.hypers.k)  # k与图size比较
            # 根据最后一列排序，返回前k个节点的特征作为图的表征
            top_k = tf.gather(graph_feature, tf.nn.top_k(graph_feature[:, -1], k=k).indices)

            # 若图size小于k，则补0行
            zeros = tf.zeros([self.hypers.k - k, sum(self.hypers.gcnn_dims)], dtype=tf.float32)
            top_k = tf.concat([top_k, zeros], 0)
            return top_k

        with tf.name_scope("sort_pooling_layer"):
            sort_pooling = tf.map_fn(sort_a_graph, self.graph_indexes, dtype=tf.float32)
        return sort_pooling

    def sortpooling_layer_fs(self, gcnn_out):
        def sort_a_graph(index_span):
            indices = tf.range(index_span[0], index_span[1])  # 获取单个图的节点特征索引
            graph_feature = tf.gather(gcnn_out, indices)  # 获取单个图的全部节点特征

            return graph_feature
        with tf.name_scope("sort_pooling_layer_fs"):
            sort_pooling = tf.map_fn(sort_a_graph, self.graph_indexes, dtype=tf.float32)
        return sort_pooling

    def cnn1d_layers(self, inputs):
        """
        两个1维cnn层
        """
        with tf.name_scope("cnn1d_layers"):
            total_dim = sum(self.hypers.gcnn_dims)
            graph_embeddings = tf.reshape(inputs, [-1, self.hypers.k * total_dim, 1])  # (batch, width, channel)

            # 第一个1d CNN层，以及MaxPooling层
            if self.hypers.conv1d_kernel_size[0] == 0:
                self.hypers.conv1d_kernel_size[0] = total_dim
            cnn1 = tf.layers.conv1d(graph_embeddings,
                                    self.hypers.conv1d_channels[0],  # channel
                                    self.hypers.conv1d_kernel_size[0],  # kernel_size
                                    self.hypers.conv1d_kernel_size[0])  # stride
            act1 = tf.nn.relu(cnn1)
            pooling1 = tf.layers.max_pooling1d(act1, 2, 2)  # (value, kernel_size, stride)

            # 第二个1d CNN层
            cnn2 = tf.layers.conv1d(pooling1, self.hypers.conv1d_channels[1], self.hypers.conv1d_kernel_size[1], 1)
            act2 = tf.nn.relu(cnn2)

        return act2


    def fc_layer(self, inputs):
        """
        全连接层
        """
        with tf.name_scope("fc_layer"):
            # for batch data reshape
            batchsize = tf.shape(self.graph_indexes)[0]
            graph_embed_dim = int((self.hypers.k - 2) / 2 + 1)
            graph_embed_dim = (graph_embed_dim - self.hypers.conv1d_kernel_size[1] + 1) * self.hypers.conv1d_channels[1]
            # reshape batch data
            cnn1d_embed = tf.reshape(inputs, [batchsize, graph_embed_dim])
            outputs = tf.layers.dense(cnn1d_embed, self.hypers.dense_dim, activation=tf.nn.relu)
        return outputs

    def fc_mmoe_layer(self, inputs, inputs_fs):
        """
        mmoe多任务模型中的全连接层
        """
        with tf.name_scope("fc_mmoe_layer"):

            # for batch data reshape
            batchsize = tf.shape(self.graph_indexes)[0]
            graph_embed_dim = int((self.hypers.k - 2) / 2 + 1)
            graph_embed_dim = (graph_embed_dim - self.hypers.conv1d_kernel_size[1] + 1) * self.hypers.conv1d_channels[1]
            # reshape batch data
            cnn1d_embed = tf.reshape(inputs, [batchsize, graph_embed_dim])
            cnn1d_embed_fs = tf.reshape(inputs_fs, [batchsize, graph_embed_dim])

            with tf.name_scope("fc_mmoe_experts_layer"):
                mixture_experts = []
                for i in range(self.hypers.mmoe_expert_num):
                    if i == 0:
                        output_expert_fs = tf.layers.dense(cnn1d_embed_fs, self.hypers.dense_dim, activation=tf.nn.relu)
                        mixture_experts.append(output_expert_fs)
                    else:
                        output_expert = tf.layers.dense(cnn1d_embed, self.hypers.dense_dim, activation=tf.nn.relu)
                        mixture_experts.append(output_expert)

            with tf.name_scope("fc_mmoe_gate_layer"):
                multi_gate = []
                for i in range(self.hypers.mmoe_expert_num):
                    if i == 0:
                        gate_fs = tf.layers.dense(cnn1d_embed_fs, self.hypers.mmoe_expert_num, activation=None)
                        multi_gate.append(gate_fs)
                    else:
                        gate = tf.layers.dense(cnn1d_embed, self.hypers.mmoe_expert_num, activation=None)
                        multi_gate.append(gate)

            with tf.name_scope("combine_gate_expert"):
                outputs = []
                for i in range(self.hypers.mmoe_expert_num):
                    gate_i = tf.transpose(multi_gate[i], [1, 0])
                    gate_i = tf.expand_dims(gate_i, axis=-1)
                    outputs.append(tf.reduce_sum(mixture_experts * gate_i, axis=0))

            return outputs


    def output_layer(self, inputs):
        """
        输出层
        """
        with tf.name_scope("output_layer"):
            drop_out = tf.nn.dropout(inputs, rate=1 - self.keep_prob)  # dropout
            outputs = tf.layers.dense(drop_out, self.hypers.class_num, activation=None)
        return outputs

    def output_layer_fs(self, inputs):
        """
        第二个输出层：故障网元
        """
        with tf.name_scope("output_layer_fs"):
            drop_out = tf.nn.dropout(inputs, rate=1 - self.keep_prob)
            outputs_fs = tf.layers.dense(drop_out, self.hypers.fault_source_num, activation=None)
        return outputs_fs

    def output_layer_bn(self, inputs):
        """
        第三个输出层：故障单板
        """
        with tf.name_scope("output_layer_bn"):
            drop_out = tf.nn.dropout(inputs, rate=1 - self.keep_prob)
            outputs_bn = tf.layers.dense(drop_out, self.hypers.board_number_num, activation=None)
        return outputs_bn

    def output_layer_pn(self, inputs):
        """
        第四个输出层：故障端口
        """
        with tf.name_scope("output_layer_pn"):
            drop_out = tf.nn.dropout(inputs, rate=1 - self.keep_prob)
            outputs_pn = tf.layers.dense(drop_out, self.hypers.port_number_num, activation=None)
        return outputs_pn

    def output_layer_ar(self, inputs):
        """
        第五个输出层：根源告警
        """
        with tf.name_scope("output_layer_ar"):
            drop_out = tf.nn.dropout(inputs, rate=1 - self.keep_prob)
            outputs_ar = tf.layers.dense(drop_out, self.hypers.alarm_root_num, activation=None)
        return outputs_ar

    def set_loss_mean(self):
        """
        平均损失函数
        """
        # softmax交叉熵损失
        with tf.name_scope("loss_scope"):
            loss_num_label = []
            # reg_loss = tf.contrib.layers.apply_regularization(  # L2正则化
            #     tf.contrib.layers.l2_regularizer(self.l2reg),
            #     tf.trainable_variables()
            # )

            loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[0], labels=self.labels))
            loss_num_label.append(loss_)

            loss_fs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[1], labels=self.labels_fs))
            loss_num_label.append(loss_fs)

            loss_bn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[2], labels=self.labels_bn))
            loss_num_label.append(loss_bn)
            
            loss_pn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[3], labels=self.labels_pn))
            loss_num_label.append(loss_pn)
            
            loss_ar = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[4], labels=self.labels_ar))
            loss_num_label.append(loss_ar)


            self.loss = tf.reduce_mean(tf.stack(loss_num_label, 0))  # + reg_loss   # ---GLOBAL---损失函数
            tf.summary.scalar('loss_mean', self.loss)
            #self.loss = loss_fs
            #self.loss = loss_

    def set_loss_uncertainty(self):
        """
        不确定性损失函数
        """
        # softmax交叉熵损失
        with tf.name_scope("loss_scope"):
            loss_num_label = []
            # reg_loss = tf.contrib.layers.apply_regularization(  # L2正则化
            #     tf.contrib.layers.l2_regularizer(self.l2reg),
            #     tf.trainable_variables()
            # )

            loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[0], labels=self.labels))
            loss_num_label.append(loss_)

            loss_fs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[1], labels=self.labels_fs))
            loss_num_label.append(loss_fs)

            loss_bn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[2], labels=self.labels_bn))
            loss_num_label.append(loss_bn)

            loss_pn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[3], labels=self.labels_pn))
            loss_num_label.append(loss_pn)

            loss_ar = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[4], labels=self.labels_ar))
            loss_num_label.append(loss_ar)

            uncertainty_weight = [
                tf.get_variable("uncertainty_weight_" + str(i), initializer=[1 / 5]
                                ) for i in range(5)
            ]

            final_loss = []
            for i in range(5):
                final_loss.append(
                    tf.div(loss_num_label[i], 2 * tf.square(uncertainty_weight[i])) + tf.log(uncertainty_weight[i]))

            self.loss = tf.reshape(tf.add_n(final_loss), shape=())  # + reg_loss   # ---GLOBAL---损失函数
            tf.summary.scalar('loss_uncertainty', self.loss)

    def set_loss_DWA(self):
        """
        DWA损失函数
        """
        # softmax交叉熵损失
        with tf.name_scope("loss_scope"):
            self.list1 = []
            self.list2 = []
            loss_num_label = []
            # reg_loss = tf.contrib.layers.apply_regularization(  # L2正则化
            #     tf.contrib.layers.l2_regularizer(self.l2reg),
            #     tf.trainable_variables()
            # )

            loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[0], labels=self.labels))
            loss_num_label.append(loss_)

            loss_fs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[1], labels=self.labels_fs))
            loss_num_label.append(loss_fs)

            loss_bn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[2], labels=self.labels_bn))
            loss_num_label.append(loss_bn)

            loss_pn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[3], labels=self.labels_pn))
            loss_num_label.append(loss_pn)

            loss_ar = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits_num_label[4], labels=self.labels_ar))
            loss_num_label.append(loss_ar)

            dwa = self.dynamic_weight_average(self.list1, self.list2)

            self.loss = tf.add_n([dwa[i] * loss_num_label[i] for i in range(len(loss_num_label))]) # + reg_loss   # ---GLOBAL---损失函数
            tf.summary.scalar('loss_DWA', self.loss)

            self.list1 = self.list2
            self.list2 = loss_num_label


    def set_accuracy(self):
        """
        准确率
        """
        with tf.name_scope("accuracy_scope"):
            correct_pred_labels = tf.equal(tf.argmax(self.logits_num_label[0], axis=1), tf.argmax(self.labels, axis=1))
            correct_pred_labels_fs = tf.equal(tf.argmax(self.logits_num_label[1], axis=1),
                                              tf.argmax(self.labels_fs, axis=1))
            correct_pred_labels_bn = tf.equal(tf.argmax(self.logits_num_label[2], axis=1),
                                              tf.argmax(self.labels_bn, axis=1))
            correct_pred_labels_pn = tf.equal(tf.argmax(self.logits_num_label[3], axis=1),
                                              tf.argmax(self.labels_pn, axis=1))
            correct_pred_labels_ar = tf.equal(tf.argmax(self.logits_num_label[4], axis=1),
                                              tf.argmax(self.labels_ar, axis=1))
            correct_pred = tf.logical_and(correct_pred_labels, correct_pred_labels_fs)
            correct_pred = tf.logical_and(correct_pred, correct_pred_labels_bn)
            correct_pred = tf.logical_and(correct_pred, correct_pred_labels_pn)
            correct_pred = tf.logical_and(correct_pred, correct_pred_labels_ar)
            #correct_pred = correct_pred_labels_fs
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # ---GLOBAL---准确率
            tf.summary.scalar('accuracy', self.accuracy)


    def set_optimizer(self):
        """
        优化器
        """
        with tf.name_scope("optimizer"):
            # self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss)
            # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def dynamic_weight_average(self, loss_t_1, loss_t_2):
        """
        :param loss_t_1: 每个task上一轮的loss列表，并且为标量
        :param loss_t_2:
        :return:
        """
        # 第1和2轮，w初设化为1，lambda也对应为1
        T = 20
        if not loss_t_1 or not loss_t_2:
            return [1/5,1/5,1/5,1/5,1/5]

        assert len(loss_t_1) == len(loss_t_2)
        task_n = len(loss_t_1)

        w = [l_1 / l_2 for l_1, l_2 in zip(loss_t_1, loss_t_2)]

        lamb = [math.exp(v / T) for v in w]

        lamb_sum = sum(lamb)

        return [task_n * l / lamb_sum for l in lamb]

    def build(self):
        """
        DNN模型构建
        """
        logits_num_label = []
        self.set_public()
        gcnns_outputs = self.gcnn_layers()
        emmbed = self.sortpooling_layer(gcnns_outputs)
        emmbed_fs = self.sortpooling_layer_fs(gcnns_outputs)
        cnn_1d = self.cnn1d_layers(emmbed)
        cnn_1d_fs = self.cnn1d_layers(emmbed_fs)

        if self.hypers.mtl_mode == 'hps':
            fc = self.fc_layer(cnn_1d)
            fc_fs = self.fc_layer(cnn_1d_fs)
            output = self.output_layer(fc)
            output_fs = self.output_layer_fs(fc_fs)
            output_bn = self.output_layer_bn(fc)
            output_pn = self.output_layer_pn(fc)
            output_ar = self.output_layer_ar(fc)

        if self.hypers.mtl_mode == 'mmoe':
            fc = self.fc_mmoe_layer(cnn_1d, cnn_1d_fs)
            output = self.output_layer(fc[1])
            output_fs = self.output_layer_fs(fc[0])
            output_bn = self.output_layer_bn(fc[2])
            output_pn = self.output_layer_pn(fc[3])
            output_ar = self.output_layer_ar(fc[4])

        logits_num_label.append(output)
        logits_num_label.append(output_fs)
        logits_num_label.append(output_bn)
        logits_num_label.append(output_pn)
        logits_num_label.append(output_ar)
        self.logits_num_label = logits_num_label
        self.predicts = tf.argmax(output, 1)

        if self.hypers.loss_algorithm == 'loss_mean':
            self.set_loss_mean()
        if self.hypers.loss_algorithm == 'loss_uncertainty':
            self.set_loss_uncertainty()
        if self.hypers.loss_algorithm == 'loss_DWA':
            self.set_loss_DWA()
        self.set_accuracy()
        self.set_optimizer()


# code for debugging
if __name__ == '__main__':
    param = tools.Parameters()
    param.set("feature_dim", 3)
    param.set("k", 10)  # 不能小于conv1d_kernel_size[1]*2，否则没法做第2个1d卷积
    param.set("class_num", 3)
    param.set("conv1d_channels", [16, 32])
    param.set("conv1d_kernel_size", [0, 5])
    param.set("dense_dim", 128)
    param.set("gcnn_dims", [32, 32, 32, 1])

    model = DGCNN(param)
    model.build()

    indices = np.array([[0, 1],
                        [1, 0],
                        [1, 2],
                        [2, 1],
                        [3, 4],
                        [3, 6],
                        [4, 3],
                        [4, 5],
                        [5, 4],
                        [6, 3],
                        ],
                       dtype=np.int64)
    values = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    shape = np.array([7, 7], dtype=np.int64)

    A_sparse = tf.SparseTensorValue(indices, values, shape)

    feature = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.2, 0.1, 0.3],
                        [0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.2, 0.1, 0.3],
                        [0.2, 0.1, 0.3]])

    indices = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=np.int64)
    values = np.array([1, 0.5, 1, 0.5, 0.5, 1, 1], dtype=np.float32)
    shape = np.array([7, 7], dtype=np.int64)
    D_inv_sparse = tf.SparseTensorValue(indices, values, shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./log', sess.graph)

        r = sess.run([model.logits, model.predicts], feed_dict={
            model.ajacent: A_sparse,
            model.features: feature,
            model.dgree_inv: D_inv_sparse,
            model.graph_indexes: [[0, 3], [3, 7]]
        })
        print(r)
