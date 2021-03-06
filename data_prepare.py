# coding: utf-8
"""
@author Liuchen
2019
"""
import logging
import random
import numpy as np
import tensorflow as tf
import networkx as nx
from tools import Parameters

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger('main.data_prepare')


class GNNGraph(object):
    """
    参考自 https://github.com/muhanzhang/pytorch_DGCNN
    辅助类，用于定义一个样本图数据，并提供处理工具
    """

    def __init__(self, g, label, fs, bn, pn, ar, node_tags=None, node_features=None):
        '''
            g: networkx 图
            label: 图标签，整数
            node_tags: 节点标签整数列表
            node_features: 节点特征numpy数组
        '''
        self.num_nodes = g.number_of_nodes() # 节点数量
        self.node_labels = node_tags         # 边标签
        self.label = label                   # 图标签
        self.fs = fs                         # 图故障站点
        self.bn = bn                         # 图故障单板位置
        self.pn = pn                         # 图故障端口位置
        self.ar = ar                         # 图故障根源告警
        self.node_features = node_features   # 节点特征 numpy array (node_num * feature_dim)

        self.degrees = list(dict(g.degree()).values())   # 节点的度列表
        self.edges = list(g.edges)           # 网络边列表


def load_data(data, test_number=0, fold=2):
    """
    加载数据（参考自 https://github.com/muhanzhang/pytorch_DGCNN）

    :param data: 数据
    :param test_number: 测试数据数量
    :param fold: 交叉验证
    :return: 训练集，测试集，数据集参数Parameter（属性维度, 节点标签数, 网络类别数）

    数据格式：
    --------txt文档开始
    网络数量
    网络1节点数 网络1标签
    节点1标签 邻接点数 邻接点1 邻接点2 ... 属性1 属性2 ...
    节点2标签 邻接点数 邻接点1 邻接点2 ... 属性1 属性2 ...
    节点3标签 邻接点数 邻接点1 邻接点2 ... 属性1 属性2 ...
    ...
    网络2节点数 网络2标签
    节点1标签 邻接点数 邻接点1 邻接点2 ... 属性1 属性2 ...
    节点2标签 邻接点数 邻接点1 邻接点2 ... 属性1 属性2 ...
    节点3标签 邻接点数 邻接点1 邻接点2 ... 属性1 属性2 ...
    ...
    ----------txt文档结构
    """

    logger.info('loading data')
    g_list = []      # 网络列表
    glabel_dict = {}  # 网络标签字典
    gfs_dict = {}
    gbn_dict = {}
    gpn_dict = {}
    gar_dict = {}
    nlabel_dict = {}  # 节点标签字典

    params = Parameters()

    with open(f'./data/{data}/{data}.txt', 'r') as f:
        n_g = int(f.readline().strip())  # 网数数量
        for _ in range(n_g):
            row = f.readline().strip().split()
            n, l, fs, bn, pn, ar = [int(w) for w in row]  # 节点数量，网络标签,故障网元，故障单板，故障端口，根告警
            if not l in glabel_dict:  # 每个标签一个编号
                mapped = len(glabel_dict)
                glabel_dict[l] = mapped
            if not fs in gfs_dict:
                mapped_fs = len(gfs_dict)
                gfs_dict[fs] = mapped_fs
            if not bn in gbn_dict:
                mapped_bn = len(gbn_dict)
                gbn_dict[bn] = mapped_bn
            if not pn in gpn_dict:
                mapped_pn = len(gpn_dict)
                gpn_dict[pn] = mapped_pn
            if not ar in gar_dict:
                mapped_ar = len(gar_dict)
                gar_dict[ar] = mapped_ar
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in nlabel_dict:
                    mapped = len(nlabel_dict)
                    nlabel_dict[row[0]] = mapped
                node_tags.append(nlabel_dict[row[0]])

                if attr is not None:
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            # assert len(g.edges()) * 2 == n_edges  (some graphs in COLLAB have self-loops, ignored here)
            assert len(g) == n
            g_list.append(GNNGraph(g, l, fs, bn, pn, ar, node_tags, node_features))
    for g in g_list:
        g.label = glabel_dict[g.label]
        g.fs = gfs_dict[g.fs]
        g.bn = gbn_dict[g.bn]
        g.pn = gpn_dict[g.pn]
        g.ar = gar_dict[g.ar]
    params.set('class_num', len(glabel_dict))
    params.set('fault_source_num', len(gfs_dict))
    params.set('board_number_num', len(gbn_dict))
    params.set('port_number_num', len(gpn_dict))
    params.set('alarm_root_num', len(gar_dict))
    params.set('node_label_dim', len(nlabel_dict))   # maximum node label (tag)

    if node_feature_flag:
        params.set('node_feature_dim', node_features.shape[1])  # dim of node features (attributes)
    else:
        params.set('node_feature_dim', 0)

    params.set('feature_dim', params.node_feature_dim + params.node_label_dim)

    if test_number == 0:
        train_idxes = np.loadtxt(f'data/{data}/10fold_idx/train_idx-{fold}.txt', dtype=np.int32).tolist()
        test_idxes = np.loadtxt(f'data/{data}/10fold_idx/test_idx-{fold}.txt', dtype=np.int32).tolist()
        return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes], params
    else:
        return g_list[: n_g - test_number], g_list[n_g - test_number:], params


def batching(graph_batch, params):
    """
    处理batch的图
    :graph_batch: 图list
    :return: (ajacent, features, batch_label, degree_inv, graph_indexes)
        ajacent 稀疏矩阵，当前batch中所有图的邻接矩阵组成的大分块邻接矩阵
        features 当前batch中所有图中节点特征
        batch_label 当前batch中图的标签
        degree_inv 稀疏矩阵，ajacent对应的度矩阵的逆矩阵
        graph_indexes 当前batch中每个图的特征 在features 中的起始和结束位置
    """

    def onehot(n, dim):
        one_hot = [0] * dim
        one_hot[n] = 1
        return one_hot
    
    # features 构造 ------------
    # 节点特征处理
    if graph_batch[0].node_features is None:
        node_features = None
    else:
        node_features = [g.node_features for g in graph_batch]
        node_features = np.concatenate(node_features, 0)
    # 节点标签处理
    node_tag_features = None
    if params.node_label_dim > 1:
        node_labes = []
        for g in graph_batch:
            node_labes += g.node_labels
        node_tag_features = np.array([onehot(n, params.node_label_dim) for n in node_labes])
    # 合并特征
    if node_features is None and node_tag_features is None:  # 若节点即无特也无标签，则以度为特征
        node_dgrees = []
        for g in graph_batch:
            node_dgrees += g.degrees
        features = node_dgrees
    else:
        if node_features is None:
            features = node_tag_features
        elif node_tag_features is None:
            features = node_features
        else:
            features = np.concatenate([node_features, node_tag_features], 1)

    # 每个图的特征的索引开始位置和结束位置 -------- 
    g_num_nodes = [g.num_nodes for g in graph_batch]
    graph_indexes = [[sum(g_num_nodes[0:i-1]), sum(g_num_nodes[0:i])] for i in range(1, len(g_num_nodes)+1)]

    # 图标签 --------
    batch_label = [onehot(g.label, params.class_num)for g in graph_batch]
    batch_label_fs = [onehot(g.fs, params.fault_source_num) for g in graph_batch]
    batch_label_bn = [onehot(g.bn, params.board_number_num) for g in graph_batch]
    batch_label_pn = [onehot(g.pn, params.port_number_num) for g in graph_batch]
    batch_label_ar = [onehot(g.ar, params.alarm_root_num) for g in graph_batch]

    # 邻接矩阵的稀疏矩阵 ---------
    total_node_degree = []
    indices = []
    indices_append = indices.append
    for i, g in enumerate(graph_batch):
        total_node_degree.extend(g.degrees)
        start_pos = graph_indexes[i][0]
        for e in g.edges:
            node_from = start_pos + e[0]
            node_to = start_pos + e[1]
            indices_append([node_from, node_to])
            indices_append([node_to, node_from])

    total_node_num = len(total_node_degree)
    values = np.ones(len(indices), dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)
    shape = np.array([total_node_num, total_node_num], dtype=np.int32)
    ajacent = tf.SparseTensorValue(indices, values, shape)

    # 度矩阵的逆的稀疏矩阵 ----------
    index_degree = [([i, i], 1.0/degree if degree >0 else 0) for i, degree in enumerate(total_node_degree)]
    index_degree = list(zip(*index_degree))
    degree_inv = tf.SparseTensorValue(index_degree[0], index_degree[1], shape)

    return ajacent, features, batch_label,batch_label_fs, batch_label_bn, batch_label_pn, batch_label_ar, degree_inv, graph_indexes


if __name__ == '__main__':
    trin, test, params = load_data('MUTAG', 1)
    print(params)
