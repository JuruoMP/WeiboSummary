# -*- coding:utf-8 -*-
import os
import sys
import glob
import pickle
import numpy as np
# import tensorflow as tf
from data_util import path_to_feature
from data_util import label_to_bilabel
from data_util import feature_normalize
from mlp_leader_detection import MlpLeaderDetection
from lstm_leader_detection import LSTMLeaderDetection


def lines_to_paths(lines):
    # split lines with '\n\n'
    paths = []
    path = []
    for line in lines:
        if not line or not line.strip():
            if path:
                paths.append(path)
                path = []
        else:
            path.append(line)
    return paths


def load_train_data(file_path, arch='MLP'):
    assert arch in ('MLP', 'LSTM')
    with open(file_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    paths = lines_to_paths(lines)
    feature_set, label_set = [], []
    for path in paths:
        # a batch of sentences of a path
        sentences, labels = [], []
        for line in path:
            label, sentence = line.strip().split(' ', 1)
            sentences.append(sentence)
            labels.append(label)
        path_feature = path_to_feature(sentences)
        if arch == 'MLP':
            path_label = label_to_bilabel(labels)
            feature_set += path_feature
            label_set += path_label
        elif arch == 'LSTM':
            path_label = labels
            feature_set.append(path_feature)
            label_set.append(path_label)
        else:
            raise Exception('Mode not supported(MLP/LSTM)')
    return feature_set, label_set


def load_eval_or_test_data(file_path, arch='MLP', mode='TEST'):
    assert arch in ('MLP', 'LSTM')
    assert mode in ('EVAL', 'TEST')
    lines = []
    if type(file_path) is list:
        for file_name in file_path:
            with open(file_name, 'r', encoding='utf-8') as fr:
                lines += fr.readlines() + ['\n']
    else:
        with open(file_path, 'r', encoding='utf-8') as fr:
            lines += fr.readlines()
    paths = lines_to_paths(lines)
    feature_set = []
    for path in paths:
        # a batch of sentences of a path
        sentences = []
        for line in path:
            sentence = line.strip()
            if mode == 'EVAL':
                _, sentence = sentence.split(' ', 1)
            sentences.append(sentence)
        path_feature = path_to_feature(sentences)
        if arch == 'MLP':
            feature_set += path_feature
        elif arch == 'LSTM':
            feature_set.append(path_feature)
        else:
            raise Exception('Mode not supported(MLP/LSTM)')
    return lines, feature_set


def train(file_path, arch='MLP'):
    assert arch in ('MLP', 'LSTM')
    feature_set, label_set = load_train_data(file_path, arch=arch)
    feature_set = feature_normalize(feature_set, mode='TRAIN', force_train=True)
    if arch == 'MLP':
        feature_dim = len(feature_set[0])
        leader_detection = MlpLeaderDetection.get_instance(feature_dim)
    elif arch == 'LSTM':
        feature_dim = len(feature_set[0][0])
        leader_detection = LSTMLeaderDetection.get_instance(feature_dim)
    train_status = leader_detection.run(mode='TRAIN', feature_set=feature_set, label_set=label_set)
    assert train_status == 1


def eval_or_test(file_path, arch='MLP', mode='TEST'):
    assert arch in ('MLP', 'LSTM')
    assert mode in ('EVAL', 'TEST')
    lines, feature_set = load_eval_or_test_data(file_path, arch=arch, mode=mode)
    feature_set = feature_normalize(feature_set, mode='EVAL')
    feature_dim = len(feature_set[0])
    assert feature_dim > 0
    if arch == 'MLP':
        leader_detection = MlpLeaderDetection.get_instance(feature_dim)
    elif arch == 'LSTM':
        leader_detection = LSTMLeaderDetection.get_instance(feature_dim)
    pred_labels = leader_detection.run(mode='EVAL', feature_set=feature_set)
    pos_lines, pos_pred_lables = 0, 0
    ret = []
    try:
        assert lines is not None
        assert pred_labels is not None
    except:
        print('error')
        pred_labels = leader_detection.run(mode='EVAL', feature_set=feature_set)
        return []
    while pos_lines < len(lines) and pos_pred_lables < len(pred_labels):
        line = lines[pos_lines].strip()
        if not line:
            ret.append('')
        else:
            mark = 1 if pred_labels[pos_pred_lables] > 0.5 else 0
            ret.append('%d, %s' % (mark, line))
            pos_pred_lables += 1
        pos_lines += 1
    return ret


def eval_path(path):
    sentences = [x.strip() for x in path]
    path_feature = path_to_feature(sentences)
    feature_set = feature_normalize(path_feature, mode='EVAL')
    feature_dim = len(feature_set[0])
    assert feature_dim > 0
    leader_detection = MlpLeaderDetection.get_instance(feature_dim)
    pred_labels = leader_detection.run(mode='EVAL', feature_set=feature_set)
    return pred_labels


def train_feature_normalizer(file_path):
    feature_set, _ = load_train_data(file_path)
    # feature_normalize(feature_set, mode='TRAIN', force_train=True)
    feature_normalize(feature_set, mode='TRAIN')


def export_data(file_path):
    feature_set, label_set = load_train_data(file_path)
    feature_set = feature_normalize(feature_set, mode='TRAIN', force_train=True)
    max_length = 0
    for features in feature_set:
        max_length = max(len(features), max_length)
    print('max_length = %d' % max_length)
    with open('data.raw', 'w', encoding='utf-8') as fw:
        print(repr(feature_set), file=fw)
        print(repr(label_set), file=fw)

if __name__ == '__main__':
    # export_data('data.data')
    # exit(-1)
    # print('Training feature normalizer parameters...')
    # train_feature_normalizer('data.data')
    # print('Training leader detection model...')
    # train('data.data', arch='LSTM')
    # print('Evaluating...')
    # eval_result = eval_or_test('eval.data', arch='MLP', mode='EVAL')
    # with open('eval.result', 'w', encoding='utf-8') as fw:
    #     for line in eval_result:
    #         print(line, file=fw)
    print('Running...')
    with open(os.path.join('..', 'WeiboDataset', 'db', 'leader_output.txt'), 'w', encoding='utf-8') as fw:
        file_list = glob.glob(os.path.join('..', 'WeiboDataset', 'weibo', '*.txt'))
        eval_result = eval_or_test(file_list, arch='MLP', mode='TEST')
        for line in eval_result:
            print(line, file=fw)
