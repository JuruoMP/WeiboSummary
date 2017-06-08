# coding: utf-8
import os
import sys
import re
import pickle as pickle
import jieba


def Jaccard_similarity(s1, s2):
    if type(s1) == str:
        s1 = '\2'.join(jieba.cut(s1)).split('\2')
    if type(s2) == str:
        s1 = '\2'.join(jieba.cut(s2)).split('\2')
    s1, s2 = set(s1), set(s2)
    return 1.0 * len(s1 & s2) / len(s1 | s2)


def path_to_feature(data_lines):
    data_features = []
    assert type(data_lines) == list
    line_no = 0
    # last_status = -1
    for line in data_lines:
        line_no += 1
        # 去括号包含的文本
        reg1 = r'\([^)）]*\)'
        reg2 = r'（[^)）]*）'
        _line = re.sub(reg1, "", line)
        _line = re.sub(reg2, "", _line)
        # number of terms
        words = [_ for _ in jieba.cut(_line)]
        num_of_terms = len(words)
        # pos
        pos = line_no
        # type, contains question mark or exclamation
        sentence_type = 0
        if '?' in line or '？' in line:
            sentence_type = 1
        elif '!' in line or '！' in line:
            sentence_type = 2
        # number of emoticons
        num_emotions = len(re.findall(r'\[[^\]]*\]', _line))
        # number of hashtags
        num_hashtags = len(re.findall(r'#[^#]*#', _line))
        # number of URLs
        num_urls = len(re.findall(r'http://', _line))
        # num of mentions, i.e. '@'
        num_mentions = 0
        for i in range(len(line)):
            if line[i] == '@':
                num_mentions += 1
        # similarity to neighbors
        similarities = []
        for i in range(-3, 4, 1):
            if i == 0:
                continue
            src_pos = line_no + i
            if src_pos < 0 or src_pos >= len(data_lines):
                similarities.append(-1)
            else:
                sim = Jaccard_similarity(line, data_lines[src_pos])
                similarities.append(sim)
        # similarity to root
        root_similarity = Jaccard_similarity(line, data_lines[0])
        features = [num_of_terms, pos, sentence_type, num_emotions, num_hashtags, num_urls, num_mentions]
        # features = [last_status, num_of_terms, pos, sentence_type, num_emotions, num_hashtags, num_urls, num_mentions]
        features += similarities
        features.append(root_similarity)
        data_features.append(features)
        # last_status = label
    return data_features


def label_to_bilabel(data_labels):
    labels = []
    #print('convert label, data_labels = %s' % data_labels)
    for label in data_labels:
        if label == '1':
            labels.append((0, 1))
        else:
            labels.append((1, 0))
    #print('after convert: labels = %s' % labels)
    return labels


def feature_normalize(feature_set, mode, force_train=False, feature_norm_path='feature_norm_params.pkl'):
    '''
    normalize input features to R(0, 1)
    :param input_data: [[feature], [feature]...]
    :return: [[feature], [feature]...] with normalization, normalization parameters
    '''
    return feature_set   # todo: do something about feature normalization
    if mode not in ('TRAIN', 'EVAL'):
        raise Exception('Error: Feature_normalize mode unknown(%s)' % mode)
    if mode == 'EVAL' and not os.path.exists(feature_norm_path):
        raise Exception('Error: Model not Trained')
    if mode != 'TRAIN' and force_train:
        raise Exception('Error: Force train should be used with mode=TRAIN')
    INF = 0xfffffff
    feature_len = len(feature_set[0])
    feature_norm_parameters = None
    if mode == 'TRAIN' and force_train:
        feature_norm_parameters = None
    elif not os.path.exists(feature_norm_path):
        feature_norm_parameters = None
    else:
        feature_norm_parameters = pickle.load(open(feature_norm_path, 'rb'))
    if mode != 'TRAIN' and not feature_norm_parameters:
        raise Exception('Error: Normalization parameters load FAILED')
    if not feature_norm_parameters:
        feature_min = [INF for _ in range(feature_len)]
        feature_max = [-INF for _ in range(feature_len)]
        for features in feature_set:
            for i, feature in enumerate(features):
                feature_min[i] = feature if feature < feature_min[i] else feature_min[i]
                feature_max[i] = feature if feature > feature_max[i] else feature_max[i]
        feature_norm_parameters = (feature_min, feature_max)
        pickle.dump(feature_norm_parameters, open(feature_norm_path, 'wb'))
    else:
        (feature_min, feature_max) = feature_norm_parameters
        assert len(feature_min) == len(feature_max)
        assert len(feature_min) == feature_len
    output_data = []
    for features in feature_set:
        output_feature = []
        for i, feature in enumerate(features):
            feature = 2.0 * (float(feature) - feature_min[i]) / (feature_max[i] - feature_min[i]) - 1.0
            output_feature.append(feature)
        output_data.append(output_feature)
    return output_data
