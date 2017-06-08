# encoding: utf-8

import os
import sys
import numpy as np
import jieba
import gensim
import logging
import pyemd as emd

assert sys.version[0] == '3'


EMBEDDING_DIM = 150


class SentencesIter(object):
    def __init__(self, file_path, file_list):
        files = [os.path.join(file_path, file_name) for file_name in file_list]
        self.file_list = files
        self.__yield_cnt = 0

    def __iter__(self):
        for file_name in self.file_list:
            with open(file_name, 'r', encoding='utf-8') as fr:
                while True:
                    try:
                        line = fr.readline()
                        if not line:
                            break
                        if not line.strip():
                            continue
                        line = line.strip().replace(' ', '').replace('\t', '')
                        words = [x for x in jieba.cut(line)]
                        yield words
                        self.__yield_cnt += 1
                        if self.__yield_cnt % 1000 == 0:
                            print('self.__yield_cnt = %d' % self.__yield_cnt)
                    except:
                        continue


class TextUtil(object):
    punctuations = set([',', '.', '?', '!', '@', '#', '$', '%', '^', '&', '*',
                        '(', ')', '[', ']', '{', '}', '`', '"', '\'', ':', ';',
                        '-', '_', '=', '+', '\\', '|',
                        '，', '。', '？', '！', '￥', '【', '】',
                        '“', '”', '’', '‘', '：', '；'])
    def __init__(self, stop_word_path):
        self.stop_words = set()
        with open(stop_word_path, 'r', encoding='utf-8') as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                self.stop_words.add(line.strip())

    def strip_stop_words(self, sentence):
        def is_ascii(s):
            return all(ord(c) < 128 for c in s)
        def is_punctuation(s):
            if len(s) == 1 and s[0] in TextUtil.punctuations:
                return True
            else:
                return False
        return [word for word in sentence if word not in self.stop_words
                and not is_ascii(word) and not is_punctuation(word)]


def WMD(model, sentence1, sentence2):
    # Word Mover's Distance
    all_words = list(set(sentence1) | set(sentence2))
    all_words_dict = dict(zip(all_words, range(len(all_words))))
    words1 = np.zeros(len(all_words), dtype=np.float)
    words2 = np.zeros(len(all_words), dtype=np.float)
    for word in sentence1:
        word_id = all_words_dict[word]
        words1[word_id] += 1.0 / len(sentence1)
    for word in sentence2:
        word_id = all_words_dict[word]
        words2[word_id] += 1.0 / len(sentence2)
    dist_mat = np.zeros([len(all_words), len(all_words)], dtype=np.float)
    for i in range(len(all_words)):
        word_i = all_words[i]
        for j in range(len(all_words)):
            word_j = all_words[j]
            try:
                vector1 = model[word_i]
                vector2 = model[word_j]
            except:
                continue
            assert vector1.shape == vector2.shape
            # Eucalid distance
            dist_mat[i, j] = ((vector1 - vector2) ** 2).sum()
            # Cosine distance
            # dist_mat[i, j] = 0.5 +
            #     (0.5 * float(vector1 * vector2.T) / (np.linalg.norm(a) * np.linalg.norm(b))
    sentence_dist = emd.emd(words1, words2, dist_mat)
    # sentence_dist, flow = emd_with_flow(words1, words2, dist_mat)
    return sentence_dist


def WCD(model, sentence1, sentence2):
    # Word Centroid Distance:
    # d_sentence1 * embedding_of_sentence1 - d_sentence2 * embedding_of_sentence2
    # where d is nCOW weight, embedding is word embedding
    vector = [0 for _ in range(EMBEDDING_DIM)]
    words_sentence1, cnt_sentences1 = {}, len(sentence1)
    words_sentence2, cnt_sentences2 = {}, len(sentence2)
    for word in sentence1:
        words_sentence1[word] = words_sentence1.get(word, 0) + 1
    for word in sentence2:
        words_sentence2[word] = words_sentence2.get(word, 0) + 1
    for word, word_cnt in words_sentence1.items():
        try:
            embedding = model[word]
            for i in range(len(vector)):
                vector[i] += 1.0 * word_cnt / cnt_sentences1 * embedding[i]
        except:
            continue
    for word, word_cnt in words_sentence2.items():
        try:
            embedding = model[word]
            for i in range(len(vector)):
                vector[i] -= 1.0 * word_cnt / cnt_sentences2 * embedding[i]
        except:
            continue
    dist_square = sum([x ** 2 for x in vector])
    return dist_square ** (1.0 / 2)


def RWMD(model, sentence1, sentence2):
    # Relaxed Word Moving Distance
    # Return minumum of two distances with two relaxed subjection
    vocabs1, vocabs2 = set(sentence1), set(sentence2)
    vocabs_dict1 = dict(zip(vocabs1, range(len(vocabs1))))
    vocabs_dict2 = dict(zip(vocabs2, range(len(vocabs2))))
    words1 = np.zeros(len(vocabs1), dtype=np.float32)
    words2 = np.zeros(len(vocabs2), dtype=np.float32)
    for word in sentence1:
        word_id = vocabs_dict1[word]
        words1[word_id] += 1.0 / len(sentence1)
    for word in sentence2:
        word_id = vocabs_dict2[word]
        words2[word_id] += 1.0 / len(sentence2)
    dist_mat = np.zeros([len(vocabs1), len(vocabs2)])
    for i in range(len(vocabs1)):
        word_i = vocabs1[i]
        for j in range(len(vocabs2)):
            word_j = vocabs2[j]
            vector1 = model[word_i]
            vector2 = model[word_j]
            assert vector1.shape == vector2.shape
            dist_mat[i, j] = ((vector1 - vector2) ** 2).sum()
    # dist1: subject to \Sigma{j=1}{n} T_{ij} = d_i
    # => T_{ij}=d_i if j=argmin_j{c(i,j)}
    cost_of_target = np.min(dist_mat, axis=1)
    dist1 = (words1 * cost_of_target).sum()
    # dist2: subject to \Sigma{i=1}{n} T_{ij} = d_j
    # => T_{ij}=d_j if i=argmin_i{c(i,j)}
    cost_of_target = np.min(dist_mat, axis=0)
    dist2 = (words2 * cost_of_target).sum()
    return min(dist1, dist2)


def sentence_dist(model, sentence1, sentence2, text_util=None, mode='WMD'):
    if text_util:
        sentence1 = text_util.strip_stop_words(sentence1)
        sentence2 = text_util.strip_stop_words(sentence2)
    dist = 0.0
    if mode == 'WMD':
        dist = WMD(model, sentence1, sentence2)
    elif mode == 'WCD':
        dist = WCD(model, sentence1, sentence2)
    elif mode == 'RWMD':
        dist = RWMD(model, sentence1, sentence2)
    else:
        raise Exception('Error: sentence_dist mode unknown')
    return dist


def find_similiar(model, target, sentences, text_util=None, k=5, n=-1):
    if text_util:
        target = text_util.strip_stop_words(target)
        sentences_content = [text_util.strip_stop_words(x) for x in sentences]
    approximate_dist = {}
    top_k, max_score = [], 0xfffffff

    def compare_top_k(top_k, sentence_id, score):
        if score >= max_score:
            return
        top_k.append((sentence_id, score))
        top_k = sorted(top_k, key=lambda x: x[1])
        top_k = top_k[:k]
        return top_k

    for sentence in sentences:
        d = sentence_dist(model, target, sentence, text_util=text_util, mode='WCD')
        approximate_dist[''.join(sentence)] = d
    approximate_dist = sorted(approximate_dist.items(), key=lambda x: x[1])
    # end_pos = n if n != -1 else 10 * k
    end_pos = n if n != -1 else len(approximate_dist)
    top_k = []
    print('k=%d, end_pos=%d' % (k, end_pos))
    for sentence, _ in approximate_dist[:end_pos]:
        d = sentence_dist(model, target, sentence, text_util=text_util, mode='WMD')
        top_k.append((sentence, d))
    top_k = sorted(top_k, key=lambda x: x[1])
    return top_k[:k]


if __name__ == '__main__':
    file_path = '.'
    file_name = ['LCSTS_train.input', 'LCSTS_test.input']
    # Word2Vec model
    model_path = os.path.join(file_path, 'Word2Vec_model.model')
    if os.path.exists(os.path.join(file_path, 'Word2Vec_model.model')):
        print('Word2Vec model \'Word2Vec_model.model\' found, loading... ')
        model = gensim.models.Word2Vec.load(model_path)
    else:
        print('Word2Vec model not found, training...')
        sentence_iter = SentencesIter(file_path, file_name)
        model = gensim.models.Word2Vec(sentence_iter, size=EMBEDDING_DIM, min_count=5, workers=32)
        model.save(model_path)

    # Test embedding
    # print(model.similar_by_vector(model['肯德基'], topn=10, restrict_vocab=None))
    # Calculate distance between sentences
    stop_word_path = 'stop_words.txt'
    text_util = TextUtil(stop_word_path)
    sentence1 = '肯德基炸鸡翅很好吃'
    sentence2 = '美味的肯德基香辣鸡翅'
    d = sentence_dist(model, sentence1, sentence2, text_util)
    print(d)

    # Find similar sentences
    test_sentence_iter = SentencesIter(file_path, [file_name[1]])
    lines = [_ for _ in test_sentence_iter]
    top_k = find_similiar(model, lines[1], lines[2:], text_util)
    print('target: %s' % ''.join(lines[1]))
    for sentence, dist in top_k:
        print('%.3f\t%s' % (dist, sentence))



