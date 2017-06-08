# encoding: utf-8

import os
import sys
import jieba
import gensim
import logging


class SentencesIter(object):
    def __init__(self, file_path, file_list):
        self.file_list = [os.path.join(file_path, file_name) for file_name in file_list]
        self.__yield_cnt = 0

    def __iter__(self):
        for file_name in self.file_list:
            with open(file_name) as fr:
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


if __name__ == '__main__':
    file_path = '.'
    file_name = ['LCSTS_train.input', 'LCSTS_test.input']
#    sentence_iter = SentencesIter(file_path, file_name)
#    for line in sentence_iter:
#        pass
#else:
    if os.path.exists(os.path.join(file_path, 'Word2Vec_model.model')):
        print('Word2Vec model \'Word2Vec_model.model\' found, loading... ')
        model = gensim.models.Word2Vec.load(os.path.join(file_path, 'Word2Vec_model.model'))
    else:
        print('Word2Vec model not found, training...')
        sentence_iter = SentencesIter(file_path, file_name)
        model = gensim.models.Word2Vec(sentence_iter, size=100, min_count=5, workers=4)
        model.save('Word2Vec_model.model')
    # Word2Vec model
    print(type(model))
    print(dir(model))
    print(model['窗户'])
    print(model.most_similar('窗户'))

