# encoding: utf-8
import os
import sys
import numpy as np
import lda
import jieba
import matplotlib.pyplot as plt


stop_words = set()
with open('stop_words.txt', 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        if not line or not line.strip():
            continue
        word = line.strip()
        stop_words.add(word)
punctuations = set([',', '.', '?', '!', '@', '#', '$', '%', '^', '&', '*',
                    '(', ')', '[', ']', '{', '}', '`', '"', '\'', ':', ';',
                    '-', '_', '=', '+', '\\', '|',
                    '，', '。', '？', '！', '￥', '【', '】',
                    '“', '”', '’', '‘', '：', '；'])
stop_words |= punctuations


def strip_stop_words(words):
    ret = []
    for word in words:
        if word in stop_words:
            continue
        elif word.isdigit():
            continue
        elif all(ord(c) < 128 for c in word):
            continue
        elif len(word) < 2:
            continue
        else:
            ret.append(word)
    return ret


raw_documents = []
with open('LCSTS_test.input', 'r', encoding='utf-8') as fr:
    while True:
        try:
            line = fr.readline()
        except:
            continue
        if not line:
            break
        elif not line.strip():
            continue
        line = line.strip().replace(' ', '')
        raw_documents.append(line)
seged_documents = []
vocabs = set()
for i, document in enumerate(raw_documents[:200000]):
    if i % 10000 == 0:
        print('Processing line %d' % i, file=sys.stderr)
    words = list(jieba.cut(document))
    words = strip_stop_words(words)
    for word in words:
        vocabs.add(word)
    seged_documents.append(words)
vocabs = tuple(vocabs)
vocabs_dict = dict(zip(vocabs, range(len(vocabs))))
documents = []
for document_word in seged_documents:
    document_vocab = [0 for _ in range(len(vocabs))]
    words = document_word
    for word in words:
        vocab_id = vocabs_dict.get(word, None)
        try:
            assert vocab_id is not None
        except:
            print('document = %s' % document)
            print('words = %s' % words)
            print('word = %s' % word)
            assert vocab_id is not None
        document_vocab[vocab_id] += 1
    documents.append(document_vocab)

# documents: len(documents) * len(vocab), X[i]: count of each document
# vocabs: tuple, storing each words
model = lda.LDA(n_topics=10, n_iter=2000, random_state=1)
model.fit(np.array(documents))
topic_word = model.topic_word_   # n_topics * num_of_words, probability desending
n_top_words = 100
for topic_id, topic_dict in enumerate(topic_word):
    ith_topic_word = np.array(vocabs)[np.argsort(topic_dict)][:-(n_top_words+1):-1]
    # ith_topic_word = np.array(vocabs)[np.argsort(topic_dict)][::-1]
    print("%d\t%s" % (topic_id, ','.join(ith_topic_word)), file=sys.stdout)
    with open(os.path.join('result', 'topic_word_%d.txt' % topic_id), 'w', encoding='utf-8') as fw:
        print("%d\t%s" % (topic_id, ','.join(ith_topic_word)), file=fw)

doc_topic = model.doc_topic_
print("type(doc_topic): {}".format(type(doc_topic)))
print("shape: {}".format(doc_topic.shape))
label = []
topic_docs = {}
for i in range(len(documents)):
    topic_most_pr = doc_topic[i].argmax()
    label.append(topic_most_pr)
    topic_docs.setdefault(topic_most_pr, [])
    topic_docs[topic_most_pr].append(i)
    # print("doc: {} topic: {}".format(n, topic_most_pr))
for topic_id in topic_docs.keys():
    with open(os.path.join('result', 'topic_%d.txt' % topic_id), 'w', encoding='utf-8') as fw:
        for doc_id in topic_docs.get(topic_id):
            print(raw_documents[doc_id], file=fw)

f, ax = plt.subplots(6, 1, figsize=(8, 8), sharex=True)
for i, k in enumerate([0, 1, 2, 3, 8, 9]):
    ax[i].stem(doc_topic[k, :], linefmt='r-', markerfmt='ro', basefmt='w-')
    ax[i].set_xlim(-1, 2)     # x坐标下标
    ax[i].set_ylim(0, 1.2)    # y坐标下标
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Document {}".format(k))
ax[5].set_xlabel("Topic")
plt.tight_layout()
plt.show()

