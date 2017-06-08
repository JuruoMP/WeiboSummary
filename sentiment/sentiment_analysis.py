# encoding: utf-8

import os
import sys
import codecs
import glob
import jieba

explain = False

output_file = sys.stdout

seg_words = []
sentiment_vocabs = {}
degree_vocabs = {}
not_vocabs = []
neg_vocabs = []
pos_vocabs = []
stop_words = []


def read_file(file_name):
    lines = []
    with codecs.open(file_name, 'r', encoding='utf-8') as read_file_handler:
        while True:
            line = read_file_handler.readline()
            line = line.encode('utf-8').decode('utf-8-sig')
            if not line:
                break
            elif line[:2] == '//':
                continue
            else:
                lines.append(line.strip())
    return lines


def calc_sentiment(line):
    def seg_sentence(sentence, seg_words):
        __seg_word = '__SEG_WORD__'
        for seg_word in seg_words:
            sentence = sentence.replace(seg_word, __seg_word)
        segs = sentence.split(__seg_word)
        return [seg for seg in segs if seg]
    def calc_score(sentence):
        words = [x for x in jieba.cut(sentence)]
        score = 0.0
        reverse = 1.0
        degree = 1.0
        for word in words:
            if word in not_vocabs:
                reverse *= -1.0
                if explain: print('\t\t\t%s\treverse' % word, file=sys.stderr)
            elif word in sentiment_vocabs.keys():
                score += sentiment_vocabs[word]
                if explain: print('\t\t\t%s\tscore += %f' % (word, sentiment_vocabs[word]), file=sys.stderr)
            elif word in degree_vocabs.keys():
                degree *= degree_vocabs[word]
                if explain: print('\t\t\t%s\tdegree *= %f' % (word, degree_vocabs[word]), file=sys.stderr)
            else:
                if explain: print('\t\t\t%s\tUNK' % word, file=sys.stderr)
        if explain: print('\t\tscore=%.2f, reverse=%.2f, degree=%.2f' % (score, reverse, degree), file=sys.stderr)
        return score * reverse * degree
    sentiment_segs = seg_sentence(line, seg_words)
    total_score = 0
    for seg in sentiment_segs:
        score = calc_score(seg)
        if explain: print('\t%f\t%s' % (score, seg), file=sys.stderr)
        total_score += score
    return total_score


# Load vocab list
seg_words = [x for x in read_file(os.path.join('vocab', 'seg.txt')) if len(x.split()) == 1]
lines = read_file(os.path.join('vocab', 'BosonNLP_sentiment_score.txt'))
for line in lines:
    try:
        word, score = line.split()
    except:
        continue
    sentiment_vocabs[word] = float(score)
lines = read_file(os.path.join('vocab', 'degree.txt'))
for line in lines:
    try:
        word, score = line.split()
    except:
        continue
    degree_vocabs[word] = float(score)
not_vocabs = [x for x in read_file(os.path.join('vocab', 'not_words.txt')) if len(x.split()) == 1]
neg_vocabs += [x for x in read_file(os.path.join('vocab', 'neg_emotion.txt')) if len(x.split()) == 1]
neg_vocabs += [x for x in read_file(os.path.join('vocab', 'neg_evaluate.txt')) if len(x.split()) == 1]
pos_vocabs += [x for x in read_file(os.path.join('vocab', 'pos_emotion.txt')) if len(x.split()) == 1]
pos_vocabs += [x for x in read_file(os.path.join('vocab', 'pos_evaluate.txt')) if len(x.split()) == 1]
stop_words = [x for x in read_file(os.path.join('vocab', 'stop_words.txt')) if len(x.split()) == 1]
print('len(not_vocabs) = %d' % len(not_vocabs), file=sys.stderr)
print('len(sentiment_vocabs) = %d' % len(sentiment_vocabs), file=sys.stderr)
print('len(degree_vocabs) = %d' % len(degree_vocabs), file=sys.stderr)
print('len(neg_vocabs) = %d' % len(neg_vocabs), file=sys.stderr)
print('len(pos_vocabs) = %d)' % len(pos_vocabs), file=sys.stderr)
# Load data
file_list = glob.glob(os.path.join('.', 'text', '*.txt'))
for file_name in file_list:
    real_name = file_name.replace('/', '\\').split('\\')[-1].split('.')[-2]
    output_file = open(os.path.join('.', 'output', real_name + '.txt'), 'w', encoding='utf-8')
    lines = read_file(file_name)
    path_no = 0
    for line in lines:
        if not line or not line.strip():
            print('', file=output_file)
            path_no += 1
            if path_no > 5:
                break
        else:
            score = calc_sentiment(line)
            print('%d\t%s' % (score, line), file=output_file)
        if explain:
            pause = input('Press any key to continue...\n\n\n')
