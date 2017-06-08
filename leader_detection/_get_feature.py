# -*- coding:utf-8 -*-
from __future__ import print_function
import sys
import re
import jieba


def Jaccard_similarity(s1, s2):
    if type(s1) == str:
        s1 = '\2'.join(jieba.cut(s1)).split('\2')
    if type(s2) == str:
        s1 = '\2'.join(jieba.cut(s2)).split('\2')
    s1, s2 = set(s1), set(s2)
    return 1.0 * len(s1 & s2) / len(s1 | s2)


lineno = 0
with open('data.data', 'r') as fr, open('features.txt', 'w', encoding='utf-8') as fw:
    rline = fr.readline()
    lines = []
    while rline:
        if not rline.strip():
            line_no = 0
            last_status = -1
            for line in lines:
                line_no += 1
                label, line = line.strip().split(' ', 1)
                # words = '\2'.join(jieba.cut(line))
                # words = words.split('\2')
                # 去括号包含的文本
                reg1 = r'\([^)）]*\)'
                reg2 = r'（[^)）]*）'
                _line = re.sub(reg1, "", line)
                _line = re.sub(reg2, "", _line)
                # number of terms
                words = '\2'.join(jieba.cut(_line)).split('\2')
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
                    if src_pos < 0 or src_pos >= len(lines):
                        similarities.append(-1)
                    else:
                        sim = Jaccard_similarity(line, lines[src_pos])
                        similarities.append(sim)
                # similarity to root
                root_similarity = Jaccard_similarity(line, lines[0])
                features = [last_status, num_of_terms, pos, sentence_type, num_emotions, num_hashtags, num_urls, num_mentions]
                features += similarities
                features.append(root_similarity)
                ret = str(label) + '\t'
                for feature in features:
                    ret += str(feature) + ','
                print(ret[:-1], file=fw)
                last_status = label
            print('', file=fw)
            lines = []
        else:
            lines.append(rline)
        rline = fr.readline()
