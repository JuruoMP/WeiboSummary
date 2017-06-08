# coding: utf-8
import os
import matplotlib.pyplot as plt

with open(os.path.join('result', 'perplexity.txt'), 'r', encoding='utf-8') as perplexity_file:
    perplexity_dict = eval(perplexity_file.readline().strip())
perplexity_list = sorted(perplexity_dict.items(), key=lambda x: x[0])
n_topics = [_[0] for _ in perplexity_list]
perplexities = [_[1] for _ in perplexity_list]

plt.title('Perplexities of LDA topic model')
plt.xlabel('Topic numbers')
plt.ylabel('Perplexity')
plt.plot(n_topics, perplexities, 'r')
plt.xticks(n_topics, [str(_) for _ in n_topics], rotation=60) 
plt.grid()
plt.show()
