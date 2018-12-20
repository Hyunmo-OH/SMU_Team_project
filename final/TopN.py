
#-*- coding: utf-8 -*-

import pandas as pd
from string import punctuation
from konlpy.tag import Okt
import gensim
from collections import Counter

q= pd.read_csv(u'H:/2018.2학기/4_경영 빅데이터 분석/프로젝트/새 폴더/암살tweetContents.csv',encoding='utf-8')

contents = list(q.contents)

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

clean_contents =[]

for sent in contents:
    clean = strip_punctuation(sent)
    clean_contents.append(clean)

twitter = Okt()

w2v_data =[]

def tokenize(data):
    for sent in data:
        tokens = twitter.pos(sent)
        new_tokens=[]
        for token in tokens:
            new_token = token[0]+'/'+token[1]
            new_tokens.append(new_token)
        w2v_data.append(new_tokens)
    return(w2v_data)

tokenize(clean_contents)

sentences_tag=[]
for sentence in clean_contents:
    morph = twitter.pos(sentence)
    sentences_tag.append(morph)

noun_adj_list=[]
for sentence1 in sentences_tag:
    for word, tag in sentence1:
        if len(word) >= 2 and tag in ['Noun','Adjective','Verb']:
            noun_adj_list.append(word)

counts = Counter(noun_adj_list)

for i, x in enumerate(counts.most_common(50)):
    print counts.most_common(50)[i][0], counts.most_common(50)[i][1]