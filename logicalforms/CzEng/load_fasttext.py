#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:25:54 2019

@author: TiffMin
"""
from gensim.models import FastText
import torch
import numpy as np
en_vocab_file = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_voc_subtitles_0.1m.p'
cz_vocab_file = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_subtitles_0.1m.p'
en_vocab = cPickle.load(open(en_vocab_file,'rb')); cz_vocab = cPickle.load(open(cz_vocab_file,'rb'))
en_w2i, en_i2w = en_vocab["en_w2i"], en_vocab["en_rm_i2w"]
cz_w2i, cz_i2w = cz_vocab["cz_w2i"], cz_vocab["cz_rm_i2w"]


cz_model = FastText.load_fasttext_format('cc.cs.300.bin')
cz_not_in_fasttext = []
for cz_word in cz_w2i :
    if not(cz_word) in model:
        cz_not_in_fasttext.append(cz_word)

en_model = FastText.load_fasttext_format('cc.en.300.bin')
en_not_in_fasttext = []
for en_word in en_w2i :
    if not(en_word) in model:
        en_not_in_fasttext.append(en_word)


en_embeddings = np.zeros((len(en_w2i), 300))
cz_embeddings = np.zeros((len(cz_w2i), 300))

for en_word, i in en_w2i.items() :
    en_embeddings[i] = en_model[en_word]

for cz_word, i in cz_w2i.items() :
    cz_embeddings[i] = cz_model[cz_word]


pickle.dump(en_embeddings, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_fasttext_0.1m.p', 'wb'))
pickle.dump(cz_embeddings, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_fasttext_0.1m.p', 'wb'))

#If pretrained, add a linear layer 

import gensim
import pickle
import numpy as np
model = gensim.models.KeyedVectors.load_word2vec_format('word2vec', binary=True)  
en_word2vec = np.zeros((len(en_w2i), 300))
for en_word, i in en_w2i.items() :
    if not(en_word) in model:
        if en_word == 'UNK':
            en_word2vec[i] = model['unk'] 
        else:
            en_word2vec[i] = [np.random.normal(scale=0.25) for i in range(300)]
    else:
        en_word2vec[i] = model[en_word] 

en_not_in_word2vec = []
for en_word in en_w2i :
    if not(en_word) in model:
        en_not_in_word2vec.append(en_word)
        
pickle.dump(en_word2vec, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_word2vec_0.1m.p', 'wb'))

#Do the same for 0.2m
en_vocab_file = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_voc_subtitles_0.25m.p'
cz_vocab_file = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_subtitles_0.25m.p'
en_vocab = cPickle.load(open(en_vocab_file,'rb')); cz_vocab = cPickle.load(open(cz_vocab_file,'rb'))
en_word2vec = np.zeros((len(en_w2i), 300))
for en_word, i in en_w2i.items() :
    if not(en_word) in model:
        if en_word == 'UNK':
            en_word2vec[i] = model['unk'] 
        else:
            en_word2vec[i] = [np.random.normal(scale=0.25) for i in range(300)]
    else:
        en_word2vec[i] = model[en_word] 

pickle.dump(en_word2vec, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_word2vec_0.2m.p', 'wb'))


#0.5m 
en_vocab_file = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_voc_subtitles_5m.p'
cz_vocab_file = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_subtitles_5m.p'
en_vocab = cPickle.load(open(en_vocab_file,'rb')); cz_vocab = cPickle.load(open(cz_vocab_file,'rb'))
en_word2vec = np.zeros((len(en_w2i), 300))
for en_word, i in en_w2i.items() :
    if not(en_word) in model:
        if en_word == 'UNK':
            en_word2vec[i] = model['unk'] 
        else:
            en_word2vec[i] = [np.random.normal(scale=0.25) for i in range(300)]
    else:
        en_word2vec[i] = model[en_word] 

pickle.dump(en_word2vec, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_word2vec_0.5m.p', 'wb'))
