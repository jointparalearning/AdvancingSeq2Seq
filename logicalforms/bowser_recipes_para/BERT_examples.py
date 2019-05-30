#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:25:59 2019

@author: TiffMin
"""

#BERT for Overnight Script 
import sys
sys.path.append("/data/scratch-oc40/symin95/packages/bert-embedidng")

from bert_embedding import BertEmbedding
#sys.path.append("/data/scratch-oc40/symin95/packages/mxnet-cu90")
#import mxnet as mx
import os 
import pickle
import json
import nltk

bert = BertEmbedding()

#ctx = mx.gpu(0)
#bert = BertEmbedding(ctx=ctx)


b1 = 'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.'
b2 = 'Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.'
b3 = 'As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.'
b_sep = [b1,b2,b3]
result_list = []
start_time_sep = time.time()
for b in b_sep:
    result_list.append(bert(b))
    end_time_sep = time.time()
    
os.chdir('/data/scratch-oc40/symin95/github_lf/logicalforms/LF-murtaza/bowser_recipes_para')

para_utterance_idx_dict = pickle.load(open('data/para_preprocessed/global_idx2_utterance_dict.p', 'rb'))
overnight_sents = [v for k, v in para_utterance_idx_dict.items()]
overnight_rec_results = bert(overnight_sents)
overnight_toks = [r[0] for r in overnight_rec_results]

nltk_toks = [nltk.word_tokenize(sent) for sent in overnight_sents ]

diff_idx = []
for i in range(len(nltk_toks)):
    if nltk_toks[i] != overnight_toks[i]:
        diff_idx.append(i)

#NLTK tokenized results 
# =============================================================================
# train_file_v1 = "data/para_preprocessed/recipes_train_v1.json"
# test_file_v1 = "data/para_preprocessed/recipes_test_v1.json"
# 
# with open(train_file_v1, "r") as f:
#     qa_list = json.load(f)
# tok_list = [cur_dict["sent_tok"] for cur_dict in qa_list]
# 
# with open(test_file_v1, "r") as f:
#     qa_list = json.load(f)
# tok_list += [cur_dict["sent_tok"] for cur_dict in qa_list]
# =============================================================================

