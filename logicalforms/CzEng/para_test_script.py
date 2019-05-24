#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:41:11 2019

@author: TiffMin
"""

#Make test set 
#First tokenize
import pickle
from mosestokenizer import *
import copy
import random
from collections import defaultdict

training_with_5m_final = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_training_with_5m_final_with_reduced_target_voc.p', 'rb'))
category = {}
for idx, cur_dict in enumerate(training_with_5m_final[:2500000]):
    string = cur_dict['selected_czeng_pair_id']
    current_cat = string.split('-')[0]      
    if not(current_cat in category):
        category[current_cat] = []
    category[current_cat].append(idx)
czeng_cat_dist = {}; tot = sum([len(v) for k,v in category.items() ])
czeng_cat_dist = {k : len(v)/tot for k,v in category.items()}
czeng_cat_tot = {k : len(v) for k,v in category.items()}



#1. First, make test set for 0.5M 
selected = [category['subtitlesM'][i] for i in range(250000)] #selected
selected_rev = {selected_idx:populated_i for populated_i, selected_idx in enumerate(selected)} 
#selected is literally "selected" tw5mfin_idxes 

cz_rm_voc = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_subtitles_5m.p','rb'))
en_rm_voc = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_voc_subtitles_5m.p','rb'))
populated_selected_training = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/populated_selected_training_halfM.p', 'rb')) 
para_test_utts = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para_test_utts_halfM.p', 'rb')) 

cz_w2i = cz_rm_voc['cz_w2i']; cz_i2w = cz_rm_voc['cz_rm_i2w']
en_w2i = en_rm_voc['en_w2i']; en_i2w = en_rm_voc['en_rm_i2w']


reduced_cz_w2i = defaultdict(lambda: cz_w2i['UNK'], cz_w2i)
reduced_en_w2i = defaultdict(lambda: en_w2i['UNK'], en_w2i)

#1.1 First Tokenize 
tokenize_en = MosesTokenizer('en')
para_tok_dict = {}
for tw5mfin_idx, para_utt_list in para_test_utts.items():
    if tw5mfin_idx in selected: 
        para_tok_list = [tokenize_en(utt) for utt in list(set(para_utt_list))]
        para_tok_dict[tw5mfin_idx] = para_tok_list


#1.2 Compare whether it exists in the training set 
para_tok_dict_nodup = {}
for tw5mfin_idx, para_tok_list in para_tok_dict.items():
    populated_i = selected_rev[tw5mfin_idx]
    para_utts_training = [''.join(tok_list[1:-1]) for tok_list in populated_selected_training[populated_i]['selected_czeng_list_para_tok']] + [''.join(populated_selected_training[populated_i]['selected_czeng_eng_moses_tok'][1:-1])]
    para_tok_list_nodup = []
    for para_idx, para_toks in enumerate(para_tok_list):
        if ''.join(para_toks) in para_utts_training:
            pass
        else:
            para_tok_list_nodup.append(para_toks)
    if para_tok_list_nodup != []:
        para_tok_dict_nodup[populated_i] = para_tok_list_nodup
    
#np.mean([len(para_tok_dict_nodup[populated_i]) for populated_i in para_tok_dict_nodup] )  -> 2.57   
#1.3 Sample 200 -> 5000
random.seed(0)
sampled = random.sample(list(para_tok_dict_nodup.keys()),2000)
# =============================================================================
# test_halfm_toks = [] #has length 5479
# for k in sampled:
#     test_halfm_toks += para_tok_dict_nodup[k] 
# =============================================================================
test_halfm = []
for populated_i in sampled:
    train_dict = populated_selected_training[populated_i]
    for para_toks in para_tok_dict_nodup[populated_i]:
        test_dict = {}
        test_dict['cz_idxes'] = train_dict['cz_idxes']; test_dict['cz_leng'] = train_dict['cz_leng']
        test_dict['eng_utts'] = ['SOS_token'] + para_toks + ['EOS_token']
        test_dict['eng_idxes'] = [reduced_en_w2i[tok] for tok in test_dict['eng_utts']]
        test_dict['eng_leng'] = len(test_dict['eng_idxes'])
        test_halfm.append(test_dict)

#save
pickle.dump(test_halfm, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para_test_dict_halfM.p','wb'))

####
####Do the same for 0.2 M and 0.1M 

#2. 0.2M
selected = [category['subtitlesM'][i] for i in range(100000)]
selected_rev = {selected_idx:populated_i for populated_i, selected_idx in enumerate(selected)}

cz_rm_voc = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_subtitles_0.25m.p','rb'))
en_rm_voc = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_voc_subtitles_0.25m.p','rb'))
populated_selected_training = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/populated_selected_training_0.25M.p', 'rb')) 
para_test_utts = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para_test_utts_0.25M.p', 'rb')) 

cz_w2i = cz_rm_voc['cz_w2i']; cz_i2w = cz_rm_voc['cz_rm_i2w']
en_w2i = en_rm_voc['en_w2i']; en_i2w = en_rm_voc['en_rm_i2w']


reduced_cz_w2i = defaultdict(lambda: cz_w2i['UNK'], cz_w2i)
reduced_en_w2i = defaultdict(lambda: en_w2i['UNK'], en_w2i)

#1.1 First Tokenize 
para_tok_dict = {}
for tw5mfin_idx, para_utt_list in para_test_utts.items():
    if tw5mfin_idx in selected: 
        para_tok_list = [tokenize_en(utt) for utt in list(set(para_utt_list))]
        para_tok_dict[tw5mfin_idx] = para_tok_list

#1.2 Compare whether it exists in the training set 
para_tok_dict_nodup = {}
for tw5mfin_idx, para_tok_list in para_tok_dict.items():
    populated_i = selected_rev[tw5mfin_idx]
    para_utts_training = [''.join(tok_list[1:-1]) for tok_list in populated_selected_training[populated_i]['selected_czeng_list_para_tok']] + [''.join(populated_selected_training[populated_i]['selected_czeng_eng_moses_tok'][1:-1])]
    para_tok_list_nodup = []
    for para_idx, para_toks in enumerate(para_tok_list):
        if ''.join(para_toks) in para_utts_training:
            pass
        else:
            para_tok_list_nodup.append(para_toks)
    if para_tok_list_nodup != []:
        para_tok_dict_nodup[populated_i] = para_tok_list_nodup
    
#np.mean([len(para_tok_dict_nodup[populated_i]) for populated_i in para_tok_dict_nodup] )  -> 2.57   
#1.3 Sample 200 -> 5000
random.seed(0)
sampled = random.sample(list(para_tok_dict_nodup.keys()),2000)
# =============================================================================
# test_halfm_toks = [] #has length 5479
# for k in sampled:
#     test_halfm_toks += para_tok_dict_nodup[k] 
# =============================================================================
test_halfm = []
for populated_i in sampled:
    train_dict = populated_selected_training[populated_i]
    for para_toks in para_tok_dict_nodup[populated_i]:
        test_dict = {}
        test_dict['cz_idxes'] = train_dict['cz_idxes']; test_dict['cz_leng'] = train_dict['cz_leng']
        test_dict['eng_utts'] = ['SOS_token'] + para_toks + ['EOS_token']
        test_dict['eng_idxes'] = [reduced_en_w2i[tok] for tok in test_dict['eng_utts']]
        test_dict['eng_leng'] = len(test_dict['eng_idxes'])
        test_halfm.append(test_dict)

#save
pickle.dump(test_halfm, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para_test_dict_0.25M.p','wb'))

#3. 0.1M
selected = [category['subtitlesM'][i] for i in range(50000)] #selected[10] = 12 -> 10번째 populated_dict 는 glb idx(tw5mfin_idx)가 12 라는것
selected_rev = {tw5mfin_idx:populated_i for populated_i, tw5mfin_idx in enumerate(selected)}

cz_rm_voc = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_subtitles_0.1m.p','rb'))
en_rm_voc = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_voc_subtitles_0.1m.p','rb'))
populated_selected_training = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/populated_selected_training_0.1M.p', 'rb')) 
para_test_utts = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para_test_utts_0.1M.p', 'rb')) 

cz_w2i = cz_rm_voc['cz_w2i']; cz_i2w = cz_rm_voc['cz_rm_i2w']
en_w2i = en_rm_voc['en_w2i']; en_i2w = en_rm_voc['en_rm_i2w']


reduced_cz_w2i = defaultdict(lambda: cz_w2i['UNK'], cz_w2i)
reduced_en_w2i = defaultdict(lambda: en_w2i['UNK'], en_w2i)

#3.1 First Tokenize 
para_tok_dict = {}
for tw5mfin_idx, para_utt_list in para_test_utts.items():
    if tw5mfin_idx in selected: 
        para_tok_list = [tokenize_en(utt) for utt in list(set(para_utt_list))]
        para_tok_dict[tw5mfin_idx] = para_tok_list
          
#3.2 Compare whether it exists in the training set 
para_tok_dict_nodup = {}
for tw5mfin_idx, para_tok_list in para_tok_dict.items():
    populated_i = selected_rev[tw5mfin_idx]
    para_utts_training = [''.join(tok_list[1:-1]) for tok_list in populated_selected_training[populated_i]['selected_czeng_list_para_tok']] 
    para_utts_training = []
    para_tok_list_nodup = []
    for para_idx, para_toks in enumerate(para_tok_list):
        if ''.join(para_toks) in para_utts_training:
            pass
        else:
            para_tok_list_nodup.append(para_toks)
    if para_tok_list_nodup != []:
        para_tok_dict_nodup[populated_i] = para_tok_list_nodup
    
#np.mean([len(para_tok_dict_nodup[populated_i]) for populated_i in para_tok_dict_nodup] )  -> 2.57   
#3.3 Sample 200 -> 5000
random.seed(0)
sampled = random.sample(list(para_tok_dict_nodup.keys()),500)
# =============================================================================
# test_halfm_toks = [] #has length 5479
# for k in sampled:
#     test_halfm_toks += para_tok_dict_nodup[k] 
# =============================================================================
test_halfm = []
for populated_i in sampled:
    train_dict = populated_selected_training[populated_i]
    for para_toks in para_tok_dict_nodup[populated_i]:
        test_dict = {}
        test_dict['cz_idxes'] = train_dict['cz_idxes']; test_dict['cz_leng'] = train_dict['cz_leng']
        test_dict['eng_utts'] = ['SOS_token'] + para_toks + ['EOS_token']
        test_dict['eng_idxes'] = [reduced_en_w2i[tok] for tok in test_dict['eng_utts']]
        test_dict['eng_leng'] = len(test_dict['eng_idxes'])
        test_halfm.append(test_dict)

entire_test_halfm = []
for populated_i in para_tok_dict_nodup:
    train_dict = populated_selected_training[populated_i]
    for para_toks in para_tok_dict_nodup[populated_i]:
        test_dict = {}
        test_dict['cz_idxes'] = train_dict['cz_idxes']; test_dict['cz_leng'] = train_dict['cz_leng']
        test_dict['eng_utts'] = ['SOS_token'] + para_toks + ['EOS_token']
        test_dict['eng_idxes'] = [reduced_en_w2i[tok] for tok in test_dict['eng_utts']]
        test_dict['eng_leng'] = len(test_dict['eng_idxes'])
        entire_test_halfm.append(test_dict)

pickle.dump(test_halfm, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/seed0_500.p','wb'))


#save
pickle.dump(entire_test_halfm, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para_entire_test_dict_0.1M.p','wb'))

pickle.dump(test_halfm, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para_test_dict_0.1M.p','wb'))



small_test_sampled = list(para_tok_dict_nodup.keys())[:30]
small_test_halfm = []
for populated_i in small_test_sampled:
    train_dict = populated_selected_training[populated_i]
    for para_toks in para_tok_dict_nodup[populated_i]:
        test_dict = {}
        test_dict['cz_idxes'] = train_dict['cz_idxes']; test_dict['cz_leng'] = train_dict['cz_leng']
        test_dict['eng_utts'] = ['SOS_token'] + para_toks + ['EOS_token']
        test_dict['eng_idxes'] = [reduced_en_w2i[tok] for tok in test_dict['eng_utts']]
        test_dict['eng_leng'] = len(test_dict['eng_idxes'])
        small_test_halfm.append(test_dict)



small_test =  small_test_halfm          
pickle.dump(small_test, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/small_test.p', 'wb'))

#News 
news_test_list_dict = cPickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/newstest_czeng.p', "rb"))  

