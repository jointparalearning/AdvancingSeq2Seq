#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 00:21:09 2019

@author: TiffMin
"""

import _pickle as cPickle
import nltk

###for PARA-NMT 


file_name = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/para-nmt-50m/para-nmt-50m.txt'
#file_name = 'para-nmt-5m-processed.txt'

para_nmt_line_list = []
with open(file_name) as open_f:
    for i, line in enumerate(open_f):
        para_nmt_line_list.append(line)
        
delimeter = '\t'
new_para_nmt_line_list = []
for line in para_nmt_line_list:
    line = line.strip('\n')
    line = line.split('\t')
    new_para_nmt_line_list.append(line)
    
para_nmt_line_list =   new_para_nmt_line_list  
   
### For Para-NMT 50M
 
para_nmt_reference_tokenized = {}  
para_nmt_reference_tokenized_and_joined = {}
para_nmt_paraphrase_tokenized = {}  
para_nmt_para_score = {}

for i, pair in enumerate(para_nmt_line_list[36300000:]):
    i = i + 36300000
    if i == 36300000:
        print(i)
    if i % 10000000 ==0:
        print(i)
        para_nmt_50m_dict = {'para_nmt_reference_tokenized':para_nmt_reference_tokenized, 
           'para_nmt_reference_tokenized_and_joined':para_nmt_reference_tokenized_and_joined, 
           'para_nmt_paraphrase_tokenized':para_nmt_paraphrase_tokenized, 
           'para_nmt_para_score':para_nmt_para_score}
        cPickle.dump(para_nmt_50m_dict, open('temp_data/para_nmt_50m_dict.p', 'wb'))
    ref = str.lower(pair[0]); par = str.lower(pair[1]); score = float(pair[2])
    para_nmt_reference_tokenized[i] = nltk.word_tokenize(ref); para_nmt_paraphrase_tokenized[i] = nltk.word_tokenize(par)
    para_nmt_reference_tokenized_and_joined[i] = ''.join(para_nmt_reference_tokenized[i])
    para_nmt_para_score[i] = score
    
### For Para-NMT 5M
para_nmt_reference_tokenized = {}  
para_nmt_reference_tokenized_and_joined = {}
para_nmt_paraphrase_tokenized = {}  

for i, pair in enumerate(para_nmt_line_list):
    if i % 1000 ==0:
        print(i)
    ref = str.lower(pair[0]); par = str.lower(pair[1])
    para_nmt_reference_tokenized[i] = nltk.word_tokenize(ref); para_nmt_paraphrase_tokenized[i] = nltk.word_tokenize(par)
    para_nmt_reference_tokenized_and_joined[i] = ''.join(para_nmt_reference_tokenized[i])

para_nmt_reference_tokenized_and_joined_rev = {}
 
for k,v in para_nmt_reference_tokenized_and_joined.items():
    if not( v in para_nmt_reference_tokenized_and_joined_rev):
        para_nmt_reference_tokenized_and_joined_rev[v] = []
    para_nmt_reference_tokenized_and_joined_rev[v].append(k)

cPickle.dump(para_nmt_reference_tokenized_and_joined_rev, open('temp_data/para_nmt_reference_tokenized_and_joined_rev.p', 'wb'))

paranmt_5m_dict = {'para_nmt_reference_tokenized':para_nmt_reference_tokenized, 
           'para_nmt_reference_tokenized_and_joined':para_nmt_reference_tokenized_and_joined, 
           'para_nmt_paraphrase_tokenized':para_nmt_paraphrase_tokenized, 
           'para_nmt_reference_tokenized_and_joined_rev':para_nmt_reference_tokenized_and_joined_rev}
cPickle.dump(paranmt_5m_dict, open("paranmt_5m_dict.p", "wb"))  

        
###for CZ-ENG
    
file_names = ['0' + str(i) + 'train' for i in range(10)]
file_names = file_names + [str(i) + 'train' for i in range(10, 98)]

train_line_list = []
for i, file_name in enumerate(file_names):
    file_name = 'para-nmt-50m/data.plaintext-format/'+ file_name
    with open(file_name) as open_f:
        for i, line in enumerate(open_f):
            train_line_list.append(line)

test_file_name = 'para-nmt-50m/data-plaintext-format.99etest'
val_file_name = 'para-nmt-50m/data-plaintext-format.98dtest'

test_line_list = []
with open(test_file_name) as open_f:
    for i, line in enumerate(open_f):
        test_line_list.append(line)
            
val_line_list = []
with open(val_file_name) as open_f:
    for i, line in enumerate(open_f):
        val_line_list.append(line)
            


delimeter = '\t'
new_train_line_list = []
for line in train_line_list:
    line = line.strip('\n')
    line = line.split('\t')
    new_train_line_list.append(line)
    
train_line_list = new_train_line_list

czeng_pair_id = {}
czeng_pair_score = {}
czeng_cz = {}
czeng_eng = {}
czeng_eng_joined = {}
for i, pair in enumerate(train_line_list[20553882:]):
    i = i + 20553882
    if i % 5000000 ==0:
        print(i)
        temp_cz = {'czeng_pair_id':czeng_pair_id, 'czeng_pair_score':czeng_pair_score, 'czeng_cz':czeng_cz, 'czeng_eng':czeng_eng, 'czeng_eng_joined':czeng_eng_joined}
        cPickle.dump(temp_cz, open("temp_data/temp_czeng/temp_cz", "wb"))  
    czeng_pair_id[i] = pair[0]; czeng_pair_score[i] = float(pair[1])
    czeng_cz[i] = nltk.word_tokenize(str.lower(pair[2])); czeng_eng[i] = nltk.word_tokenize(str.lower(pair[3]))
    czeng_eng_joined[i] = ''.join(czeng_eng[i])




#czeng_eng_joined_rev = {v:k for k,v in czeng_eng_joined.items()}

czeng_eng_joined_rev = {}
 
for k,v in czeng_eng_joined.items():
    if not( v in czeng_eng_joined_rev):
        czeng_eng_joined_rev[v] = []
    czeng_eng_joined_rev[v].append(k)


cPickle.dump(czeng_eng_joined_rev ,open("temp_data/czeng_eng_joined_rev", 'wb'))

czeng_eng_joined_rev = cPickle.load(open("../../temp_data/czeng_eng_joined_rev", 'rb'))

# =============================================================================
# #Make absolute index of ref strings that are the same 
# para_nmt_reference_absolute_idx_dict = {}
# para_nmt_reference_absolute_idx_strings = {}
# absolute_idx = 0
# for ref_string, idx_list in para_nmt_reference_tokenized_and_joined_rev.items():
#     para_nmt_reference_absolute_idx_dict[absolute_idx] = idx_list; para_nmt_reference_absolute_idx_strings[absolute_idx] = ref_string
#     absolute_idx+=1
# =============================================================================

# =============================================================================
# paranmt_idx2_czeng_idx = {}
# count_contained = 0
# for ref_string, v in para_nmt_reference_tokenized_and_joined_rev.items():
#     if ref_string in czeng_eng_joined_rev:
#         count_contained +=1
#         czeng_idx = czeng_eng_joined_rev[ref_string]
#         paranmt_idx2_czeng_idx[v] = czeng_idx 
# 
# =============================================================================

paranmt_abs_idx2_czeng_idx_list_dict = {}
count_contained = 0
for absolute_idx, idx_list in para_nmt_reference_absolute_idx_dict.items():
    ref_string = para_nmt_reference_absolute_idx_strings[absolute_idx]
    if ref_string in czeng_eng_joined_rev:
        count_contained +=1
        czeng_idx_list = czeng_eng_joined_rev[ref_string]
        paranmt_abs_idx2_czeng_idx_list_dict[absolute_idx] = czeng_idx_list


#absolute idx로 모든지 access 할 수 있게 바꿔놓자 
czeng_eng_absolute_master_dict = {}

#Make absolute index from czeng 
cz_eng_absolute_idx_2_czeng_idx_dict = {}
cz_eng_absolute_idx_eng_joined_strings = {}
absolute_idx = 0
for joined_string, idx_list in czeng_eng_joined_rev.items():
    cz_eng_absolute_idx_2_czeng_idx_dict[absolute_idx] = idx_list; cz_eng_absolute_idx_eng_joined_strings[absolute_idx] = joined_string
    absolute_idx+=1

#Among the list, just choose the one with the highest score and keep that pair 
cz_eng_absolute_idx_2_max_czeng_idx = {}
for abs_idx, idx_list in cz_eng_absolute_idx_2_czeng_idx_dict.items():
    max_idx = idx_list[np.argmax([czeng_pair_score[idx] for idx in idx_list])]
    cz_eng_absolute_idx_2_max_czeng_idx[abs_idx] = max_idx
    
#Now make new czeng_rev 
#Just reorganize everything 
czeng_pair_id_reorg = {}
czeng_pair_score_reorg = {}
czeng_cz_reorg = {}
czeng_eng_reorg = {}
czeng_eng_joined_reorg = {}
counter = 0 
for abs_idx, cz_idx in cz_eng_absolute_idx_2_max_czeng_idx.items():
    czeng_pair_id_reorg[counter] = czeng_pair_id[cz_idx]
    czeng_pair_score_reorg[counter] = czeng_pair_score[cz_idx]
    czeng_cz_reorg[counter] = czeng_cz[cz_idx]
    czeng_eng_reorg[counter] = czeng_eng[cz_idx]
    czeng_eng_joined_reorg[counter] = czeng_eng_joined[cz_idx]
    counter +=1
    
czeng_master = {'czeng_pair_id':czeng_pair_id_reorg, 'czeng_pair_score': czeng_pair_score_reorg, 'czeng_cz': czeng_cz_reorg,
                'czeng_eng': czeng_eng_reorg, 'czeng_eng_joined': czeng_eng_joined_reorg}

cPickle.dump(czeng_master, open("czeng_master.p", "wb"))  

czeng_eng_joined_rev = {v:k for k,v in czeng_eng_joined_reorg.items()}

# =============================================================================
# paranmt_idx2_czeng_idx = {}
# count_contained = 0
# for ref_string, v in para_nmt_reference_tokenized_and_joined_rev.items():
#     if ref_string in czeng_eng_joined_rev:
#         count_contained +=1
#         czeng_idx = czeng_eng_joined_rev[ref_string]
#         paranmt_idx2_czeng_idx[v] = czeng_idx 
# 
# =============================================================================

#Remove every paraphrase with score lower than 0.5 
#And count how many 
in_range_05_and_95 = {}
#argmaxes = {}
for ref_string, v in para_nmt_reference_tokenized_and_joined_rev.items():
    in_range_05_and_95[ref_string] = [nmt_idx for nmt_idx in v if para_nmt_para_score[nmt_idx]>0.5 and para_nmt_para_score[nmt_idx]<0.95]
    #argmaxes[ref_string] = v[np.argmax([para_nmt_para_score[nmt_idx] for nmt_idx in v])]

#Now compare the argmaxes and the >0.5 (whether >0.5 is acceptable)
#Count how many are >0.5  

#Distribution count for CzEng 
category = {}
for idx, string in czeng_pair_id_reorg.items():
    current_cat = string.split('-')[0]      
    if not(current_cat in category):
        category[current_cat] = []
    category[current_cat].append(idx)
    
czeng_cat_dist = {}; tot = sum([len(v) for k,v in category.items() ])
czeng_cat_dist = {k : len(v)/tot for k,v in category.items()}


#Distribution count for ParaNMT 5m
nmt_5m_category = {}

para_nmt_5m_reference_tokenized_and_joined_rev = paranmt_5m_dict['para_nmt_reference_tokenized_and_joined_rev'] 
paranmt_5m_idx2_czeng_idx = {}
count_contained = 0
for ref_string, v in para_nmt_5m_reference_tokenized_and_joined_rev.items():
    if ref_string in czeng_eng_joined_rev:
        count_contained +=1
        czeng_idx = czeng_eng_joined_rev[ref_string]
        for v_idx in v:
            paranmt_5m_idx2_czeng_idx[v_idx] = czeng_idx 

for para_idx, czeng_idx in paranmt_5m_idx2_czeng_idx.items():
    string = czeng_pair_id_reorg[czeng_idx]
    current_cat = string.split('-')[0]      
    if not(current_cat in nmt_5m_category):
        nmt_5m_category[current_cat] = []
    nmt_5m_category[current_cat].append(idx)
    
nmt_5m_cat_dist = {}; tot = sum([len(v) for k,v in nmt_5m_category.items() ])
nmt_5m_cat_dist = {k : len(v)/tot for k,v in nmt_5m_category.items()}


#Distribution count for ParaNMT 50 
nmt_50_category = {}
# =============================================================================
# paranmt_idx2_czeng_idx = {}
# count_contained = 0
# for ref_string, v in para_nmt_reference_tokenized_and_joined_rev.items():
#     if ref_string in czeng_eng_joined_rev:
#         count_contained +=1
#         czeng_idx = czeng_eng_joined_rev[ref_string]
#         paranmt_idx2_czeng_idx[v] = czeng_idx 
# =============================================================================



