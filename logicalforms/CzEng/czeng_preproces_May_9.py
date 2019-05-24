#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:02:34 2019

@author: TiffMin
"""
import os
import pickle
import random

czeng_master_dict = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_master.p', 'rb'))
para_nmt_5m_dict = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para-nmt-50m/para-nmt-5m-processed/paranmt_5m_dict.p', 'rb'))
para_nmt_50m_dict = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/temp_data/para_nmt_50m_dict.p', 'rb'))

czeng_eng_joined = czeng_master_dict['czeng_eng_joined']
czeng_eng_joined_rev = {v:k for k,v in czeng_eng_joined.items()}

para_nmt_reference_tokenized_and_joined_50m = para_nmt_50m_dict['para_nmt_reference_tokenized_and_joined']
para_nmt_reference_tokenized_and_joined_50m_rev = {}
 
for k,v in para_nmt_reference_tokenized_and_joined_50m.items():
    if not( v in para_nmt_reference_tokenized_and_joined_50m_rev):
        para_nmt_reference_tokenized_and_joined_50m_rev[v] = []
    para_nmt_reference_tokenized_and_joined_50m_rev[v].append(k)


para_nmt_reference_tokenized_and_joined_5m = para_nmt_5m_dict['para_nmt_reference_tokenized_and_joined']
para_nmt_reference_tokenized_and_joined_5m_rev = para_nmt_5m_dict['para_nmt_reference_tokenized_and_joined_rev']
 
#CZEng의 global idx와 para_nmt_5m 의 idx 맞추기 
err_joint_strs_5m = {}
glb_idx_2_paranmt_5m_idx = {}  #여기 key들은 5m에 paraphrase 있는 애들 
paranmt_5m_idx2_glb_idx = {}
for joined_str, idx_5m_list in para_nmt_reference_tokenized_and_joined_5m_rev.items():
    if joined_str in czeng_eng_joined_rev:
        glb_idx = czeng_eng_joined_rev[joined_str] 
        glb_idx_2_paranmt_5m_idx[glb_idx] = idx_5m_list
        for idx in idx_5m_list:
            paranmt_5m_idx2_glb_idx[idx] = glb_idx
    else:
        err_joint_strs_5m[joined_str] = idx_5m_list
        for idx in idx_5m_list:
            paranmt_5m_idx2_glb_idx[idx] = -1
            
#Add these to paraphrase of CZEng dict 
czeng_list_of_paraphrases = {glb_idx: -1 for glb_idx in czeng_eng_joined}
selected_glb_idx = {}
for glb_idx in glb_idx_2_paranmt_5m_idx:
    czeng_list_of_paraphrases[glb_idx] =  glb_idx_2_paranmt_5m_idx[glb_idx]
    selected_glb_idx[glb_idx] = 1

#Now select the rest glb_idxes : 5000000- 3976194 
num_res_sel = 5000000 - 3976194
temp_cz= pickle.load(open("/data/scratch-oc40/symin95/github_lf/logicalforms/data/temp_data/temp_czeng/temp_cz", "rb"))

temp_cz_pair_id = temp_cz['czeng_pair_id']
czeng_pair_id = czeng_master_dict['czeng_pair_id']

#Match temp_cz index with czeng glb_index
temp_cz_pair_id_rev = {v:k for k,v in temp_cz_pair_id.items()}
czeng_pair_id_rev = {v:k for k,v in czeng_pair_id.items()}

temp_cz_idx2_glb_idx = {}
glb_idx2_temp_cz_idx = {}

for glb_cz_pair_id, glb_idx in czeng_pair_id_rev.items():
    temp_cz_idx = temp_cz_pair_id_rev[glb_cz_pair_id]
    temp_cz_idx2_glb_idx[temp_cz_idx] = glb_idx
    glb_idx2_temp_cz_idx[glb_idx] = temp_cz_idx

for temp_cz_idx in temp_cz_pair_id:
    if not(temp_cz_idx in temp_cz_idx2_glb_idx):
        temp_cz_idx2_glb_idx[temp_cz_idx] = -1
    

#First Select the 50m 
in_range_05_and_95 = {} #These are temp_cz index 
weird_count = 0
for ref_string, v in para_nmt_reference_tokenized_and_joined_50m_rev.items():
    if ref_string in czeng_eng_joined_rev:
        glb_idx = czeng_eng_joined_rev[ref_string]
        in_range_05_and_95[glb_idx] = [nmt_idx for nmt_idx in v if para_nmt_50m_dict['para_nmt_para_score'][nmt_idx]>0.5 and para_nmt_50m_dict['para_nmt_para_score'][nmt_idx]<0.95]

    else:
        weird_count +=1


#Among in_range_05_and_95 that is not in selected_glb_idx, select num_res_sel
to_sel = [glb_idx for glb_idx in in_range_05_and_95 if not(glb_idx in selected_glb_idx)]
random.seed(0)
selected_from_50m = random.sample(to_sel, num_res_sel)
import copy
final_selected_glb_idx = copy.deepcopy(selected_glb_idx)
for idx in selected_from_50m:
    final_selected_glb_idx[idx] = 1

#Add CZ, ENG uttterance to czeng_master_dict
file_names = ['0' + str(i) + 'train' for i in range(10)]
file_names = file_names + [str(i) + 'train' for i in range(10, 98)]

train_line_list = []
for i, file_name in enumerate(file_names):
    file_name = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/para-nmt-50m/data.plaintext-format/'+ file_name
    with open(file_name) as open_f:
        for i, line in enumerate(open_f):
            train_line_list.append(line)

delimeter = '\t'
new_train_line_list = []
for line in train_line_list:
    line = line.strip('\n')
    line = line.split('\t')
    new_train_line_list.append(line)
    
train_line_list = new_train_line_list

assert (len(train_line_list) == len(temp_cz_pair_id_rev))

        
selected_czeng_cz_utt = {}; selected_czeng_eng_utt = {}
selected_czeng_cz_moses_tok = {}; selected_czeng_eng_moses_tok = {}
from mosestokenizer import *
tokenize_en = MosesTokenizer('en')
tokenize_cz = MosesTokenizer('cz')
for glb_idx in final_selected_glb_idx:
    temp_cz_idx = glb_idx2_temp_cz_idx[glb_idx]
    pair = train_line_list[temp_cz_idx]
    assert  pair[0] == temp_cz_pair_id[temp_cz_idx]
    selected_czeng_cz_utt[glb_idx] = str.lower(pair[2]); selected_czeng_eng_utt[glb_idx] = str.lower(pair[3])
    selected_czeng_cz_moses_tok[glb_idx] = tokenize_cz(selected_czeng_cz_utt[glb_idx]); selected_czeng_eng_moses_tok[glb_idx] = tokenize_en(selected_czeng_eng_utt[glb_idx])


#Get Utterance for the paraphrases 50m and 5m 
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
    
para_nmt_line_list_50m =  copy.deepcopy(new_para_nmt_line_list)  

file_name = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/para-nmt-50m/para-nmt-5m-processed/para-nmt-5m-processed.txt'
para_nmt_line_list = []
with open(file_name) as open_f:
    for i, line in enumerate(open_f):
        para_nmt_line_list.append(line)

#Run from here      
delimeter = '\t'
new_para_nmt_line_list = []
for line in para_nmt_line_list:
    line = line.strip('\n')
    line = line.split('\t')
    new_para_nmt_line_list.append(line)
    
para_nmt_line_list_5m =   copy.deepcopy(new_para_nmt_line_list)
#check length of para_nmt_line_list_5m and para_nmt_line_list_50m

############
#Now actually add para 
#REDO HERE: list of index말고 실제로 para 담기게 
tokenize_en = MosesTokenizer('en')


selected_czeng_list_para_utts = {}


for glb_idx in selected_glb_idx:
    para_idx_5m =  czeng_list_of_paraphrases[glb_idx]
    para_utts = [para_nmt_line_list_5m[i][1] for i in para_idx_5m]
    selected_czeng_list_para_utts[glb_idx] = list(set(para_utts)) 
    assert czeng_list_of_paraphrases[glb_idx] != 1

for glb_idx in selected_from_50m:
    joined_str = czeng_eng_joined[glb_idx]
    idx_50m = para_nmt_reference_tokenized_and_joined_50m_rev[joined_str]
    para_utts = [para_nmt_line_list_50m[i][1] for i in idx_50m]
    selected_czeng_list_para_utts[glb_idx] = list(set(para_utts)) 
    
    

selected_czeng_pair_id = {glb_idx :czeng_pair_id[glb_idx] for glb_idx in final_selected_glb_idx}

      
#Now attach pair id and list of paraphrases 
selected_czeng_cz_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}; selected_czeng_eng_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}
selected_czeng_cz_i2w = {v:k for k, v in selected_czeng_cz_w2i.items()}; selected_czeng_eng_i2w = {v:k for k, v in selected_czeng_eng_w2i.items()}
for glb_idx, cz_tok_list in selected_czeng_cz_moses_tok.items():
    for cz_tok in cz_tok_list:
        if cz_tok not in selected_czeng_cz_w2i:
            selected_czeng_cz_i2w[len(selected_czeng_cz_w2i)] = cz_tok
            selected_czeng_cz_w2i[cz_tok] = len(selected_czeng_cz_w2i)

#아직 Run 못함
for glb_idx, cz_tok_list in selected_czeng_eng_moses_tok.items():
    for cz_tok in cz_tok_list:
        if cz_tok not in selected_czeng_eng_w2i:
            selected_czeng_eng_i2w[len(selected_czeng_eng_w2i)] = cz_tok
            selected_czeng_eng_w2i[cz_tok] = len(selected_czeng_eng_w2i)
  
#Things to save

#entire czeng master 

czeng_master_dict['temp_cz_idx2_glb_idx'] = temp_cz_idx2_glb_idx
czeng_master_dict['glb_idx2_temp_cz_idx'] = glb_idx2_temp_cz_idx
czeng_master_dict = pickle.dump(czeng_master_dict, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_master.p', 'wb'))


#selected idxes 
selected_czeng_dict = {'selected_czeng_pair_id': selected_czeng_pair_id, 'final_selected_glb_idx':final_selected_glb_idx,
                       'selected_czeng_cz_utt': selected_czeng_cz_utt,'selected_czeng_eng_utt':selected_czeng_eng_utt,
                       'selected_czeng_cz_moses_tok':selected_czeng_cz_moses_tok, 'selected_czeng_eng_moses_tok':selected_czeng_eng_moses_tok,
                       'selected_czeng_cz_w2i':selected_czeng_cz_w2i, 'selected_czeng_cz_i2w':selected_czeng_cz_i2w,
                       'selected_czeng_list_para_utts':selected_czeng_list_para_utts}

pickle.dump(selected_czeng_dict, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/selected_czeng_dict.p', 'wb'))



###################
from mosestokenizer import *
tokenize_en = MosesTokenizer('en')

selected_czeng_dict = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/selected_czeng_dict.p', 'rb'))

#더 해야할것들 
#1. Tokenize all the English para
selected_czeng_list_para_tok = {}
for glb_idx, list_utts in selected_czeng_dict['selected_czeng_list_para_utts'].items():
    selected_czeng_list_para_tok[glb_idx] = [tokenize_en(utt) for utt in list_utts]
    selected_czeng_list_para_tok[glb_idx].append(selected_czeng_dict['selected_czeng_eng_moses_tok'][glb_idx])


#2. Make English Vocabulary 
selected_czeng_eng_moses_tok = selected_czeng_dict['selected_czeng_eng_moses_tok']
selected_czeng_eng_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}; selected_czeng_eng_i2w = {v:k for k, v in selected_czeng_eng_w2i.items()}
selected_total_eng_tok = {glb_idx: [eng_tok] + selected_czeng_list_para_tok[glb_idx] for glb_idx, eng_tok in selected_czeng_eng_moses_tok.items()}
for glb_idx, cz_tok_list in selected_total_eng_tok.items():
    for cz_tok_sent in cz_tok_list:
        for cz_tok in cz_tok_sent:
            if cz_tok not in selected_czeng_eng_w2i:
                selected_czeng_eng_i2w[len(selected_czeng_eng_w2i)] = cz_tok
                selected_czeng_eng_w2i[cz_tok] = len(selected_czeng_eng_w2i)
     
#Save to pickle
selected_czeng_dict['selected_czeng_list_para_tok'] = selected_czeng_list_para_tok
selected_czeng_dict['selected_czeng_eng_i2w'] = selected_czeng_eng_i2w
selected_czeng_dict['selected_czeng_eng_w2i'] = selected_czeng_eng_w2i

pickle.dump(selected_czeng_dict,open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/selected_czeng_dict.p', 'wb') )

#Finally convert to list of dict 
mis_sampled_czeng_final = []
for glb_idx in selected_czeng_dict['selected_czeng_list_para_utts']:
    cur_dict = {}
    cur_dict['glb_idx'] = glb_idx
    for k in selected_czeng_dict:
        if not (k in ['selected_czeng_cz_w2i', 'selected_czeng_cz_i2w', 'selected_czeng_eng_i2w', 'selected_czeng_eng_w2i']):
            cur_dict[k] = selected_czeng_dict[k][glb_idx]
    mis_sampled_czeng_final.append(cur_dict)

pickle.dump(mis_sampled_czeng_final,open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/mis_sampled_czeng_final.p', 'wb') )



########
########
########
######Try again from here
#좀 다시 해야될듯 
#0. Pull out paraphrases as official inputs
#paraphrase group make 
paraphrase_group_dict = {}
for i, cur_dict in enumerate(mis_sampled_czeng_final):
    cur_dict['']




#1. Add EOS and SOS/ length / w2i 
for i, cur_dict in enumerate(mis_sampled_czeng_final):
    cur_dict['selected_czeng_cz_moses_tok'] = ['SOS_token'] + cur_dict['selected_czeng_cz_moses_tok'] + ['EOS_token']
    cur_dict['selected_czeng_eng_moses_tok'] = ['SOS_token'] + cur_dict['selected_czeng_eng_moses_tok'] + ['EOS_token']
    
    

    cur_dict['eng_length'] = len()

#2. Add pad token       


#3. Save w2i, i2w vocab      