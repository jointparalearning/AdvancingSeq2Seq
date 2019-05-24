#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:13:00 2019

@author: TiffMin
"""

temp_cz= pickle.load(open("/data/scratch-oc40/symin95/github_lf/logicalforms/data/temp_data/temp_czeng/temp_cz", "rb"))
para_nmt_50m_dict = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/temp_data/para_nmt_50m_dict.p', 'rb'))
para_nmt_5m_dict = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para-nmt-50m/para-nmt-5m-processed/paranmt_5m_dict.p', 'rb'))
para_nmt_reference_tokenized_and_joined_5m = para_nmt_5m_dict['para_nmt_reference_tokenized_and_joined']
para_nmt_reference_tokenized_and_joined_5m_rev = para_nmt_5m_dict['para_nmt_reference_tokenized_and_joined_rev']
czeng_master_dict = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_master.p', 'rb'))
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


czeng_list_of_paraphrases = {glb_idx: -1 for glb_idx in czeng_eng_joined}
selected_glb_idx = {}
for glb_idx in glb_idx_2_paranmt_5m_idx:
    czeng_list_of_paraphrases[glb_idx] =  glb_idx_2_paranmt_5m_idx[glb_idx]
    selected_glb_idx[glb_idx] = 1







in_range_05_and_95 = {} #These are temp_cz index 
weird_count = 0
for ref_string, v in para_nmt_reference_tokenized_and_joined_50m_rev.items():
    if ref_string in czeng_eng_joined_rev:
        glb_idx = czeng_eng_joined_rev[ref_string]
        in_range_05_and_95[glb_idx] = [nmt_idx for nmt_idx in v if para_nmt_50m_dict['para_nmt_para_score'][nmt_idx]>0.5 and para_nmt_50m_dict['para_nmt_para_score'][nmt_idx]<0.95]

    else:
        weird_count +=1
 
    
len(in_range_05_and_95)
in_range_05_and_95 = {glb_idx: v for glb_idx, v in copy.deepcopy(in_range_05_and_95).items() if len(v)!=0}
len(in_range_05_and_95)



to_sel = [glb_idx for glb_idx in in_range_05_and_95 if not(glb_idx in selected_glb_idx)]
import random
random.seed(0)
num_res_sel = 5000000 - 3976194
selected_from_50m = random.sample(to_sel, num_res_sel)


#Make para nmt line list 
###########################################

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


############################################
selected_czeng_list_para_utts = {}


for glb_idx in selected_glb_idx:
    para_idx_5m =  czeng_list_of_paraphrases[glb_idx]
    para_utts = [para_nmt_line_list_5m[i][1] for i in para_idx_5m]
    selected_czeng_list_para_utts[glb_idx] = list(set(para_utts)) 
    assert czeng_list_of_paraphrases[glb_idx] != 1

for glb_idx in selected_from_50m:
    idx_50m = in_range_05_and_95[glb_idx]
    para_utts = [para_nmt_line_list_50m[i][1] for i in idx_50m]
    selected_czeng_list_para_utts[glb_idx] = list(set(para_utts)) 

final_selected_glb_idx = {idx: 1 for idx in selected_from_50m}
for k, v in selected_glb_idx.items():
    final_selected_glb_idx[k] = v
selected_czeng_pair_id = {glb_idx : czeng_master_dict['czeng_pair_id'][glb_idx] for glb_idx in final_selected_glb_idx}

#Now tokenize 
#Tokenize eng and cz 
temp_cz_pair_id = temp_cz['czeng_pair_id']
czeng_pair_id = czeng_master_dict['czeng_pair_id']
temp_cz_idx2_glb_idx = {}
glb_idx2_temp_cz_idx = {}
temp_cz_pair_id_rev = {v:k for k,v in temp_cz_pair_id.items()}
czeng_pair_id_rev = {v:k for k,v in czeng_pair_id.items()}

for glb_cz_pair_id, glb_idx in czeng_pair_id_rev.items():
    temp_cz_idx = temp_cz_pair_id_rev[glb_cz_pair_id]
    temp_cz_idx2_glb_idx[temp_cz_idx] = glb_idx
    glb_idx2_temp_cz_idx[glb_idx] = temp_cz_idx




selected_czeng_cz_utt = {}; selected_czeng_eng_utt = {}
for glb_idx in final_selected_glb_idx:
    temp_cz_idx = glb_idx2_temp_cz_idx[glb_idx]
    pair = train_line_list[temp_cz_idx]
    assert  pair[0] == temp_cz_pair_id[temp_cz_idx]
    selected_czeng_cz_utt[glb_idx] = str.lower(pair[2]); selected_czeng_eng_utt[glb_idx] = str.lower(pair[3])

selected_czeng_dict = {'selected_czeng_pair_id': selected_czeng_pair_id, 'final_selected_glb_idx':final_selected_glb_idx,
                       'selected_czeng_cz_utt': selected_czeng_cz_utt,'selected_czeng_eng_utt':selected_czeng_eng_utt,
                       'selected_czeng_list_para_utts':selected_czeng_list_para_utts}

cPickle.dump(selected_czeng_dict, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/selected_czeng_dict_05_95_inrange.p', 'wb'))

######################
#####Redo from here 
import pickle
from mosestokenizer import *
import copy
selected_czeng_dict = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/selected_czeng_dict_05_95_inrange.p', 'rb'))


selected_czeng_cz_moses_tok = {}; selected_czeng_eng_moses_tok = {}
tokenize_en = MosesTokenizer('en')
tokenize_cz = MosesTokenizer('cz')
selected_czeng_cz_utt = selected_czeng_dict['selected_czeng_cz_utt']; selected_czeng_eng_utt = selected_czeng_dict['selected_czeng_eng_utt']
for glb_idx in selected_czeng_dict['final_selected_glb_idx']:
    selected_czeng_cz_moses_tok[glb_idx] = tokenize_cz(selected_czeng_cz_utt[glb_idx]); selected_czeng_eng_moses_tok[glb_idx] = tokenize_en(selected_czeng_eng_utt[glb_idx])



#Tokenize paraphrase 
selected_czeng_list_para_tok = {}
for glb_idx, list_utts in selected_czeng_dict['selected_czeng_list_para_utts'].items():
    selected_czeng_list_para_tok[glb_idx] = [tokenize_en(utt) for utt in list_utts]
    #selected_czeng_list_para_tok[glb_idx].append(selected_czeng_dict['selected_czeng_eng_moses_tok'][glb_idx])


#Make vocab
selected_czeng_cz_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}; selected_czeng_eng_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}
selected_czeng_cz_i2w = {v:k for k, v in selected_czeng_cz_w2i.items()}; selected_czeng_eng_i2w = {v:k for k, v in selected_czeng_eng_w2i.items()}
for glb_idx, cz_tok_list in selected_czeng_cz_moses_tok.items():
    for cz_tok in cz_tok_list:
        if cz_tok not in selected_czeng_cz_w2i:
            selected_czeng_cz_i2w[len(selected_czeng_cz_w2i)] = cz_tok
            selected_czeng_cz_w2i[cz_tok] = len(selected_czeng_cz_w2i)


selected_czeng_list_para_tok_lowered = copy.deepcopy(selected_czeng_list_para_tok)

for glb_idx, list_of_tok_list in  selected_czeng_list_para_tok_lowered.items():
    for i, tok_list in enumerate(selected_czeng_list_para_tok_lowered[glb_idx]):
        for j, tok in enumerate(tok_list):
            selected_czeng_list_para_tok_lowered[glb_idx][i][j] = str.lower(selected_czeng_list_para_tok_lowered[glb_idx][i][j])
selected_czeng_list_para_tok = selected_czeng_list_para_tok_lowered 


selected_total_eng_tok = {glb_idx: [eng_tok] + selected_czeng_list_para_tok[glb_idx] for glb_idx, eng_tok in selected_czeng_eng_moses_tok.items()}
for glb_idx, cz_tok_list in selected_total_eng_tok.items():
    for cz_tok_sent in cz_tok_list:
        for cz_tok in cz_tok_sent:
            if cz_tok not in selected_czeng_eng_w2i:
                selected_czeng_eng_i2w[len(selected_czeng_eng_w2i)] = cz_tok
                selected_czeng_eng_w2i[cz_tok] = len(selected_czeng_eng_w2i)



#Now make list of dict
selected_czeng_dict['selected_czeng_cz_moses_tok'] = selected_czeng_cz_moses_tok; selected_czeng_dict['selected_czeng_eng_moses_tok'] = selected_czeng_eng_moses_tok
selected_czeng_dict['selected_czeng_list_para_tok'] = selected_czeng_list_para_tok
in_range_sampled_czeng_final = []
for glb_idx in selected_czeng_dict['selected_czeng_list_para_utts']:
    cur_dict = {}
    cur_dict['glb_idx'] = glb_idx
    for k in selected_czeng_dict:
        if not (k in ['selected_czeng_cz_w2i', 'selected_czeng_cz_i2w', 'selected_czeng_eng_i2w', 'selected_czeng_eng_w2i']):
            cur_dict[k] = selected_czeng_dict[k][glb_idx]
    in_range_sampled_czeng_final.append(cur_dict)


ref_in_range_sampled_czeng_final = copy.deepcopy(in_range_sampled_czeng_final)
#Now add paraphrases as normal inputs 
for i, ex_dict in enumerate(ref_in_range_sampled_czeng_final):
    para_tok_list = ex_dict['selected_czeng_list_para_tok'] #list of list of toks
    ori_utt_tok = ex_dict['selected_czeng_eng_moses_tok'] #list of toks
    
    for para_idx, para_tok in enumerate(para_tok_list) :
        new_cur_dict = {}
        new_cur_dict['selected_czeng_eng_moses_tok'] = para_tok
        new_para_tok_list = copy.deepcopy(para_tok_list)
        new_para_tok_list.remove(para_tok)
        new_para_tok_list.append(ori_utt_tok)
        new_cur_dict['selected_czeng_list_para_tok'] = new_para_tok_list
        for k in ex_dict:
            if k not in(['selected_czeng_list_para_tok', 'selected_czeng_eng_moses_tok', 'selected_czeng_list_para_utts']):
                new_cur_dict[k] = ex_dict[k]
                
        in_range_sampled_czeng_final.append(new_cur_dict)


#Save two versions 
pickle.dump(in_range_sampled_czeng_final, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/in_range_sampled_czeng_final_yes_para_input.p', 'wb'))
pickle.dump(ref_in_range_sampled_czeng_final, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/in_range_sampled_czeng_final_no_para_input.p', 'wb')) 
##############
#Do again from here 
###############
if 1 ==1:
    in_range_sampled_czeng_final = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/in_range_sampled_czeng_final_yes_para_input.p', 'rb'))
###########
#String lower everything
############
    for i, cur_dict in enumerate(in_range_sampled_czeng_final):
        para_tok_list = []
        for tok_list in cur_dict['selected_czeng_list_para_tok']:
            para_tok_list.append([str.lower(tok) for tok in tok_list])
        cur_dict['selected_czeng_list_para_tok'] = para_tok_list
        cur_dict['selected_czeng_eng_moses_tok'] = [str.lower(tok) for tok in cur_dict['selected_czeng_eng_moses_tok']]
###########
#Now make vocab
############
    selected_czeng_cz_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}; selected_czeng_eng_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}
    selected_czeng_cz_i2w = {v:k for k, v in selected_czeng_cz_w2i.items()}; selected_czeng_eng_i2w = {v:k for k, v in selected_czeng_eng_w2i.items()}
    for i, cur_dict in enumerate(in_range_sampled_czeng_final):
        cz_tok_list = cur_dict['selected_czeng_cz_moses_tok']
        for cz_tok in cz_tok_list:
            if cz_tok not in selected_czeng_cz_w2i:
                selected_czeng_cz_i2w[len(selected_czeng_cz_w2i)] = str.lower(cz_tok)
                selected_czeng_cz_w2i[str.lower(cz_tok)] = len(selected_czeng_cz_w2i)
    for i, cur_dict in enumerate(in_range_sampled_czeng_final):
        eng_tok_list = cur_dict['selected_czeng_eng_moses_tok']
        for eng_tok in eng_tok_list:
            if eng_tok not in selected_czeng_eng_w2i:
                selected_czeng_eng_i2w[len(selected_czeng_eng_w2i)] = str.lower(eng_tok)
                selected_czeng_eng_w2i[str.lower(eng_tok)] = len(selected_czeng_eng_w2i)          
    pickle.dump({'cz_w2i': selected_czeng_cz_w2i, 'cz_i2w':selected_czeng_cz_i2w}, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_final.p', 'wb'))
    pickle.dump({'eng_w2i':selected_czeng_eng_w2i, 'eng_i2w':selected_czeng_eng_i2w}, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/eng_voc_final.p', 'wb'))

    
    
if 1 ==1:    
    #Add EOS/SOS only to the ver with para inputs  
    for i, cur_dict in enumerate(in_range_sampled_czeng_final):
        cur_dict['selected_czeng_eng_moses_tok'] = ['SOS_token'] + cur_dict['selected_czeng_eng_moses_tok'] + ['EOS_token']
        cur_dict['selected_czeng_cz_moses_tok'] = ['SOS_token'] + cur_dict['selected_czeng_cz_moses_tok'] + ['EOS_token']
        cur_dict['selected_czeng_list_para_tok'] = [['SOS_token'] + tok_list + ['EOS_token']  for tok_list in cur_dict['selected_czeng_list_para_tok'] ]
            
    for i, cur_dict in enumerate(in_range_sampled_czeng_final):
        cur_dict['eng_leng'] = len(cur_dict['selected_czeng_eng_moses_tok']) 
        cur_dict['cz_leng'] = len(cur_dict['selected_czeng_cz_moses_tok']) 
        cur_dict['para_leng_list'] = [len(tok_list) for tok_list in cur_dict['selected_czeng_list_para_tok'] ]
        cur_dict['eng_idxes'] = [selected_czeng_eng_w2i[tok] for tok in cur_dict['selected_czeng_eng_moses_tok'] ]
        cur_dict['cz_idxes'] = [selected_czeng_cz_w2i[tok] for tok in cur_dict['selected_czeng_cz_moses_tok'] ]
        para_idxes_lists = []
        for tok_list in cur_dict['selected_czeng_list_para_tok'] :
            para_idxes_lists.append([selected_czeng_eng_w2i[tok] for tok in tok_list])
        cur_dict['para_idxes_lists'] = para_idxes_lists
    pickle.dump(in_range_sampled_czeng_final, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/in_range_sampled_czeng_final_yes_para_input.p', 'wb'))      


#Check length correct 
    
    

# =============================================================================
# #이건 지금 하지 말고 나중에 하는게 나은듯 
# if 1 ==1:        
#     #Add Padding 
#     max_len_eng = max([cur_dict['eng_leng'] for cur_dict in in_range_sampled_czeng_final])
#     max_len_cz = max([cur_dict['cz_leng'] for cur_dict in in_range_sampled_czeng_final])
#     for i, cur_dict in enumerate(in_range_sampled_czeng_final):
#         cur_dict['selected_czeng_eng_moses_tok'] = cur_dict['selected_czeng_eng_moses_tok'] + (max_len_eng - cur_dict['eng_leng']) * ['PAD_token']
#         cur_dict['selected_czeng_cz_moses_tok'] = cur_dict['selected_czeng_cz_moses_tok'] + (max_len_cz - cur_dict['cz_leng']) * ['PAD_token']
#     
#         cur_dict['eng_idxes'] = cur_dict['eng_idxes'] + (max_len_eng - cur_dict['eng_leng']) * [selected_czeng_eng_w2i['PAD_token']]
#         cur_dict['cz_idxes'] = cur_dict['cz_idxes'] + (max_len_cz - cur_dict['cz_leng'])  * [selected_czeng_cz_w2i['PAD_token']]    
#         cur_dict['selected_czeng_list_para_tok'] = [tok_list +  (max_len_eng - cur_dict['eng_leng']) * ['PAD_token'] for tok_list in cur_dict['selected_czeng_list_para_tok']]
#         cur_dict['para_idxes_lists'] = [tok_list + (max_len_eng - cur_dict['eng_leng']) * [selected_czeng_eng_w2i['PAD_token']] for tok_list in  cur_dict['para_idxes_lists']]
#     
#         
#     pickle.dump(in_range_sampled_czeng_final, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/in_range_sampled_czeng_final_yes_para_input.p', 'wb'))      
#     
# =============================================================================

###############  
##############
######## Test data file 
##############
#CzEng Test data 
test_file_name = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/para-nmt-50m/data-plaintext-format.99etest'
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
new_test_line_list = []
for line in test_line_list:
    line = line.strip('\n')
    line = line.split('\t')
    new_test_line_list.append(line)
    
test_line_list = new_test_line_list


test_pair_id = {}
test_pair_score = {}
test_cz = {}
test_eng = {}
from mosestokenizer import *
tokenize_en = MosesTokenizer('en')

for i, pair in enumerate(test_line_list):    
    test_pair_id[i] = pair[0]; test_pair_score[i] = float(pair[1])
    test_cz[i] = tokenize_en(str.lower(pair[2])); test_eng[i] = tokenize_en(str.lower(pair[3]))

#call vocab and add vocab
cz_voc = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_final.p', 'rb'))
eng_voc = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/eng_voc_final.p', 'rb'))
cz_w2i = cz_voc['cz_w2i']; cz_i2w = cz_voc['cz_i2w']
eng_w2i = eng_voc['eng_w2i']; eng_i2w = eng_voc['eng_i2w']


for i, tok_list in test_cz.items():
    for cz_tok in tok_list:
        if cz_tok not in cz_w2i:
            cz_i2w[len(cz_w2i)] = str.lower(cz_tok)
            cz_w2i[str.lower(cz_tok)] = len(cz_w2i)
            

for i, tok_list in test_eng.items():
    for en_tok in tok_list:
        if en_tok not in eng_w2i:
            eng_i2w[len(eng_w2i)] = str.lower(en_tok)
            eng_w2i[str.lower(en_tok)] = len(eng_w2i)
            

#Save voc 
pickle.dump({'cz_w2i':cz_w2i , 'cz_i2w':cz_i2w } , open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_final.p', 'wb'))
pickle.dump({'eng_w2i':eng_w2i , 'eng_i2w':eng_i2w } , open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/eng_voc_final.p', 'wb'))


#Make list of dict
test_list_dict = []
for i in test_pair_id:
    cur_dict = {}
    glb_idx = i #+ 10430445
    cur_dict['glb_idx'] = glb_idx
    cur_dict['selected_czeng_eng_moses_tok'] = ['SOS_token'] + test_eng[i] + ['EOS_token']
    cur_dict['selected_czeng_cz_moses_tok'] = ['SOS_token'] + test_cz[i] + ['EOS_token']
    
    cur_dict['eng_leng'] = len(cur_dict['selected_czeng_eng_moses_tok']) 
    cur_dict['cz_leng'] = len(cur_dict['selected_czeng_cz_moses_tok']) 
    
    cur_dict['eng_idxes'] = [eng_w2i[tok] for tok in cur_dict['selected_czeng_eng_moses_tok'] ]
    cur_dict['cz_idxes'] = [cz_w2i[tok] for tok in cur_dict['selected_czeng_cz_moses_tok'] ]
    test_list_dict.append(cur_dict)
    


cPickle.dump(test_list_dict, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/test_czeng.p', "wb"))  

    
#News data 
news_test_en_file = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/WMTtest/newstest2016-encs-src.en.sgm'
news_test_cz_file ='/data/scratch-oc40/symin95/github_lf/logicalforms/data/WMTtest/newstest2016-encs-ref.cs.sgm'


news_test_en_line_list = []
with open(news_test_en_file) as open_f:
    for i, line in enumerate(open_f):
        if '<seg id=' in line[:10]:
            new_line = str.lower(line[1:-7])
            cut_index = new_line.index('>')
            news_test_en_line_list.append(new_line[cut_index+1:])
            
toknized_news_test_en_list = [tokenize_en(sent) for sent in news_test_en_line_list]

tokenize_cz = MosesTokenizer('cz')

news_test_cz_line_list = []
with open(news_test_cz_file) as open_f:
    for i, line in enumerate(open_f):
        if '<seg id=' in line[:10]:
            new_line = str.lower(line[1:-7])
            cut_index = new_line.index('>')
            news_test_cz_line_list.append(new_line[cut_index+1:])
            
toknized_news_test_cz_list = [tokenize_cz(sent) for sent in news_test_cz_line_list]

###Now make vocabulary and add 
#Vocab 복구
cz_i2w = {k:v for k, v in cz_i2w.items() if k<1035721}
cz_w2i = {v:k for k, v in cz_i2w.items()}

eng_i2w = {k:v for k, v in eng_i2w.items() if k<608974}
eng_w2i = {v:k for k, v in eng_i2w.items()}

news_test_cz_voc = copy.deepcopy({'cz_w2i':cz_w2i , 'cz_i2w':cz_i2w })
new_test_eng_voc =  copy.deepcopy({'eng_w2i':eng_w2i , 'eng_i2w':eng_i2w })
tokenize_en = MosesTokenizer('en')


for tok_list in toknized_news_test_en_list:
    for en_tok in tok_list:
        if en_tok not in eng_w2i:
            eng_i2w[len(eng_w2i)] = str.lower(en_tok)
            eng_w2i[str.lower(en_tok)] = len(eng_w2i)
            

for tok_list in toknized_news_test_cz_list:
    for cz_tok in tok_list:
        if cz_tok not in cz_w2i:
            cz_i2w[len(cz_w2i)] = str.lower(cz_tok)
            cz_w2i[str.lower(cz_tok)] = len(cz_w2i)
  
        
pickle.dump({'cz_w2i':cz_w2i , 'cz_i2w':cz_i2w } , open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_final.p', 'wb'))
pickle.dump({'eng_w2i':eng_w2i , 'eng_i2w':eng_i2w } , open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/eng_voc_final.p', 'wb'))

news_test_list_dict = []
for i, tok_list in enumerate(toknized_news_test_en_list):
    cur_dict = {}
    glb_idx = i 
    cur_dict['selected_czeng_eng_moses_tok'] = ['SOS_token'] + toknized_news_test_en_list[i] + ['EOS_token']
    cur_dict['selected_czeng_cz_moses_tok'] = ['SOS_token'] + toknized_news_test_cz_list[i] + ['EOS_token']
    cur_dict['eng_leng'] = len(cur_dict['selected_czeng_eng_moses_tok']) 
    cur_dict['cz_leng'] = len(cur_dict['selected_czeng_cz_moses_tok']) 
    cur_dict['eng_idxes'] = [eng_w2i[tok] for tok in cur_dict['selected_czeng_eng_moses_tok'] ]
    cur_dict['cz_idxes'] = [cz_w2i[tok] for tok in cur_dict['selected_czeng_cz_moses_tok'] ]
    news_test_list_dict.append(cur_dict)

cPickle.dump(news_test_list_dict, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/newstest_czeng.p', "wb"))  
   

#######
### Pad to 50
#######

#Pad from training 
#Add Padding 
def pad(seq, pad_tok, max_len):
    if len(seq) > max_len and seq[0] ==1:
        return seq[:max_len-1] + [2]
    elif len(seq) > max_len and seq[0] =='SOS_token':
        return seq[:max_len-1] + ['EOS_token']   
    else:
        return seq + (max_len - len(seq)) * [pad_tok]

max_len_eng = 50; max_len_cz = 50
for i, cur_dict in enumerate(in_range_sampled_czeng_final):
    if i % 10000==0:
        print(i)
    cur_dict['selected_czeng_eng_moses_tok'] = pad(cur_dict['selected_czeng_eng_moses_tok'],'PAD_token', 52)
    cur_dict['selected_czeng_cz_moses_tok'] = pad(cur_dict['selected_czeng_cz_moses_tok'],'PAD_token', 52)    
    cur_dict['eng_idxes'] = pad(cur_dict['eng_idxes'],eng_w2i['PAD_token'], 52)
    cur_dict['cz_idxes'] = pad(cur_dict['cz_idxes'],cz_w2i['PAD_token'], 52)  
    cur_dict['selected_czeng_list_para_tok'] = [pad(tok_list,'PAD_token', 52) for  tok_list in cur_dict['selected_czeng_list_para_tok']]
    cur_dict['para_idxes_lists'] = [pad(tok_list,eng_w2i['PAD_token'], 52) for tok_list in  cur_dict['para_idxes_lists']]

for i, cur_dict in enumerate(test_list_dict):
    if i % 10000==0:
        print(i)
    cur_dict['selected_czeng_eng_moses_tok'] = pad(cur_dict['selected_czeng_eng_moses_tok'],'PAD_token', 52)
    cur_dict['selected_czeng_cz_moses_tok'] = pad(cur_dict['selected_czeng_cz_moses_tok'],'PAD_token', 52)    
    cur_dict['eng_idxes'] = pad(cur_dict['eng_idxes'],eng_w2i['PAD_token'], 52)
    cur_dict['cz_idxes'] = pad(cur_dict['cz_idxes'],cz_w2i['PAD_token'], 52)  

for i, cur_dict in enumerate(news_test_list_dict):
    if i % 10000==0:
        print(i)
    cur_dict['selected_czeng_eng_moses_tok'] = pad(cur_dict['selected_czeng_eng_moses_tok'],'PAD_token', 52)
    cur_dict['selected_czeng_cz_moses_tok'] = pad(cur_dict['selected_czeng_cz_moses_tok'],'PAD_token', 52)    
    cur_dict['eng_idxes'] = pad(cur_dict['eng_idxes'],eng_w2i['PAD_token'], 52)
    cur_dict['cz_idxes'] = pad(cur_dict['cz_idxes'],cz_w2i['PAD_token'], 52)  


    
pickle.dump(in_range_sampled_czeng_final, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/in_range_sampled_czeng_final_yes_para_input.p', 'wb'))      
cPickle.dump(news_test_list_dict, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/newstest_czeng.p', "wb"))  
cPickle.dump(test_list_dict, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/test_czeng.p', "wb"))  


#########Adjust the lengths
#Load

in_range_sampled_czeng_final = pickle.load( open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/in_range_sampled_czeng_final_yes_para_input.p', 'rb'))      
news_test_list_dict = cPickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/newstest_czeng.p', "rb"))  
test_list_dict = cPickle.load( open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/test_czeng.p', "rb"))  


for i, cur_dict in enumerate(in_range_sampled_czeng_final):
# =============================================================================
#     assert len(cur_dict['selected_czeng_eng_moses_tok']) == 52
#     assert len(cur_dict['selected_czeng_eng_moses_tok']) == 52
#     assert len(cur_dict['eng_idxes']) == 52
#     assert len(cur_dict['cz_idxes']) == 52
#     assert [len(x) for x in cur_dict['selected_czeng_list_para_tok']] == [52] * len(cur_dict['selected_czeng_list_para_tok'])
#     
#     
# =============================================================================
    if cur_dict['eng_leng'] > 52:
        cur_dict['eng_leng'] = 52
    if cur_dict['cz_leng'] > 52:
        cur_dict['cz_leng'] = 52
    for i in range(len(cur_dict['para_leng_list'])):
        if cur_dict['para_leng_list'][i] > 52:
            cur_dict['para_leng_list'][i] = 52

for i, cur_dict in enumerate(test_list_dict) :
    assert len(cur_dict['selected_czeng_eng_moses_tok']) == 52
    assert len(cur_dict['selected_czeng_eng_moses_tok']) == 52
    assert len(cur_dict['eng_idxes']) == 52
    assert len(cur_dict['cz_idxes']) == 52
    if cur_dict['eng_leng'] > 52:
        cur_dict['eng_leng'] = 52
    if cur_dict['cz_leng'] > 52:
        cur_dict['cz_leng'] = 52
    
          
#Modify test_list_dict with getting rev_ from paranmt 
test_joined = {}
for i, cur_dict in enumerate(test_list_dict) :
    test_joined[i] = ''.join([tok for tok in cur_dict['selected_czeng_eng_moses_tok'] if not(tok in ['SOS_token', 'EOS_token', 'PAD_token'] ) ])

test_joined_rev = {}
for k, v in test_joined.items():
    if not(v in test_joined_rev):
        test_joined_rev[v] =[] 
    test_joined_rev[v].append(k)
    
#PARANMT idxes 
test_2_paranmt_50_idxes = {}
for joined, test_idx_list in test_joined_rev.items():
    try:
        para_nmt_idxes = para_nmt_reference_tokenized_and_joined_50m_rev[joined]
        for i in test_idx_list:
            test_2_paranmt_50_idxes[i] = para_nmt_idxes
    except:
        pass
    
    
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
    
para_nmt_line_list_50m =  new_para_nmt_line_list

test_idx2_para_tok = {}
for t_idx, para_idxes in test_2_paranmt_50_idxes.items():
    if not(t_idx in test_idx2_para_tok ):
        test_idx2_para_tok[t_idx] = []
    for para_idx in para_idxes:
        if float(para_nmt_line_list_50m[para_idx][2])>0.5 and float(para_nmt_line_list_50m[para_idx][2])<0.95:
            test_idx2_para_tok[t_idx].append(tokenize_en(para_nmt_line_list_50m[para_idx][1]))
 
# =============================================================================
# small_train =  [in_range_sampled_czeng_final[i] for i in range(300)]            
# pickle.dump(small_train, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/small_train.p', 'wb'))
# =============================================================================

           
#############################################
###Just do for the ones in paranmt 50 
#############################################
#First choose until range(3976194)
training_with_5m_final = [in_range_sampled_czeng_final[i] for i in range(3976194)]          
#number of all para:
#count = sum([len(c['para_idxes_lists']) for c in training_with_5m_final])
random.seed(0)
selected = random.sample(range(3976194), 2500000)
training_with_5m_final = [training_with_5m_final[i] for i in selected]

#
new_cz_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}; new_en_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}
new_cz_i2w = {v:k for k, v in new_cz_w2i.items()}; new_en_i2w = {v:k for k, v in new_en_w2i.items()}

for i, cur_dict in enumerate(training_with_5m_final):
    cz_tok_list = cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]
    for cz_tok in cz_tok_list:
        if cz_tok not in new_cz_w2i:
            new_cz_i2w[len(new_cz_w2i)] = cz_tok
            new_cz_w2i[cz_tok] = len(new_cz_w2i)


#Now repopulate 
populated_training_with_5m_final = []
for i, ex_dict in enumerate(training_with_5m_final):
    if i %100000==1:
        print(i)
    para_tok_list = ex_dict['selected_czeng_list_para_tok'] #list of list of toks
    ori_utt_tok = ex_dict['selected_czeng_eng_moses_tok'] #list of toks
    ori_leng = ex_dict['eng_leng']
    para_lengs = ex_dict['para_leng_list']
    ori_tok_idx = ex_dict['eng_idxes']
    para_tok_idxes = ex_dict['para_idxes_lists']
    
    for para_idx, para_tok in enumerate(para_tok_list) :
        new_cur_dict = copy.deepcopy(ex_dict)
        
        #Utterance 
        new_cur_dict['selected_czeng_eng_moses_tok'] = para_tok
        new_para_tok_list = copy.deepcopy(para_tok_list)
        b = new_para_tok_list.pop(para_idx)
        new_para_tok_list.append(ori_utt_tok)
        new_cur_dict['selected_czeng_list_para_tok'] = new_para_tok_list
        
        #Length
        para_leng = para_lengs[para_idx]
        new_cur_dict['eng_leng'] = para_leng
        new_para_lengs = copy.deepcopy(para_lengs)
        b = new_para_lengs.pop(para_idx)
        new_para_lengs.append(ori_leng)
        new_cur_dict['para_leng_list'] = new_para_lengs
        
        #Idxes 
        para_tok_idx = para_tok_idxes[para_idx]
        new_cur_dict['eng_idxes'] = para_tok_idx
        new_para_tok_idxes = copy.deepcopy(para_tok_idxes)
        b = new_para_tok_idxes.pop(para_idx)
        new_para_tok_idxes.append(ori_tok_idx)
        new_cur_dict['para_idxes_lists'] = new_para_tok_idxes               
        populated_training_with_5m_final.append(new_cur_dict)


#Check 
training_with_5m_final += populated_training_with_5m_final
for i, cur_dict in enumerate(training_with_5m_final):
    en_tok_list = cur_dict['selected_czeng_eng_moses_tok'][:cur_dict['eng_leng']]
    for en_tok in en_tok_list:
        if en_tok not in new_en_w2i:
            new_en_i2w[len(new_en_w2i)] = en_tok
            new_en_w2i[en_tok] = len(new_en_w2i)

##########
########Do same for test 
##########
para_nmt_5m_dict = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para-nmt-50m/para-nmt-5m-processed/paranmt_5m_dict.p', 'rb'))
para_nmt_reference_tokenized_and_joined_5m = para_nmt_5m_dict['para_nmt_reference_tokenized_and_joined']
para_nmt_reference_tokenized_and_joined_5m_rev = para_nmt_5m_dict['para_nmt_reference_tokenized_and_joined_rev']

#PARANMT idxes 
#Need to do test_joined again 
test_2_paranmt_5_idxes = {}
for joined, test_idx_list in test_joined_rev.items():
    try:
        para_nmt_idxes = para_nmt_reference_tokenized_and_joined_5m_rev[joined]
        for i in test_idx_list:
            test_2_paranmt_5_idxes[i] = para_nmt_idxes
    except:
        pass

#Load PARANMT 5M Utterances
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

#Now add para_nmt_line_list to utts and tokenize 
#First reduce test to one with paraphrases
new_test_list_dict = [test_dict for i, test_dict in enumerate(test_list_dict) if i in  test_2_paranmt_5_idxes]
counter = 0
for k, para_idx_list in test_2_paranmt_5_idxes.items():
    cur_dict = new_test_list_dict[counter]
    cur_dict['selected_czeng_list_para_tok'] = []
    #cur_dict['para_idxes_lists'] = []
    cur_dict['para_leng_list'] = []
    for para_idx in para_idx_list:
        tokenized = tokenize_en(para_nmt_line_list_5m[para_idx][1])
        cur_dict['selected_czeng_list_para_tok'].append(tokenized)
        #cur_dict['para_idxes_lists'].append([])   
        if len(tokenized) > 52:
            cur_dict['para_leng_list'].append(52)
        else:
            cur_dict['para_leng_list'].append(len(tokenized))
    counter +=1 

test_idx2_para_utt = {}
for t_idx, para_idxes in test_2_paranmt_50_idxes.items():
    if not(t_idx in test_idx2_para_utt ):
        test_idx2_para_utt[t_idx] = []
    for para_idx in para_idxes:
        if float(para_nmt_line_list_50m[para_idx][2])>0.5 and float(para_nmt_line_list_50m[para_idx][2])<0.95:
            test_idx2_para_utt[t_idx].append(para_nmt_line_list_50m[para_idx][1])
 
#selected test idxes 
random.seed(0)
selected_test = random.sample(test_2_paranmt_50_idxes.keys(), 250000)
selected_test_list_dict = [test_list_dict[i] for i in selected_test]

#Get rid of para keys 
for i, cur_dict in enumerate(selected_test_list_dict):
    for k in ['para_leng_list','selected_czeng_list_para_tok']:
        if k in cur_dict:
            b = cur_dict.pop(k)

#Do Eng and CZ vocab with test
for i, cur_dict in enumerate(selected_test_list_dict):
    cz_tok_list = cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]
    for cz_tok in cz_tok_list:
        if cz_tok not in new_cz_w2i:
            new_cz_i2w[len(new_cz_w2i)] = cz_tok
            new_cz_w2i[cz_tok] = len(new_cz_w2i)
    

for i, cur_dict in enumerate(selected_test_list_dict):
    en_tok_list = cur_dict['selected_czeng_eng_moses_tok'][:cur_dict['eng_leng']]
    for en_tok in en_tok_list:
        if en_tok not in new_en_w2i:
            new_en_i2w[len(new_en_w2i)] = en_tok
            new_en_w2i[en_tok] = len(new_en_w2i)

#Need to convert to w2i and pad again -> pad again은 안해도 됨
#Need to both for Train and Test 
#Test 
for i, cur_dict in enumerate(selected_test_list_dict):
    cur_dict['eng_idxes'] = [new_en_w2i[tok] for tok in cur_dict['selected_czeng_eng_moses_tok']]
    cur_dict['cz_idxes'] = [new_cz_w2i[tok] for tok in cur_dict['selected_czeng_cz_moses_tok']]
            
#Train 
for i, cur_dict in enumerate(training_with_5m_final):
    cur_dict['eng_idxes'] = [new_en_w2i[tok] for tok in cur_dict['selected_czeng_eng_moses_tok']]
    cur_dict['cz_idxes'] = [new_cz_w2i[tok] for tok in cur_dict['selected_czeng_cz_moses_tok']]
    for i, para_toks in enumerate(cur_dict['selected_czeng_list_para_tok']):
        cur_dict['para_idxes_lists'][i] = [new_en_w2i[para_tok] for para_tok in para_toks]
            
#save train, test, vocab
pickle.dump(training_with_5m_final , open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_training_with_5m_final.p', 'wb'))
pickle.dump(selected_test_list_dict , open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_sampled_test_list_dict.p', 'wb'))
pickle.dump({'cz_w2i':new_cz_w2i, 'cz_i2w': new_cz_i2w}, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_sampled_voc_final.p', 'wb'))
pickle.dump({'en_w2i':new_en_w2i, 'en_i2w': new_en_i2w}, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_sampled_voc_final.p', 'wb'))

small_train =  [in_range_sampled_czeng_final[i] for i in range(40)]            
pickle.dump(small_train, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/small_train.p', 'wb'))

small_test = [selected_test_list_dict[i] for i in range(40)]
pickle.dump(small_test, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/small_test.p', 'wb'))

#Word appearance count
#Go over para and add to voc 



#Go over training and test to count 
eng_count = {v: 0 for v in new_en_w2i} ; cz_count = {v: 0 for v in new_cz_w2i}
for i, cur_dict in enumerate(selected_test_list_dict):
    for tok in cur_dict['selected_czeng_eng_moses_tok'][:cur_dict['eng_leng']]:
        eng_count[tok] +=1
    for tok in cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]:
        cz_count[tok] +=1
    
for i, cur_dict in enumerate(training_with_5m_final):
    for tok in cur_dict['selected_czeng_eng_moses_tok'][:cur_dict['eng_leng']]:
        eng_count[tok] +=1
    for tok in cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]:
        cz_count[tok] +=1
    
 
#Now remove 한 번만 일어나는 애들 
#eng_words_remaining = {v:eng_count[v]for v in eng_count if eng_count[v]>2}
#eng_words_remaining['PAD_token'] = 3 
cz_words_remaining = {v:cz_count[v] for v in cz_count if cz_count[v]>2 and (v not in ['SOS_token', 'EOS_token', 'PAD_token'])}
#cz_words_remaining['PAD_token'] = 3 
cz_words_remaining['UNK'] = 3 
len(cz_words_remaining)
reduced_cz_w2i = {}
reduced_cz_w2i['PAD_token']= 0
reduced_cz_w2i['SOS_token']= 1
reduced_cz_w2i['EOS_token'] =  2


counter = 3
for k in cz_words_remaining:
    reduced_cz_w2i[k] = counter
    counter +=1
reduced_cz_i2w = {v:k for k, v in reduced_cz_w2i.items()}

for i, cur_dict in enumerate(selected_test_list_dict):
    cz_idxes = [0.0]*len(cur_dict['selected_czeng_cz_moses_tok'])
    for j, tok in enumerate(cur_dict['selected_czeng_cz_moses_tok']):
        if tok in reduced_cz_w2i:
            cz_idxes[j] = reduced_cz_w2i[tok]
        else:
            cz_idxes[j] = reduced_cz_w2i['UNK']
    cur_dict['cz_idxes'] =  cz_idxes

for i, cur_dict in enumerate(training_with_5m_final):
    cz_idxes = [0.0]*len(cur_dict['selected_czeng_cz_moses_tok'])
    for j, tok in enumerate(cur_dict['selected_czeng_cz_moses_tok']):
        if tok in reduced_cz_w2i:
            cz_idxes[j] = reduced_cz_w2i[tok]
        else:
            cz_idxes[j] = reduced_cz_w2i['UNK']
    cur_dict['cz_idxes'] =  cz_idxes
    
pickle.dump({'cz_w2i':reduced_cz_w2i, 'cz_i2w': reduced_cz_i2w}, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_reduced_voc_final.p', 'wb'))
pickle.dump(training_with_5m_final , open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_training_with_5m_final_with_reduced_target_voc.p', 'wb'))
pickle.dump(selected_test_list_dict , open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_sampled_test_list_dict_with_reduced_target_voc.p', 'wb'))

small_train =  [training_with_5m_final[i] for i in range(40)]            
pickle.dump(small_train, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/small_train.p', 'wb'))

small_test = [selected_test_list_dict[i] for i in range(2)]
pickle.dump(small_test, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/small_test.p', 'wb'))


#Just take the first 25 thousand 
training_with_50th = training_with_5m_final[:25000]

#Populate 
populated_training_with_50th = []
for i, ex_dict in enumerate(training_with_50th):
    if i %100000==1:
        print(i)
    para_tok_list = ex_dict['selected_czeng_list_para_tok'] #list of list of toks
    ori_utt_tok = ex_dict['selected_czeng_eng_moses_tok'] #list of toks
    ori_leng = ex_dict['eng_leng']
    para_lengs = ex_dict['para_leng_list']
    ori_tok_idx = ex_dict['eng_idxes']
    para_tok_idxes = ex_dict['para_idxes_lists']
    
    for para_idx, para_tok in enumerate(para_tok_list) :
        new_cur_dict = copy.deepcopy(ex_dict)
        
        #Utterance 
        new_cur_dict['selected_czeng_eng_moses_tok'] = para_tok
        new_para_tok_list = copy.deepcopy(para_tok_list)
        b = new_para_tok_list.pop(para_idx)
        new_para_tok_list.append(ori_utt_tok)
        new_cur_dict['selected_czeng_list_para_tok'] = new_para_tok_list
        
        #Length
        para_leng = para_lengs[para_idx]
        new_cur_dict['eng_leng'] = para_leng
        new_para_lengs = copy.deepcopy(para_lengs)
        b = new_para_lengs.pop(para_idx)
        new_para_lengs.append(ori_leng)
        new_cur_dict['para_leng_list'] = new_para_lengs
        
        #Idxes 
        para_tok_idx = para_tok_idxes[para_idx]
        new_cur_dict['eng_idxes'] = para_tok_idx
        new_para_tok_idxes = copy.deepcopy(para_tok_idxes)
        b = new_para_tok_idxes.pop(para_idx)
        new_para_tok_idxes.append(ori_tok_idx)
        new_cur_dict['para_idxes_lists'] = new_para_tok_idxes               
        populated_training_with_50th.append(new_cur_dict)

training_with_50th  += populated_training_with_50th

#Just take 5000 of the test set 
test_with_5th = selected_test_list_dict[:5000]

#Now make vocab for eng and cz 
cz_w2i_50th = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}; en_w2i_50th = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}
cz_i2w_50th = {v:k for k, v in cz_w2i_50th.items()}; en_i2w_50th = {v:k for k, v in en_w2i_50th.items()}

#First make cz vocab
for i, cur_dict in enumerate(test_with_5th):
    cz_tok_list = cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]
    for cz_tok in cz_tok_list:
        if cz_tok not in cz_w2i_50th:
            cz_i2w_50th[len(cz_w2i_50th)] = cz_tok
            cz_w2i_50th[cz_tok] = len(cz_w2i_50th)
    
for i, cur_dict in enumerate(training_with_50th):
    cz_tok_list = cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]
    for cz_tok in cz_tok_list:
        if cz_tok not in cz_w2i_50th:
            cz_i2w_50th[len(cz_w2i_50th)] = cz_tok
            cz_w2i_50th[cz_tok] = len(cz_w2i_50th)

#Now make en vocab 
for i, cur_dict in enumerate(test_with_5th):
    en_tok_list = cur_dict['selected_czeng_eng_moses_tok'][:cur_dict['eng_leng']]
    for en_tok in en_tok_list:
        if en_tok not in en_w2i_50th:
            en_i2w_50th[len(en_w2i_50th)] = en_tok
            en_w2i_50th[en_tok] = len(en_w2i_50th)


for i, cur_dict in enumerate(training_with_50th):
    en_tok_list = cur_dict['selected_czeng_eng_moses_tok'][:cur_dict['eng_leng']]
    for en_tok in en_tok_list:
        if en_tok not in en_w2i_50th:
            en_i2w_50th[len(en_w2i_50th)] = en_tok
            en_w2i_50th[en_tok] = len(en_w2i_50th)

#Now delete by count 

cz_count = {v: 0 for v in cz_w2i_50th}
for i, cur_dict in enumerate(test_with_5th):
    for tok in cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]:
        cz_count[tok] +=1
    
for i, cur_dict in enumerate(training_with_50th):
    for tok in cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]:
        cz_count[tok] +=1
              


cz_words_remaining = {v:cz_count[v] for v in cz_count if cz_count[v]>2 and (v not in ['SOS_token', 'EOS_token', 'PAD_token'])}
#cz_words_remaining['PAD_token'] = 3 
cz_words_remaining['UNK'] = 3 
len(cz_words_remaining)
reduced_cz_w2i = {}
reduced_cz_w2i['PAD_token']= 0
reduced_cz_w2i['SOS_token']= 1
reduced_cz_w2i['EOS_token'] =  2
counter = 3
for k in cz_words_remaining:
    reduced_cz_w2i[k] = counter
    counter +=1
reduced_cz_i2w = {v:k for k, v in reduced_cz_w2i.items()}
reduced_cz_w2i = defaultdict(lambda: reduced_cz_w2i['UNK'], reduced_cz_w2i)

#Now convert idxes
for i, cur_dict in enumerate(test_with_5th):
    cur_dict['eng_idxes'] = [en_w2i_50th[tok] for tok in cur_dict['selected_czeng_eng_moses_tok']]
    cur_dict['cz_idxes'] = [reduced_cz_w2i[tok] for tok in cur_dict['selected_czeng_cz_moses_tok']]
             
for i, cur_dict in enumerate(training_with_50th):
    cur_dict['eng_idxes'] = [en_w2i_50th[tok] for tok in cur_dict['selected_czeng_eng_moses_tok']]
    cur_dict['cz_idxes'] = [reduced_cz_w2i[tok] for tok in cur_dict['selected_czeng_cz_moses_tok']]
    for i, para_toks in enumerate(cur_dict['selected_czeng_list_para_tok']):
        cur_dict['para_idxes_lists'][i] = [en_w2i_50th[para_tok] for para_tok in para_toks]
            
np.mean([training_with_50th[i]['eng_leng'] for i in range(len(training_with_50th))])

#Get rid of pad_tok 
for i, cur_dict in enumerate(test_with_5th):
    eng_leng = cur_dict['eng_leng']; cz_leng = cur_dict['cz_leng']
    cur_dict['eng_idxes'] = cur_dict['eng_idxes'][:eng_leng]
    cur_dict['cz_idxes'] = cur_dict['cz_idxes'][:cz_leng]
    cur_dict['selected_czeng_eng_moses_tok'] =  cur_dict['selected_czeng_eng_moses_tok'][:eng_leng]
    cur_dict['selected_czeng_cz_moses_tok'] =  cur_dict['selected_czeng_cz_moses_tok'][:cz_leng]
    
for i, cur_dict in enumerate(training_with_50th):
    eng_leng = cur_dict['eng_leng']; cz_leng = cur_dict['cz_leng']
    cur_dict['eng_idxes'] = cur_dict['eng_idxes'][:eng_leng]
    cur_dict['cz_idxes'] = cur_dict['cz_idxes'][:cz_leng]
    cur_dict['selected_czeng_eng_moses_tok'] =  cur_dict['selected_czeng_eng_moses_tok'][:eng_leng]
    cur_dict['selected_czeng_cz_moses_tok'] =  cur_dict['selected_czeng_cz_moses_tok'][:cz_leng]
    for para_idx, para_leng in enumerate(cur_dict['para_leng_list']):
        cur_dict['para_idxes_lists'][para_idx] = cur_dict['para_idxes_lists'][para_idx][:para_leng]
        cur_dict['selected_czeng_list_para_tok'][para_idx] = cur_dict['selected_czeng_list_para_tok'][para_idx][:para_leng]

small_train =  [training_with_50th[i] for i in range(2)]            
pickle.dump(small_train, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/small_train.p', 'wb'))

small_test = [training_with_50th[i] for i in range(2)]
pickle.dump(small_test, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/small_test.p', 'wb'))

#Save the entire thing and vocab
pickle.dump({'cz_w2i':reduced_cz_w2i, 'cz_i2w': reduced_cz_i2w}, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_reduced_voc_5th.p', 'wb'))
pickle.dump({'en_w2i':en_w2i_50th, 'en_i2w': en_i2w_50th}, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_reduced_voc_50th.p', 'wb'))

pickle.dump(training_with_50th, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/training_with_50th.p', 'wb'))
pickle.dump(test_with_5th, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/test_with_5th.p', 'wb'))


#See how many tokens in test was unseen in training
cz_test_count = {}
for i, cur_dict in enumerate(test_with_5th):
    for tok in cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]:
        if tok not in cz_test_count:
            cz_test_count[tok] =0
        cz_test_count[tok] +=1

cz_train_count = {}
for i, cur_dict in enumerate(training_with_50th):
    for tok in cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]:
        if tok not in cz_train_count:
            cz_train_count[tok] =0
        cz_train_count[tok] +=1

cz_no_app = {}        
for tok in cz_test_count:
    if tok not in cz_train_count:
        if cz_test_count >2 : 
        
        elif 
        cz_no_app[tok] = cz_test_count[tok]
 
en_test_count = {}
for i, cur_dict in enumerate(test_with_5th):
    for tok in cur_dict['selected_czeng_eng_moses_tok'][:cur_dict['eng_leng']]:
        if tok not in en_test_count:
            en_test_count[tok] =0
        en_test_count[tok] +=1


#########
#### Choose Domain 
########
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

eu_eng_w2i = {}
eu_cz_w2i = {}
counter = 0 
for k in category['subtitlesM']:
    if counter < 100000:
        cur_dict = training_with_5m_final[k]
        cz_tok_list = cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]
        for cz_tok in cz_tok_list:
            if cz_tok not in eu_cz_w2i:
                eu_cz_w2i[cz_tok] = 0
            eu_cz_w2i[cz_tok] +=1 
        
        en_tok_list = cur_dict['selected_czeng_eng_moses_tok'][:cur_dict['eng_leng']]
        for en_tok in en_tok_list:
            if en_tok not in eu_eng_w2i:
                eu_eng_w2i[en_tok] = 0
            eu_eng_w2i[en_tok] +=1
        counter +=1

eu_eng_rm = {v:1 for v, k in eu_eng_w2i.items() if k>2}    
cz_eng_rm = {v:1 for v, k in eu_cz_w2i.items() if k>2}    


category = {}
for idx, string in czeng_master_dict['czeng_pair_id'].items():
    current_cat = string.split('-')[0]      
    if not(current_cat in category):
        category[current_cat] = []
    category[current_cat].append(idx)
czeng_cat_dist = {}; tot = sum([len(v) for k,v in category.items() ])
czeng_cat_dist = {k : len(v)/tot for k,v in category.items()}
czeng_cat_tot = {k : len(v) for k,v in category.items()}

eu_eng_w2i = {}
eu_cz_w2i = {}
counter = 0 
for k in category['news']:
    if counter < 500000:
        cz_tok_list = czeng_master_dict['czeng_cz'][k]
        for cz_tok in cz_tok_list:
            if cz_tok not in eu_cz_w2i:
                eu_cz_w2i[cz_tok] = 0
            eu_cz_w2i[cz_tok] +=1 
        
        en_tok_list = czeng_master_dict['czeng_eng'][k]
        for en_tok in en_tok_list:
            if en_tok not in eu_eng_w2i:
                eu_eng_w2i[en_tok] = 0
            eu_eng_w2i[en_tok] +=1
        counter +=1

eu_eng_rm = {v:1 for v, k in eu_eng_w2i.items() if k>2}    
cz_eng_rm = {v:1 for v, k in eu_cz_w2i.items() if k>2}    


##########
##### First 재정비 voc
#########

cz_w2i_50th = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}; en_w2i_50th = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2}
cz_i2w_50th = {v:k for k, v in cz_w2i_50th.items()}; en_i2w_50th = {v:k for k, v in en_w2i_50th.items()}

for i, cur_dict in enumerate(training_with_50th):
    cz_tok_list = cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]
    for cz_tok in cz_tok_list:
        if cz_tok not in cz_w2i_50th:
            cz_i2w_50th[len(cz_w2i_50th)] = cz_tok
            cz_w2i_50th[cz_tok] = len(cz_w2i_50th)
            
cz_count = {v: 0 for v in cz_w2i_50th}    
for i, cur_dict in enumerate(training_with_50th):
    for tok in cur_dict['selected_czeng_cz_moses_tok'][:cur_dict['cz_leng']]:
        cz_count[tok] +=1

cz_words_remaining = {v:cz_count[v] for v in cz_count if cz_count[v]>2 and (v not in ['SOS_token', 'EOS_token', 'PAD_token'])}

cz_words_remaining['UNK'] = 3 
len(cz_words_remaining)
reduced_cz_w2i = {}
reduced_cz_w2i['PAD_token']= 0
reduced_cz_w2i['SOS_token']= 1
reduced_cz_w2i['EOS_token'] =  2
counter = 3
for k in cz_words_remaining:
    reduced_cz_w2i[k] = counter
    counter +=1
reduced_cz_i2w = {v:k for k, v in reduced_cz_w2i.items()}
reduced_cz_w2i = defaultdict(lambda: reduced_cz_w2i['UNK'], reduced_cz_w2i)

#For english, make default dict with UNK 
for i, cur_dict in enumerate(training_with_50th):
    en_tok_list = cur_dict['selected_czeng_eng_moses_tok'][:cur_dict['eng_leng']]
    for en_tok in en_tok_list:
        if en_tok not in en_w2i_50th:
            en_i2w_50th[len(en_w2i_50th)] = en_tok
            en_w2i_50th[en_tok] = len(en_w2i_50th)
en_w2i_50th['UNK'] = len(en_i2w_50th)
en_i2w_50th[en_w2i_50th['UNK'] ] = 'UNK'

en_w2i_50th =  defaultdict(lambda: en_w2i_50th['UNK'], en_w2i_50th)

###
## Convert everything 
###
for i, cur_dict in enumerate(test_with_5th):
    cur_dict['eng_idxes'] = [en_w2i_50th[tok] for tok in cur_dict['selected_czeng_eng_moses_tok']]
    cur_dict['cz_idxes'] = [reduced_cz_w2i[tok] for tok in cur_dict['selected_czeng_cz_moses_tok']]
             
for i, cur_dict in enumerate(training_with_50th):
    cur_dict['eng_idxes'] = [en_w2i_50th[tok] for tok in cur_dict['selected_czeng_eng_moses_tok']]
    cur_dict['cz_idxes'] = [reduced_cz_w2i[tok] for tok in cur_dict['selected_czeng_cz_moses_tok']]
    for i, para_toks in enumerate(cur_dict['selected_czeng_list_para_tok']):
        cur_dict['para_idxes_lists'][i] = [en_w2i_50th[para_tok] for para_tok in para_toks]
        
###
## Save Voc
###
pickle.dump({'cz_w2i':reduced_cz_w2i, 'cz_i2w': reduced_cz_i2w}, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_reduced_voc_5th.p', 'wb'))
pickle.dump({'en_w2i':en_w2i_50th, 'en_i2w': en_i2w_50th}, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_reduced_voc_50th.p', 'wb'))

        
        

###
## Make Pretrained Word Embedding Vecs and Save 
### 