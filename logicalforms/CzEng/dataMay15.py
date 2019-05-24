#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:28:29 2019

@author: TiffMin
"""

#CzENG new domain and vocab 재정비
######
#### Choose Domain 
#####
import pickle 

czeng_master_dict = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_master.p', 'rb'))

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

#1. Find Paraphrases and Match
para_nmt_50m_dict = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/temp_data/para_nmt_50m_dict.p', 'rb'))
para_nmt_reference_tokenized_and_joined_50m = para_nmt_50m_dict['para_nmt_reference_tokenized_and_joined']
para_nmt_reference_tokenized_and_joined_50m_rev = {}

czeng_eng_joined = czeng_master_dict['czeng_eng_joined']
czeng_eng_joined_rev = {v:k for k,v in czeng_eng_joined.items()}

 
for k,v in para_nmt_reference_tokenized_and_joined_50m.items():
    if not( v in para_nmt_reference_tokenized_and_joined_50m_rev):
        para_nmt_reference_tokenized_and_joined_50m_rev[v] = []
    para_nmt_reference_tokenized_and_joined_50m_rev[v].append(k)

in_range_05_and_95 = {} #These are temp_cz index 
weird_count = 0
for ref_string, v in para_nmt_reference_tokenized_and_joined_50m_rev.items():
    if ref_string in czeng_eng_joined_rev:
        glb_idx = czeng_eng_joined_rev[ref_string]
        in_range_05_and_95[glb_idx] = [nmt_idx for nmt_idx in v if para_nmt_50m_dict['para_nmt_para_score'][nmt_idx]>0.5 and para_nmt_50m_dict['para_nmt_para_score'][nmt_idx]<0.95]

    else:
        weird_count +=1
 
len(in_range_05_and_95)
in_range_05_and_95 = {glb_idx: v for glb_idx, v in in_range_05_and_95.items() if len(v)!=0}
len(in_range_05_and_95)


news_cat = {idx:1 for idx in category['news']}
news_selected = {glb_idx for glb_idx in in_range_05_and_95 if glb_idx in news_cat}
medical_cat = {idx:1 for idx in category['medical']}
medical_selected = {glb_idx for glb_idx in in_range_05_and_95 if glb_idx in medical_cat}


eu_eng_w2i = {}
eu_cz_w2i = {}
counter = 0 
for k in medical_selected:
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


###
#Really Match with Paraphrases 
file_name = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/para-nmt-50m/para-nmt-50m.txt'
if 1 ==1:
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
    in_range_05_and_95 = {} #These are temp_cz index 
    weird_count = 0
    for ref_string, v in para_nmt_reference_tokenized_and_joined_50m_rev.items():
        if ref_string in czeng_eng_joined_rev:
            glb_idx = czeng_eng_joined_rev[ref_string]
            in_range_05_and_95[glb_idx] = [nmt_idx for nmt_idx in v if para_nmt_50m_dict['para_nmt_para_score'][nmt_idx]>0.5 and para_nmt_50m_dict['para_nmt_para_score'][nmt_idx]<0.95]
    len(in_range_05_and_95)
    in_range_05_and_95 = {glb_idx: v for glb_idx, v in in_range_05_and_95.items() if len(v)!=0}
    


temp_cz= pickle.load(open("/data/scratch-oc40/symin95/github_lf/logicalforms/data/temp_data/temp_czeng/temp_cz", "rb"))

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

temp_cz_pair_id = temp_cz['czeng_pair_id']
temp_cz_pair_id_rev = {v:k for k,v in temp_cz_pair_id.items()}
news_pair_id = {glb_idx : czeng_master_dict['czeng_pair_id'][glb_idx] for glb_idx in news_selected}
news_pair_id_rev = {v:k for k,v in news_pair_id.items()}
temp_cz_idx2_glb_idx_news = {}
glb_idx2_temp_cz_idx_news = {}

for glb_cz_pair_id, glb_idx in news_pair_id_rev.items():
    temp_cz_idx = temp_cz_pair_id_rev[glb_cz_pair_id]
    temp_cz_idx2_glb_idx_news[temp_cz_idx] = glb_idx
    glb_idx2_temp_cz_idx_news[glb_idx] = temp_cz_idx

#First get Cz and ENG utterances 
news_selected_czeng_cz_utt = {}; news_selected_czeng_eng_utt = {}
for glb_idx in news_selected:
    temp_cz_idx = glb_idx2_temp_cz_idx_news[glb_idx]
    pair = train_line_list[temp_cz_idx]
    assert  pair[0] == temp_cz_pair_id[temp_cz_idx]
    news_selected_czeng_cz_utt[glb_idx] = str.lower(pair[2]); news_selected_czeng_eng_utt[glb_idx] = str.lower(pair[3])


#1.5 Just choose Subtitles 
#1) Make category
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
    if counter < 250000:
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

#First get rid of pad tokens 
for i, cur_dict in enumerate(training_with_5m_final):
    eng_leng = cur_dict['eng_leng']; cz_leng = cur_dict['cz_leng']
    cur_dict['eng_idxes'] = cur_dict['eng_idxes'][:eng_leng]
    cur_dict['cz_idxes'] = cur_dict['cz_idxes'][:cz_leng]
    cur_dict['selected_czeng_eng_moses_tok'] =  cur_dict['selected_czeng_eng_moses_tok'][:eng_leng]
    cur_dict['selected_czeng_cz_moses_tok'] =  cur_dict['selected_czeng_cz_moses_tok'][:cz_leng]
    for para_idx, para_leng in enumerate(cur_dict['para_leng_list']):
        cur_dict['para_idxes_lists'][para_idx] = cur_dict['para_idxes_lists'][para_idx][:para_leng]
        cur_dict['selected_czeng_list_para_tok'][para_idx] = cur_dict['selected_czeng_list_para_tok'][para_idx][:para_leng]


#2) Now Sample and Make Vocab
#2.0 Sample Test only for subtitles
selected_test_list_dict = pickle.load( open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_sampled_test_list_dict_with_reduced_target_voc.p', 'rb'))
test_file_name = 'para-nmt-50m/data-plaintext-format.99etest'
test_line_list = []
with open(test_file_name) as open_f:
    for i, line in enumerate(open_f):
        test_line_list.append(line)

delimeter = '\t'
new_test_line_list = []
for line in test_line_list:
    line = line.strip('\n')
    line = line.split('\t')
    new_test_line_list.append(line)

test_line_list = new_test_line_list
test_pair_id = {glb_idx: test_line_list[glb_idx][0] for glb_idx in range(len(test_line_list))}


category = {}
for idx, cur_dict in enumerate(selected_test_list_dict):
    glb_idx = cur_dict['glb_idx']
    string = test_pair_id[glb_idx]
    current_cat = string.split('-')[0]      
    if not(current_cat in category):
        category[current_cat] = []
    category[current_cat].append(idx)
    
czeng_cat_dist = {}; tot = sum([len(v) for k,v in category.items() ])
czeng_cat_dist = {k : len(v)/tot for k,v in category.items()}
czeng_cat_tot = {k : len(v) for k,v in category.items()}
       
subtitles_test_idxes = [category['subtitlesM'][i] for i in range(5000)]  
#glb_idx_rev_test = {cur_dict['glb_idx']:i for i, cur_dict in enumerate(selected_test_list_dict)} 
subtitles_test_half_M = [selected_test_list_dict[i] for i in subtitles_test_idxes]
for cur_dict in subtitles_test_half_M:
    assert 'subtitlesM' in test_pair_id[cur_dict['glb_idx']] 


  
#2.1) Total 0.5M
selected = [category['subtitlesM'][i] for i in range(250000)]
subtitles_test_idxes = [category['subtitlesM'][i] for i in range(5000)]  
subtitles_test_half_M = [selected_test_list_dict[i] for i in subtitles_test_idxes]

#2.1.1) Propagate with Paraphrases
populated_selected_training = []
for i in selected:
    ex_dict = training_with_5m_final[i]
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
        populated_selected_training.append(new_cur_dict)

populated_selected_training = [training_with_5m_final[i] for i in selected] +populated_selected_training
#2.1.2) Make Vocab only from Tokens in Training 
eng_w2i = {}
cz_w2i = {}
for k, cur_dict in enumerate(populated_selected_training):
    cz_tok_list = cur_dict['selected_czeng_cz_moses_tok']
    for cz_tok in cz_tok_list:
        if cz_tok not in cz_w2i:
            cz_w2i[cz_tok] = 0
        cz_w2i[cz_tok] +=1 
    
    en_tok_list = cur_dict['selected_czeng_eng_moses_tok']
    for en_tok in en_tok_list:
        if en_tok not in eng_w2i:
            eng_w2i[en_tok] = 0
        eng_w2i[en_tok] +=1

#2.1.3) Reduce to >2 and Save 

eng_rm = {v:1 for v, k in eng_w2i.items() if k>2 and not(v in ['SOS_token', 'EOS_token', 'PAD_token'])}    
cz_rm = {v:1 for v, k in cz_w2i.items() if k>2 and not(v in ['SOS_token', 'EOS_token', 'PAD_token'])}    
eng_rm_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2, 'UNK': 3}
counter = 4
for v in  eng_rm:
    eng_rm_w2i[v] = counter
    counter +=1

eng_rm_i2w = {v: k for k, v in eng_rm_w2i.items()}

cz_rm_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2, 'UNK':3}
counter = 4
for v in  cz_rm:
    cz_rm_w2i[v] = counter
    counter +=1

cz_rm_i2w = {v: k for k, v in cz_rm_w2i.items()}

pickle.dump({'cz_w2i': cz_rm_w2i, 'cz_rm_i2w':cz_rm_i2w }, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_subtitles_5m.p','wb'))
pickle.dump({'en_w2i': eng_rm_w2i, 'en_rm_i2w':eng_rm_i2w }, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_voc_subtitles_5m.p','wb'))

#2.1.4) Now convert everything to toks
reduced_cz_w2i = defaultdict(lambda: cz_rm_w2i['UNK'], cz_rm_w2i)
reduced_en_w2i = defaultdict(lambda: eng_rm_w2i['UNK'], eng_rm_w2i)
             
for i, cur_dict in enumerate(populated_selected_training):
    cur_dict['eng_idxes'] = [reduced_en_w2i[tok] for tok in cur_dict['selected_czeng_eng_moses_tok']]
    cur_dict['cz_idxes'] = [reduced_cz_w2i[tok] for tok in cur_dict['selected_czeng_cz_moses_tok']]
    for i, para_toks in enumerate(cur_dict['selected_czeng_list_para_tok']):
        cur_dict['para_idxes_lists'][i] = [reduced_en_w2i[para_tok] for para_tok in para_toks]

pickle.dump(populated_selected_training, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/populated_selected_training_halfM.p', 'wb')) 

#2.1.5) Make test set of paraphrases
#in_range_05_and_95 # glb idxes that has paraphrases 
selected_glb_idxes_half_M = {training_with_5m_final[i]['glb_idx'] for i in selected}

selected_i_test_half_M = {}
for i in selected:
    glb_idx = training_with_5m_final[i]['glb_idx']
    para_utts = [''.join(tok_list[1:-1]) for tok_list in training_with_5m_final[i]['selected_czeng_list_para_tok']]
    if glb_idx in in_range_05_and_95:
        paranmt_50m_idxes = in_range_05_and_95[glb_idx]
        paranmt_utts = [para_nmt_line_list_50m[nmt_idx][1] for nmt_idx in paranmt_50m_idxes]
        paranmt_remain_utts = [utt for utt in paranmt_utts if not(''.join(word_tokenize(utt)) in para_utts)]
        if len(paranmt_remain_utts)>1:
            selected_i_test_half_M[i] = paranmt_remain_utts
   
para_test_utts = selected_i_test_half_M
pickle.dump(para_test_utts, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para_test_utts_halfM.p', 'wb')) 

#2.2) Total 0.2M
selected = [category['subtitlesM'][i] for i in range(100000)]
#subtitles_test_idxes = [category['subtitlesM'][i] for i in range(5000)]  
#subtitles_test_half_M = [selected_test_list_dict[i] for i in subtitles_test_idxes]
#2.2.1) Populate with paraphrases
populated_selected_training = []
for i in selected:
    ex_dict = training_with_5m_final[i]
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
        populated_selected_training.append(new_cur_dict)

populated_selected_training = [training_with_5m_final[i] for i in selected] +populated_selected_training

#2.2.2) Make Vocab only from Tokens in Training 
eng_w2i = {}
cz_w2i = {}
for k, cur_dict in enumerate(populated_selected_training):
    cz_tok_list = cur_dict['selected_czeng_cz_moses_tok']
    for cz_tok in cz_tok_list:
        if cz_tok not in cz_w2i:
            cz_w2i[cz_tok] = 0
        cz_w2i[cz_tok] +=1 
    
    en_tok_list = cur_dict['selected_czeng_eng_moses_tok']
    for en_tok in en_tok_list:
        if en_tok not in eng_w2i:
            eng_w2i[en_tok] = 0
        eng_w2i[en_tok] +=1


#2.2.3) Reduce vocab
eng_rm = {v:1 for v, k in eng_w2i.items() if k>2 and not(v in ['SOS_token', 'EOS_token', 'PAD_token'])}    
cz_rm = {v:1 for v, k in cz_w2i.items() if k>2 and not(v in ['SOS_token', 'EOS_token', 'PAD_token'])}    
eng_rm_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2, 'UNK': 3}
counter = 4
for v in  eng_rm:
    eng_rm_w2i[v] = counter
    counter +=1

eng_rm_i2w = {v: k for k, v in eng_rm_w2i.items()}

cz_rm_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2, 'UNK':3}
counter = 4
for v in  cz_rm:
    cz_rm_w2i[v] = counter
    counter +=1

cz_rm_i2w = {v: k for k, v in cz_rm_w2i.items()}

pickle.dump({'cz_w2i': cz_rm_w2i, 'cz_rm_i2w':cz_rm_i2w }, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_subtitles_0.25m.p','wb'))
pickle.dump({'en_w2i': eng_rm_w2i, 'en_rm_i2w':eng_rm_i2w }, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_voc_subtitles_0.25m.p','wb'))
        
#2.2.4) Now convert everything to toks
reduced_cz_w2i = defaultdict(lambda: cz_rm_w2i['UNK'], cz_rm_w2i)
reduced_en_w2i = defaultdict(lambda: eng_rm_w2i['UNK'], eng_rm_w2i)
             
for i, cur_dict in enumerate(populated_selected_training):
    cur_dict['eng_idxes'] = [reduced_en_w2i[tok] for tok in cur_dict['selected_czeng_eng_moses_tok']]
    cur_dict['cz_idxes'] = [reduced_cz_w2i[tok] for tok in cur_dict['selected_czeng_cz_moses_tok']]
    for i, para_toks in enumerate(cur_dict['selected_czeng_list_para_tok']):
        cur_dict['para_idxes_lists'][i] = [reduced_en_w2i[para_tok] for para_tok in para_toks]

pickle.dump(populated_selected_training, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/populated_selected_training_0.25M.p', 'wb')) 

#2.2.5) Make test paraphrase utterances 

selected_i_test_half_M = {}
for i in selected:
    glb_idx = training_with_5m_final[i]['glb_idx']
    para_utts = [tok_list for tok_list in training_with_5m_final[i]['selected_czeng_list_para_tok']]
    if glb_idx in in_range_05_and_95:
        paranmt_50m_idxes = in_range_05_and_95[glb_idx]
        paranmt_utts = [para_nmt_line_list_50m[nmt_idx][1] for nmt_idx in paranmt_50m_idxes]
        paranmt_remain_utts = [utt for utt in paranmt_utts if not(utt in para_utts)]
        if len(paranmt_remain_utts)>1:
            selected_i_test_half_M[i] = paranmt_remain_utts
   
para_test_utts = selected_i_test_half_M
pickle.dump(para_test_utts, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para_test_utts_0.25M.p', 'wb')) 


#2.3) Total 0.1 M
selected = [category['subtitlesM'][i] for i in range(50000)]
#2.3.1 Populate
populated_selected_training = []
for i in selected:
    ex_dict = training_with_5m_final[i]
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
        populated_selected_training.append(new_cur_dict)

#Didn't do this before!
populated_selected_training = [training_with_5m_final[i] for i in selected] +populated_selected_training


#2.3.2) Make Vocab only from Tokens in Training 
eng_w2i = {}
cz_w2i = {}
for k, cur_dict in enumerate(populated_selected_training):
    cz_tok_list = cur_dict['selected_czeng_cz_moses_tok']
    for cz_tok in cz_tok_list:
        if cz_tok not in cz_w2i:
            cz_w2i[cz_tok] = 0
        cz_w2i[cz_tok] +=1 
    
    en_tok_list = cur_dict['selected_czeng_eng_moses_tok']
    for en_tok in en_tok_list:
        if en_tok not in eng_w2i:
            eng_w2i[en_tok] = 0
        eng_w2i[en_tok] +=1


#2.3.3) Reduce vocab
eng_rm = {v:1 for v, k in eng_w2i.items() if k>2 and not(v in ['SOS_token', 'EOS_token', 'PAD_token'])}    
cz_rm = {v:1 for v, k in cz_w2i.items() if k>2 and not(v in ['SOS_token', 'EOS_token', 'PAD_token'])}    
eng_rm_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2, 'UNK': 3}
counter = 4
for v in  eng_rm:
    eng_rm_w2i[v] = counter
    counter +=1

eng_rm_i2w = {v: k for k, v in eng_rm_w2i.items()}

cz_rm_w2i = {'PAD_token': 0, 'SOS_token':1, 'EOS_token':2, 'UNK':3}
counter = 4
for v in  cz_rm:
    cz_rm_w2i[v] = counter
    counter +=1

cz_rm_i2w = {v: k for k, v in cz_rm_w2i.items()}

pickle.dump({'cz_w2i': cz_rm_w2i, 'cz_rm_i2w':cz_rm_i2w }, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_subtitles_0.1m.p','wb'))
pickle.dump({'en_w2i': eng_rm_w2i, 'en_rm_i2w':eng_rm_i2w }, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_voc_subtitles_0.1m.p','wb'))


#2.3.4) Now convert everything to toks
reduced_cz_w2i = defaultdict(lambda: cz_rm_w2i['UNK'], cz_rm_w2i)
reduced_en_w2i = defaultdict(lambda: eng_rm_w2i['UNK'], eng_rm_w2i)
             
for i, cur_dict in enumerate(populated_selected_training):
    cur_dict['eng_idxes'] = [reduced_en_w2i[tok] for tok in cur_dict['selected_czeng_eng_moses_tok']]
    cur_dict['cz_idxes'] = [reduced_cz_w2i[tok] for tok in cur_dict['selected_czeng_cz_moses_tok']]
    for i, para_toks in enumerate(cur_dict['selected_czeng_list_para_tok']):
        cur_dict['para_idxes_lists'][i] = [reduced_en_w2i[para_tok] for para_tok in para_toks]

pickle.dump(populated_selected_training, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/populated_selected_training_0.1M.p', 'wb')) 

#2.3.5) Make test paraphrase utterances 

selected_i_test_half_M = {}
for i in selected:
    glb_idx = training_with_5m_final[i]['glb_idx']
    para_utts = [tok_list for tok_list in training_with_5m_final[i]['selected_czeng_list_para_tok']]
    if glb_idx in in_range_05_and_95:
        paranmt_50m_idxes = in_range_05_and_95[glb_idx]
        paranmt_utts = [para_nmt_line_list_50m[nmt_idx][1] for nmt_idx in paranmt_50m_idxes]
        paranmt_remain_utts = [utt for utt in paranmt_utts if not(utt in para_utts)]
        if len(paranmt_remain_utts)>1:
            selected_i_test_half_M[i] = paranmt_remain_utts
   
para_test_utts = selected_i_test_half_M
pickle.dump(para_test_utts, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para_test_utts_0.1M.p', 'wb')) 

small_train =  [populated_selected_training[i] for i in range(40)]            
pickle.dump(small_train, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/small_train.p', 'wb'))

small_train =  [populated_selected_training[i] for i in range(1024)]            
pickle.dump(small_train, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/small_train.p', 'wb'))



#2. Tokenize with Moses 


#3. 


















