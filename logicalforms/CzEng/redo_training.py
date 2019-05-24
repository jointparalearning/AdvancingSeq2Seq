#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 09:08:14 2019

@author: TiffMin
"""

#Make only 0.1m 

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


#Do this just for convenience and check that it's the same as before 
# =============================================================================
# for i, cur_dict in enumerate(training_with_5m_final):
#     eng_leng = cur_dict['eng_leng']; cz_leng = cur_dict['cz_leng']
#     cur_dict['eng_idxes'] = cur_dict['eng_idxes'][:eng_leng]
#     cur_dict['cz_idxes'] = cur_dict['cz_idxes'][:cz_leng]
#     cur_dict['selected_czeng_eng_moses_tok'] =  cur_dict['selected_czeng_eng_moses_tok'][:eng_leng]
#     cur_dict['selected_czeng_cz_moses_tok'] =  cur_dict['selected_czeng_cz_moses_tok'][:cz_leng]
#     for para_idx, para_leng in enumerate(cur_dict['para_leng_list']):
#         cur_dict['para_idxes_lists'][para_idx] = cur_dict['para_idxes_lists'][para_idx][:para_leng]
#         cur_dict['selected_czeng_list_para_tok'][para_idx] = cur_dict['selected_czeng_list_para_tok'][para_idx][:para_leng]
# 
# =============================================================================
#training_with_5m_final = pickle.dump(training_with_5m_final, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_training_with_5m_final_with_reduced_target_voc.p', 'wb'))
#training_with_5m_final가지고 위 line에서 잘못해서 다시하기
#czeng_training_with_5m_final_with_reduced_target_voc.p랑 아무 차이도 없음
training_with_5m_final = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_training_with_5m_final.p', 'rb'))

#Do this just for convenience and check that it's the same as before 
for i, cur_dict in enumerate(training_with_5m_final):
    eng_leng = cur_dict['eng_leng']; cz_leng = cur_dict['cz_leng']
    cur_dict['eng_idxes'] = cur_dict['eng_idxes'][:eng_leng]
    cur_dict['cz_idxes'] = cur_dict['cz_idxes'][:cz_leng]
    cur_dict['selected_czeng_eng_moses_tok'] =  cur_dict['selected_czeng_eng_moses_tok'][:eng_leng]
    cur_dict['selected_czeng_cz_moses_tok'] =  cur_dict['selected_czeng_cz_moses_tok'][:cz_leng]
    for para_idx, para_leng in enumerate(cur_dict['para_leng_list']):
        cur_dict['para_idxes_lists'][para_idx] = cur_dict['para_idxes_lists'][para_idx][:para_leng]
        cur_dict['selected_czeng_list_para_tok'][para_idx] = cur_dict['selected_czeng_list_para_tok'][para_idx][:para_leng]

#training_with_5m_final_incase = copy.deepcopy(training_with_5m_final)
pickle.dump(training_with_5m_final, open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/czeng_training_with_5m_final.p', 'wb'))

# 본격적 0.1M
selected = [category['subtitlesM'][i] for i in range(50000)]
#compare selected and selected_prev
selected[50000-1] == selected_prev[50000-1]
#2.3.1 Populate


populated_selected_training = [training_with_5m_final[i] for i in selected] +populated_selected_training
