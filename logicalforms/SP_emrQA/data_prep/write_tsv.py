#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:29:24 2019

@author: TiffMin
"""

validation_sampled = validation[:1000]
v_sampled_templq = list(set([rev_unique_templ_q_dict[i] for i in validation_sampled]))
v_sampled_lf = list(set([rev_unique_lf_dict[i] for i in validation_sampled]))
    
training_sampled_qidx = []
split_num = 1
for epoch in range(2000):
    random.seed(epoch)
    training_sampled_qidx+= random.sample(training,32*5)
training_sampled_qidx = list(set(training_sampled_qidx))

training_sampled_templq = list(set([rev_unique_templ_q_dict[i] for i in training_sampled_qidx]))
    

file = open("unlucky_validation.txt","w") 
for templ_q in v_sampled_templq:
     file.write('================================================\n')
     file.write("Q Template #"+str( templ_q)+ ": "+ " ".join(templ_qlist[unique_templ_q_dict[templ_q][0]])+"\n")
     file.write("All validation examples that belong to this template: \n")
     for i in range(len(unique_templ_q_dict_seventy[templ_q])):
             qidx = unique_templ_q_dict_seventy[templ_q][i]
             file.write(' '.join(tokenized_eng_sentences[qidx])+"\n")


file = open("training_set.txt","w") 
for templ_q in training_sampled_templq:
     file.write('================================================\n')
     file.write("Q Template #"+str( templ_q)+ ": "+ " ".join(templ_qlist[unique_templ_q_dict[templ_q][0]])+"\n")
     file.write("All validation examples that belong to this template: \n")
     for i in range(len(unique_templ_q_dict_seventy[templ_q])):
             qidx = unique_templ_q_dict_seventy[templ_q][i]
             file.write(' '.join(tokenized_eng_sentences[qidx])+"\n")



import csv

def write_tsv(f_name = 'lucky_validation.tsv', val_list = validation_sampled):
    with open(f_name, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for qidx in val_list:
            question = ' '.join(tokenized_eng_sentences[qidx])
            lf = ' '.join(OutputMasterDictbyTypeRAW['lf'][qidx])
            tsv_writer.writerow([question, lf])
        