#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:59:20 2019

@author: TiffMin
"""

file = open("scheme2split2_split.txt", "w")
for lf in lf2temp_q_dict:
    file.write('=================================================\n')
    file.write('LF: ' + str(lf) + ' '.join(templ_lflist[unique_lf_dict[lf][0]]) + '\n')
    file.write('\n')
    file.write('In Training: \n')
    tempqs = lf2temp_q_dict[lf]
    for tq in tempqs:
        if tq in training_templq:
            file.write(' '.join(templ_qlist[unique_templ_q_dict[tq][0]]) + '\n') 
    file.write('\n')
    file.write('In Test: \n')
    for tq in tempqs:
        if tq in validation_templq:
            file.write(' '.join(templ_qlist[unique_templ_q_dict[tq][0]]) + '\n')
            
file.close()