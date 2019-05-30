#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 07:29:50 2019

@author: TiffMin
"""

from scipy.stats import ks_2samp
exact_match0_sh0 =[0.6469,0.6474,0.6760, 0.7599,0.6553]  
exact_match1_sh0 =  [0.8668, 0.8630,0.8028, 0.8747, 0.8794] 

exact_match0_sh2 =[0.47,0.3851, 0.7223,0.3332,0.4102]  
exact_match1_sh2 =   [0.497,0.6566,0.7594, 0.2888,0.6030]  

syn_match0_sh0 =  [0.6647,0.6763,0.6902,0.7774,0.6836]
syn_match1_sh0 =   [0.8802, 0.8777, 0.8165, 0.8987, 0.8920]

syn_match0_sh2 =   [0.4772,0.4064,0.7299,0.3532,0.4934]   
syn_match1_sh2 =  [0.4994,0.6710, 0.8085,0.2975,0.6236]  

BLEU0_sh0 =  [0.7780,0.7851,0.7954,0.8974,0.7907]
BLEU1_sh0 =  [0.9437,0.9459,0.9028,0.9551,0.9512]

BLEU0_sh2 =    [0.6625,  0.5839,0.8138,0.5788,0.7107]
BLEU1_sh2 =   [0.6619,0.8554, 0.8977,0.5463,0.7903]







same_lf0_sh0 = [0.4850,0.4862,0.5435,0.4701,0.4832] 
same_lf1_sh0 = [0.4635,0.3559,0.4298,0.3922,0.4583]  
same_lf0_sh2 = [0.4143,0.5004,0.5099,0.4926] 
same_lf1_sh2 =  [0.4787,0.4857,0.4623,0.4656]

same_lf_diff_tq0_sh0 =  [0.3997,0.4547,0.4373,0.3973,0.4007] 
same_lf_diff_tq1_sh0 =   [0.3653,0.3082,0.3379,0.3299,0.3690]
same_lf_diff_tq0_sh2 = [0.2820,0.3916,0.3272,0.4458] 
same_lf_diff_tq1_sh2 =  [0.3648,0.3475,0.2908,0.4166] 

same_tq0_sh0 =[0.5294,0.6385,0.5705,0.5505,0.5139]    
same_tq1_sh0 = [0.5075,0.4543,0.4639,0.443,0.4946]         
same_tq0_sh2 = [0.3832,0.4620,0.5308, 0.5162] 
same_tq1_sh2 =[0.4788,0.4637,0.4991,0.4885]  

diff_lf0_sh0 =[0.3546,0.4234,0.3923,0.3323,0.3305]
diff_lf1_sh0 =[0.3415,0.2947,0.3093,0.2766, 0.3262]
diff_lf0_sh2 =[0.2951,0.3651,0.3364,0.3855]
diff_lf1_sh2 =[0.3557,0.3371,0.2701,0.2993]

index0 = []
index1 =[]
for i in range(len(diff_lf0_sh0)):
    index0.append((same_tq0_sh0[i] - same_lf_diff_tq0_sh0[i]) / (same_lf_diff_tq0_sh0[i]-diff_lf0_sh0[i]))
    index1.append((same_tq1_sh0[i] - same_lf_diff_tq1_sh0[i]) / (same_lf_diff_tq1_sh0[i]-diff_lf1_sh0[i]))

values1 = index0
values2 = index1

value, pvalue = ks_2samp(values1, values2)
print(value, pvalue)
if pvalue > 0.05:
	print('Samples are likely drawn from the same distributions (fail to reject H0)')
else:
	print('Samples are likely drawn from different distributions (reject H0)')