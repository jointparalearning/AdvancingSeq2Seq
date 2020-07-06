#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:55:53 2019

@author: TiffMin
"""

file = open("val_split_for_preethi.txt","w")     
for val_idx in validation_sampled:
     file.write("Input :")
     file.write(' '.join(tokenized_eng_sentences[val_idx]))
     file.write("\n")
     file.write("Gold LF :")
     file.write(' '.join(OutputMasterDictbyTypeRAW['lf'][val_idx]))
     file.write("\n")
     
file = open("training_split_for_preethi.txt","w")     
for val_idx in training_sampled:
     file.write("Input :")
     file.write(' '.join(tokenized_eng_sentences[val_idx]))
     file.write("\n")
     file.write("Gold LF :")
     file.write(' '.join(OutputMasterDictbyTypeRAW['lf'][val_idx]))
     file.write("\n")