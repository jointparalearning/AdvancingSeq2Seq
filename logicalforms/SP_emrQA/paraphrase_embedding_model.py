#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:23:10 2019

@author: TiffMin
"""

#Paraphrase Embedding Model
#Only need to have encoder 
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time
from models.seq2seq_Luong import *

class ParaEncoderAvg(nn.Module):

    def __init__(self, hidden_dim, vocab_size):
        super(ParaEncoderAvg, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding =  nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
            bidirectional=True)
                             
       #sentence is list of index of words
    def forward(self, sentence2idxvec,X_lengths):
        embeds = self.embedding(sentence2idxvec)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, X_lengths, batch_first=True)
        lstm_out, hidden = self.lstm(embeds)
        #print("passed gru!")
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        #lstm_out= lstm_out[:, :, :self.hidden_dim] + lstm_out[:, :, self.hidden_dim:]  # Sum bidirectional outputs

        #Put in pad_packed 반대로!
        #out = b(1) x seq x hid*2, hidden = b(1) x 1 x hid*2
        hidden = torch.mean(lstm_out, dim =1).shape
        return lstm_out, hidden
    
class ParaEncoderGRAN(nn.Module):
    
    def __init__(self, hidden_dim, vocab_size):
        super(ParaEncoderGRAN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding =  nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
            bidirectional=True)
        
        self.b = nn.Parameter(torch.randn(hidden_dim))
        self.Wx = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
                             
       #sentence is list of index of words
    def forward(self, sentence2idxvec,X_lengths):
        embeds = self.embedding(sentence2idxvec)
        b = embeds.shape[0] ; seq = embeds.shape[1]
        embeds_pad = torch.nn.utils.rnn.pack_padded_sequence(embeds, X_lengths, batch_first=True)
        lstm_out, hidden = self.lstm(embeds_pad)
        #print("passed gru!")
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        #lstm_out= lstm_out[:, :, :self.hidden_dim] + lstm_out[:, :, self.hidden_dim:]  # Sum bidirectional outputs

        #Put in pad_packed 반대로!
        #out = b(1) x seq x hid*2, hidden = b(1) x 1 x hid*2
        #conver everything to (b*seq, whatever)
        
        repeated_Wx = self.Wx.repeat(b*seq, 2, 2); repeated_Wh = self.Wh.repeat(b*seq, 2, 2)
        repeated_b = self.b.repeat(b*seq, 2).unsqueeze(2)
        repeated_embeds = embeds.repeat(1,1,2).view(b*seq, -1).unsqueeze(2)
        #print("lstm out shape:", lstm_out.shape) #the shape is [bxseqxhid*2]
        repeated_lstm_out = lstm_out.reshape(b*seq, -1).unsqueeze(2)
        
        #print("repeated_Wx.shape ", repeated_Wx.shape)
        #print("repeated_b.shape ", repeated_b.shape)
        #print("repeated_embeds.shape ", repeated_embeds.shape)
        #print("repeated_lstm_out.shape ", repeated_lstm_out.shape)
        
        
        hidden_final = repeated_embeds * F.sigmoid(torch.bmm(repeated_Wx, repeated_embeds) + torch.bmm(repeated_Wh, repeated_lstm_out) +  repeated_b )
        hidden_final = hidden_final.squeeze(2).view(b, seq, -1)
        hidden_final = torch.mean(hidden_final, dim = 1) #dim is [bx 128]
        #hidden_final = None
        return lstm_out, hidden_final
    