#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 11:41:05 2019

@author: TiffMin
"""
import pickle 
import _pickle as cPickle
import argparse
import torch
from CzENG_dataset import CzENG, MyCollator
from multiprocessing import cpu_count
from models.seq2seq_Luong import *
from collections import OrderedDict, defaultdict
from torch.utils.data import DataLoader
import nltk
import copy
import math
import numpy as np
import sys
from util_functions import numpy_to_var, toData, to_np, to_var, decoder_initial, update_logger




parser = argparse.ArgumentParser(description='Define trainig arguments')
parser.add_argument('-epoch', '--epoch', type=int, metavar='', required=True, help="load model and start with it if there is a directory")
parser.add_argument('-load_dir', '--load_dir', type=str, metavar='', required=True, help="load model and start with it if there is a directory")
parser.add_argument('-c', '--cuda', type=int, metavar='', required=False, help="split among 1,2,3,4,5")
parser.add_argument('-a', '--alpha', type=float, metavar='', default = 0, help="load model and start with it if there is a directory")
parser.add_argument('-cos_only', '--cos_only', type = int, metavar='', default=0, help = "alpha")
parser.add_argument('-cos_alph', '--cos_alph', type = int, metavar='', default=0, help = "alpha")
parser.add_argument('-which_test', '--which_test', type = str, metavar='', default='p', help = "alpha")

args = parser.parse_args()

def to_cuda(tensor):
    # turns to cuda
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def remove_pad(pair):
    pad_count = sum([1 for i in range(len(pair[1])) if pair[1][i] == 0])
    if pad_count ==0:
        return pair
    else:
        return (pair[0][:-pad_count], pair[1][:-pad_count])

def strip_SOS_EOS_if_twice(tokenized):
    new_tokenized = copy.deepcopy(tokenized) 
    if tokenized[0] == 'SOS_token' and tokenized[1] == 'SOS_token':
        new_tokenized.pop(0)
    if tokenized[-1] == 'EOS_token' and tokenized[-2] == 'EOS_token':
        new_tokenized.pop(-1)
    return new_tokenized
    

def clean_pairs(pairs_dict):
    new_dict = {}
    for k, batch_pairs_list in pairs_dict.items():
        pairs_list_list = []
        for batch_pairs in batch_pairs_list:
            batch_list0 = batch_pairs[0]
            batch_list1 = batch_pairs[1]
            for i, pairs_list in enumerate(batch_list0):
                tup = (batch_list0[i], batch_list1[i])
                #print("before", tup)
                tup = remove_pad(tup)
                #print("removed", tup)
                pairs_list_list.append((list(strip_SOS_EOS_if_twice(tup[0])),list(strip_SOS_EOS_if_twice(tup[1]))))
        new_dict[k] = pairs_list_list
    return new_dict


#entire_test_halfm = pickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/data/para_entire_test_dict_0.1M.p','rb'))
def load_model(args, en_vocab_size, cz_vocab_size):
    embed_size = 300
    AUTOhidden_dim = 256
    hidden_size = 256
    
    encoder = EncoderRNN(en_vocab_size, hidden_size, embed_size)
    decoder = LuongAttnDecoderRNN('general',hidden_size, cz_vocab_size, cz_vocab_size, embed_size,  bi=1)
    AUTOdecoder =LuongAttnDecoderRNN('general', hidden_size, en_vocab_size, en_vocab_size,embed_size, bi =1)
    
    encoder_dir = 'outputs/' + args.load_dir + '/encoder_ckpt_epoch' + str(args.epoch-1) + '.pytorch'  
    decoder_dir = 'outputs/' + args.load_dir + '/decoder_ckpt_epoch' + str(args.epoch-1) + '.pytorch'
    auto_dir = 'outputs/' + args.load_dir + '/auto_decoder_ckpt_epoch' + str(args.epoch-1) + '.pytorch'
    
    encoder.load_state_dict(torch.load(encoder_dir)) 
    decoder.load_state_dict(torch.load(decoder_dir)) 
    AUTOdecoder.load_state_dict(torch.load(auto_dir)) 
    
    return encoder, decoder, AUTOdecoder

def test(args):
    datasets = OrderedDict()
    test_file = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/para_entire_test_dict_0.1M.p'
    if args.which_test == 'N':
        test_file = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/newstest_czeng.p'

    en_vocab_file = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/en_voc_subtitles_0.1m.p'
    cz_vocab_file = '/data/scratch-oc40/symin95/github_lf/logicalforms/data/cz_voc_subtitles_0.1m.p'
    
    datasets['test'] = CzENG(
                data_file=test_file,
                split='test')

    
    batch_size = 32
    hidden_size = 256
    
    
    en_vocab = cPickle.load(open(en_vocab_file,'rb')); cz_vocab = cPickle.load(open(cz_vocab_file,'rb'))
    en_w2i, en_i2w = en_vocab["en_w2i"], en_vocab["en_rm_i2w"]
    cz_w2i, cz_i2w = cz_vocab["cz_w2i"], cz_vocab["cz_rm_i2w"]
    
    pad_token = en_w2i['PAD_token']; SOS_token = en_w2i['SOS_token'] ; EOS_token = en_w2i['EOS_token']
    en_vocab_size = len(en_w2i); cz_vocab_size = len(cz_w2i)
    encoder, decoder, AUTOdecoder = load_model(args, en_vocab_size, cz_vocab_size)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        AUTOdecoder.cuda()

    test_collate = MyCollator(True)
    data_loader_test = DataLoader(
                dataset=datasets['test'],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=test_collate,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

    with torch.no_grad():
        translation_pairs= []
        reconstruction_pairs = []
        current_loss_list = []
        
        for i, batch_list in enumerate(data_loader_test):                
            batch = {}
            batch['eng_idxes'] = torch.tensor([batch_list[i]['eng_idxes'] for i in range(len(batch_list))], dtype = torch.long)
            batch['cz_idxes'] = torch.tensor([batch_list[i]['cz_idxes'] for i in range(len(batch_list))], dtype = torch.long)
            batch['eng_leng'] = torch.tensor([batch_list[i]['eng_leng'] for i in range(len(batch_list))], dtype = torch.long)
            batch['cz_leng'] = torch.tensor([batch_list[i]['cz_leng'] for i in range(len(batch_list))], dtype = torch.long)
            
            
            sorted_lengths, sorted_idx = torch.sort(batch['eng_leng'], descending=True)
            batch['eng_idxes'] = batch['eng_idxes'][sorted_idx]
            batch['cz_idxes'] = batch['cz_idxes'][sorted_idx]
            batch['eng_leng'] = batch['eng_leng'][sorted_idx]
            batch['cz_leng'] = batch['cz_leng'][sorted_idx]
            cur_batch_size = len(batch['eng_leng'])
            
            input_vectors = to_cuda(batch['eng_idxes'])
            target_vectors = to_cuda(batch['cz_idxes'])

                
            encoded, hidden_ec = encoder(input_vectors, sorted_lengths)
            
            
            s1 = torch.cat([encoded[:, -1, :hidden_size], encoded[:, 0, hidden_size:]  ], 1)
            s1 = s1.contiguous()
        
            #Main Decoder 
            decoder_in, s, w = decoder_initial(cur_batch_size);SOS_token = 1; decoder_in = torch.LongTensor([SOS_token] * cur_batch_size).cuda()
            decoder_hidden = s1
            out_list = []
            for j in range(target_vectors.size(1)):  # for all sequences
                """
                decoder_in (Variable): [b]
                encoded (Variable): [b x seq x hid]
                input_out (np.array): [b x seq]
                s (Variable): [b x hid]
                """
                # 1st state
                
                decoder_output, decoder_hidden, decoder_attn = decoder(
                        decoder_in, decoder_hidden, encoded
                )
                decoder_output = decoder_output.cpu()
                if j ==0:
                    out = decoder_output.unsqueeze(1).cpu()
                else:
                    out = torch.cat([out,decoder_output.unsqueeze(1)],dim=1)
                decoder_in = out[:, -1].max(1)[1].cuda() 
                #out_list.append(out[:, -1].max(1)[1].squeeze().cpu().data.numpy())
                out_list.append(out[:, -1].max(1)[1].cpu().data.numpy())
            
            #AUTO DECODER
                        
            AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*cur_batch_size, device=device), input_vectors, 1
            if args.alpha > 0 and args.cos_only ==0:
                for di in range(AUTOtarget_tensor.shape[1]):
                    AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden,
                                                                         encoded)
                    topv, topi = AUTOdecoder_output.topk(1)
                    AUTOdecoder_input = topi.squeeze(1).detach()
                    AUTOdecoder_output = AUTOdecoder_output.cpu()
                    di += 1
                    if di == 1:
                        tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],
                                                                                    1,
                                                                                    AUTOdecoder_output.shape[1])
                    else:
                        tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,
                                                                       AUTOdecoder_output.view(
                                                                           AUTOdecoder_output.shape[0], 1,
                                                                           AUTOdecoder_output.shape[1])], dim=1)
            predicted = np.transpose(np.array(out_list))
            y = target_vectors.cpu().view(cur_batch_size, -1)
        
            translation_pairs.append((predicted, y.view(cur_batch_size, -1).cpu().numpy()))
            try:
                reconstruction_pairs.append((tensor_of_all_AUTOdecoded_outputs
                                                        .topk(1)[1].squeeze(2).cpu().numpy(), input_vectors.cpu().numpy()))
            except:
                pass
           
    #print(" yshape ", y.shape)
    #Verbose Results

    #for i in range(y.shape[0]):
    for i in range(min(200, y.shape[0])):
        print("target: " +str(' '.join([cz_i2w[y[i][j].item()] for j in range(y.shape[1]) if y[i][j].item() != 0])))
        temp_list = []
        for j in range(predicted.shape[1]):
            temp_list.append(cz_i2w[predicted[i][j]])
        #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
        print("predicted: " + str(' '.join(temp_list)))
        print(' ')
    try:
        for i in range(min(200, y.shape[0])):
            print("target: " +str(' '.join([en_i2w[input_vectors[i][j].item()] for j in range(input_vectors.shape[1]) if input_vectors[i][j].item() != 0])))
            temp_list = []
            for j in range(tensor_of_all_AUTOdecoded_outputs.topk(1)[1].squeeze(2).cpu().numpy().shape[1]):
                temp_list.append(en_i2w[tensor_of_all_AUTOdecoded_outputs.topk(1)[1].cpu().numpy().squeeze(2)[i][j]])
            #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
            print("predicted: " + str(' '.join(temp_list)))
            print(' ')
    except:
        pass
                
        
    
    
    #Now Calculate BLEU and show validation examples
    tr_pairs = {1: translation_pairs}
    tr_pairs = clean_pairs(tr_pairs)
    avg_BLEU = 0.0
    
    avg_counter = 0
    for e, pairs_list in tr_pairs.items():
        #print("pairs list :", pairs_list)
        for k in range(len(pairs_list)):
            #candidate = pairs_list[k][0][1:-1]; reference = pairs_list[k][1][1:-1]
            ref_len = len(pairs_list[k][1])
            candidate = pairs_list[k][0][1:ref_len-1]; reference = pairs_list[k][1][1:-1]
            #print("candiate:" , candidate)
            #print("refenrece:", reference)
            if len(reference) < 4:
                weights = [1.0/len(reference)]*len(reference)
                #print("weights : ", weights)
                BLEU_score = nltk.translate.bleu_score.sentence_bleu([reference], candidate, weights = weights)
            else:
                BLEU_score = nltk.translate.bleu_score.sentence_bleu([reference], candidate)
            avg_BLEU += BLEU_score
            #print("current BLEU: ", BLEU_score)
            avg_counter += 1
    avg_BLEU = avg_BLEU / avg_counter
    
    print("BLEU score: ", avg_BLEU * 100)
      
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.cuda.set_device(args.cuda)
    test(args)

            
        