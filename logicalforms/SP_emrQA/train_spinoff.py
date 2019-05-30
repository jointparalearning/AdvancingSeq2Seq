#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:07:08 2019

@author: TiffMin
"""
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from models.functions import numpy_to_var, toData, to_np, to_var, visualize, decoder_initial, update_logger
import time
import sys, os
import math
from numpy import inf
import random
import _pickle as cPickle
import argparse
from models.masked_cross_enropy import *
from models.plain_method2_spinoff import *
from models.seq2seq_Luong import *


#from tensorboardX import SummaryWriter




#from models.plain import 
#from models.vae_ver1 import 
#from models. import 

parser = argparse.ArgumentParser(description='Define trainig arguments')

parser.add_argument('-sh', '--shuffle_scheme', type = int, metavar='', required =True, help ="saving directory of output file")
parser.add_argument('-spl', '--split_num', type=int, metavar='', required=True, help= "split num")
parser.add_argument('-m', '--model_version', type = int, metavar='', required=True, help = "model version - vae")
parser.add_argument('-a', '--alpha', type = float, metavar='', required=True, help = "alpha")
parser.add_argument('-kl', '--loss_version', type = int, metavar='', required=True, help = "loss version")
parser.add_argument('-save_dir', '--saving_dir', type=str, metavar='', required=True, help="saving directory of output file; not required but can specify if you want to.")
parser.add_argument('-bin', '--binary', type = int, metavar='', required =True, help ="use teacher forcing for training")
parser.add_argument('-con', '--conditional_copy', type = int, metavar='', required =True, help ="conditional copy")
parser.add_argument('-loc', '--loc', type = int, metavar='', default=0, help ="conditional copy")
parser.add_argument('-D', '--D', type = int, metavar='', default=0, help ="conditional copy")
parser.add_argument('-bi', '--bi', type = int, metavar='', default = 0, help ="conditional copy")


##Optional Arguments
parser.add_argument('-H', '--hyperparams', type=str, metavar='', required=False, help = "hyperparameters")
parser.add_argument('-load_dir', '--loading_dir', type=str, metavar='', required=False, help="load model and start with it if there is a directory")
parser.add_argument('-c', '--cuda', type = int, metavar='', required=False, help = "cuda")



args = parser.parse_args()

#m_ver_dict = {0: 'models/plain.py', 1:'models/vae_ver1.py', 2:  'models/plain_explicit.py', 3:'models/vae_explicit.py', 4:'models/plain_method2.py'}
m_ver_dict = {0: 'models/plain_method2_spinoff.py', 1:'models/vae_ver1.py', 2:  'models/plain_explicit.py', 3:'models/vae_explicit.py'}


torch.manual_seed(1000)


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

def acc(pairs_dict):
    acc_list = []
    for k, pairs_list in pairs_dict.items():
        if len(pairs_list)>0:
            acc = sum([list(tup[0]) == list(tup[1]) for tup in pairs_list])/len(pairs_list)
        else:
            acc=0
        acc_list.append(acc)
    return acc_list

def train(num_epochs, args):
    def acc_bowser(dict_pairs):
        try:
            tr_pairs = dict_pairs['translation_pairs']
        except:
            tr_pairs = dict_pairs['pairs_dict']['translation_pairs']
        tr_pairs = clean_pairs(tr_pairs)
        counter = 0
        for k, pairs_list in tr_pairs.items():
            assert k == epoch
            acc = sum([list(tup[0]) == list(tup[1]) for tup in pairs_list])/len(pairs_list)
            counter +=1
        assert counter==1
        return acc
    def log(args, epoch):
        file_name = "outputs/" + args.saving_dir + "/logs.txt"
        file = open(file_name, "w")
        file.write("ran till epoch: " + str(epoch) + "\n")
        file.write("model: "+str(args.model_version)+ "\n" )
        file.write("alpha: "+ str(args.alpha)+ "\n")
        file.write("split num: " + str(args.split_num)+ "\n")
        file.write("shuffle scheme: " + str(args.shuffle_scheme)+ "\n")
        file.write("Hyperparameters are like the following: "+ "\n")
        file.write("      learning rate: "+ str(learning_rate)+ "\n")
        file.write("      weight decay: " + str(weight_decay)+ "\n")
        file.write("      batch size: " + str(batch_size)+ "\n")
        file.write("      teacher forcing ratio: " + str(teacher_forcing_prob)+ "\n")
        file.write("      embed dim: " + str(embed_dim)+ "\n")
        file.write("      latent dim: " + str(latent_dim)+ "\n")
        file.write("      hidden dim: " + str(hidden_dim)+ "\n")
        file.write("      AUTO hidden dim: " + str(AUTOhidden_dim)+ "\n")
        
        if not(kl_bool in [0,1]):
            raise NotImplementedError
        elif kl_bool == 1:
            file.write("loss: kl used in VAE autodecoder")
        else:
            file.write("loss: kl not used anywhere")
        file.close()

    def save(trans_pairs, rec_pairs, args):
        torch.save(encoder.state_dict(),'outputs/' + args.saving_dir + '/encoder.pt')
        torch.save(decoder.state_dict(),'outputs/' + args.saving_dir + '/decoder.pt')
        torch.save(AUTOdecoder.state_dict(),'outputs/' + args.saving_dir + '/auto-decoder.pt')
        pairs_dict = {'translation_pairs': trans_pairs, 'reconstruction_pairs': rec_pairs}
        loss_dict = {'training':training_loss_dict, 'validation': validation_loss_dict}
        final_dict = {'pairs_dict':pairs_dict, 'loss_dict':loss_dict}
        cPickle.dump(final_dict, open('outputs/' + args.saving_dir + '/loss_list.p',"wb"))    
        
        
    def prepare_batch_training(batch_num):
        if (batch_num+1)*batch_size <= len(training_sampled):
            end_num = (batch_num+1)*batch_size 
        else:
            end_num = len(training_sampled)
            #batch_size = end_num - batch_num*batch_size
        #sort by length
        sorted_idxes = sorted(training_sampled[batch_num*batch_size:end_num], key = lambda idx: len(sent2idxtensor(tokenized_eng_sentences[idx], idx)), reverse=True)
        #print("sorted idxes is " + str(sorted_idxes))
        max_length = len(sent2idxtensor(tokenized_eng_sentences[sorted_idxes[0]], sorted_idxes[0]))
        max_y_length = max([len(Qidx2LFIdxVec_dict[idx]) for idx in sorted_idxes])
        #print("max length is " + str(max_length))
        sentence_vectors, binary_vectors, x_lengths, labels = [],[],[],[]
        for idx in sorted_idxes:
            sent = sent2idxtensor(tokenized_eng_sentences[idx], idx)
            sentence_vectors.append(sent +[pad_token] * (max_length-len(sent)))
            if binary is 1:
                binary_vec = [SOS_token] + raw_question_binary_en[idx]  + [EOS_token]
                assert len(binary_vec) == len(sent), str(idx)
                binary_vectors.append(binary_vec + [pad_token] * (max_length-len(binary_vec)))
            else:
                pass
            x_lengths.append(len(sent))
            labels.append(Qidx2LFIdxVec_dict[idx] +[pad_token] * (max_y_length - len(Qidx2LFIdxVec_dict[idx])) )
        return torch.tensor(sentence_vectors, device=device), torch.tensor(binary_vectors, dtype=torch.float, device = device), torch.tensor(labels, device=device), x_lengths
                
    def prepare_batch_test(test_batch_num):
        if (test_batch_num+1)*batch_size <= len(validation_sampled):
            end_num = (test_batch_num+1)*batch_size 
        else:
            end_num = len(validation_sampled)
            #batch_size = end_num - batch_num*batch_size
        sorted_idxes = sorted(validation_sampled[test_batch_num*batch_size:end_num], key = lambda idx: len(sent2idxtensor(tokenized_eng_sentences[idx], idx)), reverse=True)
        #print("sorted idxes:" + str(sorted_idxes))
        max_length = len(sent2idxtensor(tokenized_eng_sentences[sorted_idxes[0]], sorted_idxes[0]))
        max_y_length = max([len(Qidx2LFIdxVec_dict[idx]) for idx in sorted_idxes])
        sentence_vectors,binary_vectors, x_lengths, labels = [],[],[],[]
        for idx in sorted_idxes:
            sent = sent2idxtensor(tokenized_eng_sentences[idx], idx)
            sentence_vectors.append(sent +[pad_token] * (max_length-len(sent)))
            if args.model_version in [2,3]:
                binary_vec = [SOS_token] + raw_question_binary_en[idx] + [EOS_token]
                binary_vectors.append(binary_vec + [pad_token] * (max_length-len(binary_vec)))
            else:
                pass
            x_lengths.append(len(sent))
            labels.append(Qidx2LFIdxVec_dict[idx] +[pad_token] * (max_y_length - len(Qidx2LFIdxVec_dict[idx])) )
        return torch.tensor(sentence_vectors, device=device),torch.tensor(binary_vectors, dtype=torch.float, device = device), torch.tensor(labels, device=device), x_lengths
    

    def joint_loss_nll(alpha, translated_predicted, translated_actual, reconstructed_predicted, reconstructed_actual, kl_true=False):
        #total_loss = 0.0
        temp_loss_function = nn.NLLLoss(ignore_index = pad_token)
        translation_loss = temp_loss_function(translated_predicted, translated_actual)
        reconstruction_loss = temp_loss_function(reconstructed_predicted, reconstructed_actual)
        if kl_true == False:
            total_loss = translation_loss + alpha* reconstruction_loss
            return translation_loss, reconstruction_loss, total_loss
        else:
            total_loss = translation_loss + alpha* (reconstruction_loss+kl_anneal(step)*kl)
            return translation_loss, reconstruction_loss, total_loss

    def kl_anneal(step_num):
        k = 0.0025; x0 = 2500
        return float(1/(1+np.exp(-k*(step_num-x0))))
  
    
    #writer = SummaryWriter()
    global shuffle_scheme
    global split_num
    global binary
    cuda, m_ver, alpha, shuffle_scheme, kl_bool, save_dir, load_dir, binary = args.cuda, args.model_version, args.alpha, args.shuffle_scheme, args.loss_version, args.saving_dir, args.loading_dir, args.binary
    
    if m_ver in [2,3]:
        binary = 1
        assert not(args.conditional_copy is 1 and binary is 1)

    
    assert args.split_num in [1,2,3,4,5]
    split_num = args.split_num
        
    ###make folder for save_dir
    save_dir = "outputs/" + save_dir 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    ###load the required arguments first
    #load shuff_scheme
    if shuffle_scheme in [0,1,2]:
        exec(open('data_prep/data_prepRAW_Shuffle.py').read(), globals(), globals())        
    else:
        raise ValueError('shuffling scheme argument wrong')
    
    #load alpha     
    alpha = alpha 
    
    ###load optional arguments and model
    #load hyperparameters
    global batch_size 
    global embed_dim 
    global hidden_dim 
    global latent_dim 
    global hidden_dim 
    global AUTOhidden_dim 
    global teacher_forcing_prob
    global weight_decay 
    global learning_rate
    global kl
    global binary_vectors
    kl = None
        
    if args.hyperparams is not None:
        exec(open('hyperparams/'+args.hyperparams).read(), globals(), globals())
    else:
        #exec(compile(open(file).read(), file, 'exec'))
        exec(open('hyperparams/default_hyperparams.py').read(), globals(), globals()) 
    original_batch_size = batch_size
    
    #load model version
    assert m_ver in m_ver_dict
    exec(open(m_ver_dict[m_ver]).read(), globals(), globals())
    #initialize model and load if needed 
    if m_ver in [0,2,4]: #DID UNTIL HERE!
        encoder = CopyEncoderRAW(hidden_dim, vocab_size)
        decoder = CopyDecoder('general', 'general', 0, 1, vocab_size, embed_dim*2, hidden_dim*2, bi = args.bi,  local_attn_cp=args.loc, D=args.D)
        #AUTOdecoder = AutoDecoder(AUTOhidden_dim, vocab_size)
        AUTOdecoder = LuongAttnDecoderRNN('general', hidden_dim, vocab_size, bi = args.bi)
        
    elif m_ver in [1,3]:
        encoder = VAEEncoderRAW(hidden_dim, vocab_size, latent_dim)
        decoder = CopyDecoder(vocab_size, embed_dim, hidden_dim, latent_dim, local_attn_cp=args.loc, D=args.D)
        AUTOdecoder = AutoDecoder(AUTOhidden_dim, vocab_size, latent_dim)
        
        
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        AUTOdecoder.cuda()

    #load save_dir
    
    #load loss function
    loss_function = joint_loss_nll  
        
    #load model 
    if load_dir:
        encoder_dir = 'outputs/' + load_dir + '/encoder.pt'  
        decoder_dir = 'outputs/' + load_dir + '/decoder.pt'
        auto_dir = 'outputs/' + load_dir + '/auto-decoder.pt'
        
        encoder.load_state_dict(torch.load(encoder_dir)) 
        decoder.load_state_dict(torch.load(decoder_dir)) 
        AUTOdecoder.load_state_dict(torch.load(auto_dir)) 
        
        encoder.train(); decoder.train(); AUTOdecoder.train()
        loss_dict = cPickle.load(open('outputs/' + load_dir + "/loss_list.p", "rb"))
        try:
            trained_until = int(list(loss_dict['translation_pairs'].keys())[-1]/100)*100
        except: 
            trained_until = int(list(loss_dict['pairs_dict']['translation_pairs'].keys())[-1]/100)*100
            
        epochs = range(trained_until+1, num_epochs)
        try:
            reconstruction_pairs = loss_dict['translation_pairs']
            translation_pairs = loss_dict['reconstruction_pairs']
        except:
            reconstruction_pairs = loss_dict['pairs_dict']['translation_pairs']
            translation_pairs = loss_dict['pairs_dict']['reconstruction_pairs']
        
    else:
        epochs = range(num_epochs)
        reconstruction_pairs = {}
        translation_pairs = {}
    
    #actual training starts
    
    training_loss_dict = {'reconstruction': [], 'translation':[], 'joint':[]}
    validation_loss_dict = {'reconstruction': [], 'translation':[], 'joint':[]}
    
    if kl_bool ==1:
        training_loss_dict['kl'] = []
        validation_loss_dict['kl'] = []
    start = time.time()
    step = 0
    for epoch in epochs:
        
        opt_e = optim.Adam(params=encoder.parameters(),  lr=learning_rate)
        opt_d = optim.Adam(params=decoder.parameters(),  lr=learning_rate)
        opt_a = optim.Adam(params=AUTOdecoder.parameters(), lr=learning_rate)

                
        batch_num = 0         
        current_loss_list = []
        print("==================================================")
        print("Epoch ",epoch)
        board_translation_loss, board_reconstruction_loss, board_joint_loss, board_kl = 0, 0, 0, 0
        while (batch_num) * batch_size < len(training_sampled):
            if batch_num % 100 ==0:
                print("==================================================")
                print("Batch Num: ",batch_num)
                print("Batch Percent: ",100 *(batch_num) * batch_size/ len(training_sampled), "%")
            sentence_vectors, binary_vectors, target, X_lengths = prepare_batch_training(batch_num)
            
            #batch_size = sentence_vectors.shape[0]
            batch_size = sentence_vectors.shape[0]
            opt_e.zero_grad()
            opt_d.zero_grad()
            opt_a.zero_grad()
    
            # obtain batch outputs
            input_out = sentence_vectors.view(batch_size,-1).data.cpu().numpy()
        
            # mask input to remove padding
            input_mask = np.array(input_out>0, dtype=int)
    
            # input and output in Variable form
            x = sentence_vectors
            y = target.view(batch_size, -1)
    
            # apply to encoder
            encoded, hidden_ec = encoder(x, X_lengths) # z is [batch_size, latent_size]
            #print("X lengths ", X_lengths)
            
            # get initial input of decoder
            decoder_in, s, w = decoder_initial(batch_size)
            SOS_token = 1
            decoder_in = torch.LongTensor([SOS_token] * batch_size)
            if args.bi==0:
                s1 = encoded[:, -1, :] 
            else:
                s1 = torch.cat([encoded[:, -1, :hidden_dim], encoded[:, 0, hidden_dim:]  ], 1) # b x seq x 2*hidden 
            s1 = s1.contiguous()
            #decoder_in = decoder_in.cpu()
            if torch.cuda.is_available():
                decoder_in = decoder_in.cuda()
            
            
            # out_list to store outputs
            use_teacher_forcing = True if random.random() < teacher_forcing_prob else False
            out_list=[]
            for j in range(y.size(1)): # for all sequences
                """
                decoder_in (Variable): [b]
                encoded (Variable): [b x seq x hid]
                input_out (np.array): [b x seq]
                s (Variable): [b x hid]
                """
                #assert binary_vectors.size(1) == y.size(1)
                # 1st state
                if j==0:
                    out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                    encoded_idx=input_out, prev_state= s1,
                                    weighted=w, order=j, X_lengths=X_lengths)
                   
                else:
                    tmp_out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                    encoded_idx=input_out, prev_state= s,
                                    weighted=w, order=j, X_lengths=X_lengths)

                    out = torch.cat([out,tmp_out],dim=1)
                    
                if not(use_teacher_forcing):
                    decoder_in = out[:,-1].max(1)[1].squeeze() # train with sequence outputs
                else:
                    decoder_in = y[:,j] # train with ground truth
                out_list.append(out[:,-1].max(1)[1].squeeze().cpu().data.numpy())
                        
            ###AUTO DECODER
            AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*batch_size, device=device), sentence_vectors, 1
            use_teacher_forcing = True if random.random() < teacher_forcing_prob else False
            
            if use_teacher_forcing:
                for di in range(AUTOtarget_tensor.shape[1]):
                    AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                        AUTOdecoder_input, AUTOdecoder_hidden, encoded)
                    #print("di is", di)
                    #print("kl is ", kl)
                    #print("AUTOdecoder_output: " + str(AUTOdecoder_output.shape))
                    AUTOdecoder_input = AUTOtarget_tensor[:,di]  # Teacher forcing
                    di+=1
                    if di ==1:
                        tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])
                    else:
                        tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,  AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])], dim=1)
    
            else:
                for di in range(AUTOtarget_tensor.shape[1]):
                    AUTOdecoder_output, AUTOdecoder_hidden, kl= AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden, encoded)
                    topv, topi = AUTOdecoder_output.topk(1) 
                    AUTOdecoder_input = topi.squeeze().detach()  
                    di +=1
                    if di ==1:
                        tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])
                    else:
                        tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,  AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])], dim=1)
    
  
            
            target = y.view(-1)
            pad_out = out.view(-1,out.shape[2])
            pad_out = pad_out + 0.0001
            pad_out = torch.log(pad_out)
            
            review_sent = sentence_vectors.view(-1)
            review_auto_out = tensor_of_all_AUTOdecoded_outputs.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2])
            
            if kl_bool == 0:
                kl = None
                translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target, review_auto_out  ,review_sent) 
                board_translation_loss += translation_loss; board_reconstruction_loss += reconstruction_loss; board_joint_loss += joint_loss

            elif kl_bool ==1:
                translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target, review_auto_out ,review_sent, kl_true=True) 
                board_translation_loss += translation_loss; board_reconstruction_loss += reconstruction_loss; board_joint_loss += joint_loss; board_kl += kl
  
            
            if len(pad_out[pad_out == -inf]) > 0 :
                joint_loss = torch.tensor(0.0, device=device)
                print("Loss broken but skipped!")
            else:
                joint_loss.backward()
            

            opt_e.step()
            opt_d.step()
            if alpha !=0:
                opt_a.step()
            step +=1
            batch_num +=1
            batch_size = original_batch_size
                        
        if epoch%1 ==0:
            learning_rate= learning_rate * weight_decay
        
        if epoch %1 ==0:
            training_loss_dict['reconstruction'].append(board_reconstruction_loss.item()/5)
            training_loss_dict['translation'].append(board_translation_loss.item()/5)
            training_loss_dict['joint'].append(board_joint_loss.item()/5)
            if kl_bool == 1:
                training_loss_dict['kl'].append(board_kl.item()/5)

        
        elapsed = time.time()
        print("Elapsed time for epoch: ",elapsed-start)
        start = time.time()
    
        ##########VALIDATION####################################
        if epoch % 1 == 0:
            translation_pairs[epoch] = []; reconstruction_pairs[epoch] = []
            #current_loss_list = []
            test_batch_num = 0
            
            board_translation_loss, board_reconstruction_loss, board_joint_loss, board_kl = 0, 0, 0, 0
            with torch.no_grad():
                while (test_batch_num) * batch_size < len(validation_sampled):
                    sentence_vectors, binary_vectors, target, X_lengths = prepare_batch_test(test_batch_num)
                    batch_size = sentence_vectors.shape[0]
                    test_batch_num +=1
                    input_out = sentence_vectors.view(batch_size,-1).data.cpu().numpy()
                    
                    # mask input to remove padding
                    input_mask = np.array(input_out>0, dtype=int)
            
                    # input and output in Variable form
                    x = sentence_vectors
                    y = target.view(batch_size, -1)
            
                    # apply to encoder
                    encoded, hidden_ec = encoder(x, X_lengths) # z is [batch_size, latent_size]
                    
                    # get initial input of decoder
                    decoder_in, s, w = decoder_initial(batch_size)
                    SOS_token = 1
                    decoder_in = torch.LongTensor([SOS_token] * batch_size)
                    if args.bi==0:
                        s1 = encoded[:, -1, :] 
                    else:
                        s1 = torch.cat([encoded[:, -1, :hidden_dim], encoded[:, 0, hidden_dim:]  ], 1) # b x seq x 2*hidden 
                    s1 = s1.contiguous()
                    #decoder_in = decoder_in.cpu()
                    if torch.cuda.is_available():
                        decoder_in = decoder_in.cuda()
            
                    # out_list to store outputs
                    out_list=[]
                    for j in range(y.size(1)): # for all sequences
                        """
                        decoder_in (Variable): [b]
                        encoded (Variable): [b x seq x hid]
                        input_out (np.array): [b x seq]
                        s (Variable): [b x hid]
                        """
                        # 1st state
                        if j==0:
                            out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                            encoded_idx=input_out, prev_state= s1,
                                            weighted=w, order=j,  X_lengths = X_lengths)
                        # remaining states
                        else:
                            tmp_out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                            encoded_idx=input_out, prev_state= s,
                                            weighted=w, order=j,  X_lengths = X_lengths)
                            out = torch.cat([out,tmp_out],dim=1)
                        decoder_in = out[:,-1].max(1)[1].squeeze() # train with sequence outputs
                        out_list.append(out[:,-1].max(1)[1].squeeze().cpu().data.numpy())
                    
                    
                    ##AUTO DECODER
                    ##
                    AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*batch_size, device=device), sentence_vectors, 1
                    use_teacher_forcing = False 
                    
                    if use_teacher_forcing:
                        for di in range(AUTOtarget_tensor.shape[1]):
                            AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                AUTOdecoder_input, AUTOdecoder_hidden, encoded)
                            AUTOdecoder_input = AUTOtarget_tensor[:,di]  # Teacher forcing
                            di+=1
                            if di ==1:
                                tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])
                            else:
                                tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,  AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])], dim=1)
            
                    else:
                        for di in range(AUTOtarget_tensor.shape[1]):
                            AUTOdecoder_output, AUTOdecoder_hidden, kl= AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden, encoded)
                            topv, topi = AUTOdecoder_output.topk(1) 
                            AUTOdecoder_input = topi.squeeze().detach()  
                            di +=1
                            if di ==1:
                                tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])
                            else:
                                tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,  AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])], dim=1)
   
                    target = y.view(-1)
                    pad_out = out.view(-1,out.shape[2])
                    pad_out = pad_out + 0.0001
                    pad_out = torch.log(pad_out)
                    
                    review_sent = sentence_vectors.view(-1)
                    review_auto_out = tensor_of_all_AUTOdecoded_outputs.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2])
                                        
                    if kl_bool == 0:
                        kl = None
                        translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target, review_auto_out  ,review_sent, kl_true=False) 
                        board_translation_loss += translation_loss; board_reconstruction_loss += reconstruction_loss; board_joint_loss += joint_loss

                    elif kl_bool ==1:
                        translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target, review_auto_out ,review_sent, kl_true=True) 
                        board_translation_loss += translation_loss; board_reconstruction_loss += reconstruction_loss; board_joint_loss += joint_loss; board_kl += kl

               
                    predicted = np.transpose(np.array(out_list))    
                    translation_pairs[epoch].append((predicted, y.cpu().numpy()))
                    reconstruction_pairs[epoch].append((tensor_of_all_AUTOdecoded_outputs.topk(1)[1].squeeze(2).cpu().numpy(), sentence_vectors.cpu().numpy()))
                    batch_size = original_batch_size
                    
                                        
                    temp_trans_pairs = {epoch: translation_pairs[epoch]}
                    temporary_dict = {'translation_pairs':temp_trans_pairs}
                    
                    for i in range(y.shape[0]):
                        print("target: " +str(' '.join([LFidx2vocab[y[i][j].item()] for j in range(y.shape[1]) if y[i][j].item() != 0])))
                        temp_list = []
                        for j in range(predicted.shape[1]):
                            temp_list.append(LFidx2vocab[predicted[i][j]])
                        #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
                        print("predicted: " + str(' '.join(temp_list)))
                        print(' ')
                        
                    print("VALIDATION ACCURACY FOR EPOCH ", epoch+1, " : ", acc_bowser(temporary_dict))
                        
            validation_loss_dict['reconstruction'].append(board_reconstruction_loss.item()/test_batch_num)
            validation_loss_dict['translation'].append(board_translation_loss.item()/test_batch_num)
            validation_loss_dict['joint'].append(board_joint_loss.item()/test_batch_num)
            if kl_bool ==1:
                validation_loss_dict['kl'].append(board_kl.item()/test_batch_num)
                    
                #test_loss_list.append(current_loss_list)
        if epoch %1 ==0:
            save(translation_pairs, reconstruction_pairs, args)
            log(args, epoch)
    
    #writer.close()
    

if __name__ == '__main__':
    print("running")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.cuda is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)

    num_epochs = 25
    train(num_epochs, args)
    
    
