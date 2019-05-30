import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
#from models.PADOUTPUTcopynet_singleMY import CopyEncoder, CopyEncoderRAW, CopyDecoder, AutoDecoder
from util_functions import numpy_to_var, toData, to_np, to_var, decoder_initial, update_logger
import time
import sys
import math
from numpy import inf
import random
import os
from bowser_constants import constants

from multiprocessing import cpu_count
import copy

import pickle as cPickle

#from bowser_model_v1 import CopyEncoder, CopyDecoder, AutoDecoder
from models.plain_method2 import *
from models.seq2seq_Luong import *

from bowser_dataset_v1 import BOWser_recipe
from collections import OrderedDict, defaultdict
from tensorboardX import SummaryWriter
import argparse
from models.masked_cross_enropy import *


parser = argparse.ArgumentParser(description='Define trainig arguments')

parser.add_argument('-a', '--alpha', type = float, metavar='', required=True, help = "alpha")
parser.add_argument('-a_k_a', '--alpha_kill_at', type = float, metavar='', default= 0, help = "alpha")

#parser.add_argument('-which_attn_g', '--which_attn_g', type = str, metavar='', required=True )
#parser.add_argument('-which_attn_c', '--which_attn_c', type = str, metavar='', required=True )

#parser.add_argument('-bahd_g', '--generate_bahd', type = int, metavar='', required=True )
#parser.add_argument('-bahd_c', '--copy_bahd', type = int, metavar='', required=True )

parser.add_argument('-l', '--cross_ent', type = int, metavar='', required=True )
parser.add_argument('-k', '--k', type = int, metavar='', required=True )
parser.add_argument('-clip', '--clip', type = float, default = 50,metavar='')
parser.add_argument('-clip_true', '--clip_true', type = float, default = 0,metavar='')
parser.add_argument('-pad', '--pad', type = float, default = 0,metavar='')


parser.add_argument('-e', '--num_epochs', type = int, default = 150,metavar='')


#parser.add_argument('-loc', '--loc', type = int, default=0,metavar='')
#parser.add_argument('-D', '--D', type = int, default=0,metavar='')
#parser.add_argument('-bi', '--bi', type = int, default=0,metavar='')


##Optional Arguments
parser.add_argument('-lr', '--lr', type=float, metavar='', required=True, help = "hyperparameters")
parser.add_argument('-d', '--decay_until', type=float, metavar='', required=True, help = "hyperparameters")
parser.add_argument('-half_end', '--half_end', type=int, metavar='', default=36, help = "hyperparameters")
parser.add_argument('-half_start', '--half_start', type=int, metavar='', default=15, help = "hyperparameters")


parser.add_argument('-c', '--cuda', type = int, metavar='', default = 0, help = "cuda")

parser.add_argument('-qp', '--q_to_p', type = int, metavar='', default=0, help = "alpha")
parser.add_argument('-s_e_e', '--save_every_epoch', type = int, metavar='', default=0, help = "alpha")
parser.add_argument('-s_from', '--save_from', type = int, metavar='', default=100, help = "alpha")

parser.add_argument('-which_attn', '--which_attn', type = str, metavar='', default = 'general', help ="conditional copy")


parser.add_argument('-cos_obj', '--cos_obj', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-cos_alph', '--cos_alph', type = float, metavar='', default = 1.0, help ="conditional copy")
parser.add_argument('-cos_only', '--cos_only', type = int, metavar='', default = 0)


parser.add_argument('-seed', '--seed', type = int, metavar='', default = 0)

parser.add_argument('-train_f', '--train_data_file', type = str , metavar='', default = 0)
parser.add_argument('-test_f', '--test_data_file', type = str, metavar='', default = 0)
parser.add_argument('-pdict_f', '--datasets_para_dict_file', type = str, metavar='', default = 0)


parser.add_argument('-hid_dim', '--hid_dim', type = int, metavar='', default = 128)

parser.add_argument('-multi_para', '--multi_para', type = int, metavar='', default = 0)
parser.add_argument('-multi_para_sim', '--multi_para_sim', type = int, metavar='', default = 0)


parser.add_argument('-adaptive_alpha', '--adaptive_alpha', type = int, metavar='', default = 0)

parser.add_argument('-v', '--result_verbose', type = int, metavar='', default = 0, help = "cuda")

parser.add_argument('-domain', '--domain', type = str, metavar='', required=True, help = "cuda")


args = parser.parse_args()


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


    
    

# Can pass in args later
def train(args):
    #file = open("results/tab" + str(args.tab) + "g_" + str(args.which_attn_g) + "c_" + str(args.which_attn_c)+  ".txt", "w")
    # Need to put things in constants and others should come from args. Then assign over here to not change the
    # code too much
    #file = open("results/" + args.which_dec + "_" + args.which_attn + "_alpha" + str(args.alpha) + "_decaytil_" + str(args.decay_until)+ ".txt", "w")
    torch.manual_seed(args.seed)
    def acc_bowser(dict_pairs):
        try:
            tr_pairs = dict_pairs['translation_pairs']
        except:
            tr_pairs = dict_pairs['pairs_dict']['translation_pairs']
        tr_pairs = clean_pairs(tr_pairs)
        counter = 0
        for k, pairs_list in tr_pairs.items():
            assert k == epoch+1
            acc = sum([list(tup[0]) == list(tup[1]) for tup in pairs_list])/len(pairs_list)
            counter +=1
        assert counter==1
        return acc
    vocab_file = 'data/para_preprocessed/' + args.domain + '/vocab.p'
    data_dir = ''
    #train_data_file = 'data/para_preprocessed/recipes_train_v1.json'
    #test_data_file = 'data/para_preprocessed/recipes_test_v1.json'
    train_data_file = 'data/para_preprocessed/' + args.train_data_file
    test_data_file = 'data/para_preprocessed/' + args.test_data_file
    model_dir = constants.model_dir
    log_dir = constants.log_dir
    pkl_dir = constants.pkl_dir
    tb_log_dir = constants.tb_log_dir
    res_dir = constants.results_dir

    log = True
    tb_log = True

    run_num = 999
    validate_at_epoch = 1
    save_model_at_epoch = 1
    save_pkl_files_at_epoch = 1
    weight_decay_at = 10

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the vocabulary
    with open(vocab_file, "rb") as f:
        vocab = cPickle.load(f)
    w2i, i2w = vocab["word2idx"], vocab["idx2word"]
    vocab_size = len(w2i)

    # Hyperparameters and other variables (like data location) - can setup using args
    embed_size = args.hid_dim
    #learning_rate = 0.0008
    learning_rate = args.lr
    AUTOhidden_dim = args.hid_dim
    weight_decay = 0.99
    hidden_size = args.hid_dim
    num_epochs = args.num_epochs
    batch_size = 32

    alpha = args.alpha

    pad_token = w2i['PAD_token']
    # print("pad token", pad_token)
    if args.domain == 'rec':
        SOS_token = w2i['SOS_token']
        EOS_token = w2i['EOS_token']
    else:
        SOS_token = w2i['[CLS]']
        UNK_token = w2i['[UNK]']
        EOS_token = w2i['[SEP]']

    # Create directories if they don't exits
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)
    if not os.path.exists(tb_log_dir):
        os.mkdir(tb_log_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    save_model_path = os.path.join(model_dir, "best_bowser_recipe")
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
        
    if log:
        run_log = open(
            os.path.join(log_dir, "logs_" + constants.add_info + constants.version + "_run" + str(run_num) + ".txt"),
            'w')
        run_log.write("Processing goequery dataset with %s and %s - Run number: " % (train_data_file, test_data_file) \
                      + str(run_num) + "\n\n")
        parameters_string = "Parameters for this run are:\nhidden_dim:" + str(hidden_size) + "\n"
        parameters_string += "batch_size:" + str(batch_size) + "\n"
        parameters_string += "learning_rate:" + str(learning_rate) + "\n"
        parameters_string += "Reduce the learning_rate in half every 5 epochs after 10 to 25. " \
                             "Then from 50 epochs use weight_decay every %d epochs"%(weight_decay_at) + "\n"
        parameters_string += "alpha - For autodecoder loss:" + str(alpha) + "\n"
        parameters_string += "save_at_model: %d, save_at_pkl: %d, " \
                             "do_validation: %d " % (
                             save_model_at_epoch, save_pkl_files_at_epoch, validate_at_epoch) + "\n"
        parameters_string += "epochs:" + str(num_epochs) + "\n\n"
        run_log.write(parameters_string)

    # Create datasets - We will create the dataloaders inside the loop for epochs
    datasets = OrderedDict()

    # data_file, data_dir, split

    if args.multi_para ==1: 
        print("Multi para running!")
        datasets['train'] = BOWser_recipe(
            data_dir=data_dir,
            data_file=train_data_file,
            split='train_with_para_multiple')
    elif args.multi_para ==0 and args.q_to_p == 1: 
        print("Sing para running!")
        datasets['train'] = BOWser_recipe(
            data_dir=data_dir,
            data_file=train_data_file,
            split='train_with_para')
    else:
        print("Plain seq2seq running!")
        datasets['train'] = BOWser_recipe(
            data_dir=data_dir,
            data_file=train_data_file,
            split='train')
    
    datasets['test'] = BOWser_recipe(
        data_dir=data_dir,
        data_file=test_data_file,
        split='test')
    
    if args.multi_para_sim ==1:
        datasets_para_dict = cPickle.load(open(args.datasets_para_dict_file,'wb'))
        datasets['datasets_para_dict'] = datasets_para_dict

    encoder = EncoderRNN(vocab_size, hidden_size)
    decoder = LuongAttnDecoderRNN(args.which_attn,hidden_size, vocab_size, bi=1)
    AUTOdecoder =LuongAttnDecoderRNN(args.which_attn, hidden_size, vocab_size, bi =1)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        AUTOdecoder.cuda()

    if tb_log:
        exp_name = "tensorboard_logs_" + constants.add_info + constants.version + "_run" + str(run_num)
        writer = SummaryWriter(os.path.join(tb_log_dir, exp_name))
        writer.add_text("model - encoder", str(encoder))
        writer.add_text("model - decoder", str(decoder))
        writer.add_text("model - autodecoder", str(AUTOdecoder))
        # writer.add_text("args", str(args))

    # Loss Function that we will be using
    def joint_loss_tensors_f(alpha, translated_predicted, translated_actual, reconstructed_predicted,
                             reconstructed_actual):
        temp_loss_function = nn.NLLLoss(ignore_index=pad_token)
        translation_loss = temp_loss_function(translated_predicted, translated_actual)
        reconstruction_loss = temp_loss_function(reconstructed_predicted, reconstructed_actual)
        total_loss = translation_loss + alpha * reconstruction_loss
        return translation_loss, reconstruction_loss, total_loss

    def to_cuda(tensor):
        # turns to cuda
        if torch.cuda.is_available():
            return tensor.cuda()
        else:
            return tensor

    # set loss
    encoder_broken = []

    reconstruction_pairs = {}
    translation_pairs = {}

    training_loss_list = []
    test_loss_list = []

    loss_function = joint_loss_tensors_f
    cos_loss_function = torch.nn.CosineEmbeddingLoss()
    temp_loss_function = nn.NLLLoss(ignore_index = 0)

    
    for epoch in range(num_epochs):
        print("==================================================")
        print("Epoch ", epoch + 1)
        start = time.time()

        opt_e = optim.Adam(params=encoder.parameters(), lr=learning_rate)
        opt_d = optim.Adam(params=decoder.parameters(), lr=learning_rate)
        opt_a = optim.Adam(params=AUTOdecoder.parameters(), lr=learning_rate)

        data_loader_train = DataLoader(
            dataset=datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        
        if args.alpha_kill_at != 0:
            if epoch >= args.alpha_kill_at:
                alpha = 0

        current_loss_list = []
        avg_translation_loss = 0.0; avg_reconstruction_loss = 0.0; avg_cos_loss = 0.0; avg_multi_loss = 0.0
        
        
        for i, batch in enumerate(data_loader_train):
            # Sort the inputs and targets in the batch in reverse sorted order
            sorted_lengths, sorted_idx = torch.sort(batch['s_length'], descending=True)
            batch['sent_tok_idx'] = batch['sent_tok_idx'][sorted_idx]
            batch['answer_tok_idx'] = batch['answer_tok_idx'][sorted_idx]
            batch['a_length'] = batch['a_length'][sorted_idx]
            batch['s_length'] = batch['s_length'][sorted_idx]
            
            if args.q_to_p == 1:
                para_sorted_lengths, para_sorted_idx = torch.sort(batch['para_length'], descending=True)
                para_vectors = batch['para_tok_idx'][sorted_idx]
                #batch['para_length'] = batch['para_length'][sorted_idx]
                #batch['para_length'] = 
        

            # Need it for the decoders
            cur_batch_size = len(batch['s_length'])

            # initialize gradient buffers
            opt_e.zero_grad()
            opt_d.zero_grad()
            opt_a.zero_grad()

            # Chop off unnecessary positions in the input and target vectors - Thus reducing sequence length
            # and increasing efficiency as decoder will be slower with more positions in the target sequences
            input_vectors = to_cuda(batch['sent_tok_idx'][:, :sorted_lengths[0]])
            if args.q_to_p == 1:
                para_vectors = to_cuda(para_vectors[:, :para_sorted_lengths[0]])
            target_vectors = to_cuda(batch['answer_tok_idx'][:, :batch["a_length"].max().item()])


            # apply to encoder
            encoded, hidden_ec = encoder(input_vectors, sorted_lengths)
            # encoded - Shape ([32, 15, 256]) - Since in this batch 15 is the max length

            

            #print("encoded: ", encoded)
            # stop if there is a problem
            if math.isnan(encoded[0][0].data[0]):
                print("encoder broken!")
                encoder_broken.append(input_vectors)
                sys.exit()
                break

            input_out = input_vectors.cpu().data.numpy()

            # get initial input of decoder
            decoder_in, s, w = decoder_initial(cur_batch_size)
            SOS_token = 1
            decoder_in = torch.LongTensor([SOS_token] * cur_batch_size)
            
            s1 = torch.cat([encoded[:, -1, :hidden_size], encoded[:, 0, hidden_size:]  ], 1) # b x seq x 2*hidden 
            #print("decodedr hidden shape", decoder_hidden.shape) #should be 1x Batch x Hidden
            s1 = s1.contiguous()
            decoder_hidden = s1
            # We will flip because we have the batch first
            if torch.cuda.is_available():
                decoder_in = decoder_in.cuda()
      
            # decoder_in = decoder_in.cpu()

            # out_list to store outputs
            out_list = []
            teacher_force_prob = 1
            use_tf =  random.random() < teacher_force_prob
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

                if j ==0:
                    out = decoder_output.unsqueeze(1)
                else:
                    out = torch.cat([out,decoder_output.unsqueeze(1)],dim=1)
                if use_tf:
                    decoder_in = target_vectors[:, j]  # Next input is current target
                else:
                    decoder_in = out[:, -1].max(1)[1]
                out_list.append(out[:, -1].max(1)[1].squeeze().cpu().data.numpy())
    
            
            ### AUTO DECODER ###
                        
            if args.q_to_p == 0:
                AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*cur_batch_size, device=device), input_vectors, 1
            else:
                AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*cur_batch_size, device=device), para_vectors, 1

            
            
            # Just going to use teach_forcing in this run
            # use_teacher_forcing = True if teacher_forcing_prob > random.random() else False
            use_teacher_forcing = True

            if use_teacher_forcing:
                for di in range(AUTOtarget_tensor.shape[1]):
                    AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden,
                                                                         encoded)
                    # print("AUTOdecoder_output: " + str(AUTOdecoder_output.shape))
                    AUTOdecoder_input = AUTOtarget_tensor[:, di]  # Teacher forcing
                    di += 1
                    if di == 1:
                        tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0], 1,
                                                                                    AUTOdecoder_output.shape[1])
                    else:
                        tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,
                                                                       AUTOdecoder_output.view(
                                                                           AUTOdecoder_output.shape[0], 1,
                                                                           AUTOdecoder_output.shape[1])], dim=1)

            else:
                for di in range(AUTOtarget_tensor.shape[1]):
                    AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden,
                                                                         encoded)
                    topv, topi = AUTOdecoder_output.topk(1)
                    AUTOdecoder_input = topi.squeeze().detach()
                    di += 1
                    if di == 1:
                        tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0], 1,
                                                                                    AUTOdecoder_output.shape[1])
                    else:
                        tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,
                                                                       AUTOdecoder_output.view(
                                                                       AUTOdecoder_output.shape[0], 1,
                                                                       AUTOdecoder_output.shape[1])], dim=1)
            
            # Loss calculation
            # You do contiguous because .view() can only be applied when your entire tensor is in the same memory block
            # Yes. Worry even about the low level details! Haha
            if args.cross_ent is 0:
                
            
                target = target_vectors.contiguous()
                target = target.view(-1)
                pad_out = torch.softmax(out.view(-1, out.shape[2]), dim=1)
                #print("pad out before log")
                #for i in range(pad_out.shape[0]):
                #    print(pad_out[i])
                pad_out = pad_out #+ 0.0001
                if args.pad == 1:
                    pad_out = pad_out + 0.0001
                pad_out = torch.log(pad_out)
                #for i in range(pad_out.shape[0]):
                #    print(pad_out[i])
    
                review_sent = AUTOtarget_tensor.contiguous().view(-1)
                #review_auto_out = tensor_of_all_AUTOdecoded_outputs.view(-1,
                #                                                         tensor_of_all_AUTOdecoded_outputs.shape[
                #                                                             2])
                review_auto_out = torch.softmax(tensor_of_all_AUTOdecoded_outputs.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2]), dim=1)
                if args.pad == 1:
                    review_auto_out = review_auto_out + 0.0001
                review_auto_out = torch.log(review_auto_out)#+0.0001)
    
                # Shape of review_sent - [544], Shape of review_auto_out - [544, 341]
                
                if args.cos_only == 0:
                    translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target,
                                                                                      review_auto_out,
                                                                                      review_sent)
                    avg_translation_loss += translation_loss; avg_reconstruction_loss += reconstruction_loss
                    avg_multi_loss += reconstruction_loss
                    
                else:
                    translation_loss = temp_loss_function(pad_out, target)
                    joint_loss = translation_loss; avg_translation_loss += translation_loss
                    
                    
                if args.cos_obj == 1:
                    para_encoder_vectors = batch['para_tok_idx'][para_sorted_idx].cuda(); para_encoder_lengths = batch['para_length'][para_sorted_idx]
                    encoded_p, hidden_ec_p = encoder(para_encoder_vectors, para_encoder_lengths)
                    p_rep = torch.cat([encoded_p[:, -1, :hidden_size], encoded_p[:, 0, hidden_size:]  ], 1) 
                    para_sorted_2_original_dict = {v:k for k, v in enumerate(para_sorted_idx)}
                    para_sorted_2_original_list = [-1]*len(para_sorted_2_original_dict)
                    for v, k in para_sorted_2_original_dict.items():
                        para_sorted_2_original_list[v] = k
                    p_rep = p_rep[para_sorted_2_original_list]
                    
                    cos_loss = cos_loss_function(p_rep, s1, torch.tensor([1.0]*cur_batch_size).cuda())
                    if args.cos_only ==1:
                        reconstruction_loss = cos_loss; avg_reconstruction_loss += reconstruction_loss
                    joint_loss += alpha* args.cos_alph* cos_loss
                    
                    avg_cos_loss = args.cos_alph* cos_loss
                    
                    
                    
                    
            
            elif args.cross_ent is 1:
                print(out.shape)
                loss = masked_cross_entropy(
                    (out).contiguous(),
                    target_vectors.contiguous(),
                    batch['a_length'],
                    True
                )
                print("this loss", loss)
                auto_loss = masked_cross_entropy(
                    (tensor_of_all_AUTOdecoded_outputs).contiguous(),
                    input_vectors.contiguous(),
                    batch['s_length'], 
                    True
                )
                #loss = temp_loss_function(torch.log((tensor_of_all_AUTOdecoded_outputs + 1e-5)), y.view(-1))
                joint_loss = loss+alpha * auto_loss
                avg_translation_loss += loss; avg_reconstruction_loss += auto_loss
                
            
# =============================================================================
#             if len(pad_out[pad_out == -inf]) > 0:
#                 joint_loss = torch.tensor(0.0, device=device)
#                 print("Loss broken but skipped!")
#             else:
#                 joint_loss.backward()
# =============================================================================
            joint_loss.backward()
            
            if args.clip_true ==1:
                clip = args.clip
                
                torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
                torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
            
            opt_e.step()
            opt_d.step()
            if alpha > 0:
                # print("Coming here")
                opt_a.step()

            current_loss_list.append(joint_loss.item())

        if (epoch + 1) % args.k == 1 and epoch+1<=args.decay_until:
            #alpha = alpha * 0.5
            learning_rate = learning_rate * 0.5

        if (epoch + 1)%args.k == (args.half_start+1)%args.k  and epoch + 1>args.half_start and epoch + 1<=args.half_end:
            learning_rate = learning_rate * 0.5 # weight decay
            run_log.write("\nAdjusting LR to:" + str(learning_rate) + " in epoch %d" % (epoch + 1))
            
        if (epoch + 1)>15 and epoch+1 % 10 == 0:
            learning_rate = learning_rate * 0.9 # weight decay
            run_log.write("\nAdjusting LR to:" + str(learning_rate) + " in epoch %d" % (epoch + 1))

# =============================================================================
#         if (epoch+1)%10==5 and 5<=(epoch+1)<=15:
#             #learning_rate /= 2
#             learning_rate /= 2
#             run_log.write("\nAdjusting LR to:" + str(learning_rate) + " in epoch %d" % (epoch + 1))
#             
# =============================================================================
# =============================================================================
#         if 9<(epoch+1)<=65:
#             learning_rate *= 0.985
#             run_log.write("\nAdjusting LR to:" + str(learning_rate) + " in epoch %d" % (epoch + 1))
#             
#         if 65<(epoch+1)<150:
#             learning_rate *= 0.995
#             run_log.write("\nAdjusting LR to:" + str(learning_rate) + " in epoch %d" % (epoch + 1))
#                 
#         if 150<=(epoch+1):
#             learning_rate *= 0.985
#             run_log.write("\nAdjusting LR to:" + str(learning_rate) + " in epoch %d" % (epoch + 1))
#     
# =============================================================================

# =============================================================================
#         if (epoch+1)%15==5 and 65<=(epoch+1)<=150:
#             #learning_rate /= 2
#             learning_rate *= 0.75
#             run_log.write("\nAdjusting LR to:" + str(learning_rate) + " in epoch %d" % (epoch + 1))
#             
#             
# =============================================================================
        

        # if epoch == 8:
        #     # half the learning rate
        #     learning_rate /= 5
        #     run_log.write("\nAdjusting LR to:" + str(learning_rate) + " in epoch %dd" % (epoch + 1))

        # if epoch == 38:
        #     learning_rate /= 2
        #
        # if 54 > epoch >= 36 and epoch % 6 == 0:
        #     learning_rate /= 2
        avg_translation_loss = avg_translation_loss.item()/(i+1); avg_reconstruction_loss =  avg_reconstruction_loss.item()/(i+1)
        try:
            avg_multi_loss = avg_multi_loss.item() / (i+1); avg_cos_loss = avg_cos_loss.item() / (i+1)
        except:
            pass
        avg_training_loss = np.mean(current_loss_list)
        print("Training Loss in epoch %d is:" % (epoch + 1), avg_training_loss)
        training_loss_list.append(avg_training_loss)
        if log:
            run_log.write("\nTraining Loss in epoch %d is: %s" % ((epoch + 1), str(avg_training_loss)))

# =============================================================================
#         if (epoch + 1) % save_model_at_epoch == 0:
#             torch.save(encoder.state_dict(),
#                        os.path.join(save_model_path, "tab%i_encoder_ckpt_epoch%i.pytorch" % (args.tab,epoch + 1)))
#             torch.save(decoder.state_dict(),
#                        os.path.join(save_model_path, "tab%i_decoder_ckpt_epoch%i.pytorch" % (args.tab, epoch + 1)))
#             torch.save(AUTOdecoder.state_dict(),
#                        os.path.join(save_model_path, "tab%i_auto_decoder_ckpt_epoch%i.pytorch" % (args.tab, epoch + 1)))
# 
# =============================================================================
        if tb_log:
            writer.add_scalar("Run%d_%s_%sEpoch/train_loss" % (run_num, constants.version, constants.add_info),
                              float(avg_training_loss), (epoch + 1))

        elapsed = time.time()
        print("Elapsed time for epoch: ", elapsed - start)

        if args.adaptive_alpha == 1:
            alpha = args.alpha *  avg_translation_loss / avg_reconstruction_loss
            

        val_translation_loss = 0.0; val_reconstruction_loss = 0.0
        if (epoch + 1) % validate_at_epoch == 0:
            ##### Validation #####
            # print("Validation begins")

            data_loader_test = DataLoader(
                dataset=datasets['test'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            with torch.no_grad():
                translation_pairs[int(epoch + 1)] = []
                reconstruction_pairs[int(epoch + 1)] = []
                current_loss_list = []

                for i, batch in enumerate(data_loader_test):
                    cur_batch_size = len(batch['s_length'])
                    # Sort the inputs and targets in the batch in reverse sorted order
                    sorted_lengths, sorted_idx = torch.sort(batch['s_length'], descending=True)
                    batch['sent_tok_idx'] = batch['sent_tok_idx'][sorted_idx]
                    batch['answer_tok_idx'] = batch['answer_tok_idx'][sorted_idx]
                    batch['a_length'] = batch['a_length'][sorted_idx]
                    batch['s_length'] = batch['s_length'][sorted_idx]

                    # Chop off unnecessary positions in the input and target vectors - Thus reducing sequence length
                    # and increasing efficiency as decoder will be slower with more positions in the target sequences
                    #para_sorted_lengths, para_sorted_idx = torch.sort(batch['para_length'], descending=True)
                    input_vectors = to_cuda(batch['sent_tok_idx'][:, :sorted_lengths[0]])
                    target_vectors = to_cuda(batch['answer_tok_idx'][:, :batch["a_length"].max().item()])

                    y = target_vectors.view(cur_batch_size, -1)

                    # apply to encoder
                    encoded, hidden_ec = encoder(input_vectors, sorted_lengths)
                    # encoded - Shape ([32, 15, 256]) - Since in this batch 15 is the max length

                    # stop if there is a problem
                    if math.isnan(encoded[0][0].data[0]):
                        print("encoder broken!")
                        encoder_broken.append(input_vectors)
                        sys.exit()
                        break

                    input_out = input_vectors.data.cpu().numpy()

                    # get initial input of decoder
                    decoder_in, s, w = decoder_initial(cur_batch_size)
                    SOS_token = 1
                    decoder_in = torch.LongTensor([SOS_token] * cur_batch_size)
                    
                    s1 = torch.cat([encoded[:, -1, :hidden_size], encoded[:, 0, hidden_size:]  ], 1) # b x seq x 2*hidden 
                    s1 = s1.contiguous()
                    decoder_hidden = s1
                    #print("decodedr hidden shape", decoder_hidden.shape) #should be 1x Batch x Hidden
                    
                    # We will flip because we have the batch first
                    if torch.cuda.is_available():
                        decoder_in = decoder_in.cuda()
                            # decoder_in = decoder_in.cpu()

                    # out_list to store outputs
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
        
                        if j ==0:
                            out = decoder_output.unsqueeze(1)
                        else:
                            out = torch.cat([out,decoder_output.unsqueeze(1)],dim=1)
                        decoder_in = out[:, -1].max(1)[1]  
                        out_list.append(out[:, -1].max(1)[1].squeeze().cpu().data.numpy())

                    # out has shape [32, 53, 353] - [batch_size, seq_len, vocab_size+oovs]
                    # outlist has 53 lists and has length of 32 in each of them for the above out
                    # So outlist has predictions at each time step

                    ### AUTO DECODER ###
                    AUTOdecoder_hidden, AUTOtarget_tensor, di = s1, input_vectors, 1

                    AUTOdecoder_input = torch.cuda.LongTensor([SOS_token] * cur_batch_size) if torch.cuda.is_available() \
                        else torch.LongTensor([SOS_token] * cur_batch_size)

                    # Just going to use teach_forcing in this run
                    # use_teacher_forcing = True if teacher_forcing_prob > random.random() else False
                    use_teacher_forcing = False

                    if use_teacher_forcing:
                        for di in range(AUTOtarget_tensor.shape[1]):
                            AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden,
                                                                                 encoded)
                            # print("AUTOdecoder_output: " + str(AUTOdecoder_output.shape))
                            AUTOdecoder_input = AUTOtarget_tensor[:, di]  # Teacher forcing
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

                    else:
                        for di in range(AUTOtarget_tensor.shape[1]):
                            AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden,
                                                                                 encoded)
                            topv, topi = AUTOdecoder_output.topk(1)
                            AUTOdecoder_input = topi.squeeze().detach()
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
                    # Shape of tensor_of_all_AUTOdecoded_outputs - [32, 15, 341]
                    # It's of the form [batch_size, question_seq_len, vocab_size]

                    # Loss calculation
                    # You do contiguous because .view() can only be applied when your entire tensor is in the same memory block
                    # Yes. Worry even about the low level details! Haha
                    target = target_vectors.contiguous()
                    target = target.view(-1)
                    pad_out = torch.softmax(out.view(-1, out.shape[2]), dim=1)
                    pad_out = pad_out
                    pad_out = torch.log(pad_out)

                    review_sent = input_vectors.contiguous().view(-1)
                    review_auto_out = torch.softmax(tensor_of_all_AUTOdecoded_outputs.view(-1,
                                                                             tensor_of_all_AUTOdecoded_outputs.shape[2]), dim=1)
                    review_auto_out = torch.log(review_auto_out)
                    # Shape of review_sent - [544], Shape of review_auto_out - [544, 341]
                    
                    
                    
                    
                    
                    if args.cos_only == 0:
                        translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target,
                                                                                      review_auto_out,
                                                                                      review_sent)
                    
                        val_translation_loss +=translation_loss; val_reconstruction_loss += reconstruction_loss
                        current_loss_list.append(joint_loss.item())
                        
                    else:
                        translation_loss = temp_loss_function(pad_out, target)
                        joint_loss = translation_loss; val_translation_loss += translation_loss
                        
                        
                    if args.cos_obj == 1:
                        pass

                    predicted = np.transpose(np.array(out_list))
                    translation_pairs[epoch + 1].append((predicted, target.view(cur_batch_size, -1).cpu().numpy()))
                    reconstruction_pairs[epoch + 1].append((tensor_of_all_AUTOdecoded_outputs
                                                            .topk(1)[1].squeeze(2).cpu().numpy(), input_vectors.cpu().numpy()))

                    # Can use the code below to check things as the model is running
                    # if i == 1:
                    #     for i in range(target.view(cur_batch_size, -1).shape[0]):
                    #         tar = target.view(cur_batch_size, -1)
                    #         print("target: " + str([i2w[tar[i][j].item()] for j in range(tar.shape[1])]))
                    #         temp_list = []
                    #         for j in range(predicted.shape[1]):
                    #             temp_list.append(i2w[predicted[i][j]])
                    #         # print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
                    #         print("predicted: " + str(temp_list))
                
                
                val_translation_loss = val_translation_loss / (i+1); val_reconstruction_loss = val_reconstruction_loss/(i+1)
                temp_trans_pairs = {epoch+1: translation_pairs[epoch + 1]}
                temporary_dict = {'translation_pairs':temp_trans_pairs}

                if args.result_verbose == 1:

                    #for i in range(y.shape[0]):
                    for i in range(5):
                        print("target: " +str(' '.join([i2w[y[i][j].item()] for j in range(y.shape[1]) if y[i][j].item() != 0])))
                        temp_list = []
                        for j in range(predicted.shape[1]):
                            temp_list.append(i2w[predicted[i][j]])
                        #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
                        print("predicted: " + str(' '.join(temp_list)))
                        print(' ')
                        
                    #file.write("VALIDATION ACCURACY FOR EPOCH " +  str(epoch+1) + " : " + str(acc_bowser(temporary_dict)) + "\n")
                    
                    
                    
                    #for i in range(input_vectors.shape[0]):
                    for i in range(5):
                        print("target: " +str(' '.join([i2w[input_vectors[i][j].item()] for j in range(input_vectors.shape[1]) if input_vectors[i][j].item() != 0])))
                        temp_list = []
                        for j in range(tensor_of_all_AUTOdecoded_outputs.topk(1)[1].squeeze(2).cpu().numpy().shape[1]):
                            temp_list.append(i2w[tensor_of_all_AUTOdecoded_outputs.topk(1)[1].cpu().numpy().squeeze(2)[i][j]])
                        #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
                        print("predicted: " + str(' '.join(temp_list)))
                        print(' ')
                    
                print("new alpha: ", alpha)
                if args.cos_only == 0 and args.cos_obj == 1:
                    print("Cos Loss: ", avg_cos_loss)
                    print("Muli Loss: ", avg_multi_loss)
                
                
                print("Training translation loss: ", avg_translation_loss)
                print("Training reconstruction loss: ", avg_reconstruction_loss)
                
                print("Validation translation loss: ", val_translation_loss.item())
                print("Validation reconstruction loss: ", val_reconstruction_loss)
                
                avg_val_loss = np.mean(current_loss_list)
                print("Validation Loss in epoch %d is:" % (epoch + 1), avg_val_loss)
                if log:
                    run_log.write("\nValidation Loss in epoch %d is: %s\n" % ((epoch + 1), str(avg_val_loss)))
                test_loss_list.append(avg_val_loss)
                
                
                print("VALIDATION ACCURACY FOR EPOCH ", epoch+1, " : ", acc_bowser(temporary_dict))
                temp_trans_pairs = {epoch+1: reconstruction_pairs[epoch+1]}
                temporary_dict = {'translation_pairs':temp_trans_pairs}
                print("Reconstruction ACCURACY FOR EPOCH ", epoch+1, " : ", acc_bowser(temporary_dict))
                
                if tb_log:
                    writer.add_scalar(
                        "Run%d_%s_%sEpoch/validation_loss" % (run_num, constants.version, constants.add_info),
                        float(avg_val_loss), (epoch + 1))

                if args.save_every_epoch is 1 and epoch>= args.save_from:
                    if (epoch + 1) % save_pkl_files_at_epoch == 0:
                        final_dict = {'translation_pairs': translation_pairs, 'reconstruction_pairs': reconstruction_pairs}
                        pkl_filname = os.path.join(pkl_dir, "validation_pairs_epoch%d" % epoch)
                        with open(pkl_filname, "wb") as f:
                            cPickle.dump(final_dict, f)
                            
                        torch.save(encoder.state_dict(),
                                   os.path.join(save_model_path, "encoder_ckpt_epoch%i.pytorch" % epoch))
                        torch.save(decoder.state_dict(),
                                   os.path.join(save_model_path, "decoder_ckpt_epoch%i.pytorch" % epoch))
                        torch.save(AUTOdecoder.state_dict(),
                                   os.path.join(save_model_path, "auto_decoder_ckpt_epoch%i.pytorch" % epoch))
            
    #file.close()

                        # print("Done with epoch -", epoch+1)
# =============================================================================
# 
#     # Save the training and test lists here. And also translation pairs just to be sure that this happens at the end
#     # Last thing to be done
#     final_dict = {'translation_pairs': translation_pairs, 'reconstruction_pairs': reconstruction_pairs,
#                   'training_loss': training_loss_list, 'validation_loss': test_loss_list}
#     pkl_filname = os.path.join(pkl_dir, "validation_pairs_%s_run%d_final" % (constants.version, run_num))
#     with open(pkl_filname, "wb") as f:
#         cPickle.dump(final_dict, f)
#     if log:
#         run_log.close()
# 
#     # Also save the model
#     torch.save(encoder.state_dict(), os.path.join(save_model_path, "encoder_ckpt_epoch%i.pytorch" % num_epochs))
#     torch.save(decoder.state_dict(), os.path.join(save_model_path, "decoder_ckpt_epoch%i.pytorch" % num_epochs))
#     torch.save(AUTOdecoder.state_dict(),
#                os.path.join(save_model_path, "auto_decoder_ckpt_epoch%i.pytorch" % num_epochs))
# 
# =============================================================================
                        
if __name__ == '__main__':
    torch.cuda.set_device(args.cuda)
    #num_epochs = 150
    train(args)

"""
Notes:
check variables are for putting breakpoints for debugging

Run numbers:
1 - version 1, alpha = 1.0, running on mkscan server - parameters in the log file
"""








