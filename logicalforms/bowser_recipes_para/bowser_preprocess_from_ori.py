import json
import os
import nltk
#from bert_embedding import BertEmbedding
from bert_serving.client import BertClient
import pickle
import _pickle as cPickle
import random
import math
import sys
import numpy as np
import copy
import random


#sys.path.append("/data/scratch-oc40/symin95/packages/bert-embedidng")
#from bert_embedding import BertEmbedding
#Night's IP address


def process_txt_and_gen_vocab_json(out_train_file, out_test_file, domain='rec', BERT_Word=False, BERT_Sent= True, ineq=False, recomb=False):
    
    if recomb:
        train_dict_name = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+ '/train_dict_recomb.p'
        test_dict_name = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+ '/test_dict_recomb.p'

    else:
        train_dict_name = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+ '/train_dict.p'
        test_dict_name = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+ '/test_dict.p'
    
    train_dict = cPickle.load(open(train_dict_name, 'rb'))
    test_dict = cPickle.load(open(test_dict_name, 'rb'))
    
    word2idx = dict()
    idx2word = dict()
    special_tokens = ['PAD_token', '[CLS]', '[SEP]', '[UNK]','GO_token']
    for st in special_tokens:
        idx2word[len(word2idx)] = st
        word2idx[st] = len(word2idx)
    
    train_data=[]
    test_data=[]
    bert_list = [nltk.word_tokenize(idx_dict['utterance']) for train_idx, idx_dict in train_dict.items()  ] + [nltk.word_tokenize(idx_dict['utterance']) for i, idx_dict in test_dict.items()]
    ans_list = [nltk.word_tokenize(idx_dict['answer']) for train_idx, idx_dict in train_dict.items()  ] + [nltk.word_tokenize(idx_dict['answer']) for i, idx_dict in test_dict.items()]
    new_ans_list = []
    if ineq:
        for tokenized in ans_list:
            RID= 0
            new_tokenized = []
            for j, tok in enumerate(tokenized):
                if len(tokenized)> j+1 and ((tokenized[j],  tokenized[j+1]) == ('<', '=') or (tokenized[j],  tokenized[j+1]) == ('>', '=') or (tokenized[j],  tokenized[j+1]) == ('!', '=')):
                    new_tokenized.append(''.join([tokenized[j],  tokenized[j+1]]))
                elif j>=1 and ((tokenized[j-1],  tokenized[j]) == ('<', '=') or (tokenized[j-1],  tokenized[j]) == ('>', '=') or (tokenized[j-1],  tokenized[j]) == ('!', '=')):
                    new_tokenized.append('RID')
                    RID = 1
                else:
                    new_tokenized.append(tok)
            if RID ==1:
                new_tokenized.remove('RID')
           
            new_ans_list.append(new_tokenized)
            
    else:
        new_ans_list = copy.deepcopy(ans_list)
                    
                    
                
    
    # Make vocab    
    if BERT_Word or BERT_Sent:
    #BERT_Word and Sent's Command should BOTH be bert-serving-start -model_dir  uncased_L-24_H-1024_A-16 -num_worker=4 -show_tokens_to_client -pooling_strategy NONE
    #bc = BertClient(ip='128.30.45.38', check_length = False)
        bc = BertClient(ip='128.30.192.12', check_length = False)
        print("bert start!")
        bert_encode = bc.encode(bert_list, is_tokenized = True, show_tokens = True)
        print("bert end!")
        
        bert_embeddings_final = [0,0,0,0,0]
        #First add for Pad token, CLS, SEP, UNK
        #Pad token
        bert_embeddings_final[0] = [random.random()] * 1024
        bert_embeddings_final[1] = bert_encode[0][0][0].tolist()
        bert_embeddings_final[2] = bert_encode[0][0][len(bert_encode[1][0])-1].tolist()
        bert_embeddings_final[4] = [random.random()] * 1024
                    
        
    
    #vocabulary index 따라서 
    for train_idx, idx_dict in train_dict.items():
        cur_dict = dict()
        sent = idx_dict['utterance']
        para_group = idx_dict['para_group']
        answer = idx_dict['answer']
        glb_idx = idx_dict['glb_idx']
        
        sent_tokens = nltk.word_tokenize(sent)
        cur_dict["sent_tok"] = sent_tokens
        if BERT_Word:
            bert_tokenized = bert_encode[1][train_idx]
            bert_embed_current_line = bert_encode[0][train_idx]
        for j, st in enumerate(sent_tokens):
            if st not in word2idx:
                idx2word[len(word2idx)] = st
                word2idx[st] = len(word2idx)
                if BERT_Word:
                    if len(bert_tokenized) > j+1 and bert_tokenized[j+1] == st:
                        bert_embeddings_final.append(bert_embed_current_line[j+1])
                    elif len(bert_tokenized) > j+1 and bert_tokenized[j+1] == '[UNK]':
                        bert_embeddings_final[3] = bert_embed_current_line[j+1]
                        bert_embeddings_final.append(bert_embed_current_line[j+1])
                    else:
                        bert_embeddings_final.append([random.random()] * 1024)
                
        #Take care of Bert Embeddings here 
        
        cur_dict["sent_tok_idx"] = [word2idx[st] for st in sent_tokens]
        cur_dict["para_group"] = para_group
        cur_dict["global_idx"] = glb_idx
        if BERT_Sent:
            cur_dict["BERT_sent_enc"] = bert_embed_current_line
        
        train_data.append(cur_dict)


    for test_idx, idx_dict in test_dict.items():
        cur_dict = dict()
        sent = idx_dict['utterance']
        para_group = idx_dict['para_group']
        answer = idx_dict['answer']
        glb_idx = idx_dict['glb_idx']
        
        sent_tokens = nltk.word_tokenize(sent)
        cur_dict["sent_tok"] = sent_tokens
        if BERT_Word:
            bert_tokenized = bert_encode[1][test_idx + len(train_dict)]
            bert_embed_current_line = bert_encode[0][test_idx + len(train_dict)]
        for j, st in enumerate(sent_tokens):
            if st not in word2idx:
                idx2word[len(word2idx)] = st
                word2idx[st] = len(word2idx)
                if BERT_Word:
                    if bert_tokenized[j+1] == st:
                        bert_embeddings_final.append(bert_embed_current_line[j+1])
                    elif bert_tokenized[j+1] == '[UNK]':
                        bert_embeddings_final[3] = bert_embed_current_line[j+1]
                        bert_embeddings_final.append(bert_embed_current_line[j+1])
                    else:
                        bert_embeddings_final.append([random.random()] * 1024)
                    
        cur_dict["sent_tok_idx"] = [word2idx[st] for st in sent_tokens]
        cur_dict["para_group"] = para_group
        cur_dict["global_idx"] = glb_idx
        if BERT_Sent:
            cur_dict["BERT_sent_enc"] = bert_embed_current_line

        test_data.append(cur_dict)
        
    for train_idx, cur_dict in enumerate(train_data):
        cur_dict['answer'] = answer
        cur_dict["answer_tokens"] =  new_ans_list[train_idx]
        for st in cur_dict["answer_tokens"]:
            if st not in word2idx:
                idx2word[len(word2idx)] = st
                word2idx[st] = len(word2idx)
                if BERT_Word:
                    bert_embeddings_final.append([random.random()] * 1024)
        cur_dict["answer_tok_idx"]   = [word2idx[st] for st in cur_dict["answer_tokens"] ]  
    
    for test_idx, cur_dict in enumerate(test_data):
        cur_dict['answer'] = answer
        cur_dict["answer_tokens"] =  new_ans_list[len(train_dict) + test_idx]
        for st in cur_dict["answer_tokens"]:
            if st not in word2idx:
                idx2word[len(word2idx)] = st
                word2idx[st] = len(word2idx)
                if BERT_Word:
                    bert_embeddings_final.append([random.random()] * 1024)
        cur_dict["answer_tok_idx"]   = [word2idx[st] for st in cur_dict["answer_tokens"] ]   
            
        
        
    print(len(idx2word)) 
    if BERT_Word:
        print(len(bert_embeddings_final))
        print("bert embeddings shape: ", np.array(bert_embeddings_final).shape)
        res = {"idx2word": idx2word, "word2idx": word2idx, 'bert_embeddings':np.array(bert_embeddings_final)}
                
    
    else:
        res = {"idx2word": idx2word, "word2idx": word2idx}
        
    if ineq:
        vocab_file = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+ '/vocab_ineq.p'
    if recomb:
        vocab_file = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+ '/vocab_recomb.p'
    else:
        vocab_file = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+ '/vocab.p'

    with open(vocab_file, "wb") as f:
        pickle.dump(res, f)
    with open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+ '/' + out_train_file, "w") as f:
        json.dump(train_data, f)
    with open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+ '/' +out_test_file, "w") as f:
        json.dump(test_data, f)


def add_BERT_emb(filename, domain='rec'):
    filename =  '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+'/'+filename

    with open(filename, "r") as f:
        qa_list = json.load(f)
        
    bert_list = [qa["sent_tok"] for qa in qa_list]
    #Ber serving  command should be bert-serving-start -model_dir  uncased_L-24_H-1024_A-16 -num_worker=4 
    bc = BertClient(ip='128.30.192.12', check_length = False)
    print("bert start!")
    bert_encode = bc.encode(bert_list, is_tokenized = True)
    print("bert end!")
    for i, qa in enumerate(qa_list):
        qa["BERT_sent_emb"] = bert_encode[i]
    
    with open(filename, 'w') as f:
        json.dump(qa_list, f)

def add_padding(filename, out_filename, domain='rec', recomb = False):
    if recomb:
        vocab_file = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+ '/vocab_recomb.p'
    
    else:
        vocab_file = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+ '/vocab.p'
    filename =  '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/'+domain+'/'+filename
    out_filename = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain +'/'+out_filename

    with open(filename, "r") as f:
        qa_list = json.load(f)
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)

    word2idx = vocab["word2idx"]

    # first add SOS and EOS tokens
    for qa in qa_list:
        qa["sent_tok"] = ['[CLS]'] + qa["sent_tok"] + ['[SEP]']
        qa["answer_tokens"] = ['[CLS]'] + qa["answer_tokens"] +  ['[SEP]']
        qa["sent_tok_idx"] = [word2idx['[CLS]']] + qa["sent_tok_idx"] + [word2idx['[SEP]']]
        qa["answer_tok_idx"] = [word2idx['[CLS]']] + qa["answer_tok_idx"] + [word2idx['[SEP]']]

    max_q_length = 0
    max_a_length = 0
    for qa in qa_list:
        max_q_length = max(max_q_length, len(qa["sent_tok"]))
        max_a_length = max(max_a_length, len(qa["answer_tokens"]))
    print(max_q_length, max_a_length)

    # Now we will modify the objects
    for i, qa in enumerate(qa_list):
        q_length = len(qa["sent_tok"])
        a_length = len(qa["answer_tokens"])

        qa["sent_tok"].extend(['PAD_token'] * (max_q_length - q_length))
        qa["answer_tokens"].extend(['PAD_token'] * (max_a_length - a_length))

        qa["sent_tok_idx"].extend([word2idx['PAD_token']] * (max_q_length - q_length))
        qa["answer_tok_idx"].extend([word2idx['PAD_token']] * (max_a_length - a_length))

        qa["s_length"] = q_length
        qa["a_length"] = a_length
        qa['cur_dict_idx'] = i

    with open(out_filename, 'w') as f:
        json.dump(qa_list, f)
    
    
   
#shuffle k is take k out for every paraphrase group with more than k paraphrases.    
def shuffle_k_split_n_preprocess(shuffle_k, split_n, vocab_file, in_train_dcs,out_train_file, domain='rec'):
    para_group_dict = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain+ '/para_group_dict.p','rb'))
    global_idx2_utterance = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain+ '/global_idx2_utterance_dict.p', 'rb')) 
    global_idx_2_shuffle0_train_idx_dict = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain + '/global_idx_2_shuffle0_train_idx_dict.p','rb'))
    global_idx_2_shuffle0_test_idx_dict = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain +'/global_idx_2_shuffle0_test_idx_dict.p','rb'))

    
    
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)

    with open(in_train_dcs, "r") as f:
        train_dcs = []
        for line in f:
            train_dcs.append(line.strip())

    word2idx = vocab["word2idx"]
    
    
    train_global_idxes = []
    for para_group, glob_idx_list in para_group_dict.items():
        #split to train an test 
                
        if len(glob_idx_list)> shuffle_k:
            random.seed(split_n*(2**para_group))
            excluded_k_idxes = random.sample(range(len(glob_idx_list)), shuffle_k)

        else:
            excluded_k_idxes = []
        shuffled_glob_idx_list = [(para_group,val) for i, val in enumerate(glob_idx_list) if not(i in excluded_k_idxes)]
        train_global_idxes += shuffled_glob_idx_list  
    #Now that we have idxes, make list of dictionary
    train_data=[]
    #train_data_list2global_idx ={}
    for i, (para_group, global_idx) in enumerate(train_global_idxes):
        #train_data_list2global_idx[i] = global_idx
        cur_dict = {}
        sent = global_idx2_utterance[global_idx]
        cur_dict["sent"] = sent
        
        sent_tokens = nltk.word_tokenize(sent)
        cur_dict["sent_tok"] = sent_tokens
        cur_dict["sent_tok_idx"] = [word2idx[st] for st in sent_tokens]
        cur_dict["para_group"] = para_group
        cur_dict["global_idx"] = global_idx

        train_data.append(cur_dict)
        
        train_idx = global_idx_2_shuffle0_train_idx_dict[global_idx]
        lf = train_dcs[train_idx]
        cur_dict["answer"] = lf
        cur_dict["answer_tokens"] = nltk.word_tokenize(lf)
        cur_dict["answer_tok_idx"] = [word2idx[st] for st in cur_dict["answer_tokens"]]  
    
    #global_idx2train_data_list ={v:k for k, v in train_data_list2global_idx.items()}     
    #Do the same for lf's
# =============================================================================
#     for train_idx, lf in enumerate(train_dcs):
#         global_idx = shuffle0_train_2_global_idx_dict[train_idx]
#         if global_idx in global_idx2train_data_list:
#             i = global_idx2train_data_list[global_idx]            
#             cur_dict = train_data[i]
#             cur_dict["answer"] = lf
#             cur_dict["answer_tokens"] = nltk.word_tokenize(lf)
#             cur_dict["answer_tok_idx"] = [word2idx[st] for st in cur_dict["answer_tokens"]]  
#         else:
#             pass
#     
#     weird_list = []
#     for i, cur_dict in enumerate(train_data):
#         try:
#             a = cur_dict["answer_tokens"]
#         except:
#             weird_list.append(i)
#     print("weird: ", weird_list)
# =============================================================================
    
    with open(out_train_file, "w") as f:
        json.dump(train_data, f)


def shuffle_k_perc_split_n_preprocess(shuffle_k_perc, split_n, vocab_file, out_train_file, out_test_file, domain = 'rec'):    
    global_idx_dict = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain+ '/global_idx_dict.p','rb'))
    para_group_dict = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain + '/para_group_dict.p','rb'))

    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    word2idx = vocab["word2idx"]
    
    #split to train and test 
    train_idxes = []; test_idxes = []
    train_idxes_for_para = {}
    for para_group, glb_index_list in para_group_dict.items():
        total = len(glb_index_list)
        random.seed(split_n * 2**para_group)
        if random.random() > 0.5:
            train_total = math.floor((1-shuffle_k_perc)*total)
        else:
            train_total = math.ceil((1-shuffle_k_perc)*total)
            
        random.seed(split_n * 2**para_group)
        trains = random.sample(glb_index_list, train_total) 
        train_idxes += trains; train_idxes_for_para[para_group] = trains
        test_idxes += [idx for idx in glb_index_list if not(idx in trains)]

    
    #Make cur dict
    train_data = []
    for cur_dict_idx, global_idx in enumerate(train_idxes):
        cur_dict = {}
        para_group = global_idx_dict['para_group'][global_idx]
        sent = global_idx_dict['utterance'][global_idx]
        cur_dict["sent"] = sent
        
        sent_tokens = nltk.word_tokenize(sent)
        cur_dict["sent_tok"] = sent_tokens
        cur_dict["sent_tok_idx"] = [word2idx[st] for st in sent_tokens]
        cur_dict["para_group"] = para_group
        cur_dict["global_idx"] = global_idx
        
        cur_dict["answer"] = global_idx_dict['lf'][global_idx]
        cur_dict["answer_tokens"] = nltk.word_tokenize(cur_dict["answer"])
        cur_dict["answer_tok_idx"] = [word2idx[st] for st in cur_dict["answer_tokens"]]  
        cur_dict['cur_dict_idx'] = cur_dict_idx
        train_data.append(cur_dict)
        
        
    test_data = []
    for cur_dict_idx, global_idx in enumerate(test_idxes):
        cur_dict = {}
        para_group = global_idx_dict['para_group'][global_idx]
        sent = global_idx_dict['utterance'][global_idx]
        cur_dict["sent"] = sent
        
        sent_tokens = nltk.word_tokenize(sent)
        cur_dict["sent_tok"] = sent_tokens
        cur_dict["sent_tok_idx"] = [word2idx[st] for st in sent_tokens]
        cur_dict["para_group"] = para_group
        cur_dict["global_idx"] = global_idx
        
        cur_dict["answer"] = global_idx_dict['lf'][global_idx]
        cur_dict["answer_tokens"] = nltk.word_tokenize(cur_dict["answer"])
        cur_dict["answer_tok_idx"] = [word2idx[st] for st in cur_dict["answer_tokens"]]  
        test_data.append(cur_dict)
    
    with open(out_train_file, "w") as f:
        json.dump(train_data, f)
    with open(out_test_file, "w") as f:
        json.dump(test_data, f)
    
        
def train_para_reduce(num_para_train_reduce, in_train_file, out_train_file,  out_para_dict_filename=None, seed = 1, domain = 'rec'):
    #make train_sampled
    #for this, don't do % but reduce by number
    global_idx_dict = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain + '/global_idx_dict.p','rb'))
    with open(in_train_file, "r") as f:
        qa_list = json.load(f)
            
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    word2idx = vocab["word2idx"]
    
    train_idxes_for_para = {}
    for i , cur_dict in enumerate(qa_list):
        global_idx = cur_dict['global_idx']; para_group = cur_dict['para_group']
        if not(para_group in train_idxes_for_para):
            train_idxes_for_para[para_group] = []
        train_idxes_for_para[para_group].append(global_idx)
    
    train_sampled_idxes = []
    large_count =0; small_count=0
    for para_group, glb_index_list in train_idxes_for_para.items():
        random.seed(num_para_train_reduce * para_group *seed)
        if len(glb_index_list)>num_para_train_reduce:
            large_count +=1 
            #trains = random.sample(glb_index_list, num_para_train_reduce-1)
            trains = random.sample(glb_index_list, len(glb_index_list)-num_para_train_reduce)
        else:
            small_count +=1
            trains = random.sample(glb_index_list,1)
        train_sampled_idxes += trains
    print("large count: ", large_count, " small count: ", small_count)
    print("length", len(train_sampled_idxes))

        
    train_reduced_para_data = []
    for cur_dict_idx, global_idx in enumerate(train_sampled_idxes):
        cur_dict = {}
        para_group = global_idx_dict['para_group'][global_idx]
        sent = global_idx_dict['utterance'][global_idx]
        cur_dict["sent"] = sent
        
        sent_tokens = nltk.word_tokenize(sent)
        cur_dict["sent_tok"] = sent_tokens
        cur_dict["sent_tok_idx"] = [word2idx[st] for st in sent_tokens]
        cur_dict["para_group"] = para_group
        cur_dict["global_idx"] = global_idx
        
        cur_dict["answer"] = global_idx_dict['lf'][global_idx]
        cur_dict["answer_tokens"] = nltk.word_tokenize(cur_dict["answer"])
        cur_dict["answer_tok_idx"] = [word2idx[st] for st in cur_dict["answer_tokens"]]  
        train_reduced_para_data.append(cur_dict)
    
    with open(out_train_file, "w") as f:
        json.dump(train_reduced_para_data , f)
        
    train_idxes_for_para_save = {}
    for i , cur_dict in enumerate(qa_list):
        para_group = cur_dict['para_group']
        if not(para_group in train_idxes_for_para_save):
            train_idxes_for_para_save[para_group] = []
        train_idxes_for_para_save[para_group].append(i)
    
    final_para_save = {}
    for i , cur_dict in enumerate(qa_list):
        para_group = cur_dict['para_group']
        final_para_save[i] = [k for k in train_idxes_for_para_save[para_group] if k!= i]
    
    if out_para_dict_filename != None:
        cPickle.dump(final_para_save, open( out_para_dict_filename, 'wb'))



def random_reduce_percent(reduce_perc, in_train_file, out_train_file, seed, domain = 'rec'):
    global_idx_dict = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain + '/global_idx_dict.p','rb'))
    with open(in_train_file, "r") as f:
        qa_list = json.load(f)
            
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    word2idx = vocab["word2idx"]    
    
    train_idxes_for_para = {}
    for i , cur_dict in enumerate(qa_list):
        global_idx = cur_dict['global_idx']; para_group = cur_dict['para_group']
        if not(para_group in train_idxes_for_para):
            train_idxes_for_para[para_group] = []
        train_idxes_for_para[para_group].append(global_idx)
                
    random.seed(reduce_perc * 100 *seed)
    train_sampled_idxes = random.sample([cur_dict['global_idx'] for cur_dict in qa_list], int(len(qa_list)*reduce_perc))
    
    train_reduced_para_data = []
    for cur_dict_idx, global_idx in enumerate(train_sampled_idxes):
        cur_dict = {}
        para_group = global_idx_dict['para_group'][global_idx]
        sent = global_idx_dict['utterance'][global_idx]
        cur_dict["sent"] = sent
        
        sent_tokens = nltk.word_tokenize(sent)
        cur_dict["sent_tok"] = sent_tokens
        cur_dict["sent_tok_idx"] = [word2idx[st] for st in sent_tokens]
        cur_dict["para_group"] = para_group
        cur_dict["global_idx"] = global_idx
        
        cur_dict["answer"] = global_idx_dict['lf'][global_idx]
        cur_dict["answer_tokens"] = nltk.word_tokenize(cur_dict["answer"])
        cur_dict["answer_tok_idx"] = [word2idx[st] for st in cur_dict["answer_tokens"]]  
        train_reduced_para_data.append(cur_dict)
    
    with open(out_train_file, "w") as f:
        json.dump(train_reduced_para_data , f)
        
    train_idxes_for_para_save = {}
    for i , cur_dict in enumerate(qa_list):
        para_group = cur_dict['para_group']
        if not(para_group in train_idxes_for_para_save):
            train_idxes_for_para_save[para_group] = []
        train_idxes_for_para_save[para_group].append(i)
        
    final_para_save = {}
    for i , cur_dict in enumerate(qa_list):
        para_group = cur_dict['para_group']
        final_para_save[i] = [k for k in train_idxes_for_para_save[para_group] if k!= i]
        
    



def add_para_and_pad_for_training(filename, vocab_file, out_filename, domain = 'rec'):

    with open(filename, "r") as f:
        qa_list = json.load(f)
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    
    global_idx_dict = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain + '/global_idx_dict.p','rb'))
    #training_paras = {qa["para_group"] for qa in qa_list}
    
    train_idxes_for_para = {}
    for i , cur_dict in enumerate(qa_list):
        global_idx = cur_dict['global_idx']; para_group = cur_dict['para_group']
        if not(para_group in train_idxes_for_para):
            train_idxes_for_para[para_group] = []
        train_idxes_for_para[para_group].append(global_idx)
    
    
    para_group_dict = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain + '/para_group_dict.p','rb'))
    global_idx2_utterance = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain +'/global_idx2_utterance_dict.p', 'rb')) 
    shuffle0_test_global_idxes = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain + '/global_idx_2_shuffle0_test_idx_dict.p','rb'))

    word2idx = vocab["word2idx"]

    
    # first add paraphrase info 
    for i, qa in enumerate(qa_list):
        para_group = qa['para_group']
        if i%100==0:
            print(i)
        #make sure nothing in test 
        list_of_paras = [pidx for pidx in para_group_dict[qa["para_group"]] if (pidx!=qa['global_idx'] and pidx in train_idxes_for_para[para_group])]
        if list_of_paras == []:
            list_of_paras = [qa['global_idx']]
        random.seed(i)
        #print("para group", qa["para_group"])
        #print("list of paras: ", list_of_paras)
        
        sampled_para_global_idx = random.sample(list_of_paras, 1)[0]
        
        qa["para_global_idx"] = sampled_para_global_idx
        para_str = global_idx2_utterance[sampled_para_global_idx]
        
        para_tokens = nltk.word_tokenize(para_str)
        qa["para_tok"] = para_tokens
        res_tok_idx = []
        for st in para_tokens:
            res_tok_idx.append(word2idx[st])
        qa["para_tok_idx"] = res_tok_idx

        
        qa["para_tok"] = ["SOS_token"] + qa["para_tok"]+ ["EOS_token"]
        qa["para_tok_idx"] = [word2idx["SOS_token"]] + qa["para_tok_idx"] + [word2idx["EOS_token"]]
        qa["para_len"] = len(qa["para_tok"])
        
        
    #Now add padding 
    max_para_len = max([qa["para_len"] for qa in qa_list])
    for i, qa in  enumerate(qa_list):
        qa["para_tok"] = qa["para_tok"]  + ['PAD_token'] * (max_para_len - len(qa["para_tok"]))
        qa["para_tok_idx"] = qa["para_tok_idx"] + [word2idx['PAD_token']] * (max_para_len - len(qa["para_tok_idx"]))
      
    for i, qa in enumerate(qa_list):
        qa['para_list'] = [j for j, another_qa in enumerate(qa_list) if another_qa['para_group'] == qa['para_group']]
        
        
    #Save 
    with open(out_filename, 'w') as f:
        json.dump(qa_list, f)
        
    
    
   


# =============================================================================
# rec_train_en = "data/overnight_rec/recipes.train.en"
# rec_train_dcs = "data/overnight_rec/recipes.train.dcs"
# rec_test_en = "data/overnight_rec/recipes.test.en"
# rec_test_dcs = "data/overnight_rec/recipes.test.dcs"
# train_file_p1 = "data/para_preprocessed/recipes_train_v1_p1.json"
# test_file_p1 = "data/para_preprocessed/recipes_test_v1_p1.json"
# 
# train_file_v1 = "data/para_preprocessed/recipes_train_v1.json"
# test_file_v1 = "data/para_preprocessed/recipes_test_v1.json"
# 
# vocab_file = "data/processed/vocab_recipes_v1.pkl"
# 
# os.chdir('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para')
# ###Normal shuffle 
# process_txt_and_gen_vocab_json(rec_train_en, rec_test_en, rec_train_dcs, rec_test_dcs, vocab_file, train_file_p1, test_file_p1)
# add_padding(train_file_p1, vocab_file, train_file_v1)   # 20, 60
# add_padding(test_file_p1, vocab_file, test_file_v1)     # 20, 60
# add_para_and_pad_for_training(train_file_v1, vocab_file, train_file_v1)
# 
# 
# ###Shuffle 2 
# #shuffle_k_perc_split_n_preprocess(shuffle_k_perc=0.25, split_n=1, vocab_file="data/processed/vocab_recipes_v1.pkl", out_train_file='data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_train_p.json', out_test_file='data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_test_p.json')
# #Todo: only did until reduce para 1, did not do reduce para 1 yet
# train_para_reduce(num_para_train_reduce=5, in_train_file='data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_train_p.json', out_train_file='data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_reduced_para=5_train_p.json')
# add_padding('data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_reduced_para=5_train_p.json', vocab_file, 'data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_reduced_para=5_train.json')
# #add_padding('data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_train_p.json', vocab_file, 'data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_train.json')
# #add_padding('data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_test_p.json', vocab_file, 'data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_test.json')
# add_para_and_pad_for_training('data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_reduced_para=5_train.json', vocab_file, 'data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_reduced_para=5_train.json')
# #add_para_and_pad_for_training('data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_train.json', vocab_file, 'data/para_preprocessed/shuffle_k_perc=0.25_split_n=1_train.json')
# 
# 
# #Official split과 same 비율 (0.5)로 balanced split만들기
# shuffle_k_perc_split_n_preprocess(shuffle_k_perc=0.5, split_n=1, vocab_file="data/processed/vocab_recipes_v1.pkl", out_train_file='data/para_preprocessed/shuffle_k_perc=0.5_split_n=1_train_p.json', out_test_file='data/para_preprocessed/shuffle_k_perc=0.5_split_n=1_test_p.json')
# add_padding('data/para_preprocessed/shuffle_k_perc=0.5_split_n=1_train_p.json', vocab_file, 'data/para_preprocessed/shuffle_k_perc=0.5_split_n=1_train.json')
# add_padding('data/para_preprocessed/shuffle_k_perc=0.5_split_n=1_test_p.json', vocab_file, 'data/para_preprocessed/shuffle_k_perc=0.5_split_n=1_test.json')
# add_para_and_pad_for_training('data/para_preprocessed/shuffle_k_perc=0.5_split_n=1_train.json', vocab_file, 'data/para_preprocessed/shuffle_k_perc=0.5_split_n=1_train.json')
# 
# 
# 
# shuffle_k_perc_split_n_preprocess(shuffle_k_perc=0.15, split_n=1, vocab_file="data/processed/vocab_recipes_v1.pkl", out_train_file='data/para_preprocessed/shuffle_k_perc=0.15_split_n=1_train_p.json', out_test_file='data/para_preprocessed/shuffle_k_perc=0.15_split_n=1_test_p.json')
# add_padding('data/para_preprocessed/shuffle_k_perc=0.15_split_n=1_train_p.json', vocab_file, 'data/para_preprocessed/shuffle_k_perc=0.15_split_n=1_train.json')
# add_padding('data/para_preprocessed/shuffle_k_perc=0.15_split_n=1_test_p.json', vocab_file, 'data/para_preprocessed/shuffle_k_perc=0.15_split_n=1_test.json')
# add_para_and_pad_for_training('data/para_preprocessed/shuffle_k_perc=0.15_split_n=1_train.json', vocab_file, 'data/para_preprocessed/shuffle_k_perc=0.15_split_n=1_train.json')
# 
# 
# 
# #Just down-sample from the official split 
# train_para_reduce(num_para_train_reduce=2, in_train_file=train_file_p1, out_train_file='data/para_preprocessed/recipes_train_official_reduced=2_p.json')
# train_para_reduce(num_para_train_reduce=3, in_train_file=train_file_p1, out_train_file='data/para_preprocessed/recipes_train_official_reduced=3_p.json')
# train_para_reduce(num_para_train_reduce=4, in_train_file=train_file_p1, out_train_file='data/para_preprocessed/recipes_train_official_reduced=4_p.json')
# add_padding('data/para_preprocessed/recipes_train_official_reduced=2_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=2.json')
# add_padding('data/para_preprocessed/recipes_train_official_reduced=3_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=3.json')
# add_padding('data/para_preprocessed/recipes_train_official_reduced=4_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=4.json')
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_reduced=2.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=2.json')
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_reduced=3.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=3.json')
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_reduced=4.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=4.json')
# 
# #2nd, 3rd, 4th, 5th, 6th seed 
# train_para_reduce(num_para_train_reduce=4, in_train_file=train_file_p1, out_train_file='data/para_preprocessed/recipes_train_official_reduced=4_seed2_p.json', seed = 2)
# add_padding('data/para_preprocessed/recipes_train_official_reduced=4_seed2_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=4_seed2.json')
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_reduced=4_seed2.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=4_seed2.json')
# 
# train_para_reduce(num_para_train_reduce=4, in_train_file=train_file_p1, out_train_file='data/para_preprocessed/recipes_train_official_reduced=4_seed3_p.json', seed = 3)
# add_padding('data/para_preprocessed/recipes_train_official_reduced=4_seed3_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=4_seed3.json')
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_reduced=4_seed3.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=4_seed3.json')
# 
# train_para_reduce(num_para_train_reduce=4, in_train_file=train_file_p1, out_train_file='data/para_preprocessed/recipes_train_official_reduced=4_seed4_p.json', seed = 4)
# add_padding('data/para_preprocessed/recipes_train_official_reduced=4_seed4_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=4_seed4.json')
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_reduced=4_seed4.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=4_seed4.json')
# 
# train_para_reduce(num_para_train_reduce=4, in_train_file=train_file_p1, out_train_file='data/para_preprocessed/recipes_train_official_reduced=4_seed5_p.json', seed = 5)
# add_padding('data/para_preprocessed/recipes_train_official_reduced=4_seed5_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=4_seed5.json')
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_reduced=4_seed5.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=4_seed5.json')
# 
# train_para_reduce(num_para_train_reduce=4, in_train_file=train_file_p1, out_train_file='data/para_preprocessed/recipes_train_official_reduced=4_seed6_p.json', seed = 6)
# add_padding('data/para_preprocessed/recipes_train_official_reduced=4_seed6_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=4_seed6.json')
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_reduced=4_seed6.json', vocab_file, 'data/para_preprocessed/recipes_train_official_reduced=4_seed6.json')
# 
# 
# 
# #Create random reduce 
# random_reduce_percent(0.5, train_file_p1, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed1_p.json', 1)
# random_reduce_percent(0.5, train_file_p1, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed2_p.json', 2)
# random_reduce_percent(0.5, train_file_p1, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed3_p.json', 3)
# random_reduce_percent(0.5, train_file_p1, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed4_p.json', 4)
# random_reduce_percent(0.5, train_file_p1, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed5_p.json', 5)
# 
# add_padding('data/para_preprocessed/recipes_train_official_random_reduced0.5_seed1_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed1.json')
# add_padding('data/para_preprocessed/recipes_train_official_random_reduced0.5_seed2_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed2.json')
# add_padding('data/para_preprocessed/recipes_train_official_random_reduced0.5_seed3_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed3.json')
# add_padding('data/para_preprocessed/recipes_train_official_random_reduced0.5_seed4_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed4.json')
# add_padding('data/para_preprocessed/recipes_train_official_random_reduced0.5_seed5_p.json', vocab_file, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed5.json')
# 
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_random_reduced0.5_seed1.json', vocab_file, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed1.json')
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_random_reduced0.5_seed2.json', vocab_file, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed2.json')
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_random_reduced0.5_seed3.json', vocab_file, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed3.json')
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_random_reduced0.5_seed4.json', vocab_file, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed4.json')
# add_para_and_pad_for_training('data/para_preprocessed/recipes_train_official_random_reduced0.5_seed5.json', vocab_file, 'data/para_preprocessed/recipes_train_official_random_reduced0.5_seed5.json')
# 
# 
# 
# ##########################################################
# ##########################################################
# 
# 
# global_idx_dict = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/global_idx_dict.p','rb'))
# 
# global_idx_2_shuffle0_training_idx_dict = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/global_idx_2_shuffle0_train_idx_dict.p','rb'))
# global_idx_2_shuffle0_test_idx_dict = cPickle.load(open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/global_idx_2_shuffle0_test_idx_dict.p','rb'))
# para_count_test = {}
# for glb_idx, test_idx in global_idx_2_shuffle0_test_idx_dict.items():
#     para_group = global_idx_dict['para_group'][glb_idx]
#     if not(para_group in para_count_test):
#         para_count_test[para_group] = 0
#     para_count_test[para_group] +=1
# 
# para_count_training = {}    
# for glb_idx, train_idx in global_idx_2_shuffle0_training_idx_dict.items():
#     para_group = global_idx_dict['para_group'][glb_idx]
#     if not(para_group in para_count_training):
#         para_count_training[para_group] = 0
#     para_count_training[para_group] +=1
# 
# para_count = copy.deepcopy(para_count_test)
#      
# for p_group in para_count_test:
#     assert p_group in para_count_training
#     para_count[p_group]+= para_count_training[p_group]
#     
# avg_para_count = np.mean([para_count[p_group] for p_group in para_count])
#     
# ratio = {}
# for p_group, v in para_count_training.items():
#     try:
#         ratio[p_group] = para_count_test[p_group]/v  
#     except:
#         ratio[p_group] = 0  
# 
# para_total_count = {para_group: len(idxes) for para_group, idxes in para_group_dict.items()}
# 
# =============================================================================
