import os
from sklearn.externals import joblib
import torch
import numpy as np
from torch.utils.data import Dataset
import json
import random
import _pickle as cPickle

class BOWser_recipe(Dataset):
    def __init__(self, data_file, data_dir, split, epoch = 1, BERT_Sent = 0):
        super().__init__()
        self.split = split #train, test, train_with_para
        self.filepath = os.path.join(data_dir, data_file)
        with open(self.filepath, "r") as f:
            self.qa_list = json.load(f)
        if split == 'train_with_para_multiple':
            self.epoch = epoch
        self.BERT_Sent = BERT_Sent
        
    def __len__(self):
        return len(self.qa_list)

    def __getitem__(self, idx):
        # The key tells you the index that was picked from the original dataset at a certain instant in time
        # print("this is the tensor", torch.tensor(self.qa_list[idx]['question_token_idx'], device=device))
        
        if self.split == 'train_with_para':
            return_dict =  {
            # We will make it the right type of tensor in the main training loop
            # Try to put only tensors in'
            
            'sent_tok_idx': torch.tensor(self.qa_list[idx]['sent_tok_idx']),
            'para_tok_idx': torch.tensor(self.qa_list[idx]["para_tok_idx"]),
            'answer_tok_idx': torch.tensor(self.qa_list[idx]['answer_tok_idx']),
            
            #'sent_tok': self.qa_list[idx]["sent_tok"],
            #'para_tok': self.qa_list[idx]["para_tok"],
            #'answer_tok': self.qa_list[idx]["answer_tokens"],
            
            's_length': self.qa_list[idx]['s_length'],
            'a_length': self.qa_list[idx]['a_length'],
            'para_length': self.qa_list[idx]['para_len'],
            
            'para_group': self.qa_list[idx]['para_group'],
            #'para_list': self.qa_list[idx]['para_list'],
            
            'idx': idx,
            'sent_global_idx': self.qa_list[idx]['global_idx'],
            'para_global_idx': self.qa_list[idx]['para_global_idx'],
            
            'cur_dict_idx': self.qa_list[idx]['cur_dict_idx']
            }
            
            if self.BERT_Sent == 1:
                return_dict['BERT_sent_emb'] = self.qa_list[idx]["BERT_sent_emb"]
                return_dict['BERT_sent_enc'] = self.qa_list[idx]["BERT_sent_enc"]
        
            return return_dict
        
        elif self.split == 'train_with_para_multiple':
            #seed
            random.seed(self.epoch*idx)
            selected_para_idx = random.sample(self.qa_list[idx]['para_list'], 1)[0]
            #sample from self.qa_list[idx]['para_list'] every time 
            return_dict =  {
            'selected_para_idx': selected_para_idx,
            
            'sent_tok_idx': torch.tensor(self.qa_list[idx]['sent_tok_idx']),
            'para_tok_idx': torch.tensor(self.qa_list[selected_para_idx]["sent_tok_idx"]),
            'answer_tok_idx': torch.tensor(self.qa_list[idx]['answer_tok_idx']),
            
            #'sent_tok': self.qa_list[idx]["sent_tok"],
            #'para_tok': self.qa_list[selected_para_idx]["sent_tok"],
            #'answer_tok': self.qa_list[idx]["answer_tokens"],
            
            's_length': self.qa_list[idx]['s_length'],
            'a_length': self.qa_list[idx]['a_length'],
            'para_length': self.qa_list[selected_para_idx]['s_length'],
            
            'para_group': self.qa_list[idx]['para_group'],
            
            'idx': idx,
            'sent_global_idx': self.qa_list[idx]['global_idx'],
            'para_global_idx': self.qa_list[selected_para_idx]['global_idx'],
            
            'cur_dict_idx': self.qa_list[idx]['cur_dict_idx']
            }
            
            if self.BERT_Sent == 1:
                return_dict['BERT_sent_emb'] = self.qa_list[idx]["BERT_sent_emb"]
                return_dict['BERT_sent_enc'] = self.qa_list[idx]["BERT_sent_enc"]

            return return_dict
        else:
            return_dict = {
                # We will make it the right type of tensor in the main training loop
                # Try to put only tensors in
                'sent_tok_idx': torch.tensor(self.qa_list[idx]['sent_tok_idx']),
                'answer_tok_idx': torch.tensor(self.qa_list[idx]['answer_tok_idx']),
                's_length': self.qa_list[idx]['s_length'],
                'a_length': self.qa_list[idx]['a_length'],
                'idx': idx,
                'sent_global_idx': self.qa_list[idx]['global_idx'],
            }
        
            if self.BERT_Sent == 1:
                return_dict['BERT_sent_emb'] = self.qa_list[idx]["BERT_sent_emb"]
                return_dict['BERT_sent_enc'] = self.qa_list[idx]["BERT_sent_enc"]

            return return_dict
