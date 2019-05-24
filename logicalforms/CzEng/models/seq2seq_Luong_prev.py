import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time

# import matplotlib
# matplotlib.use("Agg")

# Just modifying bowser model and making it a simple seq2seq network

USE_CUDA = True if torch.cuda.is_available() else False

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        # input_size is actually the vocab size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        #input_seqs: [Batch x Seq]
        #input_lengths: [Batch]
        embedded = self.embedding(input_seqs) #embedded: [Batch x Seq]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs

        return outputs, hidden #encoder outpus: [Batch x Seq x hidden], hidden: not really used so does not matter


class Attn(nn.Module):
    def __init__(self, method, hidden_size, soft=True, D= 0):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.soft = soft

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.randn(1,hidden_size,1))
            
        if self.method == 'location':
            self.attn = nn.Linear(self.hidden_size, 1)
            
        if not(soft):
            self.sig = nn.Sigmoid()
            self.D = D
            self.Vp = nn.Parameter(torch.randn(self.hidden_size, 1))
            self.Wp = nn.Linear(self.hidden_size, self.hidden_size)

    #Encoder is done once for all of B and all of S
    #Decoder is done for all B but only 1 for s at a time.
    def forward(self, hidden, encoder_outputs, X_lengths = None):
        #hidden: current hidden state of the decoder [1 x Batch x hidden] (since not bidirectional)
        #encoder_outputs : [Batch x Seq x hidden] 

        #hidden.transpose(0,1).transpose(1,2): [Batch x hidden x 1]
        #print("encoder outputs shape", encoder_outputs.shape)
        #print("hidden passed in", hidden.transpose(0,1).transpose(1,2).shape)
        if self.method in ['dot', 'general', 'concat']:
            attn_energies_3 = self.score_3(hidden.transpose(0,1).transpose(1,2), encoder_outputs)
            attn_energies_3 = attn_energies_3.view(attn_energies_3.shape[0], attn_energies_3.shape[1])
            #attn_energies_3: [Batch x Seq]
    
            if USE_CUDA:
                attn_energies_3 = attn_energies_3.cuda()
                

        else:
            hidden_2 = hidden.transpose(0,1) #hidden_2 : [Batch x 1 x hidden]
            hidden_2 = hidden_2.expand(encoder_outputs.shape[0],encoder_outputs.shape[1],encoder_outputs.shape[2]) #hidden_2 : [Batch x Seq x Hidden]
            attn_energies_3 = self.attn(hidden_2).squeeze(2) #Batch x seq
            attn_energies_3 = attn_energies_3.cuda() 
        
        
        if not(self.soft):
            pt = self.sig(F.tanh(self.Wp(hidden.squeeze(0)))).unsqueeze(1) #[Batch x 1 x hidden]
            pt = pt.bmm(self.Vp.expand(pt.shape[0], self.Vp.shape[0], self.Vp.shape[1])).unsqueeze(1) #[Batch x1]
            pt =  torch.tensor(X_lengths) * pt #pt: [Batch]
            pt = pt.expand(pt.shape[0],encoder_outputs.shape[1]) #[Batch x seq]
            s = torch.tensor([[j >= pt - self.D and j <= pt + self.D for j in range(X_lengths[i])] + [0.0]*(encoder_outputs.shape[1]- X_lengths[i]) for i in range(encoder_outputs.shape[0])]) #[batch x length of each sequence]
            pointer_coeffs = torch.exp(( s - pt )**2/(-2*(self.D/2)**2))
            attn_energies_3 = attn_energies_3 * pointer_coeffs
        
        return F.softmax(attn_energies_3, dim=1).unsqueeze(1)#[Batch x 1 x Seq]
        # Normalize energies to weights in range 0 to 1, resize to Bx1xS
       
        

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output) #energy: [Batch x Seq x hidden]
            energy = hidden.dot(energy) #hidden : [Batch x hidden x 1]
            return energy

        elif self.method == 'concat':
            hidden = hidden.transpose(1,2) #hidden: [Batch x 1 x hidden]
            energy = self.attn(torch.cat((hidden.expand(encoder_output.shape[0],encoder_output.shape[1],encoder_output.shape[2]), encoder_output), 2)) #energy: [Batch x Seq x hidden]
            energy = self.v.expand(hidden.shape[0],hidden.shape[2],1).dot(energy)
            return energy

    def score_2(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = energy.matmul(hidden)
            return energy

        elif self.method == 'concat':
           hidden = hidden.transpose(1,2) #hidden: [Batch x 1 x hidden]
           energy = self.attn(torch.cat((hidden.expand(encoder_output.shape[0],encoder_output.shape[1],encoder_output.shape[2]), encoder_output), 2)) #energy: [Batch x Seq x hidden]
           energy = energy.matmul(self.v.expand(hidden.shape[0],hidden.shape[2],1))
           return energy

    # Fully vectorized implementation - 'general' case
    def score_3(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = encoder_output.bmm(hidden)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output) #energy: [Batch x Seq x hidden] 
            energy = energy.bmm(hidden) #hidden: [Batch x hidden x 1]
            return energy

        elif self.method == 'concat':
            hidden = hidden.transpose(1,2) #hidden: [Batch x 1 x hidden]
            energy = self.attn(torch.cat((hidden.expand(encoder_output.shape[0],encoder_output.shape[1],encoder_output.shape[2]), encoder_output), 2)) #energy: [Batch x Seq x hidden]
            energy = energy.bmm(self.v.expand(hidden.shape[0],hidden.shape[2],1))
            return energy
        
            



class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1, bi=0):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
        
        if bi == 1:
            self.W_bi = nn.Linear(hidden_size*2, hidden_size)
        else:
            pass

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # input_seq : [batch_size], something like torch.LongTensor([SOS_token] * small_batch_size)
        # last_hidden: last elemnt of decoder_outputs [1 x batch_size x hidden]
        # encoder_outputs: [batch_size x seq x hidden]

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0) 
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        #No need for batch_irst = True in the gru because of the next line
        embedded = embedded.view(1, batch_size, self.hidden_size)  # embedded: [1 x Batch x hidden]
        if last_hidden.shape[-1] == self.hidden_size*2:
            last_hidden = self.W_bi(last_hidden)
        last_hidden = last_hidden.view(1,batch_size, self.hidden_size)
        embedded = embedded.contiguous(); last_hidden = last_hidden.contiguous()


        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden) #rnn output: [1 x Batch x Hidden], hidden: [1 x Batch x Hidden]

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs) #attn_weights : [Batch x 1 x Seq]
        context = attn_weights.bmm(encoder_outputs)  #context: [Batch x 1 x Hidden]

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # rnn_output is now: [Batch x Hidden]
        context = context.squeeze(1)  # context is now: [Batch x Hidden]
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output) #output is : [Batch*output_size] (output size is the number of all vocabs)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self,attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        
        assert attn_model == 'concat'
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # input_seq : [batch_size], something like torch.LongTensor([SOS_token] * small_batch_size)
        # last_hidden: last elemnt of decoder_outputs [1 x batch_size x hidden]
        # encoder_outputs: [batch_size x seq x hidden]

        
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # embedded: [1 x Batch x hidden]
        
        # Calculate attention weights and apply to encoder outputs
        #last_hidden[-1]: [Batch x hidden], last_hidden[-1].unsqueeze(0): [1 x Batch x hidden]
        last_hidden = last_hidden.view(1,batch_size, self.hidden_size)
        attn_weights = self.attn(last_hidden, encoder_outputs)  #attn_weights : [Batch x 1 x Seq]
        context = attn_weights.bmm(encoder_outputs)  #context: [Batch x 1 x Hidden]
        context = context.transpose(0, 1) #context: [1 x Batch x Hidden]
        
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((embedded, context), 2) #rnn_input: [1 x Batch x Hidden*2]
        rnn_input = rnn_input.contiguous(); last_hidden = last_hidden.contiguous()
        output, hidden = self.gru(rnn_input, last_hidden) #output: [1 x Batch x Hidden], hidden: [1 x Batch x Hidden]
        
        # Final output layer
        output = output.squeeze(0) # output: [Batch x Hidden]
        #context = context.squeeze(0)  # context: [Batch x Hidden]
        #output = F.log_softmax(self.out(torch.cat((output, context), 1))) #output:  [Batch*output_size] 
        output = F.log_softmax(self.out(output), dim=1) #output:  [Batch*output_size]
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
