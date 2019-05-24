import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
#from models.PADOUTPUTcopynet_singleMY import CopyEncoder, CopyEncoderRAW, CopyDecoder, AutoDecoder
from util_functions import decoder_initial
import time
import sys
import math
from numpy import inf
import random
import os
from bowser_constants import constants

from multiprocessing import cpu_count

import pickle as cPickle

from bowser_model_v1 import CopyEncoder, CopyDecoder, AutoDecoder
from bowser_dataset_v1 import BOWser_recipe
from collections import OrderedDict, defaultdict
from tensorboardX import SummaryWriter

# import matplotlib
# matplotlib.use("Agg")


# Can pass in args later
def train():
    # Need to put things in constants and others should come from args. Then assign over here to not change the
    # code too much
    vocab_file = constants.vocab_file
    data_dir = constants.data_dir
    train_data_file = constants.train_data_file
    test_data_file = constants.test_data_file
    model_dir = constants.model_dir
    log_dir = constants.log_dir
    pkl_dir = constants.pkl_dir
    tb_log_dir = constants.tb_log_dir
    res_dir = constants.results_dir

    log = True
    tb_log = True

    run_num = 999
    validate_at_epoch = 10
    save_model_at_epoch = 50
    save_pkl_files_at_epoch = 50
    weight_decay_at = 25

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
    embed_size = 128
    learning_rate = 0.01
    AUTOhidden_dim = 128
    weight_decay = 0.99
    hidden_size = 128
    num_epochs = 500
    batch_size = 32

    alpha = 0.0

    pad_token = w2i['PAD_token']
    # print("pad token", pad_token)
    SOS_token = w2i['SOS_token']
    GO_token = w2i['GO_token']
    EOS_token = w2i['EOS_token']

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
    save_model_path = os.path.join(model_dir, "model_" + constants.add_info + constants.version + "_run" + str(run_num))
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

    datasets['train'] = BOWser_recipe(
        data_dir=data_dir,
        data_file=train_data_file,
        split='train')
    datasets['test'] = BOWser_recipe(
        data_dir=data_dir,
        data_file=test_data_file,
        split='test')

    encoder = CopyEncoder(hidden_size, vocab_size)
    decoder = CopyDecoder(vocab_size, embed_size, hidden_size)
    AUTOdecoder = AutoDecoder(AUTOhidden_dim, vocab_size, batch_size)
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

        current_loss_list = []
        for i, batch in enumerate(data_loader_train):
            # Sort the inputs and targets in the batch in reverse sorted order
            sorted_lengths, sorted_idx = torch.sort(batch['s_length'], descending=True)
            batch['sent_tok_idx'] = batch['sent_tok_idx'][sorted_idx]
            batch['answer_tok_idx'] = batch['answer_tok_idx'][sorted_idx]
            batch['a_length'] = batch['a_length'][sorted_idx]
            batch['s_length'] = batch['s_length'][sorted_idx]

            # Need it for the decoders
            cur_batch_size = len(batch['s_length'])

            # initialize gradient buffers
            opt_e.zero_grad()
            opt_d.zero_grad()
            opt_a.zero_grad()

            # Chop off unnecessary positions in the input and target vectors - Thus reducing sequence length
            # and increasing efficiency as decoder will be slower with more positions in the target sequences
            input_vectors = to_cuda(batch['sent_tok_idx'][:, :sorted_lengths[0]])
            target_vectors = to_cuda(batch['answer_tok_idx'][:, :batch["a_length"].max().item()])

            # apply to encoder
            encoded, hidden_ec = encoder(input_vectors, sorted_lengths)
            # encoded - Shape ([32, 15, 256]) - Since in this batch 15 is the max length

            # stop if there is a problem
            if math.isnan(encoded[0][0].data[0]):
                print("encoder broken!")
                encoder_broken.append(input_vectors)
                sys.exit()
                break

            input_out = input_vectors.cpu().data.numpy()

            # get initial input of decoder
            decoder_in, s, w = decoder_initial(cur_batch_size)
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
                if j == 0:
                    out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                        encoded_idx=input_out, prev_state=s,
                                        weighted=w, order=j)
                # remaining states
                else:
                    tmp_out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                            encoded_idx=input_out, prev_state=s,
                                            weighted=w, order=j)
                    out = torch.cat([out, tmp_out], dim=1)

                # if epoch % 2 == 1:
                #     decoder_in = out[:, -1].max(1)[1].squeeze()  # train with sequence outputs
                # else:
                #     decoder_in = y[:, j]  # train with ground truth

                # Shape of out when seq_len_input = 16, seq_len_target = 53 is [32, 53, 353]
                # The reason why there is 353 even though our vocab is 341 in length is because CopyDecoder adds
                # 12 oovs to prob_g. Can take a look inside the model file.

                # Right now we will just use ground truth - otherwise uncomment the above lines
                decoder_in = target_vectors[:, j]

                out_list.append(out[:, -1].max(1)[1].squeeze().cpu().data.numpy())

            ### AUTO DECODER ###
            AUTOdecoder_hidden, AUTOtarget_tensor, di = hidden_ec, input_vectors, 1
            AUTOdecoder_input = torch.cuda.LongTensor([SOS_token] * cur_batch_size) if torch.cuda.is_available() \
                else torch.LongTensor([SOS_token] * cur_batch_size)

            # Just going to use teach_forcing in this run
            # use_teacher_forcing = True if teacher_forcing_prob > random.random() else False
            use_teacher_forcing = True

            if use_teacher_forcing:
                for di in range(AUTOtarget_tensor.shape[1]):
                    AUTOdecoder_output, AUTOdecoder_hidden = AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden,
                                                                         cur_batch_size)
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
                    AUTOdecoder_output, AUTOdecoder_hidden = AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden,
                                                                         cur_batch_size)
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
            target = target_vectors.contiguous()
            target = target.view(-1)
            pad_out = out.view(-1, out.shape[2])
            pad_out = pad_out + 0.0001
            pad_out = torch.log(pad_out)

            review_sent = input_vectors.contiguous().view(-1)
            review_auto_out = tensor_of_all_AUTOdecoded_outputs.view(-1,
                                                                     tensor_of_all_AUTOdecoded_outputs.shape[
                                                                         2])
            # Shape of review_sent - [544], Shape of review_auto_out - [544, 341]

            translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target,
                                                                              review_auto_out,
                                                                              review_sent)

            if len(pad_out[pad_out == -inf]) > 0:
                joint_loss = torch.tensor(0.0, device=device)
                print("Loss broken but skipped!")
            else:
                joint_loss.backward()

            opt_e.step()
            opt_d.step()
            if alpha > 0:
                # print("Coming here")
                opt_a.step()

            current_loss_list.append(joint_loss.item())

        if (epoch + 1) > 50 and (epoch + 1) % weight_decay_at == 0:
            learning_rate = learning_rate * weight_decay  # weight decay
            run_log.write("\nAdjusting LR to:" + str(learning_rate) + " in epoch %d" % (epoch + 1))

        if (epoch+1)%5==0 and 10<=(epoch+1)<=25:
            learning_rate /= 2
            run_log.write("\nAdjusting LR to:" + str(learning_rate) + " in epoch %d" % (epoch + 1))

        # if epoch == 8:
        #     # half the learning rate
        #     learning_rate /= 5
        #     run_log.write("\nAdjusting LR to:" + str(learning_rate) + " in epoch %d" % (epoch + 1))

        # if epoch == 38:
        #     learning_rate /= 2
        #
        # if 54 > epoch >= 36 and epoch % 6 == 0:
        #     learning_rate /= 2

        avg_training_loss = np.mean(current_loss_list)
        print("Training Loss in epoch %d is:" % (epoch + 1), avg_training_loss)
        training_loss_list.append(avg_training_loss)
        if log:
            run_log.write("\nTraining Loss in epoch %d is: %s" % ((epoch + 1), str(avg_training_loss)))

        if (epoch + 1) % save_model_at_epoch == 0:
            torch.save(encoder.state_dict(),
                       os.path.join(save_model_path, "encoder_ckpt_epoch%i.pytorch" % (epoch + 1)))
            torch.save(decoder.state_dict(),
                       os.path.join(save_model_path, "decoder_ckpt_epoch%i.pytorch" % (epoch + 1)))
            torch.save(AUTOdecoder.state_dict(),
                       os.path.join(save_model_path, "auto_decoder_ckpt_epoch%i.pytorch" % (epoch + 1)))

        if tb_log:
            writer.add_scalar("Run%d_%s_%sEpoch/train_loss" % (run_num, constants.version, constants.add_info),
                              float(avg_training_loss), (epoch + 1))

        elapsed = time.time()
        print("Elapsed time for epoch: ", elapsed - start)

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
                    input_vectors = to_cuda(batch['sent_tok_idx'][:, :sorted_lengths[0]])
                    target_vectors = to_cuda(batch['answer_tok_idx'][:, :batch["a_length"].max().item()])

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
                        if j == 0:
                            out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                                encoded_idx=input_out, prev_state=s,
                                                weighted=w, order=j)
                            # print("decoder in is : " + str(decoder_in))
                        # remaining states
                        else:
                            tmp_out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                                    encoded_idx=input_out, prev_state=s,
                                                    weighted=w, order=j)
                            out = torch.cat([out, tmp_out], dim=1)

                        # if epoch % 2 == 1:
                        #     decoder_in = out[:, -1].max(1)[1].squeeze()  # train with sequence outputs
                        # else:
                        #     decoder_in = y[:, j]  # train with ground truth

                        # Shape of out when seq_len_input = 16, seq_len_target = 53 is [32, 53, 353]
                        # The reason why there is 353 even though our vocab is 341 in length is because CopyDecoder adds
                        # 12 oovs to prob_g. Can take a look inside the model file.

                        # # Right now we will just use ground truth - otherwise uncomment the above lines
                        # decoder_in = target_vectors[:, j]

                        # Now for validation we only use the sequence outputs and not the ground truth
                        decoder_in = out[:, -1].max(1)[1].squeeze()

                        out_list.append(out[:, -1].max(1)[1].squeeze().cpu().data.numpy())

                    # out has shape [32, 53, 353] - [batch_size, seq_len, vocab_size+oovs]
                    # outlist has 53 lists and has length of 32 in each of them for the above out
                    # So outlist has predictions at each time step

                    ### AUTO DECODER ###
                    AUTOdecoder_hidden, AUTOtarget_tensor, di = hidden_ec, input_vectors, 1

                    AUTOdecoder_input = torch.cuda.LongTensor([SOS_token] * cur_batch_size) if torch.cuda.is_available() \
                        else torch.LongTensor([SOS_token] * cur_batch_size)

                    # Just going to use teach_forcing in this run
                    # use_teacher_forcing = True if teacher_forcing_prob > random.random() else False
                    use_teacher_forcing = False

                    if use_teacher_forcing:
                        for di in range(AUTOtarget_tensor.shape[1]):
                            AUTOdecoder_output, AUTOdecoder_hidden = AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden,
                                                                                 cur_batch_size)
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
                            AUTOdecoder_output, AUTOdecoder_hidden = AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden,
                                                                                 cur_batch_size)
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
                    pad_out = out.view(-1, out.shape[2])
                    pad_out = pad_out + 0.0001
                    pad_out = torch.log(pad_out)

                    review_sent = input_vectors.contiguous().view(-1)
                    review_auto_out = tensor_of_all_AUTOdecoded_outputs.view(-1,
                                                                             tensor_of_all_AUTOdecoded_outputs.shape[2])
                    # Shape of review_sent - [544], Shape of review_auto_out - [544, 341]

                    translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target,
                                                                                      review_auto_out,
                                                                                      review_sent)

                    current_loss_list.append(joint_loss.item())

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

                avg_val_loss = np.mean(current_loss_list)
                print("Validation Loss in epoch %d is:" % (epoch + 1), avg_val_loss)
                if log:
                    run_log.write("\nValidation Loss in epoch %d is: %s\n" % ((epoch + 1), str(avg_val_loss)))
                test_loss_list.append(avg_val_loss)

                if tb_log:
                    writer.add_scalar(
                        "Run%d_%s_%sEpoch/validation_loss" % (run_num, constants.version, constants.add_info),
                        float(avg_val_loss), (epoch + 1))

                if (epoch + 1) % save_pkl_files_at_epoch == 0:
                    final_dict = {'translation_pairs': translation_pairs, 'reconstruction_pairs': reconstruction_pairs}
                    pkl_filname = os.path.join(pkl_dir, "validation_pairs_%s_run%d_epoch%d" % (
                    constants.version, run_num, (epoch + 1)))
                    with open(pkl_filname, "wb") as f:
                        cPickle.dump(final_dict, f)

                        # print("Done with epoch -", epoch+1)

    # Save the training and test lists here. And also translation pairs just to be sure that this happens at the end
    # Last thing to be done
    final_dict = {'translation_pairs': translation_pairs, 'reconstruction_pairs': reconstruction_pairs,
                  'training_loss': training_loss_list, 'validation_loss': test_loss_list}
    pkl_filname = os.path.join(pkl_dir, "validation_pairs_%s_run%d_final" % (constants.version, run_num))
    with open(pkl_filname, "wb") as f:
        cPickle.dump(final_dict, f)
    if log:
        run_log.close()

    # Also save the model
    torch.save(encoder.state_dict(), os.path.join(save_model_path, "encoder_ckpt_epoch%i.pytorch" % num_epochs))
    torch.save(decoder.state_dict(), os.path.join(save_model_path, "decoder_ckpt_epoch%i.pytorch" % num_epochs))
    torch.save(AUTOdecoder.state_dict(),
               os.path.join(save_model_path, "auto_decoder_ckpt_epoch%i.pytorch" % num_epochs))


if __name__ == '__main__':
    train()

"""
Notes:
check variables are for putting breakpoints for debugging

Run numbers:
1 - version 1, alpha = 1.0, running on mkscan server - parameters in the log file
"""








