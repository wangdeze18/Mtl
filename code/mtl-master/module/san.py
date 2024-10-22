# coding=utf-8
import torch
import random
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from module.dropout_wrapper import DropoutWrapper
from module.similarity import FlatSimilarityWrapper, SelfAttnWrapper
from module.my_optim import weight_norm as WN
from bert_embedding import BertEmbedding

SMALL_POS_NUM=1.0e-30
MAX_LENGTH = 50

def generate_mask(new_data, dropout_p=0.0, is_training=False):
    if not is_training: dropout_p = 0.0
    new_data = (1-dropout_p) * (new_data.zero_() + 1)
    for i in range(new_data.size(0)):
        one = random.randint(0, new_data.size(1)-1)
        new_data[i][one] = 1
    mask = 1.0/(1 - dropout_p) * torch.bernoulli(new_data)
    mask.requires_grad = False
    return mask


class Classifier(nn.Module):
    def __init__(self, x_size, y_size, opt, prefix='decoder', dropout=None):
        super(Classifier, self).__init__()
        self.opt = opt
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(prefix), 0))
        else:
            self.dropout = dropout
        self.merge_opt = opt.get('{}_merge_opt'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)

        if self.merge_opt == 1:
            self.proj = nn.Linear(x_size * 4, y_size)
        else:
            self.proj = nn.Linear(x_size * 2, y_size)

        if self.weight_norm_on:
            self.proj = weight_norm(self.proj)

    def forward(self, x1, x2, mask=None):
        if self.merge_opt == 1:
            x = torch.cat([x1, x2, (x1 - x2).abs(), x1 * x2], 1)
        else:
            x = torch.cat([x1, x2], 1)
        x = self.dropout(x)
        scores = self.proj(x)
        return scores

class SANClassifier(nn.Module):
    """Implementation of Stochastic Answer Networks for Natural Language Inference, Xiaodong Liu, Kevin Duh and Jianfeng Gao
    https://arxiv.org/abs/1804.07888
    """
    def __init__(self, x_size, h_size, label_size, opt={}, prefix='decoder', dropout=None):
        super(SANClassifier, self).__init__()
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.prefix = prefix
        self.query_wsum = SelfAttnWrapper(x_size, prefix='mem_cum', opt=opt, dropout=self.dropout)
        self.attn = FlatSimilarityWrapper(x_size, h_size, prefix, opt, self.dropout)
        self.rnn_type = '{}{}'.format(opt.get('{}_rnn_type'.format(prefix), 'gru').upper(), 'Cell')
        self.rnn =getattr(nn, self.rnn_type)(x_size, h_size)
        self.num_turn = opt.get('{}_num_turn'.format(prefix), 5)
        self.opt = opt
        self.mem_random_drop = opt.get('{}_mem_drop_p'.format(prefix), 0)
        self.mem_type = opt.get('{}_mem_type'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.label_size = label_size
        self.dump_state = opt.get('dump_state_on', False)
        self.alpha = Parameter(torch.zeros(1, 1), requires_grad=False)
        if self.weight_norm_on:
            self.rnn = WN(self.rnn)

        self.classifier = Classifier(x_size, self.label_size, opt, prefix=prefix, dropout=self.dropout)

    def forward(self, x, h0, x_mask=None, h_mask=None):
        h0 = self.query_wsum(h0, h_mask)
        print(h0.size())
        if type(self.rnn) is nn.LSTMCell:
            c0 = h0.new(h0.size()).zero_()
        scores_list = []
        for turn in range(self.num_turn):
            att_scores = self.attn(x, h0, x_mask)
            x_sum = torch.bmm(F.softmax(att_scores, 1).unsqueeze(1), x).squeeze(1)
            print("x_sum.size = ",x_sum.size())
            scores = self.classifier(x_sum, h0)
            scores_list.append(scores)
            # next turn
            if self.rnn is not None:
                h0 = self.dropout(h0)
                if type(self.rnn) is nn.LSTMCell:
                    h0, c0 = self.rnn(x_sum, (h0, c0))
                else:
                    h0 = self.rnn(x_sum, h0)
        if self.mem_type == 1:
            mask = generate_mask(self.alpha.data.new(x.size(0), self.num_turn), self.mem_random_drop, self.training)
            mask = [m.contiguous() for m in torch.unbind(mask, 1)]
            tmp_scores_list = [mask[idx].view(x.size(0), 1).expand_as(inp) * F.softmax(inp, 1) for idx, inp in enumerate(scores_list)]
            scores = torch.stack(tmp_scores_list, 2)
            scores = torch.mean(scores, 2)
            scores = torch.log(scores)
        else:
            scores = scores_list[-1]
        if self.dump_state:
            return scores, scores_list
        else:
            return scores

class RNNDecoder_n(nn.Module):
    
    def __init__(self, x_size, h_size, label_size, opt={}, prefix='decoder', dropout=None):
        super(RNNDecoder_n, self).__init__()
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.prefix = prefix
        self.query_wsum = SelfAttnWrapper(x_size, prefix='mem_cum', opt=opt, dropout=self.dropout)
        self.attn = FlatSimilarityWrapper(x_size, h_size, prefix, opt, self.dropout)
        self.rnn_type = '{}{}'.format(opt.get('{}_rnn_type'.format(prefix), 'gru').upper(), 'Cell')
        self.rnn =getattr(nn, self.rnn_type)(x_size, h_size).cuda()
        self.num_turn = MAX_LENGTH
        self.opt = opt
        self.mem_random_drop = opt.get('{}_mem_drop_p'.format(prefix), 0)
        self.mem_type = opt.get('{}_mem_type'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.label_size = label_size
        self.dump_state = opt.get('dump_state_on', False)
        self.alpha = Parameter(torch.zeros(1, 1), requires_grad=False)
        self.teacher_forcing_ratio = 0.75
        self.hidden_size = h_size
        self.embedding = BertEmbedding()
        self.out = torch.nn.Linear(self.hidden_size,label_size)
        if self.weight_norm_on:
            self.rnn = WN(self.rnn)

        self.classifier = Classifier(x_size, self.label_size, opt, prefix=prefix, dropout=self.dropout)

    def forward(self, x0, y, tokenizer, begin_token, end_token, is_train='False',x_mask=None, h_mask=None):
        h0 = torch.zeros(self.opt.get('batch_size',8),self.hidden_size).cuda()
        if type(self.rnn) is nn.LSTMCell:
            c0 = h0.new(h0.size()).zero_()
        target_length = 512
        decoded_words = []
        use_tencher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        decoder_input = begin_token.cuda()

        for di in range(target_length-1):
                att_scores = self.attn(x0, h0, x_mask)
                x_sum = torch.bmm(F.softmax(att_scores, 1).unsqueeze(1), x0).squeeze(1)

                # next turn
                if self.rnn is not None:
                    h0 = self.dropout(h0)
                    if type(self.rnn) is nn.LSTMCell:
                        h0, c0 = self.rnn(x_sum, (h0, c0))
                    else:
                        h0 = self.rnn(x_sum, h0)

                output = torch.log_softmax(self.out(h0),dim=1)

                decoded_words.append(output.cpu())
                
        return decoded_words

class RNNDecoder(nn.Module):
    """Implementation of RNN Decoder
    """
    def __init__(self, x_size, h_size,label_size, opt={}, prefix='decoder', dropout=None):
        super(RNNDecoder, self).__init__()
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.prefix = prefix
        self.query_wsum = SelfAttnWrapper(x_size, prefix='mem_cum', opt=opt, dropout=self.dropout)
        self.attn = FlatSimilarityWrapper(x_size, h_size, prefix, opt, self.dropout)
        self.rnn_type = '{}{}'.format(opt.get('{}_rnn_type'.format(prefix), 'gru').upper(), 'Cell')
        self.rnn =getattr(nn, self.rnn_type)(x_size, h_size)
        self.num_turn = opt.get('{}_num_turn'.format(prefix), 5)
        self.opt = opt
        self.mem_random_drop = opt.get('{}_mem_drop_p'.format(prefix), 0)
        self.mem_type = opt.get('{}_mem_type'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.label_size = label_size
        self.dump_state = opt.get('dump_state_on', False)
        self.alpha = Parameter(torch.zeros(1, 1), requires_grad=False)

        self.decoder = DecoderRNN(hidden_size=h_size,output_size=30000).cuda()

        if self.weight_norm_on:
            self.rnn = WN(self.rnn)

        self.classifier = Classifier(x_size, self.label_size, opt, prefix=prefix, dropout=self.dropout)

    ##getattr(evaluation, score_function)(hypotheses, references)
    def forward(self, x0, y, tokenizer,begin_token, end_token,is_train='False'):
        decoded_words = []
        decoded_words.append(begin_token)
        max_length = MAX_LENGTH
        target_length = len(y)
        decoder_hidden = self.decoder.initHidden().cuda()
        teacher_forcing_ratio =0.75
        use_tencher_forcing = True if random.random() < teacher_forcing_ratio else False
        decoder_input = torch.tensor([[begin_token]])
        if is_train and use_tencher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                ##loss += criterion(decoder_output, target_tensor[di])
                decoder_input = y[di]  # teacher forcing
                decoded_words.append(decoder_output)
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoded_words.append(decoder_output)
                predicted_index = torch.argmax(decoder_output).item()
                predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
                decoder_input = tokenizer.convert_tokens_to_ids(predicted_token)

                ##loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == end_token:
                    break


        return decoded_words

class DecoderRNN(torch.nn.Module):
    def __init__(self,hidden_size,output_size):
        super(DecoderRNN,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(output_size,hidden_size).cuda()
        self.gru = torch.nn.GRU(hidden_size,hidden_size)
        self.out = torch.nn.Linear(hidden_size,output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
        output = self.embedding(input.cuda()).view(1,1,-1).cuda()
        output = torch.relu(output)
        output,hidden = self.gru(output,hidden)
        output = self.softmax(self.out(output[0]))
        return output,hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size)
    
class AttnDecoderRNN(torch.nn.Module):
    def __init__(self,hidden_size,output_size,dropout_p=0.1,max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p =dropout_p
        self.max_length = max_length

        self.embedding = torch.nn.Embedding(self.output_size,self.hidden_size)
        self.attn = torch.nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_conbine = torch.nn.Linear(self.hidden_size * 2,self.hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.gru = torch.nn.GRU(self.hidden_size,self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size,self.output_size)

    def forward(self,input,hidden,encoder_outputs):
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)

        attn_weights = torch.softmax(self.attn(torch.cat((embedded[0],hidden[0]),1)),dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0],attn_applied[0]),1)
        output = self.attn_conbine(output).unsqueeze(0)

        output = torch.relu(output)
        output,hidden = self.gru(output,hidden)

        output = torch.log_softmax(self.out(output[0]),dim=1)
        return output,hidden,attn_weights

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size)

class MaskLmHeader(nn.Module):
    """Mask LM
    """
    def __init__(self, embedding_weights=None, bias=False):
        super(MaskLmHeader, self).__init__()
        self.decoder = nn.Linear(embedding_weights.size(1),
                                 embedding_weights.size(0),
                                 bias=bias)
        self.decoder.weight = embedding_weights
        self.nsp = nn.Linear(embedding_weights.size(1), 2)

    def forward(self, hidden_states):
        mlm_out = self.decoder(hidden_states)
        nsp_out = self.nsp(hidden_states[:, 0, :])
        return mlm_out, nsp_out
