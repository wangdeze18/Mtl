# coding=utf-8
import os
import numpy
import torch
import torch.nn as nn
from pretrained_models import MODEL_CLASSES
from transformers import BertConfig

from module.dropout_wrapper import DropoutWrapper
from module.san import SANClassifier, MaskLmHeader,RNNDecoder,RNNDecoder_n
from module.san_model import SanModel
from data_utils.task_def import EncoderModelType, TaskType, DataFormat
import tasks
from experiments.exp_def import TaskDef
from pytorch_pretrained_bert.tokenization import BertTokenizer
from mtl.treelstm import TreeLSTM
from mtl.util import calculate_evaluation_orders
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.3):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, lens,is_train):
        batch_size, seq_len, feature_dim = input_seq.size()
        if is_train:
            input_seq = self.dropout(input_seq)
        scores = self.scorer(input_seq.contiguous().view(-1, feature_dim)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -numpy.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(input_seq).mul(input_seq).sum(1)
        return context
    
class LinearPooler(nn.Module):
    def __init__(self, hidden_size):
        super(LinearPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

def generate_decoder_opt(enable_san, max_opt):
    opt_v = 0
    if enable_san and max_opt < 3:
        opt_v = max_opt
    return opt_v
class SANBertNetwork(nn.Module):
    def __init__(self, opt, bert_config=None, initial_from_local=False):
        super(SANBertNetwork, self).__init__()
        self.dropout_list = nn.ModuleList()

        if opt['encoder_type'] not in EncoderModelType._value2member_map_:
            raise ValueError("encoder_type is out of pre-defined types")
        self.encoder_type = opt['encoder_type']
        self.preloaded_config = None

        literal_encoder_type = EncoderModelType(self.encoder_type).name.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]

        self.preloaded_config = config_class.from_dict(opt)  # load config from opt
        self.bert = model_class(self.preloaded_config)
        self.tokenizer = tokenizer_class.from_pretrained(opt['bert_model_type'], do_lower_case=opt['do_lower_case'])
        
        self.cls_emb = self.bert(torch.tensor([101]).unsqueeze(0))[0][0]
        self.cls_emb = self.cls_emb.repeat(8,1)
        #self.tokenizer = BertTokenizer.from_pretrained(opt['bert_model_type'], do_lower_case=opt['do_lower_case'])
        self.hidden_size = self.bert.config.hidden_size
        
        self.tree_model = TreeLSTM(self.hidden_size,self.hidden_size)


        if opt.get('dump_feature', False):
            self.opt = opt
            return
        if opt['update_bert_opt'] > 0:
            for p in self.bert.parameters():
                p.requires_grad = False

        task_def_list = opt['task_def_list']
        self.task_def_list = task_def_list
        self.task_types = []
        self.data_format = []
        self.decoder_opt = []
        self.n_class = []
        for task_id, task_def in enumerate(task_def_list):
            self.decoder_opt.append(0)
            self.task_types.append(task_def.task_type)
            self.data_format.append(task_def.data_type)
            self.n_class.append(task_def.n_class)

        self.tokenattn = nn.ModuleList()
        self.mul_modal_layer = nn.ModuleList()
        self.treeattn = nn.ModuleList()
        self.con_cat_layer = nn.ModuleList()
            
        # create output header
        self.scoring_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        for task_id in range(len(task_def_list)):
            task_def: TaskDef = task_def_list[task_id]
            lab = task_def.n_class
            decoder_opt = self.decoder_opt[task_id]
            task_type = self.task_types[task_id]
            task_dropout_p = opt['dropout_p'] if task_def.dropout_p is None else task_def.dropout_p
            dropout = DropoutWrapper(task_dropout_p, opt['vb_dropout'])
            self.dropout_list.append(dropout)
            task_obj = tasks.get_task_obj(task_def)
            
            temp_mul_modal_layer = nn.Linear(2 * self.hidden_size, self.hidden_size)
            temp_con_cat_layer = nn.Linear(2 * self.hidden_size, self.hidden_size)
            temp_treeAtt = SelfAttention(self.hidden_size, dropout=0.3)
            temp_tokenAtt = SelfAttention(self.hidden_size, dropout=0.3)

            self.tokenattn.append(temp_tokenAtt)
            self.mul_modal_layer.append(temp_mul_modal_layer)
            self.treeattn.append(temp_treeAtt)
            self.con_cat_layer.append(temp_con_cat_layer)
            
            if task_obj is not None:
                
                out_proj = task_obj.train_build_task_layer(decoder_opt, self.hidden_size, lab, opt, prefix='answer', dropout=dropout)
            elif task_type == TaskType.Span:
                assert decoder_opt != 1
                out_proj = nn.Linear(self.hidden_size, 2)
            elif task_type == TaskType.SeqenceLabeling:
                out_proj = nn.Linear(self.hidden_size, lab)
            elif task_type == TaskType.MaskLM:
                if opt['encoder_type'] == EncoderModelType.ROBERTA:
                    out_proj = MaskLmHeader(self.bert.embeddings.word_embeddings.weight)
                else:
                    out_proj = MaskLmHeader(self.bert.embeddings.word_embeddings.weight)
            elif task_type == TaskType.CodeSummarization:
                ##from pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
                ##decoder = BertForSeq2SeqDecoder.from_pretrained(self.bert_config)
                out_proj = RNNDecoder_n(self.hidden_size, self.hidden_size, lab, opt, prefix='decoder', dropout=dropout)
            else:
                if decoder_opt == 1:
                    out_proj = SANClassifier(self.hidden_size, self.hidden_size, lab, opt, prefix='answer', dropout=dropout)
                else:
                    out_proj = nn.Linear(self.hidden_size, lab)
            self.scoring_list.append(out_proj)

        self.opt = opt
        self._my_init()

        # if not loading from local, loading model weights from pre-trained model, after initialization
        if not initial_from_local:
            config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
            self.bert = model_class.from_pretrained(opt['init_checkpoint'],config=self.preloaded_config)

    def _my_init(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02 * self.opt['init_ratio'])
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    module.bias.data.zero_()

        self.apply(init_weights)

    def encode(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                                          attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output

    def forward(self, input_ids, token_type_ids, attention_mask,ast_features, adj,anti_ast_features=None,anti_adj=None,premise_mask=None,hyp_mask=None,task_id = 0, is_train = True):
        
        sequence_output, pooled_output = self.encode(input_ids, token_type_ids, attention_mask)
        decoder_opt = self.decoder_opt[task_id]
        task_type = self.task_types[task_id]
        data_type = self.data_format[task_id]
        task_obj = tasks.get_task_obj(self.task_def_list[task_id])
        
        tokenAtt = self.tokenattn[task_id]
        treeAtt = self.treeattn[task_id]
        one_modal_layer = self.mul_modal_layer[task_id]
        one_cat_layer = self.con_cat_layer[task_id]
        
        if self.opt['unenable_multimodal']:
            logits = task_obj.train_forward(sequence_output, pooled_output, premise_mask, hyp_mask, decoder_opt, self.dropout_list[task_id], self.scoring_list[task_id],is_train)
            return logits
        token_lens = []
        for i in range(sequence_output.size(0)):
            token_lens.append(int(sequence_output[i].size(0)))
            
        atten_token = tokenAtt(sequence_output,token_lens,is_train)
        
            
        if task_obj is not None:

            if data_type in [DataFormat.PremiseAndOneHypothesisnl,DataFormat.PremiseOnly]:

                lens = []
                for i in range(len(ast_features)):
                    lens.append(ast_features[i].size(0))

                tree_h = []
                
                for i in range(len(ast_features)):
                    ast_feature = torch.tensor(ast_features[i], device="cuda:0", dtype=torch.float32)
                    i_adj = torch.tensor(adj[i], device="cuda:0", dtype=torch.int64)

                    i_adj_array = i_adj.cpu().numpy()

                    node_order, edge_order = calculate_evaluation_orders(i_adj_array, ast_feature.size(0))
                    node_order = torch.tensor(node_order, device="cuda:0", dtype=torch.int64)
                    edge_order = torch.tensor(edge_order, device="cuda:0", dtype=torch.int64)
                    h, c = self.tree_model(ast_feature,node_order,i_adj,edge_order)
                    
                    tree_h.append(h)
                
                tree_h = tree_h
                pad_tree = rnn_utils.pad_sequence(tree_h, batch_first=True)


                atten_tree = treeAtt(pad_tree,lens,is_train)
                cat_out = torch.cat((atten_token.cuda(),atten_tree.cuda()), 1).cuda()

                mul_modal_output = one_modal_layer(cat_out.cuda()).cuda()

                logits = task_obj.train_forward(sequence_output, mul_modal_output, premise_mask, hyp_mask, decoder_opt, self.dropout_list[task_id], self.scoring_list[task_id],is_train)

            elif data_type in [DataFormat.PremiseAndOneHypothesiscode]:
                
                lens = []
                for i in range(len(ast_features)):
                    lens.append(ast_features[i].size(0))
                    
                anti_lens = []
                for i in range(len(anti_ast_features)):
                    anti_lens.append(anti_ast_features[i].size(0))
                    
                tree_h = []
                anti_tree_h = []
                for i in range(len(ast_features)):
                    ast_feature = torch.tensor(ast_features[i], device="cuda:0", dtype=torch.float32)
                    i_adj = torch.tensor(adj[i], device="cuda:0", dtype=torch.int64)
                    anti_ast_feature = torch.tensor(anti_ast_features[i], device="cuda:0", dtype=torch.float32)
                    i_anti_adj = torch.tensor(anti_adj[i], device="cuda:0", dtype=torch.int64)
                    i_adj_array = i_adj.cpu().numpy()
                    i_anti_adj_array = i_anti_adj.cpu().numpy()

                    node_order, edge_order = calculate_evaluation_orders(i_adj_array, ast_feature.size(0))
                    node_order = torch.tensor(node_order, device="cuda:0", dtype=torch.int64)
                    edge_order = torch.tensor(edge_order, device="cuda:0", dtype=torch.int64)
                    h, c = self.tree_model(ast_feature,node_order,i_adj,edge_order)
                    tree_h.append(h)
                    
                    anti_node_order, anti_edge_order = calculate_evaluation_orders(i_anti_adj_array, anti_ast_feature.size(0))
                    anti_node_order = torch.tensor(anti_node_order, device="cuda:0", dtype=torch.int64)
                    anti_edge_order = torch.tensor(anti_edge_order, device="cuda:0", dtype=torch.int64)
                    anti_h, anti_c = self.tree_model(anti_ast_feature,anti_node_order,i_anti_adj,anti_edge_order)
                    anti_tree_h.append(anti_h)
                
 
                pad_tree = rnn_utils.pad_sequence(tree_h, batch_first=True)
                pad_anti_tree = rnn_utils.pad_sequence(anti_tree_h, batch_first=True)
                atten_tree = treeAtt(pad_tree,lens,is_train)

                atten_anti_tree = treeAtt(pad_anti_tree,anti_lens,is_train)
                
                
                bi_tree = torch.cat((atten_tree.cuda(),atten_anti_tree.cuda()), 1).cuda()
                bi_tree_out = one_cat_layer(bi_tree.cuda()).cuda()
                cat_out = torch.cat((atten_token.cuda(),bi_tree_out.cuda()), 1).cuda()

                mul_modal_output = one_modal_layer(cat_out.cuda()).cuda()
                
                logits = task_obj.train_forward(sequence_output, pooled_output, premise_mask, hyp_mask, decoder_opt, self.dropout_list[task_id], self.scoring_list[task_id],is_train)
            else:
                logits = task_obj.train_forward(sequence_output, pooled_output, premise_mask, hyp_mask, decoder_opt, self.dropout_list[task_id], self.scoring_list[task_id],is_train)
            
            return logits
        elif task_type == TaskType.Span:
            assert decoder_opt != 1
            sequence_output = self.dropout_list[task_id](sequence_output)
            logits = self.scoring_list[task_id](sequence_output)
            start_scores, end_scores = logits.split(1, dim=-1)
            start_scores = start_scores.squeeze(-1)
            end_scores = end_scores.squeeze(-1)
            return start_scores, end_scores
        elif task_type == TaskType.SeqenceLabeling:
            pooled_output = sequence_output
            pooled_output = self.dropout_list[task_id](pooled_output)
            pooled_output = pooled_output.contiguous().view(-1, pooled_output.size(2))
            logits = self.scoring_list[task_id](pooled_output)
            return logits
        elif task_type == TaskType.MaskLM:
            sequence_output = self.dropout_list[task_id](sequence_output)
            logits = self.scoring_list[task_id](sequence_output)
            return logits
        elif task_type == TaskType.CodeSummarization:
            sequence_output = self.dropout_list[task_id](sequence_output)
            logits = self.scoring_list[task_id](sequence_output,target,self.tokenizer,self.cls_emb,self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token]),'True')
            return logits
        else:
            if decoder_opt == 1:
                max_query = hyp_mask.size(1)
                assert max_query > 0
                assert premise_mask is not None
                assert hyp_mask is not None
                hyp_mem = sequence_output[:, :max_query, :]
                logits = self.scoring_list[task_id](sequence_output, hyp_mem, premise_mask, hyp_mask)
            else:
                pooled_output = self.dropout_list[task_id](pooled_output)
                logits = self.scoring_list[task_id](pooled_output)
            return logits
