# coding=utf-8
import logging
import os
import nltk
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from data_utils.utils import AverageMeter
from pytorch_pretrained_bert import BertAdam as Adam
from module.bert_optim import Adamax, RAdam
from mtl.loss import LOSS_REGISTRY
from .matcher import SANBertNetwork

from data_utils.task_def import TaskType, EncoderModelType, DataFormat
import tasks
from experiments.exp_def import TaskDef
from pretrained_models import MODEL_CLASSES
from pytorch_pretrained_bert.tokenization import BertTokenizer
from nltk.translate.bleu_score import *
from torch.utils.tensorboard import SummaryWriter
from mtl.treelstm import TreeLSTM
from mtl.util import calculate_evaluation_orders
import torch.nn.utils.rnn as rnn_utils

logger = logging.getLogger(__name__)
tensorboard_logdir = os.path.join("checkpoint/", "tensorboard_logdir")
tensorboard = SummaryWriter(log_dir = tensorboard_logdir)

log_var_a = torch.zeros((1,)).cuda()
log_var_b = torch.zeros((1,)).cuda()
log_var_c = torch.zeros((1,)).cuda()

log_var_a.requires_grad = True
log_var_b.requires_grad = True
log_var_c.requires_grad = True

log_vars = [log_var_a,log_var_b,log_var_c]

class MTLModel(nn.Module):
    def __init__(self, opt, state_dict=None, num_train_step=-1):
        super(MTLModel, self).__init__()
        self.config = opt
        self.updates = state_dict['updates'] if state_dict and 'updates' in state_dict else 0
        self.local_updates = 0
        self.train_loss = AverageMeter()
        self.initial_from_local = True if state_dict else False
        self.network = SANBertNetwork(opt, initial_from_local=self.initial_from_local)
        if state_dict:
            missing_keys, unexpected_keys = self.network.load_state_dict(state_dict['state'], strict=False)
        self.mnetwork = nn.DataParallel(self.network).cuda() if opt['multi_gpu_on'] else self.network
        self.total_param = sum([p.nelement() for p in self.network.parameters() if p.requires_grad])
        
        if opt['encoder_type'] not in EncoderModelType._value2member_map_:
            raise ValueError("encoder_type is out of pre-defined types")
        self.encoder_type = opt['encoder_type']
        self.preloaded_config = None

        literal_encoder_type = EncoderModelType(self.encoder_type).name.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
        self.tokenizer = tokenizer_class.from_pretrained(opt['bert_model_type'], do_lower_case=opt['do_lower_case'])
        if opt['cuda']:
            self.network.cuda()
        optimizer_parameters = self._get_param_groups()
        self._setup_optim(optimizer_parameters, state_dict, num_train_step)
        self.para_swapped = False
        self.optimizer.zero_grad()
        self._setup_lossmap(self.config)
        self._setup_kd_lossmap(self.config)

    def _get_param_groups(self):
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.network.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.network.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        if self.config['enable_certainty']:
            return [ {'params': [p for n, p in self.network.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.network.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},{'params':log_vars, 'weight_decay': 0.0 }]
        else:
            return [
            {'params': [p for n, p in self.network.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.network.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

    def _setup_optim(self, optimizer_parameters, state_dict=None, num_train_step=-1):
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(optimizer_parameters, self.config['learning_rate'],
                                       weight_decay=self.config['weight_decay'])

        elif self.config['optimizer'] == 'adamax':
            self.optimizer = Adamax(optimizer_parameters,
                                    self.config['learning_rate'],
                                    warmup=self.config['warmup'],
                                    t_total=num_train_step,
                                    max_grad_norm=self.config['grad_clipping'],
                                    schedule=self.config['warmup_schedule'],
                                    weight_decay=self.config['weight_decay'])
            if self.config.get('have_lr_scheduler', False): self.config['have_lr_scheduler'] = False
        elif self.config['optimizer'] == 'radam':
            self.optimizer = RAdam(optimizer_parameters,
                                    self.config['learning_rate'],
                                    warmup=self.config['warmup'],
                                    t_total=num_train_step,
                                    max_grad_norm=self.config['grad_clipping'],
                                    schedule=self.config['warmup_schedule'],
                                    eps=self.config['adam_eps'],
                                    weight_decay=self.config['weight_decay'])
            if self.config.get('have_lr_scheduler', False): self.config['have_lr_scheduler'] = False
            # The current radam does not support FP16.
            self.config['fp16'] = False
        elif self.config['optimizer'] == 'adam':
            self.optimizer = Adam(optimizer_parameters,
                                  lr=self.config['learning_rate'],
                                  warmup=self.config['warmup'],
                                  t_total=num_train_step,
                                  max_grad_norm=self.config['grad_clipping'],
                                  schedule=self.config['warmup_schedule'],
                                  weight_decay=self.config['weight_decay'])
            if self.config.get('have_lr_scheduler', False): self.config['have_lr_scheduler'] = False
        elif self.config['optimizer'] == 'adadelta':
            self.optimizer = optim.Adadelta(optimizer_parameters,lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])

        if state_dict and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        if self.config['fp16']:
            try:
                from apex import amp
                global amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(self.network, self.optimizer, opt_level=self.config['fp16_opt_level'])
            self.network = model
            self.optimizer = optimizer

        if self.config.get('have_lr_scheduler', False):
            if self.config.get('scheduler_type', 'rop') == 'rop':
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['lr_gamma'], patience=3)
            elif self.config.get('scheduler_type', 'rop') == 'exp':
                self.scheduler = ExponentialLR(self.optimizer, gamma=self.config.get('lr_gamma', 0.95))
            else:
                milestones = [int(step) for step in self.config.get('multi_step_lr', '10,20,30').split(',')]
                self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=self.config.get('lr_gamma'))
        else:
            self.scheduler = None

    def _setup_lossmap(self, config):
        task_def_list: List[TaskDef] = config['task_def_list']
        self.task_loss_criterion = []
        for idx, task_def in enumerate(task_def_list):
            cs = task_def.loss
            lc = LOSS_REGISTRY[cs](name='Loss func of task {}: {}'.format(idx, cs))
            self.task_loss_criterion.append(lc)

    def _setup_kd_lossmap(self, config):
        task_def_list: List[TaskDef] = config['task_def_list']
        self.kd_task_loss_criterion = []
        if config.get('mkd_opt', 0) > 0:
            for idx, task_def in enumerate(task_def_list):
                cs = task_def.kd_loss
                assert cs is not None
                lc = LOSS_REGISTRY[cs](name='Loss func of task {}: {}'.format(idx, cs))
                self.kd_task_loss_criterion.append(lc)

    def train(self):
        if self.para_swapped:
            self.para_swapped = False

    def _to_cuda(self, tensor):
        if tensor is None: return tensor

        if isinstance(tensor, list) or isinstance(tensor, tuple):
            y = [e.cuda(non_blocking=True) for e in tensor]
            for e in y:
                e.requires_grad = False
        else:
            y = tensor.cuda(non_blocking=True)
            y.requires_grad = False
        return y

    def update(self, batch_meta, batch_data):
        self.network.train()
        y = batch_data[batch_meta['label']]
        y = self._to_cuda(y) if self.config['cuda'] else y

        task_id = batch_meta['task_id']
        inputs = batch_data[:batch_meta['input_len']]
            
        weight = None
        if self.config.get('weighted_on', False):
            if self.config['cuda']:
                weight = batch_data[batch_meta['factor']].cuda(non_blocking=True)
            else:
                weight = batch_data[batch_meta['factor']]
        logits = self.mnetwork(*inputs,task_id = task_id,is_train=True)

        # compute loss
        loss = 0
        if self.task_loss_criterion[task_id] and (y is not None):
            loss = self.task_loss_criterion[task_id](logits, y, weight, ignore_index=-1)
            if self.config['enable_certainty']:
                precision = torch.exp(-log_vars[int(task_id)]).cuda()
                loss = torch.sum(precision * loss + log_vars[int(task_id)], -1).cuda()

        self.train_loss.update(loss.item(), batch_data[batch_meta['token_id']].size(0))
        # scale loss
        loss = loss / self.config.get('grad_accumulation_step', 1)
        if self.config['fp16']:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.local_updates += 1
        if self.local_updates % self.config.get('grad_accumulation_step', 1) == 0:
            if self.config['global_grad_clipping'] > 0:
                if self.config['fp16']:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer),
                                                   self.config['global_grad_clipping'])
                else:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                  self.config['global_grad_clipping'])
            self.updates += 1
            # reset number of the grad accumulation
            self.optimizer.step()
            self.optimizer.zero_grad()

    def encode(self, batch_meta, batch_data):
        self.network.eval()
        inputs = batch_data[:3]
        sequence_output = self.network.encode(*inputs)[0]
        return sequence_output

    def extract(self, batch_meta, batch_data):
        self.network.eval()
        # 'token_id': 0; 'segment_id': 1; 'mask': 2
        inputs = batch_data[:3]
        all_encoder_layers, pooled_output = self.mnetwork.bert(*inputs)
        return all_encoder_layers, pooled_output
    
    def extract_tree(self, batch_meta, batch_data):
        self.network.eval()
        # 'token_id': 0; 'segment_id': 1; 'mask': 2
        ast_features, adj = batch_data[3],batch_data[4]
        
        tree_h = []
        for i in range(len(ast_features)):
            ast_feature = torch.tensor(ast_features[i], device="cuda:0", dtype=torch.float32)
            i_adj = torch.tensor(adj[i], device="cuda:0", dtype=torch.int64)
            i_adj_array = i_adj.cpu().numpy()
            node_order, edge_order = calculate_evaluation_orders(i_adj_array, ast_feature.size(0))
            node_order = torch.tensor(node_order, device="cuda:0", dtype=torch.int64)
            edge_order = torch.tensor(edge_order, device="cuda:0", dtype=torch.int64)
            h, c = self.network.tree_model(ast_feature,node_order,i_adj,edge_order)
                    
            tree_h.append(h)
        return tree_h
        #return a vector list
        
 
    def predict(self, batch_meta, batch_data):
        self.network.eval()
        task_id = batch_meta['task_id']
        task_def = TaskDef.from_dict(batch_meta['task_def'])
        task_type = task_def.task_type
        task_obj = tasks.get_task_obj(task_def)
        inputs = batch_data[:batch_meta['input_len']]
        score = self.mnetwork(*inputs, task_id = task_id,is_train=False)
        for name, param in self.mnetwork.named_parameters():
            if name.find("bert") == -1:
                tensorboard.add_histogram(name, param.clone().cpu().data.numpy(), 0)
                '''
                if task_def.data_type in [DataFormat.PremiseOnly]:
                    token_ids, type_ids, masks, ast_features, adj = batch_data[:batch_meta['input_len']]
                    tensorboard.add_graph(self.mnetwork, (token_ids, type_ids, masks, ast_features, adj),operator_export_type="RAW")
                elif task_def.data_type in [DataFormat.PremiseAndOneHypothesisnl]:
                    token_ids, type_ids, masks, ast_features, adj, premise_masks, hypothesis_masks = batch_data[:batch_meta['input_len']]
                    tensorboard.add_graph(self.mnetwork, (token_ids, type_ids, masks, ast_features, adj, premise_masks, hypothesis_masks),operator_export_type="RAW")
                else:
                    token_ids, type_ids, masks, ast_features, adj,anti_ast_features,anti_adj, premise_masks, hypothesis_masks = batch_data[:batch_meta['input_len']]
                    tensorboard.add_graph(self.mnetwork, (token_ids, type_ids, masks, ast_features, adj,anti_ast_features,anti_adj, premise_masks, hypothesis_masks),operator_export_type="RAW")
                '''
        if task_obj is not None:
            score, predict = task_obj.test_predict(score)
        elif task_type == TaskType.Ranking:
            score = score.contiguous().view(-1, batch_meta['pairwise_size'])
            assert task_type == TaskType.Ranking
            score = F.softmax(score, dim=1)
            score = score.data.cpu()
            score = score.numpy()
            predict = np.zeros(score.shape, dtype=int)
            positive = np.argmax(score, axis=1)
            for idx, pos in enumerate(positive):
                predict[idx, pos] = 1
            predict = predict.reshape(-1).tolist()
            score = score.reshape(-1).tolist()
            return score, predict, batch_meta['true_label']
        elif task_type == TaskType.CodeSummarization:
            def compute_bleu(predicts, labels):
                refs = []
                score_list = []

                cc = SmoothingFunction()
                predict_ts = torch.stack(predict).transpose(0,1)#8 511 30522 
                hyp_pre = predict_ts.argmax(dim=2)
                hyp_pre_ls = hyp_pre.numpy().tolist()
                print("hyp_pre.size() : ", hyp_pre.size())
                for hyp, ref in zip(hyp_pre_ls, labels):
                    hyp_all_tokens = self.tokenizer.convert_ids_to_tokens(hyp)
                    ref_all_tokens = self.tokenizer.convert_ids_to_tokens(ref)
                    score = nltk.translate.bleu([ref], hyp[1:], smoothing_function=cc.method4)
                    score_list.append(score)
                    
                return score_list

            predict = score
            scores = compute_bleu(predict,batch_meta['label'])
            return scores, predict, batch_meta['label']#scores list
        elif task_type == TaskType.SeqenceLabeling:
            mask = batch_data[batch_meta['mask']]
            score = score.contiguous()
            score = score.data.cpu()
            score = score.numpy()
            predict = np.argmax(score, axis=1).reshape(mask.size()).tolist()
            valied_lenght = mask.sum(1).tolist()
            final_predict = []
            for idx, p in enumerate(predict):
                final_predict.append(p[: valied_lenght[idx]])
            score = score.reshape(-1).tolist()
            return score, final_predict, batch_meta['label']
        elif task_type == TaskType.Span:
            start, end = score
            predictions = []
            if self.config['encoder_type'] == EncoderModelType.BERT:
                import experiments.squad.squad_utils as mrc_utils
                scores, predictions = mrc_utils.extract_answer(batch_meta, batch_data,start, end, self.config.get('max_answer_len', 5))
            return scores, predictions, batch_meta['answer']
        else:
            raise ValueError("Unknown task_type: %s" % task_type)
        return score, predict, batch_meta['label']

    def save(self, filename):
        network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
        params = {
            'state': network_state,
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }
        torch.save(params, filename)
        logger.info('model saved to {}'.format(filename))

    def load(self, checkpoint):
        model_state_dict = torch.load(checkpoint)
        self.network.load_state_dict(model_state_dict['state'], strict=False)
        self.optimizer.load_state_dict(model_state_dict['optimizer'])
        self.config.update(model_state_dict['config'])

    def cuda(self):
        self.network.cuda()

        
