# coding=utf-8
import sys
import json
import torch
import random
import numpy as np
from shutil import copyfile
from data_utils.task_def import TaskType, DataFormat
from data_utils.task_def import EncoderModelType
import tasks
from torch.utils.data import Dataset, DataLoader, BatchSampler
from experiments.exp_def import TaskDef

UNK_ID=100
BOS_ID=101

class MultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size, mix_opt, extra_task_ratio):
        self._datasets = datasets
        self._batch_size = batch_size
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        train_data_list = []
        for dataset in datasets:
            train_data_list.append(self._get_shuffled_index_batches(len(dataset), batch_size))
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [list(range(i, min(i+batch_size, dataset_len))) for i in range(0, dataset_len, batch_size)]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(self._train_data_list, self._mix_opt, self._extra_task_ratio)
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].get_task_id()
            batch = next(all_iters[local_task_idx])
            yield [(task_id, sample_id) for sample_id in batch]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices)))
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices

class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for dataset in datasets:
            task_id = dataset.get_task_id()
            assert task_id not in task_id_2_data_set_dic, "Duplicate task_id %s" % task_id
            task_id_2_data_set_dic[task_id] = dataset

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return self._task_id_2_data_set_dic[task_id][sample_id]

class SingleTaskDataset(Dataset):
    def __init__(self, 
                 path,
                 is_train=True,
                 maxlen=512,
                 factor=1.0,
                 task_id=0,
                 task_def: TaskDef =None,
                 bert_model='bert-base-uncased',
                 do_lower_case=True,
                 masked_lm_prob=0.15,
                 seed=13,
                 short_seq_prob=0.1,
                 max_seq_length=512,
                 max_predictions_per_seq=80):
        data, tokenizer = self.load(path, is_train, maxlen, factor, task_def, bert_model, do_lower_case)
        self._data = data
        self._tokenizer = tokenizer
        self._task_id = task_id
        self._task_def = task_def
        # below is for MLM
        if self._task_def.task_type is TaskType.MaskLM:
            assert tokenizer is not None
        # init vocab words
        self._vocab_words = None if tokenizer is None else list(self._tokenizer.vocab.keys())
        self._masked_lm_prob = masked_lm_prob
        self._seed = seed
        self._short_seq_prob = short_seq_prob
        self._max_seq_length = max_seq_length
        self._max_predictions_per_seq = max_predictions_per_seq
        self._rng = random.Random(seed)

    def get_task_id(self):
        return self._task_id

    @staticmethod
    def load(path, is_train=True, maxlen=512, factor=1.0, task_def=None, bert_model='bert-base-uncased', do_lower_case=True):
        task_type = task_def.task_type
        assert task_type is not None


        with open(path, 'r', encoding='utf-8') as reader:
            data = []
            cnt = 0
            for line in reader:
                sample = json.loads(line)
                sample['factor'] = factor
                cnt += 1
                if is_train:
                    task_obj = tasks.get_task_obj(task_def)
                    if task_obj is not None and not task_obj.input_is_valid_sample(sample, maxlen):
                        continue
                    if (task_type == TaskType.Ranking) and (len(sample['token_id'][0]) > maxlen or len(sample['token_id'][1]) > maxlen):
                        continue
                    if (task_type != TaskType.Ranking) and (len(sample['token_id']) > maxlen):
                        continue
                data.append(sample)
            print('Loaded {} samples out of {}'.format(len(data), cnt))
        return data, None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
            return {"task": {"task_id": self._task_id, "task_def": self._task_def}, 
                    "sample": self._data[idx]}

class Collater:
    def __init__(self, 
                 is_train=True,
                 dropout_w=0.005,
                 soft_label=False,
                 encoder_type=EncoderModelType.BERT):
        self.is_train = is_train
        self.dropout_w = dropout_w
        self.soft_label_on = soft_label
        self.encoder_type = encoder_type
        self.pairwise_size = 1

    def __random_select__(self, arr):
        if self.dropout_w > 0:
            return [UNK_ID if random.uniform(0, 1) < self.dropout_w else e for e in arr]
        else: return arr

    @staticmethod
    def patch_data(gpu, batch_info, batch_data):
        if gpu:
            for i, part in enumerate(batch_data):
                if isinstance(part, torch.Tensor):
                    batch_data[i] = part.pin_memory().cuda(non_blocking=True)
                elif isinstance(part, tuple):
                    batch_data[i] = tuple(sub_part.pin_memory().cuda(non_blocking=True) for sub_part in part)
                elif isinstance(part, list):
                    batch_data[i] = [sub_part.pin_memory().cuda(non_blocking=True) for sub_part in part]
                else:
                    raise TypeError("unknown batch data type at %s: %s" % (i, part))
                    
            if "soft_label" in batch_info:
                batch_info["soft_label"] = batch_info["soft_label"].pin_memory().cuda(non_blocking=True)

        return batch_info, batch_data

    def rebatch(self, batch):
        newbatch = []
        for sample in batch:
            size = len(sample['token_id'])
            self.pairwise_size = size
            assert size == len(sample['type_id'])
            for idx in range(0, size):
                token_id = sample['token_id'][idx]
                type_id = sample['type_id'][idx]
                uid = sample['ruid'][idx]
                olab = sample['olabel'][idx]
                newbatch.append({'uid': uid, 'token_id': token_id, 'type_id': type_id, 'label':sample['label'], 'true_label': olab})
        return newbatch
     
    def __if_pair__(self, data_type):
        return data_type in [DataFormat.PremiseAndOneHypothesisnl,DataFormat.PremiseAndOneHypothesiscode]

    def __if_code__(self,data_type):
        return data_type in [DataFormat.PremiseAndOneHypothesiscode]

    def collate_fn(self, batch):
        task_id = batch[0]["task"]["task_id"]
        task_def = batch[0]["task"]["task_def"]
        new_batch = []
        for sample in batch:
            assert sample["task"]["task_id"] == task_id
            assert sample["task"]["task_def"] == task_def
            new_batch.append(sample["sample"])
        task_type = task_def.task_type
        data_type = task_def.data_type
        batch = new_batch

        if task_type == TaskType.Ranking:
            batch = self.rebatch(batch)

        # prepare model input
        batch_info, batch_data = self._prepare_model_input(batch, data_type)
        batch_info['task_id'] = task_id  # used for select correct decoding head
        batch_info['input_len'] = len(batch_data)  # used to select model inputs
        # select different loss function and other difference in training and testing
        # DataLoader will convert any unknown type objects to dict, 
        # the conversion logic also convert Enum to repr(Enum), which is a string and undesirable
        # If we convert object to dict in advance, DataLoader will do nothing
        batch_info['task_def'] = task_def.__dict__ 
        batch_info['pairwise_size'] = self.pairwise_size  # need for ranking task

        # add label
        labels = [sample['label'] for sample in batch]
        task_obj = tasks.get_task_obj(task_def)
        if self.is_train:
            # in training model, label is used by Pytorch, so would be tensor
            if task_obj is not None:
                batch_data.append(task_obj.train_prepare_label(labels))
                batch_info['label'] = len(batch_data) - 1
            elif task_type == TaskType.Ranking:
                batch_data.append(torch.LongTensor(labels))
                batch_info['label'] = len(batch_data) - 1
            elif task_type == TaskType.Span:
                start = [sample['start_position'] for sample in batch]
                end = [sample['end_position'] for sample in batch]
                batch_data.append((torch.LongTensor(start), torch.LongTensor(end)))
                # unify to one type of label
                batch_info['label'] = len(batch_data) - 1
                #batch_data.extend([torch.LongTensor(start), torch.LongTensor(end)])
            elif task_type in (TaskType.SeqenceLabeling, TaskType.CodeSummarization):
                batch_size = self._get_batch_size(batch)
                tok_len = self._get_max_len(batch, key='token_id')
                tlab = torch.LongTensor(batch_size, 512).fill_(-1)
                for i, label in enumerate(labels):
                    ll = len(label)
                    tlab[i, : ll] = torch.LongTensor(label)
                batch_data.append(tlab)
                batch_info['label'] = len(batch_data) - 1
            elif task_type == TaskType.MaskLM:
                batch_size = self._get_batch_size(batch)
                tok_len = self._get_max_len(batch, key='token_id')
                tlab = torch.LongTensor(batch_size, tok_len).fill_(-1)
                for i, label in enumerate(labels):
                    ll = len(label)
                    tlab[i, : ll] = torch.LongTensor(label)
                labels = torch.LongTensor([sample['nsp_lab'] for sample in batch])
                batch_data.append((tlab, labels))
                batch_info['label'] = len(batch_data) - 1

            # soft label generated by ensemble models for knowledge distillation
            if self.soft_label_on and 'softlabel' in batch[0]:
                sortlabels = [sample['softlabel'] for sample in batch]
                sortlabels = task_obj.train_prepare_soft_labels(sortlabels)
                batch_info['soft_label'] = sortlabels
        else:
            # in test model, label would be used for evaluation
            if task_obj is not None:
                task_obj.test_prepare_label(batch_info, labels)
            else:
                batch_info['label'] = labels
                if task_type == TaskType.Ranking:
                    batch_info['true_label'] = [sample['true_label'] for sample in batch]
                if task_type == TaskType.Span:
                    batch_info['token_to_orig_map'] = [sample['token_to_orig_map'] for sample in batch]
                    batch_info['token_is_max_context'] = [sample['token_is_max_context'] for sample in batch]
                    batch_info['doc_offset'] = [sample['doc_offset'] for sample in batch]
                    batch_info['doc'] = [sample['doc'] for sample in batch]
                    batch_info['tokens'] = [sample['tokens'] for sample in batch]
                    batch_info['answer'] = [sample['answer'] for sample in batch]

        batch_info['uids'] = [sample['uid'] for sample in batch]  # used in scoring
        return batch_info, batch_data

    def _get_max_len(self, batch, key='token_id'):
        tok_len = max(len(x[key]) for x in batch)
        return tok_len

    def _get_batch_size(self, batch):
        return len(batch)

    def _prepare_model_input(self, batch, data_type):
        batch_size = self._get_batch_size(batch)
        tok_len = self._get_max_len(batch, key='token_id')
        #tok_len = max(len(x['token_id']) for x in batch)
        premise_len = max(len(x['type_id']) - sum(x['type_id']) for x in batch)
        if self.encoder_type == EncoderModelType.ROBERTA:
            token_ids = torch.LongTensor(batch_size, tok_len).fill_(1)
            type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            masks = torch.LongTensor(batch_size, tok_len).fill_(0)
        else:
            token_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            masks = torch.LongTensor(batch_size, tok_len).fill_(0)
            
        
        ast_features = []
        adj = []
        anti_ast_features = []
        anti_adj = []
        
        if self.__if_pair__(data_type):
            hypothesis_masks = torch.ByteTensor(batch_size, tok_len).fill_(1)
            premise_masks = torch.ByteTensor(batch_size, premise_len).fill_(1)
        for i, sample in enumerate(batch):
            select_len = min(len(sample['token_id']), tok_len)
            tok = sample['token_id']
            if self.is_train:
                tok = self.__random_select__(tok)
            token_ids[i, :select_len] = torch.LongTensor(tok[:select_len])
            type_ids[i, :select_len] = torch.LongTensor(sample['type_id'][:select_len])
            masks[i, : select_len] = torch.LongTensor([1] * select_len)
            
            ast_features.append(torch.Tensor(sample['ast_features']))
            adj.append(torch.Tensor(sample['adj']))
            
            if self.__if_pair__(data_type):
                if self.__if_code__(data_type):
                    anti_ast_features.append(torch.Tensor(sample['anti_ast_features']))
                    anti_adj.append(torch.Tensor(sample['anti_adj']))
                plen = len(sample['type_id']) - sum(sample['type_id'])
                premise_masks[i, :plen] = torch.LongTensor([0] * plen)
                for j in range(plen, select_len):
                    hypothesis_masks[i, j] = 0
        if self.__if_pair__(data_type):
            if self.__if_code__(data_type):
                batch_info = {
                    'token_id': 0,
                    'segment_id': 1,
                    'mask': 2,
                    'ast_features': 3,
                    'adj':4,
                    'anti_ast_features':5,
                    'anti_adj':6,
                    'premise_mask': 7,
                    'hypothesis_mask': 8

                }
                batch_data = [token_ids, type_ids, masks, ast_features, adj,anti_ast_features,anti_adj, premise_masks, hypothesis_masks]
            else:
                batch_info = {
                    'token_id': 0,
                    'segment_id': 1,
                    'mask': 2,
                    'ast_features': 3,
                    'adj': 4,
                    'premise_mask': 5,
                    'hypothesis_mask': 6
                }
                batch_data = [token_ids, type_ids, masks, ast_features, adj, premise_masks, hypothesis_masks]
        else:
            batch_info = {
                'token_id': 0,
                'segment_id': 1,
                'mask': 2,
                'ast_features': 3,
                'adj': 4
            }
            batch_data = [token_ids, type_ids, masks, ast_features, adj]
        return batch_info, batch_data
