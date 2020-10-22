# coding=utf-8
import yaml
import os
import numpy as np
import argparse
import json
import sys
from data_utils import load_data
from data_utils.task_def import TaskType, DataFormat
from data_utils.log_wrapper import create_logger
from experiments.exp_def import TaskDefs, EncoderModelType
from pretrained_models import *


DEBUG_MODE = False
MAX_SEQ_LEN = 512
DOC_STRIDE = 180
MAX_QUERY_LEN = 64
MRC_MAX_SEQ_LEN = 384

logger = create_logger(
    __name__,
    to_disk=True,
    log_file='mtl_data_proc_{}.log'.format(MAX_SEQ_LEN))

def feature_extractor(tokenizer, text_a, text_b=None, max_length=512, model_type=None, enable_padding=False, pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=False): # set mask_padding_with_zero default value as False to keep consistent with original setting
    inputs = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_length,
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)

    if enable_padding:
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    if model_type.lower() in ['bert', 'roberta']:
        attention_mask = None

    if model_type.lower() not in ['distilbert','bert', 'xlnet'] :
        token_type_ids = [0] * len(token_type_ids)

    return input_ids,attention_mask, token_type_ids # input_ids, input_mask, segment_id

def build_data(data, dump_path, tokenizer, data_format=DataFormat.PremiseOnly,
               max_seq_len=MAX_SEQ_LEN, encoderModelType=EncoderModelType.BERT, lab_dict=None):
    def build_data_premise_only(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT):
        """Build data of single sentence tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                label = sample['label']
                ast_features = sample['features']
                adj = sample['adj']
                if len(premise) > max_seq_len - 2:
                    premise = premise[:max_seq_len - 2]
                input_ids, input_mask, type_ids = feature_extractor(tokenizer, premise, max_length=max_seq_len, model_type=encoderModelType.name)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids,
                    'ast_features': ast_features,
                    'adj': adj}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_premise_and_one_hypo(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT):
        """Build data of sentence pair tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis = sample['hypothesis']
                label = sample['label']
                input_ids, input_mask, type_ids = feature_extractor(tokenizer, premise, text_b=hypothesis, max_length=max_seq_len,
                                                                    model_type=encoderModelType.name)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids}
                writer.write('{}\n'.format(json.dumps(features)))
    
    def build_data_premise_and_one_hypo_nl(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT):
        """Build data of sentence pair tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis = sample['hypothesis']
                label = sample['label']
                ast_features = sample['features']
                adj = sample['adj']
                
                input_ids, input_mask, type_ids = feature_extractor(tokenizer, premise, text_b=hypothesis, max_length=max_seq_len,
                                                                    model_type=encoderModelType.name)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids,
                    'ast_features': ast_features,
                    'adj': adj}
                writer.write('{}\n'.format(json.dumps(features)))
    
    def build_data_premise_and_one_hypo_code(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT):
        """Build data of sentence pair tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis = sample['hypothesis']
                label = sample['label']
                ast_features = sample['features']
                adj = sample['adj']
                anti_ast_features = sample['anti_features']
                anti_adj = sample['anti_adj']

                input_ids, input_mask, type_ids = feature_extractor(tokenizer, premise, text_b=hypothesis, max_length=max_seq_len,
                                                                    model_type=encoderModelType.name)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids,
                    'ast_features': ast_features,
                    'adj': adj,
                    'anti_ast_features': anti_ast_features,
                    'anti_adj': anti_adj}
                writer.write('{}\n'.format(json.dumps(features)))
    
    def build_data_premise_and_multi_hypo(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT):
        """Build QNLI as a pair-wise ranking task
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis_list = sample['hypothesis']
                label = sample['label']
                input_ids_list = []
                type_ids_list = []
                for hypothesis in hypothesis_list:
                    input_ids, mask, type_ids = feature_extractor(tokenizer,
                                                                        premise, hypothesis, max_length=max_seq_len,
                                                                        model_type=encoderModelType.name)
                    input_ids_list.append(input_ids)
                    type_ids_list.append(type_ids)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids_list,
                    'type_id': type_ids_list,
                    'ruid': sample['ruid'],
                    'olabel': sample['olabel']}
                writer.write('{}\n'.format(json.dumps(features)))


    def build_Code_Sum(data, dump_path, tokenizer, max_seq_len=MAX_SEQ_LEN):
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                label = sample['label']

                tokens = []
                for i, word in enumerate(premise):
                    subwords = tokenizer.tokenize(word)
                    tokens.extend(subwords)
                if len(premise) >  max_seq_len - 2:
                    tokens = tokens[:max_seq_len - 2]

                input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])

                #  convert labels to ids
                labels = []
                for i,word in enumerate(label):
                    subwords = tokenizer.tokenize(word)
                    labels.extend(subwords)
                if len(labels) > max_seq_len - 2:
                    labels = labels[:max_seq_len-2]

                label_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + labels + [tokenizer.sep_token])

                type_ids = [0] * len(input_ids)
                features = {'uid': ids, 'label': label_ids, 'token_id': input_ids, 'type_id': type_ids}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_sequence(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT, label_mapper=None):
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                tokens = []
                labels = []
                for i, word in enumerate(premise):
                    subwords = tokenizer.tokenize(word)
                    tokens.extend(subwords)
                    for j in range(len(subwords)):
                        if j == 0:
                            labels.append(sample['label'][i])
                        else:
                            labels.append(label_mapper['X'])
                if len(premise) >  max_seq_len - 2:
                    tokens = tokens[:max_seq_len - 2]
                    labels = labels[:max_seq_len - 2]

                label = [label_mapper['CLS']] + labels + [label_mapper['SEP']]
                input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
                assert len(label) == len(input_ids)
                type_ids = [0] * len(input_ids)
                features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
                writer.write('{}\n'.format(json.dumps(features)))



    if data_format == DataFormat.PremiseOnly:
        build_data_premise_only(
            data,
            dump_path,
            max_seq_len,
            tokenizer,
            encoderModelType)
    elif data_format == DataFormat.PremiseAndOneHypothesis:
        build_data_premise_and_one_hypo(
            data, dump_path, max_seq_len, tokenizer, encoderModelType)
    elif data_format == DataFormat.PremiseAndOneHypothesisnl:
        build_data_premise_and_one_hypo_nl(
            data, dump_path, max_seq_len, tokenizer, encoderModelType)
    elif data_format == DataFormat.PremiseAndOneHypothesiscode:
        build_data_premise_and_one_hypo_code(
            data, dump_path, max_seq_len, tokenizer, encoderModelType)
    elif data_format == DataFormat.PremiseAndMultiHypothesis:
        build_data_premise_and_multi_hypo(
            data, dump_path, max_seq_len, tokenizer, encoderModelType)
    elif data_format == DataFormat.Seqence:
        build_data_sequence(data, dump_path, max_seq_len, tokenizer, encoderModelType, lab_dict)
    elif data_format == DataFormat.CodeSum:
        build_Code_Sum(data, dump_path, tokenizer, max_seq_len)
    else:
        raise ValueError(data_format)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocessing dataset.')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help='support all BERT, XLNET and ROBERTA family supported by HuggingFace Transformers')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--root_dir', type=str, default='data/canonical_data')
    parser.add_argument('--task_def', type=str, default="experiments/task/task_def.yml")

    args = parser.parse_args()
    return args


def main(args):
    # hyper param
    do_lower_case = args.do_lower_case
    root = args.root_dir
    assert os.path.exists(root)

    literal_model_type = args.model.split('-')[0].upper()
    encoder_model = EncoderModelType[literal_model_type]
    literal_model_type = literal_model_type.lower()
    mtl_suffix = literal_model_type
    if 'base' in args.model:
        mtl_suffix += "_base"
    elif 'large' in args.model:
        mtl_suffix += "_large"

    config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model, do_lower_case=do_lower_case)

    if 'uncased' in args.model:
        mtl_suffix = '{}_uncased'.format(mtl_suffix)
    else:
        mtl_suffix = '{}_cased'.format(mtl_suffix)

    if do_lower_case:
        mtl_suffix = '{}_lower'.format(mtl_suffix)

    mtl_root = os.path.join(root, mtl_suffix)
    if not os.path.isdir(mtl_root):
        os.mkdir(mtl_root)

    task_defs = TaskDefs(args.task_def)

    for task in task_defs.get_task_names():
        task_def = task_defs.get_task_def(task)
        logger.info("Task %s" % task)
        file_path = os.path.join(root, "%s.tsv" % (task))
        if not os.path.exists(file_path):
            logger.warning("File %s doesnot exit")
            sys.exit(1)
            
        rows = load_data(file_path, task_def)
        dump_path = os.path.join(mtl_root, "%s.json" % (task))
        logger.info(dump_path)
        build_data(rows,
                dump_path,
                tokenizer,
                task_def.data_type,
                encoderModelType=encoder_model,
                lab_dict=task_def.label_vocab)


if __name__ == '__main__':
    args = parse_args()
    main(args)
