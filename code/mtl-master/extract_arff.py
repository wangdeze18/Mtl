# coding=utf-8
import argparse
import json
import os
import random
from datetime import datetime
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from pretrained_models import *
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
from experiments.exp_def import TaskDefs
from mtl.inference import eval_model, extract_encoding
from data_utils.log_wrapper import create_logger
from data_utils.task_def import EncoderModelType
from data_utils.utils import set_environment
from mtl.batcher import SingleTaskDataset, MultiTaskDataset, Collater, MultiTaskBatchSampler
from mtl.model import MTLModel

token_features_list = []
tree_features_list = []
label_list = []

def model_config(parser):
    parser.add_argument('--update_bert_opt', default=0, type=int)
    parser.add_argument('--multi_gpu_on', action='store_true')
    parser.add_argument('--mem_cum_type', type=str, default='simple',
                        help='bilinear/simple/defualt')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=0, help='0,1')
    parser.add_argument('--mtl_opt', type=int, default=1)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--init_ratio', type=float, default=1)
    parser.add_argument('--encoder_type', type=int, default=EncoderModelType.BERT)
    parser.add_argument('--num_hidden_layers', type=int, default=-1)

    # BERT pre-training
    parser.add_argument('--bert_model_type', type=str, default='bert-base-uncased')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15)
    parser.add_argument('--short_seq_prob', type=float, default=0.2)
    parser.add_argument('--max_predictions_per_seq', type=int, default=128)
    return parser


def data_config(parser):
    parser.add_argument('--log_file', default='mtl-train.log', help='path for log file.')
    parser.add_argument('--tensorboard', default='True')
    parser.add_argument('--tensorboard_logdir', default='tensorboard_logdir')
    parser.add_argument("--init_checkpoint", default='mtl_models/bert_model_base_uncased.pt', type=str)
    parser.add_argument('--data_dir', default='data/canonical_data/bert_base_uncased_lower/')
    parser.add_argument('--data_sort_on', action='store_true')
    parser.add_argument('--name', default='farmer')
    parser.add_argument('--task_def', type=str, default="experiments/task/validation_task_def.yml")
    parser.add_argument('--train_datasets', default='CommentClassify,AuthorAttr,dfd')#CommentClassify,AuthorAttr,dfd
    parser.add_argument('--test_datasets', default='CommentClassify,AuthorAttr,dfd')#CommentClassify,AuthorAttr,dfd
    parser.add_argument('--mkd-opt', type=int, default=0, 
                        help=">0 to turn on knowledge distillation, requires 'softlabel' column in input data")
    return parser


def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--save_per_updates', type=int, default=10000)
    parser.add_argument('--save_per_updates_on', action='store_true')
    parser.add_argument('--epochs', type=int, default = 100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')
    parser.add_argument('--adam_eps', type=float, default=1e-6)

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.000)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)
    # loading
    parser.add_argument("--model_ckpt", default='checkpoint/model_40.pt', type=str)###############
    parser.add_argument("--resume", action='store_true')

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', default='True')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--freeze_layers', type=int, default=-1)
    parser.add_argument('--embedding_opt', type=int, default=0)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--bert_l2norm', type=float, default=0.0)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_dir', default='input_examples')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--grad_accumulation_step', type=int, default=1)
    
    parser.add_argument('--enable_certainty', action='store_true')
    parser.add_argument('--unenable_multimodal', action='store_true')

    #fp 16
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    return parser
    


def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def write_line(file, text):
    file.write("{}\n".format(text))

def write_attributes( file):
    # Ignore file name and class value
    num_features = 768 * 2

    for i in range(num_features):
        write_line(file, "@ATTRIBUTE x{} NUMERIC".format(i))

def write_class_attribute(file):
    #class_vals = ["1", "2", "3", "4", "5", "6", "9"]
    class_vals = ["0","1"]

    write_line(file, "@ATTRIBUTE class { " + ",".join(class_vals) + " }")

def write_data(file):
    
    write_line(file, "\n\n@DATA")
    for i in range(len(label_list)):
        line = ""
        #line += ','.join(map(str,token_features_list[i]))
        line += ','.join("%.5f" % x for x in token_features_list[i])
        line += ','
        line += ','.join("%.5f" % x for x in tree_features_list[i])
        line += ','
        line += str(label_list[i])
        write_line(file, line)
    ## write feature vector and label        

def main():
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)
    parser.add_argument('--encode_mode', action='store_true', help="only encode test data")
    parser.add_argument('--task_name', type=str, default='algorithm')

    args = parser.parse_args()

    output_dir = args.output_dir
    
    task_name = args.task_name
    prefix = task_name
    data_dir = args.data_dir + prefix + '.json'

    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    set_environment(args.seed, args.cuda)
    log_path = args.log_file
    logger = create_logger(__name__, to_disk=True, log_file=log_path)
    logger.info(args.answer_opt)

    encoder_type = args.encoder_type
    
    opt = vars(args)
    # update data dir
    opt['data_dir'] = data_dir
    batch_size = args.batch_size

    tasks = {}
    task_def_list = []
    dropout_list = []

    datasets = []
        
    task_defs = TaskDefs(args.task_def)
    task_def = task_defs.get_task_def(prefix)
    task_def_list.append(task_def)
    opt['task_def_list'] = task_def_list


    data_path = data_dir
    collater = Collater(is_train=False, encoder_type=encoder_type)
    data_set = SingleTaskDataset(data_path, False, maxlen=args.max_seq_len, task_def=task_def)
    batcher = DataLoader(data_set, batch_size=args.batch_size, collate_fn=collater.collate_fn, pin_memory=args.cuda)

    num_all_batches = len(batcher)

    init_model = args.init_checkpoint
    state_dict = None

    if os.path.exists(init_model):
        state_dict = torch.load(init_model)
        config = state_dict['config']
    else:
        if opt['encoder_type'] not in EncoderModelType._value2member_map_:
            raise ValueError("encoder_type is out of pre-defined types")
        literal_encoder_type = EncoderModelType(opt['encoder_type']).name.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
        config = config_class.from_pretrained(init_model).to_dict()

    config['attention_probs_dropout_prob'] = args.bert_dropout_p
    config['hidden_dropout_prob'] = args.bert_dropout_p
    config['multi_gpu_on'] = opt["multi_gpu_on"]
    if args.num_hidden_layers != -1:
        config['num_hidden_layers'] = args.num_hidden_layers
    opt.update(config)

    model_dict = state_dict
    model = MTLModel(opt, state_dict=state_dict, num_train_step=num_all_batches)
    model_dict = model.state_dict()
    if args.resume and args.model_ckpt:
        logger.info('loading model from {}'.format(args.model_ckpt))
        
        params = torch.load(args.model_ckpt)
        model.load_state_dict(params,strict=False)
    
    for batch_meta, batch_data in batcher:
        batch_meta, batch_data = Collater.patch_data(args.cuda, batch_meta, batch_data)
        all_encoder_layers, _ = model.extract(batch_meta, batch_data) 
        embeddings = torch.mean(all_encoder_layers,dim=1).detach().cpu().numpy()

        token_repre = model.extract_tree(batch_meta, batch_data)
        tree_repre = []
        for i in range(len(token_repre)):
            sum_repre = torch.mean(token_repre[i],dim=0)
            tree_repre.append(sum_repre.detach().cpu().numpy())

        uids = batch_meta['uids']
        y = batch_meta['label']

        
        masks = batch_data[batch_meta['mask']].detach().cpu().numpy().tolist()
        for idx, uid in enumerate(uids):
            token_features_list.append(embeddings[idx].tolist())
            tree_features_list.append(tree_repre[idx].tolist())
            label_list.append(y[idx])

     
    # dump arff features
    file_name = task_name + '_' + args.model_ckpt[args.model_ckpt.find('/') + 1:]
    with open(os.path.join(output_dir, file_name + ".arff"), mode='w') as open_file:
        # Write the relation header
        write_line(open_file, "@RELATION {}\n\n".format(task_name))

        # Write each attribute
        write_attributes(open_file)

        # Write the class attribute
        write_class_attribute(open_file)

        # Write the actual data
        write_data(open_file)
    

if __name__ == '__main__':
    main()


