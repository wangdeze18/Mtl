import json
import numpy as np

from data_utils.task_def import TaskType, DataFormat
import tasks
import pickle
import torch
import random
def load_data(file_path, task_def):
    data_format = task_def.data_type
    task_type = task_def.task_type
    label_dict = task_def.label_vocab
    path = task_def.path
    
    if task_type == TaskType.Ranking:
        assert data_format == DataFormat.PremiseAndMultiHypothesis

    rows = []
    index = 0##for the location of the AST feature document
    for line in open(file_path, encoding="utf-8"):
        fields = line.strip("\n").split("\t")
        if data_format == DataFormat.PremiseOnly:
            
            ast_path = path + '/' + str(index)
            index += 1 

            with open(ast_path,'rb') as df:
                ast_dict = pickle.load(df)
            
            features = ast_dict['features']

            #if len(features) > 500:
            #    continue 
            #if int(fields[1]) == 1:
            #    temp_int = random.randint(1,5)
            #    if temp_int > 1:
            #        continue
            for i in range(len(features)):
                features[i] = features[i].tolist()
            
            
            assert len(fields) == 3
            row = {"uid": fields[0], "label": fields[1], "premise": fields[2],
                "features": features,
                "adj": ast_dict['adj']}
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            
            assert len(fields) == 4
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3]}
        elif data_format == DataFormat.PremiseAndOneHypothesisnl:
            ast_path = path + '/' + str(index)
            index += 1 

            with open(ast_path,'rb') as df:
                ast_dict = pickle.load(df)

            features = ast_dict['features']
            for i in range(len(features)):
                features[i] = features[i].tolist()
            
            #no change
            assert len(fields) == 4
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3],
                "features": features,
                "adj": ast_dict['adj']}
        elif data_format == DataFormat.PremiseAndOneHypothesiscode:
            ast_path = path + '/' + str(index)
            index += 1 
            with open(ast_path,'rb') as df:
                ast_dict = pickle.load(df)
            
            features = ast_dict['features']
            for i in range(len(features)):
                features[i] = features[i].tolist()
            
            anti_features = ast_dict['anti_features']
            for i in range(len(anti_features)):
                anti_features[i] = anti_features[i].tolist()
            
            assert len(fields) == 4
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3],
                "features": features,
                "adj": ast_dict['adj'],
                "anti_features": anti_features,
                "anti_adj": ast_dict['anti_adj']
            }
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            assert len(fields) > 5
            row = {"uid": fields[0], "ruid": fields[1].split(","), "label": fields[2], "premise": fields[3],
                   "hypothesis": fields[4:]}
        elif data_format == DataFormat.Seqence:
            row = {"uid": fields[0], "label": eval(fields[1]),  "premise": eval(fields[2])}

        elif data_format == DataFormat.MRC:
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3]}
        elif data_format == DataFormat.CodeSum:
            row = {
                "uid": fields[0],
                "premise": fields[1],
                "label": fields[2]}
        else:
            raise ValueError(data_format)

        task_obj = tasks.get_task_obj(task_def)
        if task_obj is not None:
            row["label"] = task_obj.input_parse_label(row["label"])
        elif task_type == TaskType.Ranking:
            labels = row["label"].split(",")
            if label_dict is not None:
                labels = [label_dict[label] for label in labels]
            else:
                labels = [float(label) for label in labels]
            row["label"] = int(np.argmax(labels))
            row["olabel"] = labels
        elif task_type == TaskType.Span:
            pass  # don't process row label
        elif task_type == TaskType.SeqenceLabeling:
            assert type(row["label"]) is list
            row["label"] = [label_dict[label] for label in row["label"]]
        elif task_type == TaskType.CodeSummarization:
            str_label = row["label"]
            str_label = str_label.strip().split(' ')
            row["label"] = str_label

        rows.append(row)
        
    return rows

def load_split_data(file_path, task_def,split_name):
    data_format = task_def.data_type
    task_type = task_def.task_type
    label_dict = task_def.label_vocab
    path = task_def.path
    #print("path = ", path)
    
    if task_type == TaskType.Ranking:
        assert data_format == DataFormat.PremiseAndMultiHypothesis

    rows = []
    index = 0##for the location of the AST feature document
    for line in open(file_path, encoding="utf-8"):
        fields = line.strip("\n").split("\t")
        if data_format == DataFormat.PremiseOnly:
            
            ast_path = path + '/' + split_name + '/' + str(index)
            #print (features,adj,nl)
            with open(ast_path,'rb') as df:
                ast_dict = pickle.load(df)
            
            features = ast_dict['features']
            for i in range(len(features)):
                features[i] = features[i].tolist()
            
            
            assert len(fields) == 3
            row = {"uid": fields[0], "label": fields[1], "premise": fields[2],
                "features": features,
                "adj": ast_dict['adj']}
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            
            assert len(fields) == 4
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3]}
        elif data_format == DataFormat.PremiseAndOneHypothesisnl:
            ast_path = path + '/' + split_name + '/' + str(index)
            with open(ast_path,'rb') as df:
                ast_dict = pickle.load(df)
            features = ast_dict['features']
            for i in range(len(features)):
                features[i] = features[i].tolist()
            
            #no change
            assert len(fields) == 4
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3],
                "features": features,
                "adj": ast_dict['adj']}
        elif data_format == DataFormat.PremiseAndOneHypothesiscode:
            ast_path = path + '/' + split_name + '/' + str(index)
            with open(ast_path,'rb') as df:
                ast_dict = pickle.load(df)
            
            features = ast_dict['features']
            for i in range(len(features)):
                features[i] = features[i].tolist()
            
            anti_features = ast_dict['anti_features']
            for i in range(len(anti_features)):
                anti_features[i] = anti_features[i].tolist()
            
            assert len(fields) == 4
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3],
                "features": features,
                "adj": ast_dict['adj'],
                "anti_features": anti_features,
                "anti_adj": ast_dict['anti_adj']
            }
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            assert len(fields) > 5
            row = {"uid": fields[0], "ruid": fields[1].split(","), "label": fields[2], "premise": fields[3],
                   "hypothesis": fields[4:]}
        elif data_format == DataFormat.Seqence:
            row = {"uid": fields[0], "label": eval(fields[1]),  "premise": eval(fields[2])}

        elif data_format == DataFormat.MRC:
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3]}
        elif data_format == DataFormat.CodeSum:
            row = {
                "uid": fields[0],
                "premise": fields[1],
                "label": fields[2]}
        else:
            raise ValueError(data_format)

        task_obj = tasks.get_task_obj(task_def)
        if task_obj is not None:
            row["label"] = task_obj.input_parse_label(row["label"])
        elif task_type == TaskType.Ranking:
            labels = row["label"].split(",")
            if label_dict is not None:
                labels = [label_dict[label] for label in labels]
            else:
                labels = [float(label) for label in labels]
            row["label"] = int(np.argmax(labels))
            row["olabel"] = labels
        elif task_type == TaskType.Span:
            pass  # don't process row label
        elif task_type == TaskType.SeqenceLabeling:
            assert type(row["label"]) is list
            row["label"] = [label_dict[label] for label in row["label"]]
        elif task_type == TaskType.CodeSummarization:
            str_label = row["label"]
            #print(str_label)
            str_label = str_label.strip().split(' ')
            #print(str_label[0])
            row["label"] = str_label

        rows.append(row)
        index += 1 
    return rows

def load_score_file(score_path, n_class):
    sample_id_2_pred_score_seg_dic = {}
    score_obj = json.loads(open(score_path, encoding="utf-8").read())
    assert (len(score_obj["scores"]) % len(score_obj["uids"]) == 0) and \
           (len(score_obj["scores"]) / len(score_obj["uids"]) == n_class), \
        "scores column size should equal to sample count or multiple of sample count (for classification problem)"

    scores = score_obj["scores"]
    score_segs = [scores[i * n_class: (i+1) * n_class] for i in range(len(score_obj["uids"]))]
    for sample_id, pred, score_seg in zip(score_obj["uids"], score_obj["predictions"], score_segs):
        sample_id_2_pred_score_seg_dic[sample_id] = (pred, score_seg)
    return sample_id_2_pred_score_seg_dic