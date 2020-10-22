import sys
from tqdm import tqdm
from glob import glob
from utils import Node, traverse_label, traverse
import numpy as np
import pickle
import os
import torch
from collections import Counter
import re
from os.path import abspath
import nltk
from transformers import *
import warnings
warnings.filterwarnings("ignore")

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]
    
tokenizer = tokenizer_class.from_pretrained("./bert-base-uncased")

model = model_class.from_pretrained('./bert-base-uncased')



def parse(path):
    features = []
    adj = []

    with open(path, "r",errors='ignore') as f:
        num_objects = f.readline()
        nodes = [Node(num=i, children=[]) for i in range(int(num_objects))]
        for i in range(int(num_objects)):
            label = " ".join(f.readline().split(" ")[1:])[:-1]
            tokens = tokenizer.tokenize(label)
            small_tokens = []
            for token in tokens:
                if token in ['(', ')', '=', '\'', '_']:
                    continue
                small_tokens.append(token)
            input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(small_tokens)).unsqueeze(0)  

            if input_ids.size()[1] > 100:
                input_ids = input_ids[:,:100]

            outputs = model(input_ids)
            last_hidden_states = outputs[0]
            last_hidden_array = last_hidden_states.detach().numpy()
            feature_multi_len = last_hidden_array[0]
            feature = feature_multi_len.sum(0)
            features.append(feature)
            nodes[i].label = label

        while 1:
            line = f.readline()
            if line == "\n":
                break
            p, c = map(int, line.split(" "))
            adj.append([p,c])
            nodes[p].children.append(nodes[c])
            nodes[c].parent = nodes[p]

        nl = f.readline()[:-1]

    return nodes[0],features,adj

def get_method_name(root):
    for c in root.children:
        if c.label == "name (SimpleName)":
            return c.children[0].label[12:-1]

def is_invalid_tree(root):
    labels = traverse_label(root)
    if root.label == 'root (ConstructorDeclaration)':
        return True
    if len(labels) >= 100:
        return True
    method_name = get_method_name(root)
    for word in ["test", "Test", "set", "Set", "get", "Get"]:
        if method_name[:len(word)] == word:
            return True
    return False


def parse_dir(data_dir,path_to_dir):
    files = sorted(glob(path_to_dir + "/*"))
    set_name = path_to_dir.split("/")[-1]

    #skip = 0

    for file in tqdm(files, "parsing {}".format(path_to_dir)):
        number = int(file.split("/")[-1])
        tree,features,adj = parse(file)

        new_dict = {"features": features, "adj": adj}
        with open(data_dir + "/" + "extractfeatures/" + set_name + "/" + str(number), "wb", 1) as f:
            pickle.dump(new_dict, f)

def pickling():
    args = sys.argv

    if len(args) <= 1:
        raise Exception("(usage) $ python preprocessing_task.py [dir]")

    data_dir = args[1]

    dirs = [
        data_dir + "/" + "extractfeatures",
        data_dir + "/" + "extractfeatures/train",
        data_dir + "/" + "extractfeatures/dev",
        data_dir + "/" + "extractfeatures/test"
    ]

    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    for path in [data_dir + "/" + s for s in ["train","test","dev"]]:
        parse_dir(data_dir,path)  

if __name__ == "__main__":
    #nltk.download('punkt')
    sys.setrecursionlimit(10000)
    pickling()
