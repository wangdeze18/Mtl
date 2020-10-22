from transformers import *
#from pytorch_pretrained_bert import *
from module.san_model import SanModel
MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "san": (BertConfig, SanModel, BertTokenizer),
}