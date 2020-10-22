import nltk
from enum import Enum

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import classification_report
from nltk.translate.bleu_score import *

def compute_acc(predicts, labels):
    return 100.0 * accuracy_score(labels, predicts)

def compute_f1(predicts, labels):
    return 100.0 * f1_score(labels, predicts)

def compute_multiF1(predicts, labels):
    return 100.0 * f1_score(labels, predicts,average='weighted')

def compute_f1mac(predicts, labels):
    return 100.0 * f1_score(labels, predicts, average='macro')

def compute_f1mic(predicts, labels):
    return 100.0 * f1_score(labels, predicts, average='micro')

def compute_mcc(predicts, labels):
    return 100.0 * matthews_corrcoef(labels, predicts)

def compute_pearson(predicts, labels):
    pcof = pearsonr(labels, predicts)[0]
    return 100.0 * pcof

def compute_spearman(predicts, labels):
    scof = spearmanr(labels, predicts)[0]
    return 100.0 * scof

def compute_auc(predicts, labels):
    auc = roc_auc_score(labels, predicts)
    return 100.0 * auc

def compute_seqacc(predicts, labels, label_mapper):
    y_true, y_pred = [], []
    def trim(predict, label):
        temp_1 =  []
        temp_2 = []
        for j, m in enumerate(predict):
            if j == 0:
                continue
            if label_mapper[label[j]] != 'X':
                temp_1.append(label_mapper[label[j]])
                temp_2.append(label_mapper[m])
        temp_1.pop()
        temp_2.pop()
        y_true.append(temp_1)
        y_pred.append(temp_2)
    for predict, label in zip(predicts, labels):
        trim(predict, label)
    report = classification_report(y_true, y_pred,digits=4)
    return report


def compute_bleu(predicts, labels):
    refs = []
    count = 0
    total_score = 0.0

    cc = SmoothingFunction()

    for hyp, ref in zip(predicts, labels):
        hyp = hyp.split()
        ref = ref.split()

        score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
        total_score += score
        count += 1

    avg_score = total_score / count
    print ('avg_score: %.4f' % avg_score)
    return avg_score

def compute_kappa(predicts, labels):
    acc = 1.0 * accuracy_score(labels, predicts)
    num = len(predicts)
    num_aa = 0 
    num_bb = 0
    num_ab = 0
    num_ba = 0
    for i in range(num):
        if predicts[i] == 0 and labels[i] == 0:
            num_aa += 1
        elif predicts[i] == 1 and labels[i] == 1:
            num_bb += 1
        elif predicts[i] == 0 and labels[i] == 1:
            num_ab += 1
        else:
            num_ba += 1

    p_e = 1.0 * ((num_aa + num_ab) *(num_aa + num_ba) + (num_bb + num_ab) * (num_bb + num_ba)) / (num * num)

    kappa = (acc-p_e)/(1-p_e)
    return kappa

class Metric(Enum):
    ACC = 0
    F1 = 1
    MCC = 2
    Pearson = 3
    Spearman = 4
    AUC = 5
    SeqEval = 7
    F1MAC = 8
    F1MIC = 9
    Bleu = 10
    Kappa = 11
    multiF1 = 12


METRIC_FUNC = {
    Metric.ACC: compute_acc,
    Metric.F1: compute_f1,
    Metric.MCC: compute_mcc,
    Metric.Pearson: compute_pearson,
    Metric.Spearman: compute_spearman,
    Metric.AUC: compute_auc,
    Metric.SeqEval: compute_seqacc,
    Metric.F1MAC: compute_f1mac,
    Metric.F1MIC: compute_f1mic,
    Metric.Bleu: compute_bleu,
    Metric.Kappa: compute_kappa,
    Metric.multiF1: compute_multiF1
}


def calc_metrics(metric_meta, golds, predictions, scores, label_mapper=None):
    """Label Mapper is used for NER/POS etc. 
    TODO: a better refactor, by xiaodl
    """
    metrics = {}
    for mm in metric_meta:
        metric_name = mm.name
        metric_func = METRIC_FUNC[mm]
        if mm in (Metric.ACC, Metric.F1, Metric.MCC, Metric.F1MAC, Metric.F1MIC, Metric.Bleu, Metric.Kappa, Metric.multiF1):
            #print(predictions)
            #print(golds)
            metric = metric_func(predictions, golds)
        elif mm == Metric.SeqEval:
            metric = metric_func(predictions, golds, label_mapper)
        else:
            if mm == Metric.AUC:
                assert len(scores) == 2 * len(golds), "AUC is only valid for binary classification problem"
                scores = scores[1::2]
            metric = metric_func(scores, golds)
        metrics[metric_name] = metric
    return metrics
