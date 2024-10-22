from enum import IntEnum
class TaskType(IntEnum):
    Classification = 1
    Regression = 2
    Ranking = 3
    Span = 4
    SeqenceLabeling = 5
    MaskLM = 6
    codeClassify = 7
    CodeSummarization = 8
    CommentClassify = 9
    algorithm = 10
    javabugs = 11
    opencv = 12

class DataFormat(IntEnum):
    PremiseOnly = 1
    PremiseAndOneHypothesis = 2
    PremiseAndMultiHypothesis = 3
    MRC = 4
    Seqence = 5
    MLM = 6
    CodeSum = 7
    PremiseAndOneHypothesisnl = 8
    PremiseAndOneHypothesiscode = 9

class EncoderModelType(IntEnum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3
    SAN = 4