#!/bin/sh

list_checkpoint="checkpoint/model_0.pt checkpoint/model_10.pt checkpoint/model_20.pt checkpoint/model_30.pt checkpoint/model_40.pt checkpoint/model_50.pt checkpoint/model_60.pt" 
list_task="algorithm opencv javabugs" 
for checkpoint in $list_checkpoint
do
    for task in $list_task
    do
         python extract_arff.py --do_lower_case --bert_model bert-base-uncased --model_ckpt $checkpoint --resume --task_name $task
    done
done
      