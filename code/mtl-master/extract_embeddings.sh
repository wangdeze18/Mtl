python prepro.py --model bert-base-uncased --root_dir data/canonical_data --task_def experiments/task/validation_task_def.yml --do_lower_case $1
python extract_arff.py --bert_model bert-base-uncased --model_ckpt checkpoint/model_40.pt --task_name algorithm