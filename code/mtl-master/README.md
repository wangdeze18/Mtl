# Mtl

## requirement

1. python3.6

2. Run `pip install -r requirement.txt`

## Usage

1. Preprocess the raw data:
	run `sh prepro.sh`

2. Training:
	run `python train.py`


## validation experiment for generalization

1. Extract embeddings from the pretrained sequence encoder and structure encoder:
	run `sh extract_embeddings.sh`

2. The resulting datasets will be in `.arff` format in the `input_examples/` folder. 

3. Perform 10-fold cross-validation on the dataset using any off-the-shelf [WEKA](https://www.cs.waikato.ac.nz/ml/weka/) classifier.
