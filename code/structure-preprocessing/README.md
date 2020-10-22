# parser

Run `java -jar parser.jar -f [filename] -d [dirname]`.

## example
`cd parser`
`java -jar parser.jar -f algorithm.json -d algorithm`

## requirement
Java 1.8


# structure preprocesing

1. Preprocess the parsed data using the pretrained BERT:
	run `python preprocessing_task.py [dir]`.

2. Move the processed data folder `parser/task` to the `../mtl/data` directory

## example

1. `python preprocessing_AACC.py parser/AA`

2. `mv parser/AA ../mtl-master/data`

## requirement

Run `pip install -r requirement.txt`
