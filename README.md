## Named Entity Aware Transfer Learning for Biomedical Factoid Question Answering 
## Installation
Please note that this repository is based on the [BioASQ_BioBERT repository](https://github.com/dmis-lab/bioasq-biobert).
### Pre-trained weights (Pre-trained first on NCBI-disease then SQuAD)
We are releasing the pre-trained weights for BioBERT system in the paper. The weights are pre-trained on `SQuAD v1.1` dataset on top of `BioBERT v1.1`(1M steps pre-trained on PubMed corpus).
We only used training set of NCBI-disease and SQuAD datasets.
*   **[`BioBERT v1.1 - NCBI-disease - SQuAD v1.1`](https://drive.google.com/drive/folders/1JSat9F1wC8LzQI3kmy_fsSuQ3lFL5OUj?usp=sharing)
*   **[`bert_config.json`](https://drive.google.com/open?id=17fX1-oChZ5rxu-e-JuaZl2I96q1dGJO4) [`vocab.txt`](https://drive.google.com/open?id=1GQUvBbXvlI_PeUPsZTqh7xQDZMOXh7ko)** : Essential files.

As an alternative option, you may wish to pre-train from scratch. In that case, please follow :
```
1. Fine-tune BioBERT on NCBI-disease dataset
2. Refine-tune BioBERT on SQuAD dataset
2. Use the resulting ckpt of 1 as an initial checkpoint for fine-tuning BioASQ datasets. 
```

## Datasets
We use the pre-processed version of BioASQ 6b/7b - Phase B datasets which provided by the author of the [BioASQ_BioBERT repository](https://github.com/dmis-lab/bioasq-biobert):
*   **[`BioASQ 6b/7b`](https://drive.google.com/open?id=1-KzAQzaE-Zd4jOlZG_7k7D4odqPI3dL1)** (23 MB) Last update : 15th Oct. 2019 

Due to the copyright issue, we can not provide golden answers for BioASQ 6b test dataset at the moment. 
**However, you can extract golden answers for 6b from original BioASQ 7b dataset.**
To use original BioASQ datasets, you should register in [BioASQ website](http://participants-area.bioasq.org). 

For details on the datasets, please see **An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition (Tsatsaronis et al. 2015)**.

## Fine-tuning BioBERT
After downloading one of the pre-trained models, unpack it to any directory you want, which we will denote as `pretrained_model`.
You need to download other essential files ([`bert_config.json`](https://drive.google.com/open?id=17fX1-oChZ5rxu-e-JuaZl2I96q1dGJO4) and [`vocab.txt`](https://drive.google.com/open?id=1GQUvBbXvlI_PeUPsZTqh7xQDZMOXh7ko)) to `$BIOBERT_DIR` as well. 

Please download our pre-processed version of BioASQ-6/7b datasets, and unpack it to `pretrained_model`.

### Training and predicting

Please use `run_factoid_bilstm.py`, `run_factoid_bilstm.py` and `run_factoid_bilstm_bagging.py` for our basic model, basic model with Bilstm and full model respectively.
Use `BioASQ-*.json` as training and testing dataset which we pre-processed the original BioASQ data to SQuAD dataset form. 
This is necessary as the input data format of BioBERT is different from BioASQ dataset format. 
Also, please be informed that the do_lower_case flag should be set as `--do_lower_case=False` since BioBERT model is based on `BERT-BASE (CASED)` model. 

As an example, the following command runs fine-tuning and predicting code on factoid questions (6b; _full abstract_ method) with default arguments.

``` 
export BIOBERT_DIR=$HOME/BioASQ/pretrained_model
export BIOASQ_DIR=$HOME/BioASQ/data-release

python run_factoid.py \
     --do_train=True \
     --do_predict=True \
     --vocab_file=$BIOBERT_DIR/vocab.txt \
     --bert_config_file=$BIOBERT_DIR/bert_config.json \
     --init_checkpoint=$BIOBERT_DIR/model.ckpt-14599 \
     --max_seq_length=384 \
     --train_batch_size=12 \
     --learning_rate=5e-6 \
     --doc_stride=128 \
     --num_train_epochs=5.0 \
     --do_lower_case=False \
     --train_file=$BIOASQ_DIR/BioASQ-6b/train/Full-Abstract/BioASQ-train-factoid-6b-full-annotated.json \
     --predict_file=$BIOASQ_DIR/BioASQ-6b/test/Full-Abstract/BioASQ-test-factoid-6b-3.json \
     --output_dir=/tmp/factoid_output/
```
You can change the arguments as you want. Once you have trained your model, you can use it in inference mode by using `--do_train=false --do_predict=true` for evaluating other json file with identical structure.

The predictions will be saved into a file called `predictions.json` and `nbest_predictions.json` in the `output_dir`.
Run transform file `transform_n2b_factoid.py` in `./biocodes/` folder to convert `nbest_predictions.json` or `predictions.json` to BioASQ JSON format, which will be used for the official evaluation.
```
python ./biocodes/transform_n2b_factoid.py --nbest_path={QA_output_dir}/nbest_predictions.json --output_path={output_dir}
```
This will generate `BioASQform_BioASQ-answer.json` in `{output_dir}`.
Clone **[`evaluation code`](https://github.com/BioASQ/Evaluation-Measures)** from BioASQ github and run evaluation code on `Evaluation-Measures` directory. Please note that you should put 5 as parameter for -e if you are evaluating the system for BioASQ 5b/6b/7b dataset .
```
cd Evaluation-Measures
java -Xmx10G -cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 \
    $BIOASQ_DIR/6B1_golden.json \
    RESULTS_PATH/BioASQform_BioASQ-answer.json
```
As our example is on factoid questions, the result will be like
``` 
0.0 0.4358974358974359 0.6153846153846154 0.5072649572649572 0.0 0.0 0.0 0.0 0.0 0.0
```
## Requirement
* GPU (Our setting was Titan Xp with 12Gb graphic memory)
* Python 3 (Not working on python 2; encoding issues for run_yesno.py)
* TensorFlow v1.11 (Not working on TF v2)
* For other software requirement details, please check `requirements.txt` 

## License and Disclaimer
Please see and agree `LICENSE` file for details. Downloading data indicates your acceptance of our disclaimer.

