# nnfornlp-project
Repository for CMU 11-747 Course Project

## Steps to run:
Steps to run Assignment 3
1. Use the dataset constraint_Hindi_Train - Sheet1_combined.csv.csv where the train and validation datasets were combined.
2. Run main_bin_classification.py for binary classification to classify a post as either hostile or non hostile.
3. Run main_multitask_learning.py for multi-class classification. It will run and generate weights for all sub classes- Fake, Hate, Defamation and Offensive.
4. Run generate_csv.py to test the models on the test dataset- Test Set Complete - test.csv.

Steps to run Assignment 4
1. Use the scripts in preprocessing folder to get the augmented datasets.
2. Run aux_bin_classification.py and aux_eval.py for training and evaluating the data on the auxillary model.
3. Run coarse_bin_classification.py and coarse_eval.py for training and evaluating the data for the coarse grained task.
4. Run ensemble_classification.py and ensemble_eval.py for training and evaluating the data on the ensemble model.
5. Run lstm_bin_classification.py and lstm_eval.py for training and evaluating the data on the BERT + LSTM model.

## Dependencies


| Dependency | Version | Installation Command |
| ---------- | ------- | -------------------- |
| Python     | 3.8     | `conda create --name hostile_speech python=3.8` and `conda activate hostile_speech` |
| PyTorch, cudatoolkit    | >=1.5.0, 10.1   | `conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch` |
| Transformers (Huggingface) | 3.5.1 | `pip install transformers==3.5.1` |
| Scikit-learn | >=0.23.1 | `pip install scikit-learn==0.23.1` |
| Pandas | 0.24.2 | `pip install pandas==0.24.2` |
| Numpy | 1.18.5 | `pip install numpy==1.18.5` |
| Emoji | 0.6.0 | `pip install emoji==0.6.0` |
| Deep-translator | | `pip install -U deep_translator` |

## Acknowledgments

We would like to acknowledge the guidance of Ojasv Kamal, a student in IIT Kharagpur, who authored the [paper](https://arxiv.org/abs/2101.05494) we chose to implement, for clarifying parameters used in re-creating mentioned results. We also thank Professor Graham Neubig, Pengfei Liu and Ritam Dutt for providing valuable feedback and suggestions. 



Here is the link to our trained model weights that gave us best results for assignment 3-
https://drive.google.com/drive/folders/1YsIoU-3PHgaMm3hUd8ypIJwH0x0OFyre?usp=sharing

Here is the link to preprocessed datasets, emoji embeddings and trained model weights that gave us best results for assignment 4-
https://drive.google.com/drive/folders/1sjyGIkMwBT9WoyXuvNria0v4xGrgknyP?usp=sharing

