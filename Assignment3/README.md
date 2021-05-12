# nnfornlp-project
Repository for CMU 11-747 Course Project

## Steps to run:
1. Use the dataset constraint_Hindi_Train - Sheet1_combined.csv.csv where the train and validation datasets were combined.
2. Run main_bin_classification.py for binary classification to classify a post as either hostile or non hostile.
3. Run main_multitask_learning.py for multi-class classification. It will run and generate weights for all sub classes- Fake, Hate, Defamation and Offensive.
4. Run generate_csv.py to test the models on the test dataset- Test Set Complete - test.csv.

## Dependencies


| Dependency | Version | Installation Command |
| ---------- | ------- | -------------------- |
| Python     | 3.8     | `conda create --name covid_entities python=3.8` and `conda activate covid_entities` |
| PyTorch, cudatoolkit    | >=1.5.0, 10.1   | `conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch` |
| Transformers (Huggingface) | 3.5.1 | `pip install transformers==3.5.1` |
| Scikit-learn | >=0.23.1 | `pip install scikit-learn==0.23.1` |
| Pandas | 0.24.2 | `pip install pandas==0.24.2` |
| Numpy | 1.18.5 | `pip install numpy==1.18.5` |
| Emoji | 0.6.0 | `pip install emoji==0.6.0` |
| Tqdm | 4.48.2| `pip install tqdm==4.48.2` |

## Acknowledgments

We would like to acknowledge the guidance of Ojasv Kamal, a student in IIT Kharagpur, who authored the [paper](https://arxiv.org/abs/2101.05494) we chose to implement, for clarifying parameters used in re-creating mentioned results.  



Here is the link to our trained model weights that gave us best results-
https://drive.google.com/drive/folders/1YsIoU-3PHgaMm3hUd8ypIJwH0x0OFyre?usp=sharing
