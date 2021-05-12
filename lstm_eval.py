from torch.nn.modules import linear
import pandas as pd
import numpy as np
import torch
import random
from sklearn import metrics
import transformers
import re
import emoji
import os
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel,AutoTokenizer


# Set seed
random.seed(46)
np.random.seed(46)
torch.manual_seed(46)
torch.cuda.manual_seed_all(46)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Define Dataset Class
class PostDataset(Dataset):
    def __init__(self,dataframe,tokenizer,max_len,labeltype):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.post = self.data['Post+url']
        self.labels = self.data[labeltype]
        self.max_len = max_len

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        post = str(self.post[index])

        inputs = self.tokenizer.encode_plus(
            post,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.labels[index], dtype=torch.float)
        }


# Model Definition
class ModelClass(torch.nn.Module):
    def __init__(self,pretrain_path,dropout,target_labels):
        super(ModelClass, self).__init__()
        self.layer1 = AutoModel.from_pretrained(pretrain_path)
        self.hidden_size = 768 # For most BERT models
        self.lstm1 = torch.nn.LSTM(768,256,batch_first=True,bidirectional=True,dropout=dropout,num_layers=2)
        self.linear = torch.nn.Linear(256*2,target_labels)

    def forward(self,ids,mask,token_type_ids):
        output_1 = self.layer1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = output_1[0]
        output_lstm1 = self.lstm1(output_2)[0]
        output = self.linear(output_lstm1[:,0,:])
        return output


# Class for evaluation
class evaluator():
    def __init__(self,test_data,model_name,model_loc,target):
        self.test_df = test_data
        self.model_loc = model_loc
        self.model_path = model_name 
        self.tokenizer_path = model_name
        self.max_len = 200
        self.TEST_BATCH_SIZE = 8
        self.target_labels = 2
        self.target = target
        self.device = 'cuda' if cuda.is_available() else 'cpu'

        # Define tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        # Create instances of DataSet class
        test_set = PostDataset(self.test_df,tokenizer,self.max_len,self.target)

        # Define dataloaders for test set
        test_params = {'batch_size': self.TEST_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

        self.testing_loader = DataLoader(test_set, **test_params)

        # Load saved model for inference and move it to the GPU
        self.model = torch.load(self.model_loc)
        self.model.to(self.device)

    # Function for running inference on the test set
    def evaluate(self):
        device = self.device
        self.model.eval()
        fin_targets=[]
        fin_outputs=[]
        for iter,data in enumerate(self.testing_loader,0):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            targets = data['targets'].to(device)
            outputs = self.model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.argmax(outputs,1).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets

    # Utility function for generating f1 score
    def generate_f1(self):
        outputs,targets = self.evaluate()
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='binary')
        print()
        print(f"Test Accuracy = {accuracy}")
        print(f"Test F1 Score (Weighted) = {f1_score_micro}")


# Main program:

# Load test set
test_dataframe = pd.read_csv('preprocessing/test_cleaned_new.csv')

# mBERT

# Defamation
inference = evaluator(test_dataframe,"bert-base-multilingual-uncased",'models/al-uncased_defamation_lstm2_final_model.pt','defamation')
inference.generate_f1()

# Fake
inference = evaluator(test_dataframe,"bert-base-multilingual-uncased",'models/al-uncased_fake_lstm2_final_model.pt','fake')
inference.generate_f1()

# Hate
inference = evaluator(test_dataframe,"bert-base-multilingual-uncased",'models/al-uncased_hate_lstm2_final_model.pt','hate')
inference.generate_f1()

# Offensive
inference = evaluator(test_dataframe,"bert-base-multilingual-uncased",'models/al-uncased_offensive_lstm2_final_model.pt','offensive')
inference.generate_f1()
