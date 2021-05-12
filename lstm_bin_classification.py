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
from torchnlp.nn import LockedDropout


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
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.labels[index], dtype=torch.long)
        }


# Model Definition
class ModelClass(torch.nn.Module):
    def __init__(self,pretrain_path,dropout,target_labels):
        super(ModelClass, self).__init__()
        self.layer1 = AutoModel.from_pretrained(pretrain_path)
        self.hidden_size = 768 # For most BERT models
        self.lstm1 = torch.nn.LSTM(768,256,batch_first=True,bidirectional=True,dropout=dropout,num_layers=2) # Bi-LSTM layer
        self.linear = torch.nn.Linear(256*2,target_labels)

    def forward(self,ids,mask,token_type_ids):
        output_1 = self.layer1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = output_1[0]
        output_lstm1 = self.lstm1(output_2)[0]
        output = self.linear(output_lstm1[:,0,:])
        return output


# Class for binary classification
class binary_classifier():
    def __init__(self,train_data,valid_data,model_name,target,epochs,lr):
        self.train_df = train_data
        self.valid_df = valid_data
        self.output_dir = './models'
        self.model_path = model_name 
        self.tokenizer_path = model_name
        self.max_len = 200
        self.TRAIN_BATCH_SIZE = 32
        self.VALID_BATCH_SIZE = 4
        self.epochs = epochs
        self.lr = lr
        self.target_labels = 2
        self.dropout = 0.3
        self.target = target
        self.device = 'cuda' if cuda.is_available() else 'cpu'

        # Define tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        # Create instances of DataSet class
        training_set = PostDataset(self.train_df,tokenizer,self.max_len,self.target)
        valid_set = PostDataset(self.valid_df,tokenizer,self.max_len,self.target)

        # Define dataloaders for train and validation sets
        train_params = {'batch_size': self.TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

        valid_params = {'batch_size': self.VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

        self.training_loader = DataLoader(training_set, **train_params)
        self.testing_loader = DataLoader(valid_set, **valid_params)

        # Create instance of Model class and move it to the GPU
        self.model = ModelClass(self.model_path, self.dropout, self.target_labels)
        self.model.to(self.device)

        # Define Optimizer and loss function for training
        self.optimizer = torch.optim.AdamW(params = self.model.parameters(), lr=self.lr)
        self.loss = torch.nn.CrossEntropyLoss()
        os.makedirs(self.output_dir, exist_ok=True)

    # Function for training the model
    def train(self,epoch):
        device = self.device
        self.model.train()
        for iter,data in enumerate(self.training_loader, 0):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            targets = data['targets'].to(device)

            outputs = self.model(ids, mask, token_type_ids)

            self.optimizer.zero_grad()
            loss = self.loss(outputs, targets)
                
            if iter%100==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # Function for evaluating the validation set
    def validation(self, epoch):
        device = self.device
        self.model.eval()
        fin_targets=[]
        fin_outputs=[]
        for iter, data in enumerate(self.testing_loader, 0):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            targets = data['targets'].to(device)
            outputs = self.model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.argmax(outputs,1).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets

    # Function for training + validation
    def train_model(self):
        best_score = 0
        for epoch in range(self.epochs):
            self.train(epoch)
            outputs, targets = self.validation(epoch)
            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average='binary')
            print(f"Accuracy = {accuracy}")
            print(f"F1 Score = {f1_score_micro}")
            print()

            # Save model with highest validation f1 score so far
            if f1_score_micro>=best_score:
                torch.save(self.model,os.path.join(self.output_dir, self.model_path[-10:] + "_" + self.target + '_lstm2_final' + "_model.pt"))
                best_score = f1_score_micro
 

# # Main program

# Load validation set
valid_dataframe = pd.read_csv('preprocessing/valid_cleaned_new.csv')

# Train mBERT-based bi-LSTM models

# Defamation
train_dataframe = pd.read_csv('preprocessing/augmented_defame.csv')
classifier = binary_classifier(train_dataframe,valid_dataframe,"bert-base-multilingual-uncased",'defamation',15,1e-5)
classifier.train_model()

# Fake
train_dataframe = pd.read_csv('preprocessing/augmented_fake.csv')
classifier = binary_classifier(train_dataframe,valid_dataframe,"bert-base-multilingual-uncased",'fake',15,1e-5)
classifier.train_model()

# Hate
train_dataframe = pd.read_csv('preprocessing/augmented_hate.csv')
classifier = binary_classifier(train_dataframe,valid_dataframe,"bert-base-multilingual-uncased",'hate',15,1e-5)
classifier.train_model()

# Offense
train_dataframe = pd.read_csv('preprocessing/augmented_offense.csv')
classifier = binary_classifier(train_dataframe,valid_dataframe,"bert-base-multilingual-uncased",'offensive',15,1e-5)
classifier.train_model()

