# Note: Please refer to https://github.com/uclnlp/emoji2vec for additional instructions on ways to extract embeddings.
# Download file 'emoji2vec.bin' from the above repository
# Python version required - 3.5
# gensim >= 2.3.0

import pandas as pd
import string
import gensim.models as gsm
import emoji
import numpy as np

#Function to extract embeddings from emojis and emoticons
def save_emoji2vec(clean_path,e2v):
    df = pd.read_csv(clean_path)
    emoji_matrix = np.zeros((df.shape[0],300))
    for index,row in df.iterrows():
        emoji_text = row['emoji']
        if type(emoji_text) is not float:
            vector = np.zeros(300)
            count = 0
            for symbol in emoji_text:
                try:
                    vector += e2v[symbol]
                    count += 1
                except KeyError:
                    pass
            if (count > 0):
                vector = vector/count
            emoji_matrix[index,:] = np.reshape(vector,(1,300))
    return emoji_matrix


# Define emoji2vec object
e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)

# Get train emoji embeddings
train_matrix = save_emoji2vec('train_cleaned.csv',e2v) # Change file path to generate embeddings for other training datasets
np.save('train_emoji',train_matrix)

# Get validation emoji embeddings
valid_matrix = save_emoji2vec('valid_cleaned.csv',e2v) # Change file path to generate embeddings for other validation datasets
np.save('valid_emoji',valid_matrix)

# Get test emoji embeddings
test_matrix = save_emoji2vec('test_cleaned.csv',e2v) # Change file path to generate embeddings for other test datasets
np.save('test_emoji',test_matrix)