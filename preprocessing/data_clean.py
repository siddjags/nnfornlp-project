import pandas as pd
import string
import emoji

class preprocess():

    def __init__(self,csv_file_path):
        self.df = pd.read_csv(csv_file_path)

    # Function to add column for each class and remove unwanted columns
    def clean_csv(self):
        self.df['Post+url'] = ""
        self.df['Post(cleaned)'] = ""
        self.df['Post+emoji'] = ""
        self.df['emoji'] = ""
        self.df['non-hostile'] = 0
        self.df['defamation'] = 0
        self.df['fake'] = 0
        self.df['hate'] = 0
        self.df['offensive'] = 0

        for index,row in self.df.iterrows():
            if 'non-hostile' in row['Labels Set']:
                self.df.at[index,'non-hostile'] = 1

            if 'defamation' in row['Labels Set']:
                self.df.at[index,'defamation'] = 1

            if 'fake' in row['Labels Set']:
                self.df.at[index,'fake'] = 1

            if 'hate' in row['Labels Set']:
                self.df.at[index,'hate'] = 1

            if 'offensive' in row['Labels Set']:
                self.df.at[index,'offensive'] = 1

        self.df.drop(columns = ['Unique ID'],inplace=True)

    # Function that removes punctuation and stop words
    def clean_post(self):
        prefix = 'http'
        for index, row in self.df.iterrows():
            for seperator in string.punctuation:
                text = row['Post'].replace(seperator,'')
            words = []
            words_url = []
            for word in text.split():
                word = word.strip()
                if word:
                    if prefix not in word:
                        words.append(word)
                    words_url.append(word)
            clean_text = ' '.join(words)
            clean_text_url = ' '.join(words_url)
            self.df.at[index,'Post+emoji'] = clean_text
            self.df.at[index,'Post+url'] = clean_text_url

    # Function that separates emojis from post
    def process_emoji(self):
        for index,row in self.df.iterrows():
            allchars = [str for str in row['Post+emoji']]
            emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
            clean_text = ' '.join([str for str in row['Post+emoji'].split() if not any(i in str for i in emoji_list)])
            clean_text_url = ' '.join([str for str in row['Post+url'].split() if not any(i in str for i in emoji_list)])
            emoji_text = ''.join(emoji_list)
            self.df.at[index,'Post(cleaned)'] = clean_text
            self.df.at[index,'Post+url'] = clean_text_url
            self.df.at[index,'emoji'] = emoji_text


train = preprocess('train.csv') # Replace file name with path to downloaded dataset if needed
train.clean_csv()
train.clean_post()
train.process_emoji()
train.df.to_csv('train_cleaned_new.csv',index = False)

valid = preprocess('valid.csv') # Replace file name with path to downloaded dataset if needed
valid.clean_csv()
valid.clean_post()
valid.process_emoji()
valid.df.to_csv('valid_cleaned_new.csv', index = False)

test = preprocess('test.csv') # Replace file name with path to downloaded dataset if needed
test.clean_csv()
test.clean_post()
test.process_emoji()
test.df.to_csv('test_cleaned_new.csv', index = False)
