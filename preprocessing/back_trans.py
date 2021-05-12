import deep_translator
from deep_translator import GoogleTranslator
import pandas as pd

df = pd.read_csv('train_cleaned_new.csv') # Path to cleaned/pre-processsed training set
df_new = pd.DataFrame(columns=df.columns)
df_defame = pd.DataFrame(columns=df.columns)
df_hate = pd.DataFrame(columns=df.columns)
df_offense = pd.DataFrame(columns=df.columns)
df_fake = pd.DataFrame(columns=df.columns)

count = 0

# Perform data augmentation by back-translating posts having hostile labels

for index,row in df.iterrows():
    if row['fake'] == 1 or row['defamation'] == 1 or row['hate'] == 1 or row['offensive'] == 1:
        clean_text = row['Post(cleaned)']
        clean_text_url = row['Post+url']
        try:
            clean_text_trans = GoogleTranslator(source='auto', target='en').translate(clean_text)
            clean_text_back = GoogleTranslator(source='auto', target='hi').translate(clean_text_trans)
            clean_text_trans_url = GoogleTranslator(source='auto', target='en').translate(clean_text_url)
            clean_text_back_url = GoogleTranslator(source='auto', target='hi').translate(clean_text_trans_url)
            row['Post(cleaned)'] = clean_text_back
            row['Post+url'] = clean_text_back_url
            df_new = df_new.append(row)
            if row['defamation'] == 1:
                df_defame = df_defame.append(row)
            if row['hate'] == 1:
                df_hate = df_hate.append(row)
            if row['offensive'] == 1:
                df_offense = df_offense.append(row)
            if row['fake'] == 1:
                df_fake = df_fake.append(row)
            count += 1
            print('row ' + str(count) + ' done!')
        except deep_translator.exceptions.NotValidPayload:
            print('Exception handling')

# The following lines of code should create an augmented dataset for each hostile label in the form of a csv file

df_defame = df.append(df_defame)
df_defame.to_csv('augmented_defame.csv',index=False)

df_fake = df.append(df_fake)
df_fake.to_csv('augmented_fake.csv',index=False)

df_hate = df.append(df_hate)
df_hate.to_csv('augmented_hate.csv',index=False)

df_offense = df.append(df_offense)
df_offense.to_csv('augmented_offense.csv',index=False)


