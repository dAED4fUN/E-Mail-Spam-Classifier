import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import nltk


df=pd.read_csv(r'D:\VSCodeProjects\sms-spam-classifier\E-Mail-Spam-Classifier\spam.csv', encoding='latin-1')
# print(df.sample(5))
# print(df.shape)

# DATA CLEANING
# print(df.info())

df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
# print(df.sample(5))

df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
# print(df.sample(5))

encoder=LabelEncoder()
df['target']=encoder.fit_transform(df['target'])
# print(df.head())

# print(df.isnull().sum())
# print(df.duplicated().sum())
df=df.drop_duplicates(keep='first')
# print(df.duplicated().sum())
# print(df.shape)

#EDA
# print(df['target'].value_counts())
# plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
# plt.show()
#data is imbalanced

df['num_characters'] = df['text'].apply(len)
# print(df.head())

df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))

df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
# print(df.head())

# print(df[['num_characters', 'num_words', 'num_sentences']].describe())

# print(df[df['target']==0][['num_characters', 'num_words', 'num_sentences']].describe())
# print(df[df['target']==1][['num_characters', 'num_words', 'num_sentences']].describe())

# sns.pairplot(df, hue='target')
# plt.show()

# sns.heatmap(df.drop(columns='text').corr(), annot=True)
# plt.show()