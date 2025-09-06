import numpy as np
import pandas as pd

df=pd.read_csv(r'D:\VSCodeProjects\sms-spam-classifier\E-Mail-Spam-Classifier\spam.csv', encoding='latin-1')
# print(df.sample(5))
# print(df.shape)

# DATA CLEANING
# print(df.info())

df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
# print(df.sample(5))

df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
# print(df.sample(5))

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['target']=encoder.fit_transform(df['target'])
# print(df.head())

# print(df.isnull().sum())
# print(df.duplicated().sum())
df=df.drop_duplicates(keep='first')
# print(df.duplicated().sum())
print(df.shape)