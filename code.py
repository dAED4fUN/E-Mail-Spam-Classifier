import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from wordcloud import WordCloud
from collections import Counter

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

ps = PorterStemmer()
#Data Preprocessing
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i) 
        # if i.isalnum()==False:
        #     text.remove(i)
    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# print(transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."))
# print(df['text'][10])
df['transformed_text']=df['text'].apply(transform_text)
# print(df.head())

# wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
# spam_wc = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))
# plt.imshow(spam_wc)
# plt.axis("off")
# plt.show()
# ham_wc = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))
# plt.imshow(ham_wc)
# plt.axis("off")
# plt.show()

spam_corpus = []
for msg in df['transformed_text'][df['target']==1].tolist():
    for word in msg.split():
        spam_corpus.append(word)
# print(len(spam_corpus))

# sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
# plt.xticks(rotation='vertical')
# plt.show()

ham_corpus = []
for msg in df['transformed_text'][df['target']==0].tolist():
    for word in msg.split():
        ham_corpus.append(word)