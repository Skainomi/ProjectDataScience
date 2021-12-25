import os

from matplotlib import use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from logging import root
from tkinter import *
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
seed = 0
np.random.seed(seed)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid')
from pprint import pprint
import twint
import nest_asyncio
nest_asyncio.apply()

import datetime as dt
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# Scraping 25000 tweets and then store to csv file using twint (https://github.com/twintproject/twint)

# c = twint.Config()
# c.Search = '"kuliah online" lang:id'
# c.Limit = 25000
# c.Store_csv = True
# c.Output = '25k_tweets_data.csv'
# twint.run.Search(c)

# Load data from a CSV file into pandas DataFrame

tweets_data = pd.read_csv('data/25k_tweets_data.csv', nrows=100)
tweets = tweets_data[['id', 'username', 'created_at', 'tweet', 'replies_count', 'retweets_count', 'likes_count']]
# tweets

# Some functions for preprocessing text

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers

    text = text.replace('\n', ' ') # replace new line into space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip(' ') # remove characters space from both left and right text
    return text

def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower() 
    return text

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text) 
    return text

def filteringText(text): # Remove stopwors in a text
    listStopwords = set(stopwords.words('indonesian'))
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered 
    return text

def stemmingText(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text

def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence

# Preprocessing tweets data

tweets['text_clean'] = tweets['tweet'].apply(cleaningText)
tweets['text_clean'] = tweets['text_clean'].apply(casefoldingText)
tweets.drop(['tweet'], axis = 1, inplace = True)

tweets['text_preprocessed'] = tweets['text_clean'].apply(tokenizingText)
tweets['text_preprocessed'] = tweets['text_preprocessed'].apply(filteringText)
tweets['text_preprocessed'] = tweets['text_preprocessed'].apply(stemmingText)

# drop duplicates/spams tweets
tweets.drop_duplicates(subset = 'text_clean', inplace = True)

# Export to csv file
tweets.to_csv(r'data/25k_tweets_data_clean.csv', index = False, header = True,index_label=None)

# Loads lexicon positive and negative data
lexicon_positive = dict()
import csv
with open('data/lexicon_positive.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = dict()
import csv
with open('data/lexicon_negative.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])
        
# Function to determine sentiment polarity of tweets        
def sentiment_analysis_lexicon_indonesia(text):
    #for word in text:
    score = 0
    for word in text:
        if (word in lexicon_positive):
            score = score + lexicon_positive[word]
    for word in text:
        if (word in lexicon_negative):
            score = score + lexicon_negative[word]
    polarity=''
    if (score > 0):
        polarity = 'positive'
    elif (score < 0):
        polarity = 'negative'
    else:
        polarity = 'neutral'
    return score, polarity

# Results from determine sentiment polarity of tweets

results = tweets['text_preprocessed'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
tweets['polarity_score'] = results[0]
tweets['polarity'] = results[1]

# Export to csv file
tweets.to_csv(r'data/25k_tweets_data_clean_polarity.csv', index = False, header = True,index_label=None)


def words_with_sentiment(text):
    positive_words=[]
    negative_words=[]
    for word in text:
        score_pos = 0
        score_neg = 0
        if (word in lexicon_positive):
            score_pos = lexicon_positive[word]
        if (word in lexicon_negative):
            score_neg = lexicon_negative[word]
        
        if (score_pos + score_neg > 0):
            positive_words.append(word)
        elif (score_pos + score_neg < 0):
            negative_words.append(word)
            
    return positive_words, negative_words

def wordCloud():
    sentiment_words = tweets['text_preprocessed'].apply(words_with_sentiment)
    sentiment_words = list(zip(*sentiment_words))
    positive_words = sentiment_words[0]
    negative_words = sentiment_words[1]

    fig, ax = plt.subplots(1, 2,figsize = (12, 10))
    list_words_postive=''
    for row_word in positive_words:
        for word in row_word:
            list_words_postive += ' '+(word)
    wordcloud_positive = WordCloud(width = 800, height = 600, background_color = 'black', colormap = 'Greens'
                                , min_font_size = 10).generate(list_words_postive)
    ax[0].set_title('Word Cloud of Positive Words on Tweets Data \n (based on Indonesia Sentiment Lexicon)', fontsize = 14)
    ax[0].grid(False)
    ax[0].imshow((wordcloud_positive))
    fig.tight_layout(pad=0)
    ax[0].axis('off')

    list_words_negative=''
    for row_word in negative_words:
        for word in row_word:
            list_words_negative += ' '+(word)
    wordcloud_negative = WordCloud(width = 800, height = 600, background_color = 'black', colormap = 'Reds'
                                , min_font_size = 10).generate(list_words_negative)
    ax[1].set_title('Word Cloud of Negative Words on Tweets Data \n (based on Indonesia Sentiment Lexicon)', fontsize = 14)
    ax[1].grid(False)
    ax[1].imshow((wordcloud_negative))
    fig.tight_layout(pad=0)
    ax[1].axis('off')

    plt.show()

def showPie():
    fig, ax = plt.subplots(figsize = (6, 6))
    sizes = [count for count in tweets['polarity'].value_counts()]
    labels = list(tweets['polarity'].value_counts().index)
    explode = (0.1, 0, 0)
    ax.pie(x = sizes, labels = labels, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
    ax.set_title('Sentiment Polarity on Tweets Data \n (total = 100 tweets)', fontsize = 16, pad = 20)
    plt.show()

print(tweets)

usernameData = tweets['username'];
newListUsername = []
for i in list(range(0, len(usernameData))):
    newListUsername.append(usernameData[i])
    
root = Tk()
root.title("Analisis Sentimen Perkuliahan Daring di Indonesia")
def buttonClick():
    showPie()
    

listbox = Listbox(root, listvariable=StringVar(value=newListUsername))


def clickEvent(self):
    index = listbox.curselection()[0]
    text = Text(root)
    tweet = """"{0}"""
    text.insert(END, tweet.format(tweets['text_clean'][index]))
    text.grid(row=1, column=2, rowspan=2)
    Label(root, text=tweets['polarity_score'][index]).grid(row=3, column=2)
    Label(root, text=tweets['polarity'][index]).grid(row=4, column=2)
    
Label(root, text="Select Username").grid(row=0, column=0)
listbox.grid(row=1, column=0, rowspan=6)
listbox.bind('<<ListboxSelect>>', clickEvent)
Label(root, text="Tweet : ").grid(row=1, column=1)
Label(root, text="Polarity Score : ").grid(row=3, column=1)
Label(root, text="Polarity : ").grid(row=4, column=1)
Button(root, text="Show PieChart",command=buttonClick, width=100).grid(row=7, column=0, columnspan=18)
Button(root, text="WordCloud",command=wordCloud, width=100).grid(row=8, column=0, columnspan=18)

root.mainloop()