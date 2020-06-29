## Nayef Abou Tayoun
##20141101
##MMAI
##2020
##MMAI891
##28/6/2020

## Q1 all part(1)




## 1- Preparing the Jupyter Environment
### library to install 
#### pip install sklearn
#### pip install regex
#### pip install nltk
#### pip install autocorrect
#### pip install gensim
#### pip install  simpletransformers

import warnings
warnings.filterwarnings("ignore")
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import f1_score
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from autocorrect import Speller
nltk.download('punkt')
nltk.download('stopwords')


##2- Loading the Training and Test Data
df_train = pd.read_csv("sentiment_train.csv")
df_train_EDA=df_train
print(df_train.info())

df_test = pd.read_csv("sentiment_test.csv")
print(df_train.info())


## 3- EDA
### is it balanced data ?
df_train['Polarity'].value_counts().plot(kind='bar',figsize=(7,4));
plt.title('Number of Polarity');
plt.xlabel('Polarity');
plt.ylabel('number');
train_pos = df_train[ df_train['Polarity'] == 1]
train_pos

!pip install wordcloud


from wordcloud import WordCloud,STOPWORDS

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()])
    wordcloud = WordCloud(stopwords=STOPWORDS,regexp=r"\w[\w']+",
                      background_color=color,
                      width=2500,
                      height=2000,
                      max_words=100
                         ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("What is in there ?")
##### To draw worldcloud please use Jupyter notebook for visualization  
##### wordcloud_draw(df_train['Sentence'],'white')


# 4- Pre-processing the Text
## Lowercase, remove tags, remove special characters and number
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("","",text)
    
    # remove special characters and numbers
    text=re.sub("(\\d|\\W)+"," ",text)
    return text

df_train['Sentence'] = df_train['Sentence'].apply(lambda x:pre_process(x))
df_test['Sentence'] = df_test['Sentence'].apply(lambda x:pre_process(x))
df_train['Sentence'][142]

## Tokenizing, Stop-words, Stemming, Lemmatization, Spelling
def stemming(text):
    #nltk.download('wordnet')
    stemmer = SnowballStemmer("english")
    tokenized_word1=tokening(text)
    stemmed_words=[]
    for w in tokenized_word1:
        stemmed_words.append(stemmer.stem(w))
    return " ".join(stemmed_words)

def tokening(text):
    
    #nltk.download('punkt')
    tokenized_word=word_tokenize(text)
    return tokenized_word


def removing_stopwords(text):
    
    #nltk.download('punkt')
    tokenized_word1=tokening(text)
    #nltk.download('stopwords')
    stop_words=set(stopwords.words("english"))
    filtered_sent=[]
    for w in tokenized_word1:
        if w not in stop_words:
            filtered_sent.append(w)
    return " ".join(filtered_sent)
from autocorrect import Speller

def Lemmating(text):
    #nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    tokenized_word1=tokening(text)
    stemmed_words=[]
    for w in tokenized_word1:
        stemmed_words.append(lemmatizer.lemmatize(w))
    return " ".join(stemmed_words)

check = Speller(lang='en')
#df_train['Sentence'] = df_train['Sentence'].apply(lambda x: Speller(x))
#s=df_train['Sentence'][0]
s=df_train['Sentence'][1]
df_train['Sentence'] = df_train['Sentence'].apply(lambda x: check(x))
df_test['Sentence'] = df_test['Sentence'].apply(lambda x: check(x))
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

# Use English stemmer.
#stemmer = SnowballStemmer("english")
#df_train = df_train['Sentence'].apply(lambda x: stemmer.stem(x))
#df_test = df_test.apply(lambda r: TaggedDocument(words=Lemmating(r['Sentence']), tags=[r.Polarity]), axis=1)

#df_test = df_test.apply(lambda r: TaggedDocument(words=stemming(r['Sentence']), tags=[r.Polarity]), axis=1)


#df_train = df_train.apply(lambda r: TaggedDocument(words=Lemmating(r['Sentence']), tags=[r.Polarity]), axis=1)
#df_test = df_test.apply(lambda r: TaggedDocument(words=Lemmating(r['Sentence']), tags=[r.Polarity]), axis=1)

# Use English stemmer.
#stemmer = SnowballStemmer("english")
#df_test = df_test.apply(lambda r: TaggedDocument(words=stemmer.stem(r['Sentence']), tags=[r.Polarity]), axis=1)
#df_train = df_train.apply(lambda r: TaggedDocument(words=stemmer.stem(r['Sentence']), tags=[r.Polarity]), axis=1)


#df_train['Sentence'] = df_train['Sentence'].apply(lambda x: stemming(x))

df_train['Sentence'] = df_train['Sentence'].apply(lambda x: removing_stopwords(x))
train_tagged = df_train.apply(lambda r: TaggedDocument(words=tokening(r['Sentence']), tags=[r.Polarity]), axis=1)
test_tagged = df_test.apply(lambda r: TaggedDocument(words=tokening(r['Sentence']), tags=[r.Polarity]), axis=1)

df_train['Sentence']

# 5- Running Vader library  

nltk.download('vader_lexicon')
%matplotlib inline 

analyser = SentimentIntensityAnalyzer()


def print_sentiment_scores(sentence):
    snt = analyser.polarity_scores(sentence)  #Calling the polarity analyzer
    print("{:-<40} {}".format(sentence, str(snt)))

vadersc = [ ]
for i in range(len(df_train['Sentence'])):
    k = analyser.polarity_scores(df_train.iloc[i]['Sentence'])
    vadersc.append(k['compound'])

df_train['Vader_score']=vadersc
df_train.head(40)

vaderpos = [ ]
for i in range(len(df_train['Sentence'])):
    k = analyser.polarity_scores(df_train.iloc[i]['Sentence'])
    vaderpos.append(k['pos'])
vaderneg = [ ]
for i in range(len(df_train['Sentence'])):
    k = analyser.polarity_scores(df_train.iloc[i]['Sentence'])
    vaderneg.append(k['neg'])

df_train['vader_pos']=vaderpos
df_train['vader_neg']=vaderneg
df_train

i=0
predicted_value = [ ]
for i in range(len(df_train['Sentence'])):
    if ((df_train.iloc[i]['Vader_score'] > 0.0)):
        predicted_value.append(1)
    elif ((df_train.iloc[i]['Vader_score'] <= 0.0)):
        predicted_value.append(0)

df_train['Vader_polarity_comp']=predicted_value


i=0
vader_binary = [ ]

for i in range(len(df_train['Sentence'])):
    if ((df_train.iloc[i]['vader_pos'] > df_train.iloc[i]['vader_neg'])):
        vader_binary.append(1)
    elif ((df_train.iloc[i]['vader_pos'] <= df_train.iloc[i]['vader_neg'])):
        vader_binary.append(0)

df_train['vader_binary']=vader_binary
df_train

# 6- Model Evaluation 

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
  

 
predicted=df_train['vader_binary']
actual=df_train['Polarity']


results = confusion_matrix(actual,predicted)
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(actual, predicted))
print ('F1_score :',f1_score(actual, predicted)) 
print ('Report : ')
print (classification_report(actual, predicted)) 

df_1 = pd.read_csv("sentiment_train.csv")

df_train['Actual Sentence']=df_1['Sentence']
df_train
train_pos = df_train[df_train['Polarity'] == 0]
unmatched_polarity=train_pos[train_pos['vader_binary'] == 1]
unmatched_polarity.head(10)


# Example-1
d2=unmatched_polarity.iloc[1]
d21=unmatched_polarity.iloc[1]

print ('processed Sentence :',d2['Sentence']) 
print ('Actual Sentence :',d21['Actual Sentence']) 
print ('Actual Polarity :',d2['Polarity'])       
print ('Predicted Polarity :',d2['vader_binary']) 
# Example-2
d3=unmatched_polarity.iloc[6]
d31=unmatched_polarity.iloc[6]
print ('processed Sentence :',d3['Sentence']) 
print ('Actual Sentence :',d31['Actual Sentence']) 
print ('Actual Polarity :',d3['Polarity'])       
print ('Predicted Polarity :',d3['vader_binary']) 
# Example-3
d4=unmatched_polarity.iloc[19]
d41=unmatched_polarity.iloc[19]

print ('processed Sentence :',d4['Sentence']) 
print ('Actual Sentence :',d41['Actual Sentence']) 
print ('Actual Polarity :',d4['Polarity'])       
print ('Predicted Polarity :',d4['vader_binary'])  
# Example-4
d5=unmatched_polarity.iloc[30]
d51=unmatched_polarity.iloc[30]

print ('processed Sentence :',d5['Sentence']) 
print ('Actual Sentence :',d51['Actual Sentence']) 
print ('Actual Polarity :',d5['Polarity'])       
print ('Predicted Polarity :',d5['vader_binary']) 

# Example-5
d5=unmatched_polarity.iloc[40]
d51=unmatched_polarity.iloc[40]

print ('processed Sentence :',d5['Sentence']) 
print ('Actual Sentence :',d51['Actual Sentence']) 
print ('Actual Polarity :',d5['Polarity'])       
print ('Predicted Polarity :',d5['vader_binary']) 

df_train['Actual Sentence']=df_1['Sentence']
df_train
train_pos = df_train[df_train['Polarity'] == 0]
unmatched_polarity=train_pos[train_pos['vader_binary'] == 1]
unmatched_polarity.head(20)