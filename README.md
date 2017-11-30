# Data Preprocessing

## Scaling [0,1]- Tree based models dont depend on scaling. Non Tree models depend on scaling.


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
xtrain= scaler.fit_transform(train[['Age'],'Sbup']]
pd.Dataframe(xtrain).hist(figsize=(10,4))


## Standard scaler[mean 0 and std=1]

from sklearn.preprocessing import StardardScaler


## Preprocessing outliers
define upperbound and lowerbound
UPPERBOUND, LOWERBOUND= np.percentile(x,[1,99])
y= np.clip(x,UPPERBOUND,LOWERBOUND)
pd.Series(y).hist(bins=30)

## Log
np.log(1+x)


## FEATURE GENERATION
. prior knowledge
. EDA


## CATEGORICAL AND ORDINAL FEATURES PREPROCESSING
Ordinal- categorical values that are sorted- for example: Ticket: 1,2,3. Ticket 1 is expensive where ticket 3 is cheap. But ticket 3-ticket 3 is not ticket 2
Drivers license A,B,C,D

## LABEL CODING- TREE BASED- YES

## Bag of words( Post processing)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus)
  which means TF- termed frequency
  
  tf= 1/x.sum(axis=1)[:,None]
  x=x*tf
  IDF= Inverse document frequency
  scales down high frequency words
  
## Text proprocessing( DO this first before BOW)
1. Lowercase - Use count vectorizer-from sklearn.feature_extraction.text import CountVectorizer
2. lemmitization
3. Stemming
I had a car and we have cars become I have a car and we have car
Both L and S can achieve this.
Saw becomes s by stemming
Saw becomes see or saw by lemmitization depending upon context

4. Stopwords
words that dont contain useful info. Articles
NLTK 
CountVectorizer max_df
## Word2vec
Text<< vector
https://rare-technologies.com/word2vec-tutorial/

## Exploratory Data Analysis - Week2











