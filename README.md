# Data Preprocessing
Basic imports
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import timedelta
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.filterwarnings('ignore')

Good Notebook to review
https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners/notebook

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
# Save and drop labels
y = train.y
X = X.drop('y', axis=1)

# fill NANs 
X = X.fillna(-999)

# Label encoder
for c in train.columns[train.dtypes == 'object']:
    X[c] = X[c].factorize()[0]
    
rf = RandomForestClassifier()
rf.fit(X,y)


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


### Plotting feature importance
rf = random forest classifier
plt.plot(rf.feature_importances_)
plt.xticks(np.arrange(X.shape[1]),X.columns.tolist())

Investigate the most important features

1. plt.hist(n_bins=50)
2. plt.scatter(x1,x2)
3. df.corr(),plt.matshow()
4. scatter matrix- pd.scatter_matrix(df)

Removing duplicate columns
df.T.drop_duplicates()

Remove duplicate rows







