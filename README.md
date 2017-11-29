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


# FEATURE GENERATION
. prior knowledge
. EDA


# CATEGORICAL AND ORDINAL FEATURES PREPROCESSING



















