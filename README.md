# Data Preprocessing

## Scaling


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
xtrain= scaler.fit_transform(train[['Age'],'Sbup']]
pd.Dataframe(xtrain).hist(figsize=(10,4))

























