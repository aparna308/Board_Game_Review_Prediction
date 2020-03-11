# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 23:35:03 2020

@author: dell
"""

import sys
import pandas
import matplotlib
import seaborn
import sklearn

print(sys.version)
print(pandas.__version__)
print(matplotlib.__version__)
print(seaborn.__version__)
print(sklearn.__version__)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

games=pandas.read_csv("games.csv")
print(games.columns)
print(games.shape)

plt.hist(games["average_rating"])
plt.show()

games = games[games["users_rated"] > 0]
games=games.dropna(axis=0)
plt.hist(games["average_rating"])
plt.show()

corrmat=games.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True)
plt.show()

columns=games.columns.tolist()
columns=[c for c in columns if c not in["bayes_average_rating","average_rating","type","name","id"]]
target="average_rating"
train=games.sample(frac=0.8,random_state=1)
test=games.loc[~games.index.isin(train.index)]

print(train.shape)
print(test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr=LinearRegression()
lr.fit(train[columns],train[target])

predictions=lr.predict(test[columns])
mean_squared_error(predictions,test[target])

from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)
rfr.fit(train[columns],train[target])

predictions=rfr.predict(test[columns])
mean_squared_error(predictions,test[target])

rating_lr=lr.predict(test[columns].iloc[1].values.reshape(1,-1))
rating_rfr=rfr.predict(test[columns].iloc[1].values.reshape(1,-1))

print(rating_lr)
print(rating_rfr)
test[target].iloc[0]



