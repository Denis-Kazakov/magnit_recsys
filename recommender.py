#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import pandas as pd
from surprise.dump import load


# Data for the recommendation model
train = pd.read_csv('my_project/model/train_joke_df.csv')
test = pd.read_csv('my_project/model/test_joke_df_nofactrating.csv', index_col='InteractionID')
full = pd.concat((train, test), axis=0, join='outer', ignore_index=True)

model = load('my_project/model/model_svd')[1]

def prediction(JID, UID):
    return model.predict(str(int(UID)), str(int(JID))).est


def ranking(uid, downrate=10):
    '''Joke ratings for a specific user'''
    result = pd.DataFrame({'JID': range(1, 101)})
    # Ratings predicted by the model
    result['predicted_rating'] = result.JID.apply(prediction, UID=uid)

    # Add true ratings
    true_ratings = train.query('UID == @uid').drop(columns='UID')
    result = result.merge(true_ratings, how='left', on='JID')
    missing = result.Rating.isna()
    result.loc[missing, 'Rating'] = result.loc[missing, 'predicted_rating']

    best_joke_index = result.Rating.idxmax()
    best_joke_number = result.JID[best_joke_index]
    best_joke_rating = result.Rating[best_joke_index]

    #Downrate jokes that have already been rated
    rated = full.query('UID == @uid').drop(columns=['UID', 'Rating'])
    rated['downrate'] = downrate
    result = result.merge(rated, how='left', on='JID')
    result.fillna(0, inplace=True)
    result.loc[:, 'Rating'] = result.Rating - result.downrate

    # Select top 10 jokes to recommend
    result.sort_values(by='Rating', ascending=False, inplace=True)
    top10 =  result.JID.iloc[0:10].tolist()
    return [{best_joke_number: best_joke_rating}, top10]


# path = 'my_project/data/' + sys.argv[1]
path = 'my_project/data/test.csv'

df = pd.read_csv(path, index_col=0)
df['recommendations'] = df.UID.apply(ranking)
df.to_csv('my_project/data/recommendations.csv')

