# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:40:28 2019

@author: Sean
"""
import pandas as pd

file_path= r'C:\Users\Sean\Documents\Work\Miracle Soft\Hands_on_yelp_1\yelp-dataset\yelp_academic_dataset_review.json'

f = open( file_path, encoding="ISO-8859-1")

lst = []
#read json as json reader object and append list with each chunk
reader = pd.read_json( f, lines=True, chunksize=10000)
for num, chunk in zip(range(200),reader):
    x = pd.DataFrame(chunk)
    lst.append(x)

#turn list of dataframes into single dataframe
y = pd.DataFrame(columns=x.columns)
for item in lst:
    y = pd.concat([y,item])

#drop duplicates
print(y.shape)
y = y.drop_duplicates(keep='first')
print(y.shape)

#for generating test file
if 0 == 1:
    y = y.drop(y.index[[0,999999]])

#drop unneeded cols    
y = y.drop([ 'review_id','business_id', 'cool', 'date', 'funny', 'useful', 'user_id'], axis=1)

print(y.head())

y.to_csv(r'C:\Users\Sean\Documents\Work\Miracle Soft\Hands_on_yelp_1\review1m_test.csv')
