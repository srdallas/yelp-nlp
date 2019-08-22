# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:54:33 2019

@author: Sean
"""

'''
Utilizes/tests a number of classifiers and regressors
change 0 to 1 if you want to test
Input of preprocessed data with 3 stars
'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt

import time
start = time.time()

#specify file path to read test data from and read it into a dataframe
file_path = r'C:\Users\Sean\Documents\Work\Miracle Soft\Hands_on_yelp_1\review50k_with_3.csv'
f = open( file_path, encoding="ISO-8859-1")
data = pd.read_csv(f)

#seperate stars and text data
star = data['stars'].values.tolist()
reviews_text = data['text'].astype(str).values.tolist()
#reviews_text = list(data.iloc[:,0].astype(str))
#star = list(data.iloc[:,1])

#setup vectorizer
vectorizer = TfidfVectorizer(max_features = 5000, min_df=5)
vector_x = vectorizer.fit_transform(reviews_text)
vector_x_array = vector_x.toarray()

X_train, X_test, y_train, y_test = train_test_split( vector_x_array, star, test_size=0.2, random_state=42)

if 0 == 1:
    """
    GridSearch LogReg
    C of 1,2,3,4,5,6,7,8,9,10
    https://www.youtube.com/watch?v=Gol_qOgRqfA
    """
    model = LogisticRegression(solver='lbfgs', max_iter=350, random_state=42)
    c_range = range(1,11)
    param_grid = dict(C = c_range)
    grid = GridSearchCV(model, param_grid,cv=10,scoring='accuracy')
    grid.fit(vector_x_array,star)
    print (grid.best_score_)
    print (grid.best_params_)
    print (grid.best_estimator_)
    
if 0 == 1:
    """
    XGBoost randomsearch
    """
    model=xgb.XGBClassifier(random_state=1,learning_rate=0.05, verbose=True,
                            n_estimators=1000, num_boost_round=150)
    params = {
        'min_child_weight': [1, 5, 10],
#        'gamma': [0.5, 1, 1.5, 2, 5],
#        'subsample': [0.6, 0.8, 1.0],
#        'colsample_bytree': [0.6, 0.8, 1.0],
#        'max_depth': [3, 4, 5]
        }
    grid = RandomizedSearchCV(model, params,cv=10,scoring='accuracy')
    grid.fit(vector_x_array,star)
    print (grid.best_score_)
    print (grid.best_params_)
    print (grid.best_estimator_)
    
if 0 == 1:
    """
    --XGBoost--
    Finished with 78.4% @50k, took 15076.8 seconds or 4.25 hours
    https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
    https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost
    
    """
    model=xgb.XGBClassifier(random_state=1,learning_rate=0.05, verbose=True,
                            n_estimators=1000, max_depth=6, gamma=5, num_boost_round=150)
    print('Fitting Model with Data')
    model.fit(X_train, y_train)
    print('Predicting')
    pred = model.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.4f" % score)
    
    print("Confusion Matrix")
    print (metrics.confusion_matrix(y_test, pred))
    
    end = time.time()
    print(end - start)
    
if 0 == 1:
    """
    --AdaBoost--
    78.0% @ 10k
    78.1% @ 50k
    """
    model = AdaBoostClassifier(random_state=1)
    print('Fitting Model with Data')
    model.fit(X_train, y_train)
    print('Predicting')
    pred = model.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.4f" % score)
    print("Confusion Matrix")
    print (metrics.confusion_matrix(y_test, pred))
    end = time.time()
    print(end - start)
    
if 1 == 1:
    """
    --VotingClassifier--
    """
    model1 = LogisticRegression(solver='lbfgs', random_state=1, max_iter=1000)
    model2 = DecisionTreeClassifier(random_state=1)
    model3 = MultinomialNB()
    model4 = BernoulliNB()
    print('Models Created')
    model = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('mnb', model3), ('bnb', model4)], voting='hard')
    print('Fitting Model with Data')
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.4f" % score)
    
if 0 == 1:
    """
    --Logisitic Regression--
    83.1% @ 10k
    84.4% @ 50k
    85.2% @ 200k max_features=5000, min_df = 5
    """
    model = LogisticRegression(solver='lbfgs', C=1, max_iter=350, random_state=42, verbose=1)
    print('Fitting Model with Data')
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.4f" % score)
    print("Confusion Matrix")
    print (metrics.confusion_matrix(y_test, pred))
    end = time.time()
    print(end - start)
    
if 0 == 1:
    """
    """
    model = RandomForestClassifier( random_state=42, verbose=1)
    print('Fitting Model with Data')
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.4f" % score)
    print("Confusion Matrix")
    print (metrics.confusion_matrix(y_test, pred))
    end = time.time()
    print(end - start)
#    predictions = cross_val_predict(model, X, y, cv=3)
#    cross_val_accuracy = metrics.accuracy_score(y, predictions)
#    print(cross_val_accuracy)