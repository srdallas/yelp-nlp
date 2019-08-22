# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:08:47 2019

@author: Sean
https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras import backend as K
from keras import callbacks
from keras import optimizers
from keras import layers

import time

"""
https://datascience.stackexchange.com/questions/36238/what-does-the-output-of-model-predict-function-from-keras-mean
"""

start = time.time()

#specify file path to read test data from and read it into a dataframe
file_path = r'C:\Users\Sean\Documents\Work\Miracle Soft\Hands_on_yelp_1\review50k_with_3.csv'
f = open( file_path, encoding="ISO-8859-1")
data = pd.read_csv(f)

#same for test path
test_path =  r'C:\Users\Sean\Documents\Work\Miracle Soft\Hands_on_yelp_1\review200k_test_predone.csv'
g = open( test_path, encoding="ISO-8859-1")
test_data = pd.read_csv(g)
#seperate stars and text data
star = data['stars'].values.tolist()
reviews_text = data['text'].astype(str).values.tolist()

#do the same for test data
star_test = test_data['stars'].values.tolist()
review_text_test = test_data['text'].astype(str).values.tolist()

#setup vectorizer
vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df=20 ,max_features = 5000)
vector_x = vectorizer.fit_transform(reviews_text)
vector_x_array = vector_x.toarray()

#setup test strings
vector_test = vectorizer.transform(review_text_test)
vector_test = vector_test.toarray()

#standard scalar
if 0 == 1:
    scaler = StandardScaler()
    vector_x_array = scaler.fit_transform(vector_x_array)
    vector_test = scaler.transform(vector_test)
#stars
star = to_categorical(star)
star_test = to_categorical(star_test)

X_train, X_test, y_train, y_test = train_test_split( vector_x_array, star, test_size=0.2, random_state=42)

if 1 == 1:
    """
    Layers: 128,64,32,3 .0005 lr, epoch 10, batch size 10 84-85%
    Layers: 128,64,32,3 .0005 lr, epoch 3, batch size 100, test acc 85%, 200k train, and 200k test
    Layers: 512,256,256,128,3 .0005 lr, epoch 3, batch size 16, test acc 85%, 200k train, and 200k test
    """
    model = Sequential()
    model.add(Dense(512, kernel_initializer='random_uniform', input_dim=5000, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dropout(0.2))
    #model.add(Dense(16, activation='relu'))
    model.add(Dense(3, kernel_initializer='random_uniform', activation='softmax'))
    
    savebest=callbacks.ModelCheckpoint(filepath='checkpoint-{val_acc:.2f}.h5',monitor='val_acc',save_best_only=True)
    callbacks_list=[savebest]
    
    adm = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    model.compile(loss="categorical_crossentropy", optimizer= adm, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=2, batch_size=16, validation_data=(X_test, y_test), callbacks=callbacks_list)
    
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    end = time.time()
    print(end - start)
    
    pred = model.predict(vector_test)
    pred = np.argmax(pred, axis=1)
    star_tester = np.argmax(star_test, axis=1)
    print("Confusion Matrix")
    print (metrics.confusion_matrix(star_tester, pred))
    print (metrics.accuracy_score(star_tester, pred))
        
    #model.save('model.h5')
    
if 0 == 1:
    model = Sequential()
    
    model.add(layers.Embedding(input_dim=2500, output_dim=64))
    model.add(layers.GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(18, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    model.summary()
    
    savebest=callbacks.ModelCheckpoint(filepath='checkpoint-{val_acc:.2f}.h5',monitor='val_acc',save_best_only=True)
    callbacks_list=[savebest]
    
    adm = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    model.compile(loss="categorical_crossentropy", optimizer= adm, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test), callbacks=callbacks_list)
    