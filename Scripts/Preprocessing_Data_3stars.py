# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:56:02 2019

@author: Sean
"""
import re
from nltk.stem import WordNetLemmatizer
import pandas as pd


stop_words =['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself',
            'yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself',
            'they','them','their','theirs','themselves','what','which','who','whom','this','that',
            'these','those','am','is','are','was','were','be','been','being','have','has','had',
            'having','do','does','did','doing','a','an','the','and','but','if','or','because','as',
            'until','while','of','at','by','for','with','about','against','between','into','through',
            'during','before','after','above','below','to','from','up','down','in','out','on','off',
            'over','under','again','further','then','once','here','there','when','where','why','how',
            'all','any','both','each','few','more','most','other','some','such','no','nor','not',
            'only','own','same','so','than','too','very','s','t','can','will','just','don','should',
            'now','uses','use','using','used','one','also']

def tokenized(data):
    reviews_tokens = []
    for review in data:
        review = review.lower() #Convert to lower-case words
        raw_word_tokens = re.findall(r'(?:\w+)', review,flags = re.UNICODE) #remove punctuation
        word_tokens = [w for w in raw_word_tokens if not w in stop_words] # do not add stop words
        reviews_tokens.append(word_tokens)
    return reviews_tokens #return all tokens


def lemmatize_list(list_of_words):
    lemmatizer = WordNetLemmatizer()
    place_holder2 = [[] for x in range(0,len(list_of_words))]
    #print (place_holder2)
    
    for num,i in zip(range(len(list_of_words)),list_of_words):
        for j in i:
            place_holder2[num].append(lemmatizer.lemmatize(j))
            
    return place_holder2

def toStringFromList(lst):
    place_holder = ["" for x in range(0,len(lst))]
    for num,i in zip(range(len(lst)),lst):
        hold = " ".join(i)
        place_holder[num] = hold
            
    return place_holder

def transform_star(lst):
    place_holder = []
    for num, i in zip(range(len(lst)),lst):
        if i >= 4:
            place_holder.append(2)
        elif i == 3:
            place_holder.append(1)
        else:
            place_holder.append(0)
    return place_holder

#specify file path to read test data from and read it into a dataframe
file_path = r'C:\Users\Sean\Documents\Work\Miracle Soft\Hands_on_yelp_1\review1m_test.csv'
f = open( file_path, encoding="ISO-8859-1")
data = pd.read_csv(f)


#seperate stars and text data
star = data['stars']
reviews_text = data['text']


#preprocess data
print('Preprocessing Data')
test = tokenized(reviews_text)
test2 = lemmatize_list(test)
test = toStringFromList(test2)
star = transform_star(star)

print('Saving to CSV')
destination = r'C:\Users\Sean\Documents\Work\Miracle Soft\Hands_on_yelp_1\review1m_test_p.csv'
frame = {'text':test,'stars':star}
df = pd.DataFrame(frame)
df.to_csv(destination)