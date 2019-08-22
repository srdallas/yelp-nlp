# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:17:55 2019

@author: Sean
"""

file_path = r'C:\Users\Sean\Documents\Work\Miracle Soft\Hands_on_yelp_1\review50k.json'

#function to cut json file to 10k entries
def shrink_json(path_name):
    f = open( path_name, encoding='utf-8')
    g = open("new_json_test_10k.json", 'w+')
    
    for i in range(0,10000):
        lol = f.readline()
        g.write(lol)

shrink_json(file_path)

# function to find length of json file
#def length_of_json():
#    
#    x = 0
#    y = 1
#    while y == 1:
#        lol = f.readline()
#        if lol == "":
#            print (x)
#            y = 2
#        x += 1
        
# length_of_json()
