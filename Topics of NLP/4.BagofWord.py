#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 23:24:51 2021

@author: quantum
"""

import nltk
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


 
paragraph='''Topic sentences are similar to mini thesis statements. 
Like a thesis statement, a topic sentence has a specific main point. 
Whereas the thesis is the main point of the essay, the topic sentence is the main point of the paragraph.
 Like the thesis statement, a topic sentence has a unifying function. But a thesis statement or topic sentence alone doesn’t guarantee unity. 
 An essay is unified if all the paragraphs relate to the thesis, whereas a paragraph is unified if all the sentences relate to the topic sentence. 
 Note: Not all paragraphs need topic sentences. In particular, opening and closing paragraphs, which serve different functions from body paragraphs, generally don’t have topic sentences.
'''


# Stemming
stem_ps=PorterStemmer()

#Lemmatization

lem=WordNetLemmatizer()


#sentences

sentence=nltk.sent_tokenize(paragraph)

corpus=[]


for i in range(len(sentence)):
    clean_data=re.sub('[^a-zA-z]', ' ',sentence[i])
    clean_data=clean_data.lower()
    clean_data=clean_data.split()
    clean_data=[lem.lemmatize(word) for word in clean_data if word not in set(stopwords.words('english'))]
    clean_data=' '.join(clean_data)
    corpus.append(clean_data)
    
    
#Creating Bag Of Words

from sklearn.feature_extraction.text import CountVectorizer

#Creating an object of CV
cv=CountVectorizer()

#Creating document matrix
X=cv.fit_transform(corpus).toarray()
    