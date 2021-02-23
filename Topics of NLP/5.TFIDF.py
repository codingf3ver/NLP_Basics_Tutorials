#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:03:42 2021

@author: quantum
"""

import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


 
paragraph='''Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.

Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.  Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.

The best way to prevent and slow down transmission is to be well informed about the COVID-19 virus, the disease it causes and how it spreads. Protect yourself and others from infection by washing your hands or using an alcohol based rub frequently and not touching your face. 

The COVID-19 virus spreads primarily through droplets of saliva or discharge from the nose when an infected person coughs or sneezes, so it’s important that you also practice respiratory etiquette (for example, by coughing into a flexed elbow).
'''


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
    
from sklearn.feature_extraction.text import TfidfVectorizer

tfid=TfidfVectorizer()
features=tfid.fit_transform(corpus).toarray()