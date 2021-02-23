#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 22:04:48 2021

@author: quantum
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


paragraph='''Topic sentences are similar to mini thesis statements. 
Like a thesis statement, a topic sentence has a specific main point. 
Whereas the thesis is the main point of the essay, the topic sentence is the main point of the paragraph.
 Like the thesis statement, a topic sentence has a unifying function. But a thesis statement or topic sentence alone doesn’t guarantee unity. 
 An essay is unified if all the paragraphs relate to the thesis, whereas a paragraph is unified if all the sentences relate to the topic sentence. 
 Note: Not all paragraphs need topic sentences. In particular, opening and closing paragraphs, which serve different functions from body paragraphs, generally don’t have topic sentences.
'''


sentence=nltk.sent_tokenize(paragraph)

stemmer=PorterStemmer()


#Stemming


for i in range(len(sentence)):
    words=nltk.word_tokenize(sentence[i])
    words=[stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentence[i]=' '.join(words)