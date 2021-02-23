#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:21:17 2021

@author: quantum
"""

import nltk
#nltk.download()

paragraph='''Topic sentences are similar to mini thesis statements. 
Like a thesis statement, a topic sentence has a specific main point. 
Whereas the thesis is the main point of the essay, the topic sentence is the main point of the paragraph.
 Like the thesis statement, a topic sentence has a unifying function. But a thesis statement or topic sentence alone doesn’t guarantee unity. 
 An essay is unified if all the paragraphs relate to the thesis, whereas a paragraph is unified if all the sentences relate to the topic sentence. 
 Note: Not all paragraphs need topic sentences. In particular, opening and closing paragraphs, which serve different functions from body paragraphs, generally don’t have topic sentences.
'''

#convert the pargraph into sentence 
sentence=nltk.sent_tokenize(paragraph)


#find out the number of words as list

words=nltk.word_tokenize(paragraph)


