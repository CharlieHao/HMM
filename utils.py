#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 31 15:54:33 2017

@author: zehao
"""

# Description:
# utility functions for mixture gaussian 

import numpy as np
from sklearn.metrics import f1_score

def softmax(x):
	return x/x.sum(axis=1,keepdims=True)


def accuracy(T,Y):
	total = 0
	correct = 0 
	for t, y in zip(T,Y):
		correct += np.sum(t==y)
		total += len(y)
	return correct/total

def sum_F1_score(T,Y):
	# F1 score is used to test the general performance of the classifier
	T = np.concatenate(T)
	Y = np.concatenate(Y)
	return f1_score(T,Y,average=None).mean()

def get_chunk_data(split_sequences=False):
	# Data structure: Three columns seperated by spaces. 
	#				  The first column contains the current word, 
	#				  the second its part-of-speech tag as derived by the Brill tagger 
	#				  and the third its chunk tag as derived from the WSJ corpus.
	#				  We are interested in the first column
	
	# get train data set
	word2idx = {}
	tag2idx ={}
	word_idx = 0
	tag_idx = 0
	Xtrain = []
	Ytrain = []
	currentX = []
	currentY = []
	for line in open('chunking_data/train.txt'):
		line = line.rstrip()
		if line:
			r = line.split()
			word, tag, _ = r
			if word not in word2idx:
				word2idx[word] = word_idx
				word_idx += 1
			currentX.append(word2idx[word])

			if tag not in tag2idx:
				tag2idx[tag] = tag_idx
				tag_idx += 1
			currentY.append(tag2idx[tag])
		elif split_sequences:
			Xtrain.append(currentX)
			Ytrain.append(currentY)
			currentX = []
			currentY = []

	if not split_sequences:
		Xtrain = currentX
		Ytrain = currentY

	# get test data set
	Xtest = []
	Ytest = []
	currentX = []
	currentY = []
	for line in open('chunking_data/test.txt'):
		line = line.rstrip()
		if line:
			r = line.split()
			word, tag, _ = r
			if word in word2idx:
				currentX.append(word2idx[word])
			else:
				currentX.append(word_idx) 
			currentY.append(tag2idx[tag])
		elif split_sequences:
			Xtest.append(currentX)
			Ytest.append(currentY)
			currentX = []
			currentY = []
	if not split_sequences:
		Xtest = currentX
		Ytest = currentY

	return Xtrain, Ytrain, Xtest, Ytest, word2idx



