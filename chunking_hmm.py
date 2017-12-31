#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zehao
"""

# Descriptions: Parts of speech tagging project
#               Text chunking consists of dividing a text in syntactically correlated parts of words
# 
# Method: Hidden Markov model	
# 			 			  
# Data structure: Three columns seperated by spaces. 
#				  The first column contains the current word, 
#				  the second its part-of-speech tag as derived by the Brill tagger 
#				  and the third its chunk tag as derived from the WSJ corpus.

# Tips: we already know the hidden states in the training data set
#       Use viberti algorithm to find the most likely hidden states chain    
import numpy as np 
import matplotlib.pyplot as plt 
 
from Multinomial_hmm import Multinomial_HMM
from utils import accuracy, sum_F1_score, get_chunk_data

def main(smoothing_param = 1e-1):
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_chunk_data(split_sequences=True)
    V = len(word2idx) + 1

    # find hidden state transition matrix and pi
    K = len(set(np.concatenate(Ytrain)))
    A = np.ones((K, K))*smoothing_param
    pi = np.zeros(K)
    for y in Ytrain:
        pi[y[0]] += 1
        for i in range(len(y)-1):
            A[y[i], y[i+1]] += 1
    # turn it into a probability matrix
    A /= A.sum(axis=1, keepdims=True)
    pi /= pi.sum()

    # find the observation matrix
    B = np.ones((K, V))*smoothing_param # add-one smoothing
    for x, y in zip(Xtrain, Ytrain):
        for xi, yi in zip(x, y):
            B[yi, xi] += 1
    B /= B.sum(axis=1, keepdims=True)

    hmm = Multinomial_HMM(K,V)
    hmm.pi = pi
    hmm.A = A
    hmm.B = B

    # get predictions
    Ptrain = []
    for x in Xtrain:
        p = hmm.Viterbi(x)
        Ptrain.append(p)

    Ptest = []
    for x in Xtest:
        p = hmm.Viterbi(x)
        Ptest.append(p)

    # print results
    print( "train accuracy:", accuracy(Ytrain, Ptrain))
    print( "test accuracy:", accuracy(Ytest, Ptest))
    print( "train f1:", sum_F1_score(Ytrain, Ptrain))
    print ("test f1:", sum_F1_score(Ytest, Ptest))

if __name__ == '__main__':
    main()


