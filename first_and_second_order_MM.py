#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:54:33 2017

@author: zehao
"""

'''
Use dictionary to describe Markov Models
First case is for general one-ordr markov model
Second case is for second-order markov model
'''

import numpy as np 
import string

# One-order markov model
'''
Description of the data set:
there are total 10 web pages
-1 means the start page
B and C means bounce(B) and close(C)
dataset is .csv
'''

# Need two dictionary to describe the 1-order markov chain
transitions = {}
start_state_sum = {}

for line in open('filename.csv'):
	s,e = line.rstrip().split(',')
	transitions[(s,e)] = transitions.get((s,e),0)+1
	row_sums[s] = row_sums.get(s,0)+1

for k, v in transitions.iteritems():
	s, e = k
	transitions[k] = v / row_sums[s]

# initial state distribution
print "initial state distribution:"
for k, v in transitions.iteritems():
	s, e = k
	if s == '-1':
		print e, v

# which page has the highest bounce?
for k, v in transitions.iteritems():
	s, e = k
	if e == 'B':
		print "bounce rate for %s: %s" % (s, v)



# Second-order markov model
'''
Suppose this is a second order markov chain
dataset is .txt
'''

# Need 3 dictionary to describe the 2-order markov model
initial = {}
second_word = {}
transitions = {}

def remove_punctuation(s):
return s.translate(None,string.punctuation) 

def add2dict(dic, k, v):
	# key is a tuole of (V_t-2,V_t-1)(in transitions), and value is a list of verbal after that 
    if k not in d:
        d[k] = []
    d[k].append(v)

for line in open('file_name.txt'):
    tokens = remove_punctuation(line.rstrip().lower()).split()

    T = len(tokens)
    for i in xrange(T):
        t = tokens[i]
        if i == 0:
            # measure the distribution of the first word
            initial[t] = initial.get(t, 0.) + 1
        else:
            t_1 = tokens[i-1]
            if i == T - 1:
                # measure probability of ending the line
                add2dict(transitions, (t_1, t), 'END')
            if i == 1:
                # measure distribution of second word
                # given only first word
                add2dict(second_word, t_1, t)
            else:
                t_2 = tokens[i-2]
                add2dict(transitions, (t_2, t_1), t)
## above, generalize initial, second_word and transitions to store the informations 

# normalize the distributions
initial_total = sum(initial.values())
for t, c in initial.iteritems():
    initial[t] = c / initial_total

def list2pdict(ts):
    # turn each list of possibilities into a dictionary of probabilities
    d = {}
    n = len(ts)
    for t in ts:
        d[t] = d.get(t, 0.) + 1
    for t, c in d.iteritems():
        d[t] = c / n
    return d

for t_1, ts in second_word.iteritems():
    # replace list with dictionary of probabilities
    second_word[t_1] = list2pdict(ts)

for k, ts in transitions.iteritems():
	# output: the value of the dict is a dict(ket=next word, value=conditional transition probability)
	# corresponding to the key in the outer dictionary
    transitions[k] = list2pdict(ts)







