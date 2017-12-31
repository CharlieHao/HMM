#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:54:45 2017

@author: zehao
"""

# Descriptions:
# HMM for multivariate observations in Tensorflow
# Algorithms: Banum Welch algorithm, 
#			  in tensorflow, M step can be achieved by automatic gradient
#			  So, just need to set parameters as tf.Variable, set observations as tf.placeholder
# Data structure: jagged array: a combination of observed sequences
#                               each sequence could have different length
'''
Tips:tensorflow scan
Allow numbers of iterations to be part of the symbolic structure
minnimizes the numbr of GPU transfers
Computes gradients through sequential step
lower memory usage
'''
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

class Multinomial_HMM(object):
	def __init__(self,K):
		self.K = K

	def init_session(self,session):
		self.session = session

	def init_params(self,V):
		'''V is the number of states in the multinimial distribution(emission dis)'''
		raw_pi = np.zeros(self.K).astype(np.float32)
		raw_A = np.random.randn(self.K,self.K).astype(np.float32)
		raw_B = np.random.randn(self.K,V).astype(np.float32)

		self.build(raw_pi,raw_A,raw_B)

	def build(self,raw_pi,raw_A,raw_B):
		K,V = raw_B.shape
		self.raw_pi = tf.Variable(raw_pi)
		self.raw_A = tf.Variable(raw_A)
		self.raw_B = tf.Variable(raw_B)

		self.pi = tf.nn.softmax(self.raw_pi)
		self.A  = tf.nn.softmax(self.raw_A)
		self.B  = tf.nn.softmax(self.raw_B)

		self.tfx = tf.placeholder(tf.int32,shape=(None,),name='x')
		#use tensorflow scan to defin the train_op and cost
		def recurrence(last_output,current_input):
			'''
			Inputs: last_output is alpha[t-1], current_input is observaton of current x
			output: two dim: alpha[t] and scale[t]
			'''
			last_alpha = tf.reshape(last_output[0],(1,self.K))
			alpha = tf.matmul(last_alpha,self.A)*self.B[:,current_input]
			alpha = tf.reshape(alpha,(self.K,))
			scale = tf.reduce_sum(alpha)
			return (alpha/scale),scale

		alpha, scale = tf.scan(
			fn = recurrence,
			elems = self.tfx[1:],
			initializer = (self.pi*self.B[:,self.tfx[0]],np.float32(1.0)),
		)

		self.cost = -tf.reduce_sum(tf.log(scale))
		self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.cost)


	def fit(self,X,max_iter=30,print_period=2):
		N = len(X) # number of observed sequences

		minus_log_likelihoodS = []
		for ite in range(max_iter):
			for n in range(N):
				ml = self.minus_log_likelihood(X).sum()
				minus_log_likelihoodS.append(ml)
				self.session.run(self.train_op,feed_dict={self.tfx:X[n]})

			if ite % print_period == 0:
				print('iteration:', ite, 'loglikelihhod', -ml)

		plt.plot(minus_log_likelihoodS)
		plt.title('-log likelihood')
		plt.show()

	def generate_cost(self,x):
		return self.session.run(self.cost,feed_dict={self.tfx:x})

	def minus_log_likelihood(self,X):
		return np.array([self.generate_cost(x) for x in X])

	def test(self,X,raw_pi,raw_A,raw_B):
		assign1_op = self.raw_pi.assign(raw_pi)
		assign2_op = self.raw_A.assign(raw_A)
		assign3_op = self.raw_B.assign(raw_B)
		self.session.run([assign1_op,assign2_op,assign3_op])
		return self.minus_log_likelihood(X).sum()




## Blow is the benchmark of how to use this class object
'''
def main():
	# read data
	X

	# benchmark
	hmm = Multinomial_HMM(2)

	hmm.init_params(2)
	init = tf.global_variables_initializer() # initialize all tf variables
	with tf.Session() as session:
		session.run(init)
		hmm.init_session(session)
		hmm.fit(X,max_iter=20)
		C = hmm.minus_log_likelihood(X).sum()
		print('cost in fitted model:',C)

	
if __name__ == '__main__':
	main()
'''

