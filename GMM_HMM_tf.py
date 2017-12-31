#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:14:13 2017

@author: zehao
"""
# Descriptions:
# HMM for mixture gaussian observations in Tensorflow
# Algorithms: Banum Welch algorithm, 
#			  in tensorflow, M step can be achieved by automatic gradient
#			  So, just need to set parameters as tf.Variable, set observations as tf.placeholder
# Data structure: jagged array: a combination of observed sequences
#                               each sequence could have different length, each element in the sequence is D-dim
#								X: (N*T*D), X[n]: [T*D], T is length of each sequence. each row is an observation
# Parmeters in this modek:
#		hidden states parameters: transition matrix A(K*V), initial distribution matrix: pi
#		emission distribution: r(K*V):K,hidden layer state, V number of miture, probability, sum row =1
#							   MU(K*V*D), Mu(k,v),hidden state k, mixture v, has D dimension, mean
#							   SIGMA(K*V*D*D), Sigma(k,v), hidden state k, mixture v, D*D dim, covariance matrix

'''
Tips:tensorflow scan
Allow numbers of iterations to be part of the symbolic structure
minnimizes the numbr of GPU transfers
Computes gradients through sequential step
lower memory usage
And tensorflow deal with underflow automatically
'''
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

MVN = tf.contrib.distributions.MultivariateNormalDiag
'''details in tensorflow'''

class GMM_HMM(object):
	def __init__(self,K,V,D):
		self.K = K
		self.V = V
		self.D = D

	def init_session(self,session):
		self.session = session

	def init_params(self,X):
		N = X.shape[0]
		T = X.shape[1]
		# hidden units parameters
		raw_pi = np.ones(self.K).astype(np.float32)
		raw_A = np.random.randn(self.K,self.K).astype(np.float32)
		# emission distribution: gaussian mixture
		raw_R = np.ones((self.K,self.V)).astype(np.float32)

		Mu0 = np.zeros((self.K,self.V,self.D))
		for k in range(self.K):
			for v in range(self.V):
				n = np.random.randint(N)
				t = np.random.randint(T)
				Mu0[k,v] = X[n,t]
		Mu0 = Mu0.astype(np.float32)

		logSigma = np.random.randn(self.D,self.D)
		self.build(raw_pi,raw_A,raw_R,Mu0,logSigma)

	def build(self,nonsoftmax_pi,nonsoftmax_A,nonsoftmax_R,Mu,logSigma):
		# use initialized values to build temsorflow structure
		self.nonsoftmax_pi = tf.Variable(nonsoftmax_pi)
		self.nonsoftmax_A = tf.Variable(nonsoftmax_A)
		self.nonsoftmax_R = tf.Variable(nonsoftmax_R)
		self.Mu = tf.Variable(Mu)
		self.logSigma = tf.Variable(logSigma)

		self.tfx = tf.placeholder(tf.float32,shape=(None,D),name='x')

		self.pi = tf.nn.softmax(self.nonsoftmax_pi)
		self.A = tf.nn.softmax(self.nonsoftmax_A)
		self.R = tf.nn.softmax(self.nonsoftmax_R)
		self.Sigma = tf.exp(self.logSigma)
		# X will be transformed into the structure of (∑Ti,D)
		self.mvns = []
		for k in range(self.K):
			self.mvns.append([])
			for v in range(self.V):
				self.mvns[k].append(
					MVN(self.Mu[k,v],self.Sigma[k,v])
				)
		B = []
		# B has same meaning as Multinomial HMM, so we need to deal with same problem
		# but B is more powerful, since this is kind of expectation over observation, which leads to little bit difference
		for k in range(self.K):
			# component[k,v,t] = R[k,v]* N(x(t)|Mu[k,v],Sigma[k,v])	
			# B[j,t] = ∑_v	component[k,v,t] 
			component = [] 
			for v in range(self.V):
				component.append(self.mvns[k][v].prob(self.tfx))
				# mvn.pdf(input).eval() return the pdf of mvn(defined by MVN) at input
				# tensors can only be stacked as a list
				# stack in tensorflow like concatenate in numpy

			component = tf.stack(component) # V * T
			R_j = tf.reshape(R[j],[1,self.V])
			component = tf.matmul(R_j,component) # 1 * T
			component = tf.reshape(component,[-1]) # T
			B.append(component) # B: will be K * T
		B = tf.stack(B) # (KT) * 1
		B = tf.reshape(B,[1,0]) # T * K

		def recurrence(last_output,B_t):
			'''
			Inputs: last_output is alpha[t-1], current_input is observaton of current x
			output: two dim: alpha[t] and scale[t]
			'''
			last_alpha = tf.reshape(last_output[0],(1,self.K))
			alpha = tf.matmul(last_alpha,self.A)*B_t
			alpha = tf.reshape(alpha,(self.K,))
			scale = tf.reduce_sum(alpha)
			return (alpha/scale),scale

		alpha, scale = tf.scan(
			fn = recurrence,
			elems = B[1:],
			initializer = (self.pi*B[0],np.float32(1.0)),
		)

		self.cost_op = -tf.reduce_sum(tf.log(scale))
		self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.cost_op)

	def fit(self,X,max_iter = 30,print_period=1):
		N = len(X)
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



















