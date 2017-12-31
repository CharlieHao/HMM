#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:54:33 2017

@author: zehao
"""

# Descriptions:
# HMM for multivariate observations
# Algorithms: Banum Welch algorithm (In EM, M step)
#             Forward Backward algorithm (In EM, E step),(same result of sum-product algorithm )
#		      Vitergi Algorithm (trace back for the most likely sequence)
# 
# Method fit: general case
# Method fit_jagged_array: deal with jagged array (Multiple observation)

import numpy as np 
import matplotlib.pyplot as plt
from numpy import matrix as m

def normalized_random(d1,d2):
	x = np.random.random((d1,d2))
	return x/x.sum(axis=1,keepdims=True)

def one_of_K_encoding(y):
	K = len(set(y))
	N = len(y)
	Y = m(np.zeros((K,N)))
	for i in range(N):
		Y[y[i]-1,i]=1
	return np.array(Y)

class Multinomial_HMM(object):
	def __init__(self,K,V):
		''' K # hidden states
			V # categories of observation 
			pi initial distribution, 
			A trnsition matrix of hidden states K*K 
			B emission distribution for Multinomial distribution K*V
		'''
		self.K = K
		self.V = V
		self.pi = np.ones(self.K)/self.K
		self.A = normalized_random(self.K,self.K)
		self.B = normalized_random(self.K,self.V)

	def forward_backward(self,X):
		'''
		X should be an array of shape 1*N, and values should be 0,1...,V-1
		sum-product algorithm
		M step in EM algorithm
		'''
		N = len(X)

		# Forward Process
		alpha = np.zeros((N,self.K))
		alpha[0] = self.pi * self.B[:,X[0]]
		for t in range(1,N):
			alpha[t] = alpha[t-1].dot(self.A)*self.B[:,X[t]]
		
		# Backward
		beta = np.zeros((N,self.K))
		beta[-1] = 1
		for t in range(N-2,-1,-1):
			'''generate a decreasing array'''
			beta[t] = self.A.dot(self.B[:,X[t+1]]*beta[t+1])

		# gamma and epsilon are used for update of parameters in M step
		# gamma and epsilon are defined as Pattern Recognition
		# epsilon[n,j,k] = P(Zn-1=j,Zn=k | theta and X), posterior dis
		gamma = np.zeros([N,self.K])
		epsilon = np.zeros([N-1,self.K,self.K])
		P = sum(alpha[-1])
		for t in range(N):
			gamma[t] = alpha[t]*beta[t]
			# gamma[t] = alpha[t] * beta[t]/P
		for t in range(N-1):
			epsilon[t] = alpha[t].dot((beta[t+1]*self.B[:,X[t+1]]).T)*self.A/P

		return gamma, epsilon, P

	def Viterbi(self,X):
		'''Used for track the most probable hidden states chain'''
		N = len(X)
		delta = np.zeros((N,self.K))
		psi = np.zeros((N,self.K)) # psi[t,j] is the value of hidden unit at time t-1, which 
		                           #  correspond the max alpha(or message) at time t in position j
		delta[0] = self.pi * self.B[:,X[0]]
		for t in range(1,N):
			for j in range(self.K):
				delta[t,j] = np.max(delta[t-1]*self.A[:,j]) *  self.B[j,X[t]]
				psi[t,j] = np.argmax(delta[t-1]*self.A[:,j])

		# backtracking
		tracking_states = np.zeros(N,dtype = np.int32)
		tracking_states[N-1] = np.argmax(delta[N-1])
		for t in range(N-2, -1 ,-1):
			tracking_states[t] = psi[t+1, tracking_states[t+1]]

		return tracking_states

	def scaling_factor(self,X):
		'''In case of underflow of alpha'''
		N = len(X)
		scale = np.zeros(N)

		# forward process
		alpha = np.zeros((N,self.K))
		alpha[0] = self.pi*self.B[:,X[0]]
		scale[0] = alpha[0].sum()
		alpha[0] /= scale[0]
		for t in range(1,N):
			alpha_non_scale = alpha[t-1].dot(self.A)*self.B[:,X[t]]
			scale[t] = alpha_non_scale.sum()
			alpha[t] = alpha_non_scale/scale[t]

		beta = np.zeros((N,self.K))
		beta[-1] = 1
		for t in range(N-2,-1,-1):
			beta[t] = self.A.dot(self.B[:,X[t+1]]*beta[t+1])/scale[t+1]

		# scale is the conditional probability
		logP = np.log(scale).sum()

		gamma = np.zeros([N,self.K])
		epsilon = np.zeros([N-1,self.K,self.K])
		for t in range(N):
			gamma[t] = alpha[t]*beta[t]
		for t in range(N-1):
			epsilon[t] = alpha[t].dot((beta[t+1]*self.B[:,X[t+1]]).T)*self.A*scale[t+1]

		return gamma, epsilon, logP

	def Banum_welch_update(self,X,gamma,epsilon):
		# update initial distribution of Z0, initial hidden unit state
		self.pi = gamma[0]/gamma[0].sum()
		self.A = epsilon.sum(axis=0)/epsilon.sum(axis=0).sum(axis=1,keepdims=True)
		# update the emission distribution parameters
		X_ind = one_of_K_encoding(X)
		self.B = X_ind.dot(gamma)/gamma.sum(axis=0,keepdims=True)

	def fit(self,X,max_iter=50, scaling=True):
		np.random.seed(120)

		self.pi = np.ones(self.K)/self.K
		self.A = normalized_random(self.K,self.K)
		self.B = normalized_random(self.K,self.V)

		likelihood = []
		log_likelihood = []
		for ite in range(max_iter):
			if ite%10 == 0:
				print('iteration:',ite)
			
			# E step
			if scaling :
				gamma, epsilon, logP =self.scaling_factor(X)
				log_likelihood.append(logP)
			else:
				gamma, epsilon, P = self.forward_backward(X)
				likelihood.append(P)

			# M step
			self.Banum_welch_update(X,gamma,epsilon)

		print('transition matrix of hidden markov chain:',self.A)
		print('multinomial emission distribution:',self.B)

		if scaling:
			plt.plot(log_likelihood)
		else:
			plt.plot(likelihood)
		plt.show()

	def likelihood_multi(self,X):
		gamma,epsilon,logP = self.scaling_factor(X)
		return np.exp(logP)

	def fit_jagged_array(self,X,max_iter=50):
		'''
		V1 the number of the vocabulary
		N  the length of observaed sequence
		'''
		np.random.seed(121)
		V1 = max(max(x) for x in X) +1
		N =len(X)

		self.pi = np.ones(self.K)/self.K
		self.A = normalized_random(self.K,self.K)
		self.B = normalized_random(self.K,V1)

		costs=[]
		for ite in range(max_iter):
			if ite%10==0:
				print('iteration:',ite)

			# E step
			alphas = []
			betas = []
			scales = []
			logP = np.zeros(N) # record for each line

			for n in range(N):
				x = X[n]
				T =len(x)
				scale = np.zeros(T)

				alpha = np.zeros((T,self.K))
				alpha[0] = self.pi * self.B[:,x[0]]
				scale[0] = alpha[0].sum()
				alpha[0] /= scale[0]
				for t in range(1,T):
					alpha_non_scale = alpha[t-1].dot(self.A)*self.B[:,x[t]]
					scale[t] = alpha_non_scale.sum()
					alpha[t] = alpha_non_scale / scale[t]

				logP[n] = np.log(scale).sum()
				alphas.append(alpha)
				scales.append(scale)

				beta = np.zeros((T,self.K))
				beta[-1] = 1
				for t in range(T-2,-1,-1):
					beta[t] = self.A.dot(self.B[:,x[t+1]]*beta[t+1])/scale[t+1]
				betas.append(beta)

			cost = np.sum(logP)
			costs.append(cost)

			# M step
			self.pi = np.sum((alphas[n][0]*betas[n][0]) for n in range(N))/N

			d1 = np.zeros((self.K,1))
			d2 = np.zeros((self.K,1))
			a_num = np.zeros((self.K,self.K))
			b_num = np.zeros((self.K,V1))
			for n in range(N):
				x = X[n]
				T = len(x)
				d1 += (alphas[n][:-1]*betas[n][:-1]).sum(axis=0,keepdims=True).T
				d2 += (alphas[n]*betas[n]).sum(axis=0,keepdims=True).T

				for i in range(self.K):
					for j in range(self.K):
						for t in range(T-1):
							a_num[i,j] += alphas[n][t,i]*betas[n][t+1,j]*self.A[i,j]*self.B[j,x[t+1]]/scales[n][t+1]

				for i in range(self.K):
					for t in range(T):
						b_num[i,x[t]] += alphas[n][t,i] * betas[n][t,i]

			self.A = a_num/d1
			self.B = b_num/d2

		print('initial distribtion:',self.pi)
		print('transition matrix of hidden markov chain:',self.A)
		print('multinomial emission distribution:',self.B)

		plt.plot(costs)
		plt.show()

	def likelihood(self,x):
		'''likelihood for a single observation '''
		N = len(x)
		alpha = np.zeros((N,self.K))
		alpha[0] = self.pi * self.B[:,x[0]]
		for t in lrange(1,N):
			alpha[t] = alpha[t-1].dot(self.A)*self.B[:,x[t]]
		return alpha[-1].sum()

	def log_likelihood_jagged_array(self,X):
		'''log likelihood for a combination of several observed sequences'''
		Y = np.array([self.likelihood(x) for x in X])
		return np.log(Y).sum()



