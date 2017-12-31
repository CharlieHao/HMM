#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 9:04:53 2017

@author: zehao
"""
# Descriptions:
# HMM for mixture gaussian observations 
# Algorithms: Banum Welch algorithm, 
# Data structure: jagged array: a combination of observed sequences
#                               each sequence could have different length
# Parmeters in this model:
#		hidden states parameters: transition matrix A(K*V), initial distribution matrix: pi
#		emission distribution: r(K*V):K,hidden layer state, V number of miture, probability, sum row =1
#							   Mu(K*V*D), Mu(k,v),hidden state k, mixture v, has D dimension, mean
#							   SIGMA(K*V*D*D), Sigma(k,v), hidden state k, mixture v, D*D dim, covariance matrix


import numpy as np
import matplotlib.pyplot as plt 

from scipy.stats import multivariate_normal as mvn 

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

class GMM_HMM(object):
	def __init__(self,K,V):
		self.K = K
		self.V = V

	def reshape_and_index(self,X):
		sequence_length = []
		for x in X:
			sequence_length.append(len(x))
		X_concat = np.concatenate(X)
		'''
		X = [x1,x2,x3...xn], x1.shape = (Ti,D): Ti row, each row is the observation of a D-dim gaussian 
		np.concatenate(X) can transfer it to (∑Ti,D), each line is an observation
		'''
		T = len(X_concat)
		start_points = np.zeros(len(X_concat),dtype=np.bool)
		end_points = np.zeros(len(X_concat),dtype=np.bool)
		start_points_val = []
		ind = 0
		for leng in sequence_length:
			start_points_val.append(ind)
			start_points[ind] = 1
			if ind > 0:
				end_points[ind-1] = 1
			ind += leng
		return X_concat,start_points,start_points_val,end_points

	def init_params(self,X_concat,X):
		T = len(X_concat) # T total number of obsevation
		D = X[0].shape[1]
		self.D = D
		self.pi = np.ones(self.K)/self.K
		self.A = normalized_random(self.K,self.K)
		self.R = np.ones((self.K,self.V))/self.V
		self.Mu = np.zeros((self.K,self.V,self.D))
		for k in range(self.K):
			for v in range(self.V):
				index = np.random.choice(T)
				self.Mu[k,v] = X_concat[index]
		self.Sigma = np.zeros((self.K,self.V,self.D,self.D))
		for k in range(self.K):
			for v in range(self.V):
				self.Sigma[k,v] = np.eye(D)

	def Viterbi(self,X):
		'''Used for track the most probable hidden states chain'''
		N = len(X)
		delta = np.zeros((N,self.K))
		psi = np.zeros((N,self.K)) # psi[t,j] is the value of hidden unit at time t-1, which 
		                           #  correspond the max alpha(or message) at time t in position j
		delta[0] = self.pi*self.B[:,X[0]]
		for t in range(1,N):
			for j in range(self.K):
				delta[t,j] = np.max(delta[t-1]*self.A[:,j])*self.B[j,x[t]]
				psi[t,j] = np.argmax(delta[t-1]*self.A[:j])

		# backtracking
		tracking_states = np.zeros(N,dtype = np.int32)
		tracking_states[N-1] = np.argmax(delta[N-1])
		for t in range(N-2, -1 ,-1):
			tracking_states[t] = psi[t+1, tracking_states[t+1]]

		return tracking_states

	def fit(self,X,max_iter=30,print_period=1,eps=1e-1):
		# transform input data X
		X_concat,start_points,start_points_val,end_points = self.reshape_and_index(X)
		# initialization
		self.init_params(X_concat,X)
		# EM (Banum Welch algorithm)

		minus_loglikelihoodS = []
		T = len(X_concat)
		for ite in range(max_iter):
			B = np.zeros((self.K,T)) # B is equivalent to B in the Discrete HMM, while B does not exist actually
			component = np.zeros((self.K,self.V,T)) # In order to store information
			for k in range(self.K):
				for v in range(self.V):
					prob = self.R[k,v] * mvn.pdf(X_concat,self.Mu[k,v],self.Sigma[k,v])
					component[k,v,:] = prob
					B[k,:] += prob

			## forward backward algorithm
			scale = np.zeros(T)
			alpha = np.zeros((T,self.K))
			alpha[0] = self.pi*B[:,0]
			scale[0] = alpha[0].sum()
			alpha[0] /= scale[0]

			for t in range(1,T):
				if start_points[t] == 0:
					alpha_non_scale = alpha[t-1].dot(self.A)*B[:,t] # difference with before: can be treated as known state in B
				else: 												# started a new line in original X
					alpha_non_scale = self.pi*B[:,t]
				scale[t] = alpha_non_scale.sum()
				alpha[t] = alpha_non_scale/scale[t]
			logP = np.log(scale).sum()
			minus_loglikelihoodS.append(logP)
			if ite%print_period == 0:
				print('iteration:',ite,'cost',logP)

			beta = np.zeros((T,self.K))
			beta[-1] = 1
			for t in range(T-2,-1,-1):
				if start_points[t+1] == 1:
					beta[t] = 1
				else: 												
					beta[t] = self.A.dot(B[:,t+1]*beta[t+1])/scale[t+1]

			## M step: do the update use component information 
			### store information
			a_denom = np.zeros((self.K,1))
			a_numer = np.zeros((self.K,self.K))
			R_denom = np.zeros(self.K)
			R_numer = np.zeros((self.K,self.V))
			Mu_numer = np.zeros((self.K,self.V,self.D))
			Sigma_numer = np.zeros((self.K,self.V,self.D,self.D))

			### generate gamma information
			gamma = np.zeros((T,self.K,self.V)) 
			for t in range(T):
				phi = alpha[t,:].dot(beta[t,:])
				for k in range(self.K):
					factor = alpha[t,k]*beta[t,k]/phi
					for v in range(self.V):
						gamma[t,k,v] = factor * component[k,v,t]/B[k,t]

			### update hidden state parameters
			self.pi = np.sum((alpha[t]*beta[t] for t in start_points_val)) / len(start_points_val)
			'''categorize end points and non end points'''
			non_end_points = (1- end_points).astype(np.bool)
			a_denom += (alpha[non_end_points]*beta[non_end_points]).sum(axis=0,keepdims=True).T
			for k in range(self.K):
				for v in range(self.K):
					for t in range(T-1):
						if end_points[t] != 1:
							a_numer[k,v] += alpha[t,k] * beta[t+1,v] *self.A[k,v]*B[v,t+1]/scale[t+1]
			self.A = a_numer/a_denom

			### GMM parameters update
			for k in range(self.K):
				for v in range(self.V):
					for t in range(T):
						R_numer[k,v] += gamma[t,k,v]
						R_denom[k] += gamma[t,k,v]

			for k in range(self.K):
				for v in range(self.V):
					for t in range(T):
						Mu_numer[k,v]+= gamma[t,k,v]*X_concat[t]
						Sigma_numer[k,v]+= gamma[t,k,v]*np.outer(X_concat[t]-self.Mu[k,v],X_concat[t]-self.Mu[k,v])

			for k in range(self.K):
				for v in range(self.V):
					self.R[k,v] = R_numer[k,v]/R_denom[k]
					self.Mu[k,v] = Mu_numer[k,v]/R_numer[k,v]
					self.Sigma[k,v] = Sigma_numer[k,v]/R_numer[k,v] + np.eye(self.D)*eps


			### make diagnostic
			assert(np.all(self.R<=1))
			assert(np.all(self.A<=1))

			

		print('transition matrix of hidden states:', self.A)
		print('initial distribution:',self.pi)
		print('emission mixture allocation:',self.R)
		print('emission gaussian mean:',self.Mu)
		print('emission gaussian covariance matrix:',self.Sigma) 

		plt.plot(minus_loglikelihoodS)
		plt.show()

	def log_likelihood(self,x):
		T = len(x)
		scale = np.zeros(T)
		B = np.zeros((self.K, T))
		for k in range(self.K):
			for v in range(self.V):
				p = self.R[k,v] * mvn.pdf(x, self.Mu[k,v], self.Sigma[k,v])
				B[k,:] += p

		alpha = np.zeros((T, self.K))
		alpha[0] = self.pi*B[:,0]
		scale[0] = alpha[0].sum()
		alpha[0] /= scale[0]
		for t in range(1, T):
			alpha_non_scale = alpha[t-1].dot(self.A) * B[:,t]
			scale[t] = alpha_non_scale.sum()
			alpha[t] = alpha_non_scale / scale[t]
		return np.log(scale).sum()

	def log_likelihood_multi(self, X):
		return np.array([self.log_likelihood(x) for x in X])



