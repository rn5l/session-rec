from __future__ import division
from __future__ import print_function
import numpy as np
import math
import random
import re
import os
import glob
import sys
from time import time
from .mf_base import MFBase

class BPRMF(MFBase):
	''' Implementation of the algorithm presented in "BPR: Bayesian personalized ranking from implicit feedback", by Rendle S. et al., 2009.

	The adaptive sampling algorithm is adapted from "Improving pairwise learning for item recommendation from implicit feedback", by Rendle S. et al., 2014
	'''

	def __init__(self, k = 100, adaptive_sampling=True, sampling_bias=500, **kwargs):

		super(BPRMF, self).__init__(**kwargs)

		self.name = 'BPRMF'
		self.k = k
		self.adaptive_sampling = adaptive_sampling
		self.sampling_bias = sampling_bias # lambda parameter in "Improving pairwise learning for item recommendation from implicit feedback", by Rendle S. et al., 2014

	def _get_model_filename(self, epochs):
		'''Return the name of the file to save the current model
		'''
		filename = "bprmf_ne"+str(epochs)+"_lr"+str(self.init_learning_rate)+"_an"+str(self.annealing_rate)+"_k"+str(self.k)+"_reg"+str(self.reg)+"_ini"+str(self.init_sigma)
		if self.adaptive_sampling:
			filename += "_as"+str(self.sampling_bias)
		return filename+".npz"

	def init_model(self):
		''' Initialize the model parameters
		'''
		self.V = self.init_sigma * np.random.randn(self.n_users, self.k).astype(np.float32)
		self.H = self.init_sigma * np.random.randn(self.n_items, self.k).astype(np.float32)
		self.bias = np.zeros(self.n_items).astype(np.float32)

	def sgd_step(self, user, true_item, false_item):
		''' Make one SGD update, given that the interaction between user and true_item exists, 
		but the one between user and false_item does not.
		user, true_item and false_item are all user or item ids.

		return error
		'''

		# Compute error
		x_true = self.bias[true_item] + np.dot(self.V[user, :], self.H[true_item, :]) 
		x_false = self.bias[false_item] + np.dot(self.V[user, :], self.H[false_item, :]) 
		delta = 1 - 1 / (1 + math.exp(min(10, max(-10, x_false - x_true)))) # Bound x_true - x_false in [-10, 10] to avoid overflow
		
		# Update CF
		V_mem = self.V[user, :]
		self.V[user, :] += self.learning_rate * ( delta * (self.H[true_item, :] - self.H[false_item, :]) - self.reg * self.V[user, :])
		self.H[true_item, :] += self.learning_rate * ( delta * V_mem - self.reg * self.H[true_item, :])
		self.H[false_item, :] += self.learning_rate * ( -delta * V_mem - self.reg / 10 * self.H[false_item, :])
		self.bias[true_item] += self.learning_rate * (delta - self.reg * self.bias[true_item])
		self.bias[false_item] += self.learning_rate * (- delta - self.reg * self.bias[false_item])

		return delta

	def compute_factor_rankings(self):
		'''Rank items according to each factor in order to do adaptive sampling
		'''

		self.ranks = np.argsort(self.H, axis=0)
		self.var = np.var(self.H, axis=0)

	def get_training_sample(self):
		'''Pick a random triplet from self.triplets and a random false next item.
		returns a tuple of ids : (user, true_item, false_item)
		'''

		user_id = random.randrange(self.n_users)
		while self.users[user_id,1] < 2:
			user_id = random.randrange(self.n_users)
		user_items = self.items[self.users[user_id,0]:self.users[user_id,0]+self.users[user_id,1]]
		true_item = random.choice(user_items)
		if self.adaptive_sampling:
			while True:
				rank = np.random.exponential(scale=self.sampling_bias)
				while rank >= self.n_items:
					rank = np.random.exponential(scale=self.sampling_bias)
				factor_signs = np.sign(self.V[user_id, :])
				factor_prob = np.abs(self.V[user_id, :]) * self.var
				f = np.random.choice(self.k, p=factor_prob/sum(factor_prob))
				false_item = self.ranks[int(rank) * int(factor_signs[f]),f]
				if false_item not in user_items:
					break
		else:
			false_item = random.randrange(self.n_items)
			while false_item in user_items:
				false_item = random.randrange(self.n_items)

		return (user_id, true_item, false_item)

	def top_k_recommendations(self, sequence, user_id=None, k=10, exclude=None):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''

		if exclude is None:
			exclude = []

		last_item = sequence[-1][0]
		output = self.bias + np.dot(self.V[user_id, :], self.H.T)

		# Put low similarity to viewed items to exclude them from recommendations
		output[[i[0] for i in sequence]] = -np.inf
		output[exclude] = -np.inf

		# find top k according to output
		return list(np.argpartition(-output, range(k))[:k])
	
	def recommendations(self, sequence, user_id=None, k=10, exclude=None, session=None):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''

		if exclude is None:
			exclude = []
		
		if user_id is None:
			uv = self.H[session].mean(axis=0)
		else:
			uv = self.V[user_id, :]
		
		output = self.bias + np.dot(uv, self.H.T)

		return output

	def training_step(self, iterations):
		if self.adaptive_sampling and iterations%int(self.n_items * np.log(self.n_items)) == 0:
			self.compute_factor_rankings()

		# Train with a new batch
		return self.sgd_step(*self.get_training_sample())

	def save(self, filename):
		'''Save the parameters of a network into a file
		'''
		print('Save model in ' + filename)
		if not os.path.exists(os.path.dirname(filename)):
			os.makedirs(os.path.dirname(filename))
		np.savez(filename, V=self.V, H=self.H, bias=self.bias)
		

	def load(self, filename):
		'''Load parameters values form a file
		'''
		f = np.load(filename)
		self.V = f['V']
		self.H = f['H']
		self.bias = f['bias']