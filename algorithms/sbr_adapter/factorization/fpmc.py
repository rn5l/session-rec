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
from algorithms.sbr_adapter.helpers import evaluation
from .mf_base import MFBase

class FPMC(MFBase):
	''' Code based on work by Rendle et al., Factorizing Personalized Markov Chains for Next-basket Recommendation. WWW 2010.

	The adaptive sampling algorithm is adapted from "Improving pairwise learning for item recommendation from implicit feedback", by Rendle S. et al., 2014
	'''

	def __init__(self, k_cf = 100, k_mc = 100, adaptive_sampling=True, sampling_bias=500, **kwargs):

		super(FPMC, self).__init__(**kwargs)

		self.name = 'FPMC'
		self.k_cf = k_cf
		self.k_mc = k_mc
		self.adaptive_sampling = adaptive_sampling
		self.sampling_bias = sampling_bias # lambda parameter in "Improving pairwise learning for item recommendation from implicit feedback", by Rendle S. et al., 2014
		self.max_length = np.inf # For compatibility with the RNNs

	def _get_model_filename(self, epochs):
		'''Return the name of the file to save the current model
		'''
		filename = "fpmc_ne"+str(epochs)+"_lr"+str(self.init_learning_rate)+"_an"+str(self.annealing_rate)+"_kcf"+str(self.k_cf)+"_kmc"+str(self.k_mc)+"_reg"+str(self.reg)+"_ini"+str(self.init_sigma)
		if self.adaptive_sampling:
			filename += "_as"+str(self.sampling_bias)
		return filename+".npz"

	def init_model(self):
		''' Initialize the model parameters
		'''
		self.V_user_item = self.init_sigma * np.random.randn(self.n_users, self.k_cf).astype(np.float32)
		self.V_item_user = self.init_sigma * np.random.randn(self.n_items, self.k_cf).astype(np.float32)
		self.V_prev_next = self.init_sigma * np.random.randn(self.n_items, self.k_mc).astype(np.float32)
		self.V_next_prev = self.init_sigma * np.random.randn(self.n_items, self.k_mc).astype(np.float32)

	def sgd_step(self, user, prev_item, true_next, false_next):
		''' Make one SGD update, given that the transition from prev_item to true_next exist in user history,
		But the transition prev_item to false_next does not exist.
		user, prev_item, true_next and false_next are all user or item ids.

		return error
		'''

		# Compute error
		x_true = np.dot(self.V_user_item[user, :], self.V_item_user[true_next, :]) + np.dot(self.V_prev_next[prev_item, :], self.V_next_prev[true_next, :])
		x_false = np.dot(self.V_user_item[user, :], self.V_item_user[false_next, :]) + np.dot(self.V_prev_next[prev_item, :], self.V_next_prev[false_next, :])
		delta = 1 - 1 / (1 + math.exp(min(10, max(-10, x_false - x_true)))) # Bound x_true - x_false in [-10, 10] to avoid overflow

		# Update CF
		V_user_item_mem = self.V_user_item[user, :]
		self.V_user_item[user, :] += self.learning_rate * ( delta * (self.V_item_user[true_next, :] - self.V_item_user[false_next, :]) - self.reg * self.V_user_item[user, :])
		self.V_item_user[true_next, :] += self.learning_rate * ( delta * V_user_item_mem - self.reg * self.V_item_user[true_next, :])
		self.V_item_user[false_next, :] += self.learning_rate * ( -delta * V_user_item_mem - self.reg * self.V_item_user[false_next, :])

		# Update MC
		V_prev_next_mem = self.V_prev_next[prev_item, :]
		self.V_prev_next[prev_item, :] += self.learning_rate * ( delta * (self.V_next_prev[true_next, :] - self.V_next_prev[false_next, :]) - self.reg * self.V_prev_next[prev_item, :])
		self.V_next_prev[true_next, :] += self.learning_rate * ( delta * V_prev_next_mem - self.reg * self.V_next_prev[true_next, :])
		self.V_next_prev[false_next, :] += self.learning_rate * ( -delta * V_prev_next_mem - self.reg * self.V_next_prev[false_next, :])

		return delta

	def compute_factor_rankings(self):
		'''Rank items according to each factor in order to do adaptive sampling
		'''

		CF_rank = np.argsort(self.V_item_user, axis=0)
		MC_rank = np.argsort(self.V_next_prev, axis=0)
		self.ranks = np.concatenate((CF_rank, MC_rank), axis=1)

		CF_var = np.var(self.V_item_user, axis=0)
		MC_var = np.var(self.V_next_prev, axis=0)
		self.var = np.concatenate((CF_var, MC_var))

	def get_training_sample(self):
		'''Pick a random triplet from self.triplets and a random false next item.
		returns a tuple of ids : (user, prev_item, true_next, false_next)
		'''

		# user_id, prev_item, true_next = random.choice(self.triplets)
		user_id = random.randrange(self.n_users)
		while self.users[user_id,1] < 2:
			user_id = random.randrange(self.n_users)
		r = random.randrange(self.users[user_id,1]-1)
		prev_item = self.items[self.users[user_id,0]+r]
		true_next = self.items[self.users[user_id,0]+r+1]
		if self.adaptive_sampling:
			while True:
				rank = np.random.exponential(scale=self.sampling_bias)
				while rank >= self.n_items:
					rank = np.random.exponential(scale=self.sampling_bias)
				factor_signs = np.sign(np.concatenate((self.V_user_item[user_id, :], self.V_prev_next[prev_item, :])))
				factor_prob = np.abs(np.concatenate((self.V_user_item[user_id, :], self.V_prev_next[prev_item, :]))) * self.var
				f = np.random.choice(self.k_cf+self.k_mc, p=factor_prob/sum(factor_prob))
				#print( int(rank), ' ', factor_signs[f], ' ', f )
				false_next = self.ranks[int(rank) * int(factor_signs[f]),f]
				if false_next != true_next:
					break
		else:
			false_next = random.randrange(self.n_items-1)
			if false_next >= true_next: # To make sure false_next != true_next
				false_next += 1

		return (user_id, prev_item, true_next, false_next)

	def top_k_recommendations(self, sequence, user_id=None, k=10, exclude=None, session=None):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''

		if exclude is None:
			exclude = []
		
		last_item = sequence[-1][0]
		if user_id is None:
			uv = self.V_item_user[session].mean(axis=0)
		else:
			uv = self.V_user_item[user_id, :]
		output = np.dot(uv, self.V_item_user.T) + np.dot(self.V_prev_next[last_item, :], self.V_next_prev.T)

		# Put low similarity to viewed items to exclude them from recommendations
		output[[i[0] for i in sequence]] = -np.inf
		output[exclude] = -np.inf

		# find top k according to output
		return list(np.argpartition(-output, range(k))[:k])
	
	def recommendations(self, sequence, user_id=None,exclude=None, session=None):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''

		if exclude is None:
			exclude = []
		
		last_item = sequence[-1][0]
		if user_id is None:
			uv = self.V_item_user[session].mean(axis=0)
		else:
			uv = self.V_user_item[user_id, :]
		output = np.dot(uv, self.V_item_user.T) + np.dot(self.V_prev_next[last_item, :], self.V_next_prev.T)

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
		np.savez(filename, V_user_item=self.V_user_item, V_item_user=self.V_item_user, V_prev_next=self.V_prev_next, V_next_prev=self.V_next_prev)

	def load(self, filename):
		'''Load parameters values form a file
		'''
		f = np.load(filename)
		self.V_user_item = f['V_user_item']
		self.V_item_user = f['V_item_user']
		self.V_prev_next = f['V_prev_next']
		self.V_next_prev = f['V_next_prev']
		