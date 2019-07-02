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

class Fossil(MFBase):
	''' Code based on work by He and McAuley. Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation. CoRR abs/1609.09152, 2016.
	'''

	def __init__(self, k = 100, order=1, alpha=0.2, **kwargs):

		super(Fossil, self).__init__(**kwargs)

		self.name = 'Fossil'
		self.k = k
		self.order = order #markov chain order
		self.alpha = alpha

	def _get_model_filename(self, epochs):
		'''Return the name of the file to save the current model
		'''
		filename = "fossil_ne"+str(epochs)+"_lr"+str(self.init_learning_rate)+"_an"+str(self.annealing_rate)+"_k"+str(self.k)+"_o"+str(self.order)+"_reg"+str(self.reg)+"_ini"+str(self.init_sigma)
		
		return filename+".npz"

	def init_model(self):
		''' Initialize the model parameters
		'''
		self.V = self.init_sigma * np.random.randn(self.n_items, self.k).astype(np.float32)
		self.H = self.init_sigma * np.random.randn(self.n_items, self.k).astype(np.float32)
		self.eta = self.init_sigma * np.random.randn(self.n_users, self.order).astype(np.float32)
		self.eta_bias = np.zeros(self.order).astype(np.float32)
		self.bias = np.zeros(self.n_items).astype(np.float32)

	def item_score(self, user_id, user_items, item=None):
		''' Compute the prediction score of the Fossil model for the item "item", based on the list of items "user_items".
		'''

		long_term = np.power(len(user_items), -self.alpha) * self.V[user_items, :].sum(axis=0)
		effective_order = min(self.order, len(user_items))
		if user_id is None:
			short_term = np.dot((self.eta_bias + self.eta.mean(axis=0))[:effective_order], self.V[user_items[:-effective_order-1:-1], :])
		else:
			short_term = np.dot((self.eta_bias + self.eta[user_id, :])[:effective_order], self.V[user_items[:-effective_order-1:-1], :])

		if item is not None:
			return self.bias[item] + np.dot(long_term + short_term, self.H[item, :])
		else:
			return self.bias + np.dot(long_term + short_term, self.H.T)

	def sgd_step(self, user_id, user_items, false_item):
		''' Make one SGD update, given that the interaction between user and true_item exists, 
		but the one between user and false_item does not.

		return error
		'''

		true_item = user_items[-1]
		user_items = user_items[:-1]
		effective_order = min(self.order, len(user_items))

		long_term = np.power(len(user_items), -self.alpha) * self.V[user_items, :].sum(axis=0)
		short_term = np.dot((self.eta_bias + self.eta[user_id, :])[:effective_order], self.V[user_items[:-effective_order-1:-1], :])

		# Compute error
		x_true = self.item_score(user_id, user_items, true_item)
		x_false = self.item_score(user_id, user_items, false_item)
		delta = 1 / (1 + math.exp(-min(10, max(-10, x_false - x_true)))) # sigmoid of the error
		
		# Compute Update
		V_update = self.learning_rate * ( delta * np.power(len(user_items), -self.alpha) * (self.H[true_item, :] - self.H[false_item, :]) - self.reg * self.V[user_items, :])
		V_update2 = self.learning_rate * delta *  np.outer((self.eta_bias + self.eta[user_id, :])[:effective_order], self.H[true_item, :] - self.H[false_item, :])
		H_true_up = self.learning_rate * ( delta * (long_term + short_term) - self.reg * self.H[true_item, :])
		H_false_up = self.learning_rate * ( -delta * (long_term + short_term) - self.reg * self.H[false_item, :])
		bias_true_up = self.learning_rate * (delta - self.reg * self.bias[true_item])
		bias_false_up = self.learning_rate * (- delta - self.reg * self.bias[false_item])
		eta_bias_up = self.learning_rate * (delta * np.dot(self.V[user_items[:-effective_order-1:-1], :], self.H[true_item, :] - self.H[false_item, :]) - self.reg * self.eta_bias[:effective_order])
		eta_up = self.learning_rate * (delta * np.dot(self.V[user_items[:-effective_order-1:-1], :], self.H[true_item, :] - self.H[false_item, :]) - self.reg * self.eta[user_id, :effective_order])


		# Update
		self.V[user_items, :] += V_update
		self.V[user_items[:-effective_order-1:-1], :] += V_update2
		self.H[true_item, :] += H_true_up
		self.H[false_item, :] += H_false_up
		self.bias[true_item] += bias_true_up
		self.bias[false_item] += bias_false_up
		self.eta_bias[:effective_order] += eta_bias_up
		self.eta[user_id, :effective_order] += eta_up

		return delta

	def get_training_sample(self):
		'''Pick a random triplet from self.triplets and a random false next item.
		returns a tuple of ids : (user_items, true_item, false_item)
		'''

		user_id = random.randrange(self.n_users)
		while self.users[user_id,1] < 2:
			user_id = random.randrange(self.n_users)
		user_items = self.items[self.users[user_id,0]:self.users[user_id,0]+self.users[user_id,1]]
		
		t = random.randrange(1, len(user_items))
		
		false_item = random.randrange(self.n_items)
		while false_item in user_items[:t+1]:
			false_item = random.randrange(self.n_items)

		return (user_id, user_items[:t+1], false_item)


	def top_k_recommendations(self, sequence, user_id=None, k=10, exclude=None):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''

		if exclude is None:
			exclude = []

		user_items = [i[0] for i in sequence]
		output = self.item_score(user_id, user_items)

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

		output = self.item_score(user_id, session)
		
		return output

	def training_step(self, iterations):
		return self.sgd_step(*self.get_training_sample())

	def save(self, filename):
		'''Save the parameters of a network into a file
		'''
		print('Save model in ' + filename)
		if not os.path.exists(os.path.dirname(filename)):
			os.makedirs(os.path.dirname(filename))
		np.savez(filename, V=self.V, H=self.H, bias=self.bias, eta=self.eta, eta_bias=self.eta_bias)

	def load(self, filename):
		'''Load parameters values form a file
		'''
		f = np.load(filename)
		self.V = f['V']
		self.H = f['H']
		self.bias = f['bias']
		self.eta = f['eta']
		self.eta_bias = f['eta_bias']