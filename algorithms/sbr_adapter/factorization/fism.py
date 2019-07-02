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

class FISM(MFBase):
	''' Code based on work by Kabbur et al., FISM: Factored Item Similarity Models for top-N Recommender Systems, KDD 2013.
	'''

	def __init__(self, k = 100, alpha=0.5, loss="BPR", **kwargs):

		super(FISM, self).__init__(**kwargs)

		self.name = 'FISM'
		self.k = k
		self.loss = loss
		if loss not in ['RMSE', 'BPR']:
			raise ValueError('Unknown loss for FISM: ', loss)
		self.alpha = alpha

	def _get_model_filename(self, epochs):
		'''Return the name of the file to save the current model
		'''
		filename = "fism_" + self.loss + "_ne"+str(epochs)+"_lr"+str(self.init_learning_rate)+"_an"+str(self.annealing_rate)+"_k"+str(self.k)+"_reg"+str(self.reg)+"_ini"+str(self.init_sigma)
		
		return filename+".npz"

	def init_model(self):
		''' Initialize the model parameters
		'''
		self.V = self.init_sigma * np.random.randn(self.n_items, self.k).astype(np.float32)
		self.H = self.init_sigma * np.random.randn(self.n_items, self.k).astype(np.float32)
		self.bias = np.zeros(self.n_items).astype(np.float32)

	def item_score(self, user_items, item = None):
		''' Compute the prediction score of the FISM model for the item "item", based on the list of items "user_items".
		'''
		if item is not None:
			return self.bias[item] + np.power(len(user_items), -self.alpha) * np.dot(self.V[user_items, :].sum(axis=0), self.H[item, :])
		else:
			return self.bias + np.power(len(user_items), -self.alpha) * np.dot(self.V[user_items, :].sum(axis=0), self.H.T)

	def auc_sgd_step(self, user_items, true_item, false_item):
		''' Make one SGD update, given that the interaction between user and true_item exists, 
		but the one between user and false_item does not.
		user, true_item and false_item are all user or item ids.

		return error
		'''

		# Compute error
		x_true = self.item_score(user_items, true_item)
		x_false = self.item_score(user_items, false_item)
		delta = 1 - 1 / (1 + math.exp(min(10, max(-10, x_false - x_true)))) # Original BPR error
		#delta = (x_true - x_false - 1) # error proposed in the FISM paper
		
		# Update CF
		V_sum = self.V[user_items, :].sum(axis=0)
		self.V[user_items, :] += self.learning_rate * ( delta * np.power(len(user_items), -self.alpha) * (self.H[true_item, :] - self.H[false_item, :]) - self.reg * self.V[user_items, :])
		self.H[true_item, :] += self.learning_rate * ( delta * np.power(len(user_items), -self.alpha) * V_sum - self.reg * self.H[true_item, :])
		self.H[false_item, :] += self.learning_rate * ( -delta * np.power(len(user_items), -self.alpha) * V_sum - self.reg * self.H[false_item, :])
		self.bias[true_item] += self.learning_rate * (delta - self.reg * self.bias[true_item])
		self.bias[false_item] += self.learning_rate * (- delta - self.reg * self.bias[false_item])

		return delta

	def rmse_sgd_step(self, user_items, item, rating):
		'''

		return error
		'''

		# Compute error
		prediction = self.item_score(user_items, item)
		delta = (rating - prediction) # error proposed in the FISM paper

		print(delta)
		if delta != delta:
			raise ValueError('NaN')
		
		# print(prediction)
		# y = raw_input()
		
		# Update CF
		V_sum = self.V[user_items, :].sum(axis=0)

		self.V[user_items, :] += self.learning_rate * ( delta * np.power(len(user_items), -self.alpha) * self.H[item, :] - self.reg * self.V[user_items, :])
		self.H[item, :] += self.learning_rate * ( delta * np.power(len(user_items), -self.alpha) * V_sum - self.reg * self.H[item, :])
		self.bias[item] += self.learning_rate * ( delta - self.reg * self.bias[item])

		return delta

	def get_auc_training_sample(self):
		'''Pick a random triplet from self.triplets and a random false next item.
		returns a tuple of ids : (user_items, true_item, false_item)
		'''

		user_id = random.randrange(self.n_users)
		while self.users[user_id,1] < 2:
			user_id = random.randrange(self.n_users)
		user_items = self.items[self.users[user_id,0]:self.users[user_id,0]+self.users[user_id,1]]
		
		true_item = random.choice(user_items)
		
		false_item = random.randrange(self.n_items)
		while false_item in user_items:
			false_item = random.randrange(self.n_items)

		return ([i for i in user_items if i is not true_item], true_item, false_item)

	def get_rmse_training_sample(self):

		neg_to_pos_ratio = 3
		user_items, true_item, false_item = self.get_auc_training_sample()

		if random.random() < 1 / (neg_to_pos_ratio + 1):
			return (user_items, true_item, 1)
		else:
			return (user_items, false_item, 0)


	def top_k_recommendations(self, sequence, user_id=None, k=10, exclude=None):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''

		if exclude is None:
			exclude = []

		user_items = [i[0] for i in sequence]
		output = self.item_score(user_items)

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

		user_items = session
		output = self.item_score(user_items)

		return output
	
	def training_step(self, iterations):
		if self.loss == "BPR":
			return self.auc_sgd_step(*self.get_auc_training_sample())
		else: 
			return self.rmse_sgd_step(*self.get_rmse_training_sample())

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