from __future__ import division

import numpy as np
import scipy.sparse as ssp
import os.path
import theano
import theano.tensor as T
import random
import operator
import collections
#import matplotlib.pyplot as plt

# Plot multiple figures at the same time
#plt.ion()

class Evaluator(object):
	'''Evaluator is a class to compute metrics on tests

	It is used by first adding a series of "instances" : pairs of goals and predictions, then metrics can be computed on the ensemble of instances:
	average precision, percentage of instance with a correct prediction, etc.

	It can also return the set of correct predictions.
	'''
	def __init__(self, dataset, k=10):
		super(Evaluator, self).__init__()
		self.instances = []
		self.dataset = dataset
		self.k = k

		self.metrics = {'sps': self.short_term_prediction_success,
			'recall': self.average_recall, 
			'precision': self.average_precision,
			'ndcg': self.average_ndcg,
			'item_coverage': self.item_coverage,
			'user_coverage': self.user_coverage,
			'assr': self.assr,
			'blockbuster_share': self.blockbuster_share}
	
	def add_instance(self, goal, predictions):
		#print('isntance:')
		#print( '-- goal:'+str(goal) )
		#print( '-- predictions:'+str(predictions) )
		self.instances.append([goal, predictions])

	def _load_interaction_matrix(self):
		'''Load the training set as an interaction matrix between items and users in a sparse format.
		'''
		filename = self.dataset.dirname + 'data/train_set_triplets'
		if os.path.isfile(filename + '.npy'):
			file_content = np.load(filename + '.npy')
		else:
			file_content = np.loadtxt(filename)
			np.save(filename, file_content)

		self._interactions = ssp.coo_matrix((np.ones(file_content.shape[0]), (file_content[:,1], file_content[:,0]))).tocsr()

	def _intra_list_similarity(self, items):
		'''Compute the intra-list similarity of a list of items.
		'''
		if not hasattr(self, "_interactions"):
			self._load_interaction_matrix()

		norm = np.sqrt(np.asarray(self._interactions[items, :].sum(axis=1)).ravel())
		sims = self._interactions[items, :].dot(self._interactions[items, :].T).toarray()
		S = 0
		for i in range(len(items)):
			for j in range(i):
				S += sims[i, j] / norm[i] / norm[j]

		return S

	def average_intra_list_similarity(self):
		'''Return the average intra-list similarity, as defined in "Auralist: Introducing Serendipity into Music Recommendation"
		'''

		ILS = 0
		for goal, prediction in self.instances:
			if len(prediction) > 0:
				ILS += self._intra_list_similarity(prediction[:min(len(prediction), self.k)])

		return ILS / len(self.instances)


	def blockbuster_share(self):
		'''Return the percentage of correct long term predictions that are about items in the top 1% of the most popular items.
		'''

		correct_predictions = self.get_correct_predictions()
		nb_pop_items = self.dataset.n_items // 100
		pop_items = np.argpartition(-self.dataset.item_popularity, nb_pop_items)[:nb_pop_items]

		if len(correct_predictions) == 0:
			return 0
		return len([i for i in correct_predictions if i in pop_items])/len(correct_predictions)

	def average_novelty(self):
		'''Return the average novelty measure, as defined in "Auralist: Introducing Serendipity into Music Recommendation"
		'''

		nb_of_ratings = sum(self.dataset.item_popularity)

		novelty = 0
		for goal, prediction in self.instances:
			if len(prediction) > 0:
				novelty += sum(map(np.log2, self.dataset.item_popularity[prediction[:min(len(prediction), self.k)]] / nb_of_ratings)) / min(len(prediction), self.k)

		return -novelty / len(self.instances)

	def average_precision(self):
		'''Return the average number of correct predictions per instance.
		'''
		precision = 0
		for goal, prediction in self.instances:
			if len(prediction) > 0:
				precision += float(len(set(goal) & set(prediction[:min(len(prediction), self.k)]))) / min(len(prediction), self.k)

		return precision / len(self.instances)

	def average_recall(self):
		'''Return the average recall.
		'''
		recall = 0
		for goal, prediction in self.instances:
			if len(goal) > 0:
				recall += float(len(set(goal) & set(prediction[:min(len(prediction), self.k)]))) / len(goal)

		return recall / len(self.instances)

	def average_ndcg(self):
		ndcg = 0.
		for goal, prediction in self.instances:
			if len(prediction) > 0:
				dcg = 0.
				max_dcg = 0.
				for i, p in enumerate(prediction[:min(len(prediction), self.k)]):
					if i < len(goal):
						max_dcg += 1. / np.log2(2 + i)

					if p in goal:
						dcg += 1. / np.log2(2 + i)

				ndcg += dcg/max_dcg

		return ndcg / len(self.instances)

	def short_term_prediction_success(self):
		'''Return the percentage of instances for which the first goal was in the predictions
		'''
		score = 0
		for goal, prediction in self.instances:
			score += int(goal[0] in prediction[:min(len(prediction), self.k)])

		return score / len(self.instances)
	
	def sps(self):
		return self.short_term_prediction_success()

	def user_coverage(self):
		'''Return the percentage of instances for which at least one of the goals was in the predictions
		'''
		score = 0
		for goal, prediction in self.instances:
			score += int(len(set(goal) & set(prediction[:min(len(prediction), self.k)])) > 0)

		return score / len(self.instances)

	def get_all_goals(self):
		'''Return a concatenation of the goals of each instances
		'''
		return [g for goal, _ in self.instances for g in goal]

	def get_strict_goals(self):
		'''Return a concatenation of the strict goals (i.e. the first goal) of each instances
		'''
		return [goal[0] for goal, _ in self.instances]

	def get_all_predictions(self):
		'''Return a concatenation of the predictions of each instances
		'''
		return [p for _, prediction in self.instances for p in prediction[:min(len(prediction), self.k)]]

	def get_correct_predictions(self):
		'''Return a concatenation of the correct predictions of each instances
		'''
		correct_predictions = []
		for goal, prediction in self.instances:
			correct_predictions.extend(list(set(goal) & set(prediction[:min(len(prediction), self.k)])))
		return correct_predictions

	def item_coverage(self):
		return len(set(self.get_correct_predictions()))

	def get_correct_strict_predictions(self):
		'''Return a concatenation of the strictly correct predictions of each instances (i.e. predicted the first goal)
		'''
		correct_predictions = []
		for goal, prediction in self.instances:
			correct_predictions.extend(list(set([goal[0]]) & set(prediction[:min(len(prediction), self.k)])))
		return correct_predictions

	def get_rank_comparison(self):
		'''Returns a list of tuple of the form (position of the item in the list of goals, position of the item in the recommendations)
		'''
		all_positions = []
		for goal, prediction in self.instances:
			position_in_predictions = np.argsort(prediction)[goal]
			all_positions.extend(list(enumerate(position_in_predictions)))

		return all_positions

	def assr(self):
		'''Returns the average search space reduction.
		It is defined as the number of items in the dataset divided by the average number of dot products made during testing.
		'''

		if hasattr(self, 'nb_of_dp') and self.nb_of_dp > 0:
			return self.dataset.n_items / self.nb_of_dp
		else:
			return 1 # If nb_of_dp is not defined, clustering is probably not used, return default assr: 1

class DistributionCharacteristics(object):
	"""DistributionCharacteristics computes and plot certain characteristics of a list of movies, such as the distribution of popularity.
	"""
	def __init__(self, movies):
		super(DistributionCharacteristics, self).__init__()
		self.movies = collections.Counter(movies)

	def plot_frequency_distribution(self):
		'''Plot the number of items versus the frequency
		'''
		frequencies = self.movies.values()
		freq_distribution = collections.Counter(frequencies)
		#plt.figure()
		#plt.loglog(freq_distribution.keys(), freq_distribution.values(), '.')
		#plt.show()

	def plot_popularity_distribution(self):
		'''Bar plot of the number of movies in each popularity category
		'''
		pass
# 		bars = np.zeros(10)
# 		for key, val in self.movies.items():
# 			popularity_index = OTHER_FEATURES[key, 3] - 1 # minus 1 to shift from 1-based to 0-based counting
# 			bars[popularity_index] += val

		# plt.figure()
		# plt.bar(np.arange(10) + 0.5, bars, width=1)
		# plt.show() 

	def number_of_movies(self):
		return len(self.movies)

		