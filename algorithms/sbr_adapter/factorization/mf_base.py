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

class MFBase(object):
	'''Base class for methods based on matrix factorization
	'''

	def __init__(self, reg = 0.0025, learning_rate = 0.05, annealing=1., init_sigma = 1):
		self.name = 'Base for matrix factorization'
		self.reg = reg
		self.learning_rate = learning_rate # self.learning_rate will change due to annealing.
		self.init_learning_rate = learning_rate # self.init_learning_rate keeps the original value (for filename)
		self.annealing_rate = annealing
		self.init_sigma = init_sigma
		self.max_length = np.inf # For compatibility with the RNNs

		self.metrics = {'recall': {'direction': 1},
			'sps': {'direction': 1},
			'user_coverage' : {'direction': 1},
			'item_coverage' : {'direction': 1},
			'ndcg' : {'direction': 1},
			#'blockbuster_share' : {'direction': -1}
		}

	def prepare_model(self, dataset):
		'''Must be called before using train, load or top_k_recommendations
		'''
		self.dataset = dataset
		self.valset = self.dataset[ np.in1d( dataset.SessionId, dataset.SessionId.unique()[-1000:] ) ]
		self.n_items = dataset.ItemId.nunique()
		self.n_users = dataset.SessionId.nunique()

	def change_data_format(self, dataset):
		'''Gets a generator of data in the sequence format and save data in the csr format
		'''
		
		self.users = np.zeros((self.n_users,2), dtype=np.int32 )
		self.items = np.zeros( len(dataset), dtype=np.int32 )
		
		index_session = dataset.columns.get_loc( 'SessionId' )
		index_item = dataset.columns.get_loc( 'ItemId' )
		
		session_list = []
		
		self.user_map = {}
		self.user_count = 0
		
		self.item_map = {}
		self.item_list = []
		self.item_count = 0
		
		last_session = -1
		
		cursor = 0
		
		for row in dataset.itertuples(index=False): 
			
			item, session = row[index_item], row[index_session]
			
			if not session in self.user_map:
				self.user_map[session] = self.user_count
				self.user_count += 1
			
			if not item in self.item_map:
				self.item_map[item] = self.item_count
				self.item_list.append(item)
				self.item_count += 1
				
			if last_session != session:
				muser = self.user_map[session]
				
				if last_session > 0:
					self.users[muser, :] = [cursor, len(session_list)]
					self.items[cursor:cursor+len(session_list)] = session_list
					cursor += len(session_list)
				
				session_list = []
			
			last_session = session
			session_list.append( self.item_map[item] )
		
		self.users[muser, :] = [cursor, len(session_list)]
		self.items[cursor:cursor+len(session_list)] = session_list
		cursor += len(session_list)
		
	
	def prepare_model_old(self, dataset):
		'''Must be called before using train, load or top_k_recommendations
		'''
		self.dataset = dataset
		self.n_items = dataset.n_items
		self.n_users = dataset.n_users

	def change_data_format_old(self, dataset):
		'''Gets a generator of data in the sequence format and save data in the csr format
		'''
		
		self.users = np.zeros((self.n_users,2), dtype=np.int32)
		self.items = np.zeros(dataset.training_set.n_interactions, dtype=np.int32)
		cursor = 0
		with open(dataset.training_set.filename, 'r') as f:
			for sequence in f:
				sequence = sequence.split()
				items = map(int, sequence[1::2])
				self.users[int(sequence[0]), :] = [cursor, len(list(items))]
				self.items[cursor:cursor+len(list(items))] = items
				cursor += len(list(items))
	
	def get_pareto_front(self, metrics, metrics_names):
		costs = np.zeros((len(metrics[metrics_names[0]]), len(metrics_names)))
		for i, m in enumerate(metrics_names):
			costs[:, i] = np.array(metrics[m]) * self.metrics[m]['direction']
		is_efficient = np.ones(costs.shape[0], dtype = bool)
		for i, c in enumerate(costs):
			if is_efficient[i]:
				is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)
		return np.where(is_efficient)[0].tolist()


	def _compute_validation_metrics(self, metrics):
		ev = evaluation.Evaluator(self.dataset, k=10)
		
		session_idx = self.valset.columns.get_loc( 'SessionId' )
		item_idx = self.valset.columns.get_loc( 'ItemId' )
		
		last_item = -1
		last_session = -1
		for row in self.valset.itertuples(index=False):
			item, session = row[item_idx], row[session_idx]
			
			if last_session != session:
				last_item = -1
			
			if last_item != -1:
				seq = [[self.item_map[last_item]]]
				top_k = self.top_k_recommendations(seq, user_id=self.user_map[session] )
				ev.add_instance([self.item_map[item]], top_k)
			
			last_item = item
			last_session = session
			
		metrics['recall'].append(ev.average_recall())
		metrics['sps'].append(ev.sps())
		metrics['ndcg'].append(ev.average_ndcg())
		metrics['user_coverage'].append(ev.user_coverage())
		metrics['item_coverage'].append(ev.item_coverage())
		#metrics['blockbuster_share'].append(ev.blockbuster_share())

		return metrics

	def train(self, dataset, 
		max_time=1000, 
		progress=100000,
		time_based_progress=False, 
		autosave='Best', 
		save_dir='mdl/', 
		min_iterations=100000,
		max_iter=2000000, 
		max_progress_interval=np.inf,
		load_last_model=False,
		early_stopping=None,
		validation_metrics=['sps']):
		'''Train the model based on the sequence given by the training_set

		max_time is used to set the maximumn amount of time (in seconds) that the training can last before being stop.
			By default, max_time=np.inf, which means that the training will last until the training_set runs out, or the user interrupt the program.
		
		progress is used to set when progress information should be printed during training. It can be either an int or a float:
			integer : print at linear intervals specified by the value of progress (i.e. : progress, 2*progress, 3*progress, ...)
			float : print at geometric intervals specified by the value of progress (i.e. : progress, progress^2, progress^3, ...)

		max_progress_interval can be used to have geometric intervals in the begining then switch to linear intervals. 
			It ensures, independently of the progress parameter, that progress is shown at least every max_progress_interval.

		time_based_progress is used to choose between using number of iterations or time as a progress indicator. True means time (in seconds) is used, False means number of iterations.

		autosave is used to set whether the model should be saved during training. It can take several values:
			All : the model will be saved each time progress info is printed.
			Best : save only the best model so far
			None : does not save

		min_iterations is used to set a minimum of iterations before printing the first information (and saving the model).

		save_dir is the path to the directory where models are saved.

		load_last_model: if true, find the latest model in the directory where models should be saved, and load it before starting training.

		early_stopping : should be a callable that will recieve the list of validation error and the corresponding epochs and return a boolean indicating whether the learning should stop.
		'''

		# Change data format
		self.change_data_format(dataset)
		#del dataset.training_set.lines
		if len(set(validation_metrics) & set(self.metrics.keys())) < len(validation_metrics):
			raise ValueError('Incorrect validation metrics. Metrics must be chosen among: ' + ', '.join(self.metrics.keys()))

		# Load last model if needed, else initialise the model
		iterations = 0
		epochs_offset = 0
		if load_last_model:
			epochs_offset = self.load_last(save_dir)
		if epochs_offset == 0:
			self.init_model()

		start_time = time()
		next_save = int(progress)
		train_costs = []
		current_train_cost = []
		epochs = []
		metrics = {name:[] for name in self.metrics.keys()}
		filename = {}

		while (time() - start_time < max_time and iterations < max_iter):

			cost = self.training_step(iterations)

			current_train_cost.append(cost)

			# Cool learning rate
			if iterations % len(dataset) == 0:
				self.learning_rate *= self.annealing_rate

			# Check if it is time to save the model
			iterations += 1

			if time_based_progress:
				progress_indicator = int(time() - start_time)
			else:
				progress_indicator = iterations

			if progress_indicator >= next_save:

				if progress_indicator >= min_iterations:
					
					# Save current epoch
					epochs.append(epochs_offset + iterations / len(dataset) )

					# Average train cost
					train_costs.append(np.mean(current_train_cost))
					current_train_cost = []

					# Compute validation cost
					metrics = self._compute_validation_metrics( metrics )

					# Print info
					self._print_progress(iterations, epochs[-1], start_time, train_costs, metrics, validation_metrics)

					# Save model
					run_nb = len(metrics[list(self.metrics.keys())[0]])-1
					if autosave == 'All':
						filename[run_nb] = save_dir + self._get_model_filename(round(epochs[-1], 3))
						self.save(filename[run_nb])
					elif autosave == 'Best':
						pareto_runs = self.get_pareto_front(metrics, validation_metrics)
						if run_nb in pareto_runs:
							filename[run_nb] = save_dir + self._get_model_filename(round(epochs[-1], 3))
							self.save(filename[run_nb])
							to_delete = [r for r in filename if r not in pareto_runs]
							for run in to_delete:
								try:
									os.remove(filename[run])
									print('Deleted ', filename[run])
								except OSError:
									print('Warning : Previous model could not be deleted')
								del filename[run]

					if early_stopping is not None:
						# Stop if early stopping is triggered for all the validation metrics
						if all([early_stopping(epochs, metrics[m]) for m in validation_metrics]):
							break 


				# Compute next checkpoint
				if isinstance(progress, int):
					next_save += min(progress, max_progress_interval)
				else:
					next_save += min(max_progress_interval, next_save * (progress - 1))

		best_run = np.argmax(np.array(metrics[validation_metrics[0]]) * self.metrics[validation_metrics[0]]['direction'])
		return ({m: metrics[m][best_run] for m in self.metrics.keys()}, time()-start_time, filename[best_run])

	def _print_progress(self, iterations, epochs, start_time, train_costs, metrics, validation_metrics):
		'''Print learning progress in terminal
		'''
		print(self.name, iterations, "batchs, ", epochs, " epochs in", time() - start_time, "s")
		print("Last train cost : ", train_costs[-1])
		for m in self.metrics:
			print(m, ': ', metrics[m][-1])
			if m in validation_metrics:
				print('Best ', m, ': ', max(np.array(metrics[m])*self.metrics[m]['direction'])*self.metrics[m]['direction'])
		print('-----------------')

		# Print on stderr for easier recording of progress
		#print(iterations, epochs, time() - start_time, train_costs[-1], ' '.join(map(str, [metrics[m][-1] for m in self.metrics])), file=sys.stderr)

	def load_last(self, save_dir):
		'''Load last model from dir
		'''
		def extract_number_of_epochs(filename):
			m = re.search('_ne([0-9]+(\.[0-9]+)?)_', filename)
			return float(m.group(1))

		# Get all the models for this RNN
		file = save_dir + self._get_model_filename("*")
		file = np.array(glob.glob(file))

		if len(file) == 0:
			print('No previous model, starting from scratch')
			return 0

		# Find last model and load it
		last_batch = np.amax(np.array(map(extract_number_of_epochs, file)))
		last_model = save_dir + self._get_model_filename(last_batch)
		print('Starting from model ' + last_model)
		self.load(last_model)

		return last_batch