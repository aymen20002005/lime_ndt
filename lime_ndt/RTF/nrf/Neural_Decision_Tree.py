# -*- coding: utf-8 -*-
"""
Neural Decision Tree implementation
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.activations import tanh
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, mean_squared_error

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_functions import (get_list_split_phi,
									  get_parents_nodes_leaves_dic,
							  		  print_decision_path)


BIAS = 1 # Index for the bias value in split information
DIM = 0 # Index for the feature dimension in split information
RIGHT = 1 # Value representing the right child or right direction in the tree
LEFT = -1 # Value representing the left child or left direction in the tree
LEAVES = -1 # Special value indicating a leaf node
LEAVES_THRESHOLDS = -2 # Special value indicating a leaf node threshold
LEAVES_FEATURES = -2 # Special value indicating a leaf node feature
EMPTY_NODE = -5 # Special value indicating an empty or non-existent node


class SparseNDT(Callback):
	"""
	Callback to apply sparsity to the neural decision tree weights after each batch end.
	The goal is to keep the behaviour of the network close to the original decision tree.
	"""
	def __init__(self, NDT):
		super().__init__()

		self.weight_masks = []
		weight_matrices = [NDT.W_in_nodes.values, NDT.W_nodes_leaves.values]

		for w in weight_matrices:
			mat = np.copy(w)
			idx = mat.nonzero()
			mat[idx] = 1
			self.weight_masks.append(w)

	def on_batch_end(self, batch, logs=None):
		layers = [2, 5] # Apply sparsity on the Activation, maybe we can try to change it to [1,3]

		for i, layer in enumerate(layers):
			wb = self.model.layers[layer].get_weights()
			w = wb[0]
			b = wb[1]

			w = w*self.weight_masks[i]

			self.model.layers[layer].set_weights(weights=[w, b])


class tanh_gamma(Layer):

	def __init__(self, gamma=1, **kwargs):
		super(tanh_gamma, self).__init__(**kwargs)
		self.gamma = K.cast_to_floatx(gamma)

	def call(self, inputs):
		return K.tanh(inputs*self.gamma)

class CustomEarlyStopping(Callback):
    """
    Custom Early stopping callback that restores the best weights even is there
    was no early stopping during training. (Most of the code is the same as in
    the EarlyStopping callback in Keras)

    Args:
        monitor (str, optional): quantity to monitor. Defaults to 'val_loss'.
        min_delta (float, optional): minimum change in the monitored quantity
            to qualify as an improvement. Defaults to 0.
        patience (float, optional): number of epochs that produced the monitored
            quantity with no improvement after which training will
            be stopped. Defaults to 0.
        verbose (int, optional): verbosity mode. Defaults to 0.
        mode ('auto', 'min', or 'max'; optional): In `min` mode, training will
            stop when the quantity monitored has stopped decreasing; in `max`
            mode it will stop when the quantity monitored has stopped increasing;
            in `auto` mode, the direction is automatically inferred from the name
            of the monitored quantity.. Defaults to 'auto'.
        baseline (float, optional): baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement over the baseline.
            Defaults to None.
        restore_best_weights (bool, optional): whether to restore model weights from
            the epoch with the best value of the monitored quantity. If False,
            the model weights obtained at the last step of training are used. If
            True, the weights where the best value of the monitored quantity is achieved
            are restored, even if the training did not stop before the last epoch
            of training. Defaults to False.
    """
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
		
		# Initialize the callback with monitoring settings, patience, baseline, and options for restoring best weights.
        super(CustomEarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_weights_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Reset internal state at the start of training (wait counter, stopped epoch, best weights, etc.).
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_epoch_end(self, epoch, logs=None):
		# Check if the monitored metric has improved. If not, increment wait counter and stop training if patience is exceeded. Save best weights if required.
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:  # keep track of the best weights and epoch
                self.best_weights_epoch = epoch
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
		# At the end of training, optionally print early stopping info and restore the best weights if requested.
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        if self.restore_best_weights:  # restore best weights even if training did not stop
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch:', self.best_weights_epoch)
            self.model.set_weights(self.best_weights)

    def get_monitor_value(self, logs):
		# Retrieve the value of the monitored metric from the logs dictionary for the current epoch.
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        return monitor_value

class ndt:
	def __init__(self, D, gammas=[10, 1], tree_id=None,
				 sigma=0, gamma_activation=True, is_classifier=False):
		"""
		Create a neural decision tree

		Args:
		gammas (list): Metaparameter for each layer of the neural decision tree
					   (slope of the tanh function).
					   High gamma -> behavior of NN is closer to tree (and also
					   harder to change).
		tree_id (str or int): identifier for the tree.
		sigma (float): STD for the initial weights

		Returns:
		ndt : neural decision tree
		"""
		self.gammas = gammas
		self.use_gamma_activation=gamma_activation
		self.D = D
		self.tree_id = tree_id
		self.sigma = sigma
		self.is_classifier = is_classifier
		if self.is_classifier:
			self._estimator_type = "classifier"
		else:
			self._estimator_type = "regressor"

	def compute_matrices_and_biases(self, decision_tree):
		"""
		Compute synaptic weights and biases according to a decision tree

		Args:
				decision_tree (sklearn.tree.DecisionTreeClassifier): scikit-learn decision
				tree
		"""
		self.decision_tree = decision_tree
		self.splits = pd.DataFrame(get_list_split_phi(decision_tree)).T
		self.leaves = get_parents_nodes_leaves_dic(decision_tree)
		self.N = self.splits.shape[0]
		self.L = len(self.leaves)
		# Fill Input -> Nodes layer matrix
		self.W_in_nodes = pd.DataFrame(np.zeros((self.D, self.N)),
									   index=list(range(self.D)),
									   columns=self.splits.index)
		for node, dim in self.splits[DIM].items():
			self.W_in_nodes.loc[dim, node] = 1.
		# Fill Input -> Nodes layer biases

		self.b_nodes = pd.DataFrame(- self.splits[BIAS])
		self.b_nodes.columns = ["NODES_BIASES"]
		# Fill Nodes -> Leaves layer matrix
		self.W_nodes_leaves = pd.DataFrame(np.zeros((self.N, self.L)),
													 index=self.splits.index,
													 columns=self.leaves.keys())
		for leave, node_sides in self.leaves.items():
			for node, r_l in node_sides:
				self.W_nodes_leaves.loc[node, leave] = r_l
		# Fill Nodes -> Leaves layer biases
		b_leaves = {k: -len(x)+0.5 for k, x in self.leaves.items()}
		self.b_leaves = pd.DataFrame(list(b_leaves.values()),
									 index=b_leaves.keys(),
									 columns=["LEAVES_BIASES"])

		if not self.use_gamma_activation:
			self.W_in_nodes = self.W_in_nodes*self.gammas[0]
			self.b_nodes = self.b_nodes*self.gammas[0]

			self.W_nodes_leaves = self.W_nodes_leaves*self.gammas[1]
			self.b_leaves = self.b_leaves*self.gammas[1]

		if self.is_classifier:
			# Fill Leaves -> class matrix
			self.classes = decision_tree.classes_
			self.C = len(self.classes)

			class_counts_per_leaf = decision_tree.tree_.value[list(self.leaves.keys())]
			class_counts_per_leaf = class_counts_per_leaf.reshape(self.L, self.C)
			class_probas_all_leaves = class_counts_per_leaf * 1/np.sum(class_counts_per_leaf)
			
			self.W_leaves_out = pd.DataFrame(class_probas_all_leaves,
												index=self.leaves.keys(),
												columns=self.classes)
			self.W_leaves_out = self.W_leaves_out * 0.5

			# Fill Leaves -> class biases
			self.b_out = pd.DataFrame(np.sum(class_probas_all_leaves, axis=0),
										index=self.classes,
										columns=["CLASS_BIASES"])
			self.b_out = self.b_out * 0.5


		else:
			self.C = 1
			mean_leaf_values = decision_tree.tree_.value[list(self.leaves.keys())]
			# Debug
			print("mean_leaf_values shape:", mean_leaf_values.shape)
			print("self.L:", self.L, "self.C:", self.C)
			mean_leaf_values = np.squeeze(mean_leaf_values, axis=1)
			print("mean_leaf_values shape after squeeze:", mean_leaf_values.shape)
			self.W_leaves_out = pd.DataFrame(mean_leaf_values[:, :1],  # Ensure shape is (186, 1)
											 index=self.leaves.keys(),
											 columns=["Mean values/1"])
			self.W_leaves_out = self.W_leaves_out / 2

			self.b_out = pd.DataFrame(np.sum(mean_leaf_values),
									  index=[self.C],
									  columns=["BIAS"])
			self.b_out = self.b_out/2

	def to_keras(self, loss,
				 metrics=[], optimizer=optimizers.Adam,
				 kernel_regularizer=[None, None, None],
				 optimizer_params={"learning_rate": 0.001, "beta_1": 0.9,
								   "beta_2": 0.999, "epsilon": 1e-8,
								   "decay": 1e-6}):
		"""
		Creates a keras neural network

		Args:
				loss (str): loss function
				optimizer (keras.optimizers): keras optimizer
				kernel_regularizer (keras.regularizers): regularization constrains for
														 each layer
				optimizer_params (dict): dictionnary of parameters for the optimizer
		"""

		self.count_ops = 0

		self.input_layer = Input(shape=(self.D,))
		self.nodes_layer = Dense(self.N,
								 kernel_regularizer=kernel_regularizer[0])(self.input_layer)

		self.count_ops = self.count_ops+2*self.D*self.N

		if self.use_gamma_activation:
			self.act_layer_tanh_gamma1 = tanh_gamma(gamma=self.gammas[0])(self.nodes_layer)
			self.count_ops = self.count_ops+self.N*26
		else:
			self.act_layer_tanh_gamma1 = tanh_gamma(gamma=1)(self.nodes_layer)
			self.count_ops = self.count_ops+self.N*25

		self.leaves_layer = Dense(self.L,
								  kernel_regularizer=kernel_regularizer[1])(self.act_layer_tanh_gamma1)

		self.count_ops = self.count_ops+2*self.N*self.L

		if self.use_gamma_activation:
			self.act_layer_tanh_gamma2 = tanh_gamma(gamma=self.gammas[1])(self.leaves_layer)
			self.count_ops = self.count_ops+self.L*26
		else:
			self.act_layer_tanh_gamma2 = tanh_gamma(gamma=1)(self.leaves_layer)
			self.count_ops = self.count_ops+self.L*25

		if self.is_classifier:
			kr = kernel_regularizer[2]
			self.output_layer = Dense(self.C, activation='softmax',
									  kernel_regularizer=kr)(self.act_layer_tanh_gamma2)

			self.count_ops = self.count_ops+2*self.L*self.C
			self.count_ops = self.count_ops+self.C*(11+self.C*10)

		else:
			kr = kernel_regularizer[2]
			self.output_layer = Dense(self.C,
									  kernel_regularizer=kr)(self.act_layer_tanh_gamma2)

			self.count_ops = self.count_ops+2*self.L*self.C
			self.count_ops = self.count_ops+1

		self.model = Model(inputs=self.input_layer, outputs=self.output_layer)
		self.model_nodes = Model(inputs=self.input_layer, outputs=self.nodes_layer)
		self.model_leaves = Model(inputs=self.input_layer, outputs=self.leaves_layer)

		self.sgd = optimizer(**optimizer_params)
		self.model.compile(loss=loss, optimizer=self.sgd, metrics=metrics)

		# print(self.model.summary())

		flat_b_nodes = self.b_nodes.values.flatten()
		flat_b_leaves = self.b_leaves.values.flatten()
		flat_b_out = self.b_out.values.flatten()

		self.model.layers[1].set_weights(weights=[self.W_in_nodes+np.random.randn(*self.W_in_nodes.shape)*self.sigma,
										 flat_b_nodes+np.random.randn(*flat_b_nodes.shape)*self.sigma])
		self.model.layers[3].set_weights(weights=[self.W_nodes_leaves+np.random.randn(*self.W_nodes_leaves.shape)*self.sigma,
										 flat_b_leaves+np.random.randn(*flat_b_leaves.shape)*self.sigma])
		self.model.layers[5].set_weights(weights=[self.W_leaves_out+np.random.randn(*self.W_leaves_out.shape)*self.sigma,
										 flat_b_out+np.random.randn(*flat_b_out.shape)*self.sigma])

	def fit(self, X, y, sparse=False, epochs=100, min_delta=0, patience=10,
		 	earlyStopping=True, monitor="loss", validation_data=None,
			 restore_best_weights=True, verbose=0, **fit_params):
		"""
		Fit the neural decision tree

		Args:
		X (numpy.array or pandas.DataFrame): Training set
		y (numpy.array or pandas.Series): training set labels
		epochs (int): number of epochs
		min_delta (float): stoping criteria delta
		patience (int): stoping criteria patience
		to_categorical_conversion (bool): If True turn y to categorical
		"""

		callbacks_list = []

		if earlyStopping:
			early_stopping = CustomEarlyStopping(monitor=monitor,
										min_delta=min_delta,
										patience=patience,
										verbose=0,
										mode='auto',
										restore_best_weights=restore_best_weights)

			callbacks_list.append(early_stopping)

		if sparse:

			sparse_ndt = SparseNDT(self)
			callbacks_list.append(sparse_ndt)

		if not callbacks_list:
			callbacks_list = None

		history = self.model.fit(x=X,
								 y=y,
								 callbacks=callbacks_list,
								 epochs=epochs,
								 verbose=verbose,
								 validation_data=validation_data,
								 **fit_params)
		if earlyStopping:
			if early_stopping.stopped_epoch == 0:
				self.stopped_epoch = epochs
			else:
				if restore_best_weights:
					self.stopped_epoch = early_stopping.best_weights_epoch
				else:
					self.stopped_epoch = early_stopping.stopped_epoch


		return history.history

	def get_activations(self,
							X,
							y=None):
		"""
		Get the activation for each layer

		Args:
				X (np.array or pandas.DataFrame): data set
				y (np.array or pandas.Series): labels

		Returns:

		"""
		nodes_a = pd.DataFrame(self.model_nodes.predict(X),
							   columns=self.b_nodes.index)
		leaves_a = pd.DataFrame(self.model_leaves.predict(X),
								columns=self.b_leaves.index)
		output_a = pd.DataFrame(self.model.predict(X),
								columns=self.b_out.index)
		return nodes_a, leaves_a, output_a

	def get_weights_from_NN(self):
		"""
		Get the weights from the keras NN, and load them into attributes of the neural decision tree
		"""
		w_2 = self.model.layers[1].get_weights()

		self.W_in_nodes_nn = pd.DataFrame(w_2[0],
										  index=self.W_in_nodes.index,
										  columns=self.W_in_nodes.columns)
		self.b_nodes_nn = pd.DataFrame(w_2[1],
									   index=self.b_nodes.index,
									   columns=self.b_nodes.columns)

		w_5 = self.model.layers[3].get_weights()
		self.W_nodes_leaves_nn = pd.DataFrame(w_5[0],
											  index=self.W_nodes_leaves.index,
											  columns=self.W_nodes_leaves.columns)
		self.b_leaves_nn = pd.DataFrame(w_5[1],
										index=self.b_leaves.index,
										columns=self.b_leaves.columns)

		w_8 = self.model.layers[5].get_weights()
		self.W_leaves_out_nn = pd.DataFrame(w_8[0],
											index=self.W_leaves_out.index,
											columns=self.W_leaves_out.columns)
		self.b_out_nn = pd.DataFrame(w_8[1],
									   index=self.b_out.index,
									   columns=self.b_out.columns)

	def compute_weights_differences(self):
		"""
		Computes the difference between the original tree weights and those after training
		"""
		self.get_weights_from_NN()
		self.diff_W_in_nodes = self.W_in_nodes - self.W_in_nodes_nn
		self.diff_b_nodes = self.b_nodes - self.b_nodes_nn
		self.diff_W_nodes_leaves = self.W_nodes_leaves - self.W_nodes_leaves_nn
		self.diff_b_leaves = self.b_leaves - self.b_leaves_nn
		self.diff_W_leaves_output = self.W_leaves_out - self.W_leaves_out_nn
		self.diff_b_out = self.b_out - self.b_out_nn

	def predict(self, X, **kwargs):
		return self.model.predict(X, **kwargs)

	def score(self, X, y, **kwargs):
		"""
		Compute prediction score

		Args:
		X (numpy.array or pandas.DataFrame): dataset
		y (numpy.array or pandas.Series): labels
		"""
		if self.is_classifier:
			return accuracy_score(y, self.predict(X, **kwargs))
		else:
			return mean_squared_error(y, self.predict(X, **kwargs))

	def plot_differences(self):
		"""
		Plot the difference between the original tree weights and those after training
		"""
		if "diff_W_in_nodes" not in dir(self):
			self.compute_weights_differences()
		fig = plt.figure(figsize=(3, 2))
		columns = 3
		rows = 2
		ax1a = fig.add_subplot(rows, columns, 1)
		plt.imshow(self.diff_W_in_nodes, aspect="auto", cmap="gray")
		ax1a.set_title("diff W in nodes")

		ax2a = fig.add_subplot(rows, columns, 2)
		plt.imshow(self.diff_b_nodes, aspect="auto", cmap="gray")
		ax2a.set_title("diff b nodes")

		ax3a = fig.add_subplot(rows, columns, 3)
		plt.imshow(self.diff_W_nodes_leaves, aspect="auto", cmap="gray")
		ax3a.set_title("diff W nodes leaves")

		ax4a = fig.add_subplot(rows, columns, 4)
		plt.imshow(self.diff_b_leaves, aspect="auto", cmap="gray")
		ax4a.set_title("diff b leaves")

		ax5a = fig.add_subplot(rows, columns, 5)
		plt.imshow(self.diff_W_leaves_output, aspect="auto", cmap="gray")
		ax5a.set_title("diff W leaves out")

		ax6a = fig.add_subplot(rows, columns, 6)
		plt.imshow(self.diff_b_out, aspect="auto", cmap="gray")
		ax6a.set_title("diff b class")
		plt.show()

	def plot_W_nn_quantiles(self, quantiles=np.arange(0, 99.999, 0.001)):
		"""
		Plot the weights and biases quantiles
		"""
		fig = plt.figure(figsize=(3, 2))
		columns = 3
		rows = 2
		ax1a = fig.add_subplot(rows, columns, 1)
		plt.plot(quantiles, np.percentile(self.W_in_nodes_nn, quantiles))
		plt.plot(quantiles, np.percentile(self.W_in_nodes, quantiles))
		ax1a.set_title("W in nodes")

		ax2a = fig.add_subplot(rows, columns, 2)
		plt.plot(quantiles, np.percentile(self.b_nodes_nn, quantiles))
		plt.plot(quantiles, np.percentile(self.b_nodes, quantiles))
		ax2a.set_title("b nodes")

		ax3a = fig.add_subplot(rows, columns, 3)
		plt.plot(quantiles, np.percentile(self.W_nodes_leaves_nn, quantiles))
		plt.plot(quantiles, np.percentile(self.W_nodes_leaves, quantiles))
		ax3a.set_title("W nodes leaves")

		ax4a = fig.add_subplot(rows, columns, 4)
		plt.plot(quantiles, np.percentile(self.b_leaves_nn, quantiles))
		plt.plot(quantiles, np.percentile(self.b_leaves, quantiles))
		ax4a.set_title("b leaves")

		ax5a = fig.add_subplot(rows, columns, 5)
		plt.plot(quantiles, np.percentile(self.W_leaves_out_nn, quantiles))
		plt.plot(quantiles, np.percentile(self.W_leaves_out, quantiles))
		ax5a.set_title("W leaves out")

		ax6a = fig.add_subplot(rows, columns, 6)
		plt.plot(quantiles, np.percentile(self.b_out_nn, quantiles))
		plt.plot(quantiles, np.percentile(self.b_out, quantiles))
		ax6a.set_title("b class")
		plt.show()

	def neural_network_to_tree(self, node_leaves_matrix=None, in_nodes_matrix=None,
							   nodes_biases=None, threshold=0.9):
		def insert_node(parent,
						right_left,
						children_right,
						children_left,
						node_leaves_matrix,
						threshold):

			print("_____________")
			print("parent", parent)
			print("right_left", right_left)
			print("nodes_leaves_sum")
			print(node_leaves_matrix)

			# Find parent's child
			node_leaves_matrix = node_leaves_matrix[np.abs(node_leaves_matrix).sum(axis=1) > 0]
			count_nb_leaves = np.abs(node_leaves_matrix).sum(axis=1)

			child = count_nb_leaves.argmax()
			print("child", child, count_nb_leaves[child])

			# Remaining nodes are those that have at least one leaf
			remaining_nodes = list(count_nb_leaves.index[count_nb_leaves > 0])

			print("nodes_leaves_sum")
			print(np.abs(node_leaves_matrix).sum(axis=1))
			print("remaining nodes")
			print(remaining_nodes)

			# Add child to tree
			if parent is not None:
				if right_left == RIGHT:
					children_right[parent] = child
				elif right_left == LEFT:
					children_left[parent] = child

			# get leaves of current child
			current_node_leaves = node_leaves_matrix.loc[child]
			print("current node leaves")
			print(current_node_leaves)
			# Remove nodes that have been already included
			if parent in remaining_nodes:
				remaining_nodes.remove(parent)
			if child in remaining_nodes:
				remaining_nodes.remove(child)

			if remaining_nodes:
				node_leaves_matrix = node_leaves_matrix.loc[remaining_nodes]
			leaves = node_leaves_matrix.columns
			right_leaves = leaves[(current_node_leaves >= threshold).values]
			left_leaves = leaves[(current_node_leaves <= -threshold).values]

			print("leaves", leaves)
			print("right_leaves", right_leaves)
			print("left_leaves", left_leaves)

			# Apply the same method to right and left children
			right_node_leaves_matrix = None
			if len(remaining_nodes):
				right_node_leaves_matrix = node_leaves_matrix[right_leaves]
				if np.abs(right_node_leaves_matrix).sum().sum() <= threshold:
					right_node_leaves_matrix = None
				else:
					insert_node(child,
								RIGHT,
								children_right,
								children_left,
								right_node_leaves_matrix,
								threshold)
			if right_node_leaves_matrix is None:
				if not len(right_leaves):
					children_right[child] = EMPTY_NODE
				else:
					children_right[child] = right_leaves[0]
					if len(right_leaves) > 1:
						print("more than one leaf", right_leaves)
					print("adding to", child, "children right", right_leaves)

			left_node_leaves_matrix = None
			if len(remaining_nodes):
				left_node_leaves_matrix = node_leaves_matrix[left_leaves]
				if np.abs(left_node_leaves_matrix).sum().sum() <= threshold:
					left_node_leaves_matrix = None
				else:
					insert_node(child,
								LEFT,
								children_right,
								children_left,
								left_node_leaves_matrix,
								threshold)

			if left_node_leaves_matrix is None:
				if not len(left_leaves):
					children_left[child] = EMPTY_NODE
				else:
					children_left[child] = left_leaves[0]
					if len(left_leaves) > 1:
						print("more than one leaf", left_leaves)
				print("adding to", child, "children left", left_leaves)

		if node_leaves_matrix is None:
			node_leaves_matrix = self.W_nodes_leaves
		if in_nodes_matrix is None:
			in_nodes_matrix = self.W_in_nodes
		if nodes_biases is None:
			nodes_biases = self.b_nodes

		children_right = pd.Series(LEAVES, index=list(node_leaves_matrix.index)+list(node_leaves_matrix.columns))
		children_left = pd.Series(LEAVES, index=list(node_leaves_matrix.index)+list(node_leaves_matrix.columns))
		thresholds = pd.Series(LEAVES_THRESHOLDS, index=list(node_leaves_matrix.index)+list(node_leaves_matrix.columns))
		features = pd.Series(LEAVES_FEATURES, index=list(node_leaves_matrix.index)+list(node_leaves_matrix.columns))
		node_leaves_matrix_local = np.copy(node_leaves_matrix)
		node_leaves_matrix_local = node_leaves_matrix_local * (np.abs(node_leaves_matrix) >= threshold*self.gammas[1])
		insert_node(None,
					None,
					children_right,
					children_left,
					node_leaves_matrix_local,
					threshold)
		# Compute the threshold of each node
		# print features
		# print thresholds
		# print nodes_biases
		thresholds[nodes_biases.index] = nodes_biases["NODES_BIASES"]*1./self.gammas[0]
		# Retrieve the most important feature for each node (one could imagine to create new features if necessary)
		features[in_nodes_matrix.columns] = in_nodes_matrix.index[in_nodes_matrix.values.argmax(axis=0)]
		return children_right, children_left, thresholds, features

	def assert_sample(self, X):
		"""
		Print the decision path for the samples as well as the activation functions.

		Args:
				X (pandas.DataFrame or numpy.array): dataset
		"""
		print_decision_path(self.decision_tree, X)
		print(self.get_activations(X))

	def print_tree_weights(self):
		"""
		Print tree weights
		"""
		print("W: Input -> Nodes")
		print(self.W_in_nodes)
		print("b: Input -> Nodes")
		print(self.b_nodes)
		print("W: Nodes -> Leaves")
		print(self.W_nodes_leaves)
		print("b: Nodes -> Leaves")
		print(self.b_leaves)
		print("W: Leaves -> Out")
		print(self.W_leaves_out)
		print("b: Leaves -> Out")
		print(self.b_out)

	def print_nn_weights(self):
		"""
		Print NN weights
		"""
		print("W: Input -> Nodes")
		print(self.W_in_nodes_nn)
		print("b: Input -> Nodes")
		print(self.b_nodes_nn)
		print("W: Nodes -> Leaves")
		print(self.W_nodes_leaves_nn)
		print("b: Nodes -> Leaves")
		print(self.b_leaves_nn)
		print("W: Leaves -> Out")
		print(self.W_leaves_out_nn)
		print("b: Leaves -> Out")
		print(self.b_out_nn)

	def plot_old_new_network(self):
		"""
		Plot new and old network side by side
		"""
		if "W_in_nodes" not in dir(self):
			self.get_weights_from_NN()
		fig = plt.figure(figsize=(6, 2))
		columns = 6
		rows = 2
		ax1a = fig.add_subplot(rows, columns, 1)
		plt.imshow(self.W_in_nodes, aspect="auto", cmap="gray")
		ax1a.set_title("W in nodes")
		ax1b = fig.add_subplot(rows, columns, 2)
		plt.imshow(self.W_in_nodes_nn, aspect="auto", cmap="gray")
		ax1b.set_title("W in nodes nn")

		ax2a = fig.add_subplot(rows, columns, 3)
		plt.imshow(self.b_nodes, aspect="auto", cmap="gray")
		ax2a.set_title("b nodes ")
		ax2b = fig.add_subplot(rows, columns, 4)
		plt.imshow(self.b_nodes_nn, aspect="auto", cmap="gray")
		ax2b.set_title("b nodes nn")

		ax3a = fig.add_subplot(rows, columns, 5)
		plt.imshow(self.W_nodes_leaves, aspect="auto", cmap="gray")
		ax3a.set_title("W nodes leaves")
		ax3b = fig.add_subplot(rows, columns, 6)
		plt.imshow(self.W_nodes_leaves_nn, aspect="auto", cmap="gray")
		ax3b.set_title("W nodes leaves nn")

		ax4a = fig.add_subplot(rows, columns, 7)
		plt.imshow(self.b_leaves, aspect="auto", cmap="gray")
		ax4a.set_title("b leaves")
		ax4b = fig.add_subplot(rows, columns, 8)
		plt.imshow(self.b_leaves_nn, aspect="auto", cmap="gray")
		ax4b.set_title("b leaves nn")

		ax5a = fig.add_subplot(rows, columns, 9)
		plt.imshow(self.W_leaves_out, aspect="auto", cmap="gray")
		ax5a.set_title("W leaves out")
		ax5b = fig.add_subplot(rows, columns, 10)
		plt.imshow(self.W_leaves_out_nn, aspect="auto", cmap="gray")
		ax5b.set_title("W leaves out nn")

		ax6a = fig.add_subplot(rows, columns, 11)
		plt.imshow(self.b_out, aspect="auto", cmap="gray")
		ax6a.set_title("b class")
		ax6b = fig.add_subplot(rows, columns, 12)
		plt.imshow(self.b_out_nn, aspect="auto", cmap="gray")
		ax6b.set_title("b class nn")
		plt.show()


class NDTClassifier(ndt):
	def __init__(self, D, gammas=[10, 1], tree_id=None, sigma=0, gamma_activation=True):
		super().__init__(D, gammas=gammas, tree_id=tree_id,
						 sigma=sigma, gamma_activation=gamma_activation,
						 is_classifier=True)

	def fit(self, X, y, to_categorical_conversion=True, sparse=False, epochs=100,
			min_delta=0, patience=10, earlyStopping=True,
			monitor='loss', validation_data=None, verbose=0,
			**fit_params):

		if to_categorical_conversion:
			self.to_categorical = True
			y = to_categorical(y)
			if validation_data is not None:
				yv = validation_data[1]
				yv = to_categorical(yv)
				validation_data = (validation_data[0], yv)

		return super().fit(X, y, sparse=sparse, epochs=epochs, min_delta=min_delta,
						   patience=patience, earlyStopping=earlyStopping, monitor=monitor,
						   validation_data=validation_data, verbose=verbose, **fit_params)

	def predict_proba(self, X, **kwargs):
		return super().predict(X, **kwargs)

	def predict_classes(self, X, **kwargs):
		"""
		Predict class membership with the neural decision tree

		Args:
		X (numpy.array or pandas.DataFrame): dataset
		y (numpy.array or pandas.Series): labels
		"""
		class_indexes = np.argmax(self.predict_proba(X, **kwargs), axis=1)
		return self.classes[class_indexes]

	def predict(self, X, **kwargs):
		return self.predict_classes(X, **kwargs)


class NDTRegressor(ndt):
	def __init__(self, D, gammas=[10, 1], tree_id=None, sigma=0, gamma_activation=True):
		super().__init__(D, gammas=gammas, tree_id=tree_id,
						 sigma=sigma, is_classifier=False)


def generateNNClassifier(arch, acts, input_size, n_classes):

	model = Sequential()
	count_ops = 0

	if acts[0] == 'tanh':
		act_ops0 = 25
	else:
		act_ops0 = 1

	if acts[1] == 'tanh':
		act_ops1 = 25
	else:
		act_ops1 = 1

	for index, size in enumerate(arch):
		if index == 0:
			model.add(Dense(size, input_shape=(input_size,), activation=acts[0]))
			count_ops = count_ops+2*input_size*size
			count_ops = count_ops+act_ops0*size

		else:
			model.add(Dense(size, activation=acts[1]))
			count_ops = count_ops+2*arch[index-1]*size
			count_ops = count_ops+act_ops1*size

	optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-6)

	count_ops = count_ops+2*arch[-1]*n_classes

	if n_classes == 2:
		model.add(Dense(1, activation='sigmoid'))
		count_ops = count_ops+13*n_classes
		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	else:
		# print(n_classes)
		model.add(Dense(n_classes, activation='softmax'))
		count_ops = count_ops+n_classes*(11+n_classes*10)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy',
					  metrics=['sparse_categorical_accuracy'])

	model.count_ops = count_ops

	return model


def generateNNRegressor(arch, acts, input_size, n_classes=None):

	model = Sequential()
	count_ops = 0

	if acts[0] == 'tanh':
		act_ops0 = 25
	else:
		act_ops0 = 1

	if acts[1] == 'tanh':
		act_ops1 = 25
	else:
		act_ops1 = 1

	for index, size in enumerate(arch):
		if index == 0:
			model.add(Dense(size, input_shape=(input_size,), activation=acts[0]))
			count_ops = count_ops+2*input_size*size
			count_ops = count_ops+act_ops0*size

		else:
			model.add(Dense(size, activation=acts[1]))
			count_ops = count_ops+2*arch[index-1]*size
			count_ops = count_ops+act_ops1*size

	optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-6)

	count_ops = count_ops+2*arch[-1]*n_classes

	model.add(Dense(1))
	model.compile(optimizer=optimizer, loss='mean_squared_error')

	model.count_ops = count_ops

	return model


if __name__ == "__main__":
	from sklearn.tree import DecisionTreeClassifier  # , export_graphviz
	# import matplotlib.pyplot as plt
	dataset_length = 10000
	D = 2
	X = np.random.randn(dataset_length, D)*0.1
	X[0:dataset_length//2, 0] += 0.1
	X[0:dataset_length//2, 0] += 0.2
	Y = np.ones(dataset_length)
	Y[0:dataset_length//2] *= 0

	X_test = np.random.randn(dataset_length, D)*0.1
	X_test[0:dataset_length//2, 0] += 0.1
	X_test[0:dataset_length//2, 0] += 0.2
	Y_test = np.ones(dataset_length)
	Y_test[0:dataset_length//2] *= 0
	# Train a Tree
	clf = DecisionTreeClassifier(max_depth=10)
	clf = clf.fit(X, Y)

	a = ndt(D=2, gammas=[1, 100, 100], tree_id=0)
	a.compute_matrices_and_biases(clf)
	a.to_keras(loss='mean_squared_error')
	print("FLOPs before:", a.count_ops)
	a.fit(X, Y)
	print("FLOPs after:", a.count_ops)

	print("scores before training")
	print(a.score(X_test, Y_test))
	print(a.score(X, Y))

	print(clf.score(X_test, Y_test))
	print(clf.score(X, Y))
	errors = a.fit(X, Y, epochs=10)
	print("scores after training")
	print(a.score(X_test, Y_test))
	print(a.score(X, Y))