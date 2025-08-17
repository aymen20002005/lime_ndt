import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ndt.model.Neural_Decision_Tree import NDTRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor

class NDTRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, D, gammas=[1, 100], tree_id=None, sigma=0, 
                 gamma_activation=True, max_depth=5, random_state=42, epochs=10):
        self.D = D
        self.gammas = gammas
        self.tree_id = tree_id
        self.sigma = sigma
        self.gamma_activation = gamma_activation
        self.max_depth = max_depth
        self.random_state = random_state
        self.epochs = epochs
        self.ndt = None

    def fit(self, X, y, sample_weight=None):
        self.ndt = NDTRegressor(num_features=self.D, gammas=self.gammas, tree_id=self.tree_id, 
                                sigma=self.sigma, gamma_activation=self.gamma_activation)
        tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                     random_state=self.random_state).fit(X, y)
        self.ndt.compute_matrices_and_biases(tree)
        self.ndt.to_keras(loss='mean_squared_error')
        self.ndt.fit(X, y, epochs=self.epochs)
        self.intercept_ = np.mean(y)
        return self

    def predict(self, X):
        return self.ndt.predict(X).flatten()

    @property
    def coef_(self):
        # Multiplication des 3 matrices pour obtenir l'impact global des features
        W1 = self.ndt.W_in_nodes.values            # features -> nodes
        W2 = self.ndt.W_nodes_leaves.values        # nodes -> leaves
        W3 = self.ndt.W_leaves_out.values          # leaves -> output
        coef_matrix = W1 @ W2 @ W3
        return coef_matrix.mean(axis=1)            # importance moyenne par feature