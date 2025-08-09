import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def generateNewDataClassification(oracle, X, y, ratio, replace_labels=False, by_class=True, **params):
	'''
	Generate new data using an oracle to label randomly generated examples.
	The quantity of new data depends on the input ratio
	'''

	newX = np.copy(X)
	classes = np.unique(y)

	if hasattr(oracle, "predict_classes"):
		predict_func = oracle.predict_classes
	else:
		predict_func = oracle.predict

	if replace_labels:
		newY = predict_func(X, **params)
		if not isinstance(newY, list):
			newY = np.reshape(newY, (newY.shape[0],))

	else:
		newY = np.copy(y)

	if ratio > 0:

		if by_class:  #### distinguish between classes when sampling values for new examples

			for c in classes:
				idx = np.nonzero(y == c)[0]
				X_cl = X[idx, :]

				size = np.ceil(ratio*X_cl.shape[0]).astype(int)  ### at least one example from each class

				extra_X = np.zeros((size, X_cl.shape[1]))

				idx = np.random.choice(X_cl.shape[0], size=extra_X.shape,
									   replace=True)

				extra_X = X_cl[idx, np.arange(X_cl.shape[1])[None, :]]

				newX = np.vstack((newX, extra_X))
				# print(predict_func)
				preds = predict_func(extra_X, **params)
				if not isinstance(preds, list):
					if len(preds.shape) > 1:
						if preds.shape[1] == 1:
							preds = preds.reshape((preds.shape[0],))
					# print(preds.shape)
					newY = np.concatenate((newY, preds))  ### generate labels for
														  ### the new examples
				else:
					newY = np.concatenate((newY, preds))

		else:

			size = np.ceil(ratio*X.shape[0]).astype(int)  ### at least one example from each class

			extra_X = np.zeros((size, X.shape[1]))

			idx = np.random.choice(X.shape[0], size=extra_X.shape, replace=True)

			extra_X = X[idx, np.arange(X.shape[1])[None, :]]

			newX = np.vstack((newX, extra_X))
			preds = predict_func(extra_X, **params)
			preds = np.reshape(preds, (preds.shape[0],))
			newY = np.concatenate((newY, preds))  ### generate labels for
												  ### the new examples

	classes_Ynew = np.unique(newY)
	not_represented = set(classes) - set(classes_Ynew)

	if bool(not_represented):
		for c in not_represented:
			idx = np.nonzero(y == c)[0]
			# rand_idx = np.random.choice(idx, size=1)

			newX = np.vstack((newX, X[idx, :]))
			newY = np.concatenate((newY, y[idx]))

	idx = np.arange(len(newY))
	np.random.shuffle(idx)  ### shuffle examples
	newX = newX[idx, :]
	newY = np.array([newY[i] for i in idx])

	# print("#################")
	# print(idx)
	# print(newX.shape)
	# print(len(newY))

	return(newX, newY)


def generateNewDataRegression(oracle, X, y, ratio, replace_labels=False, by_class=None, **params):
	'''
	Generate new data using an oracle to label randomly generated examples.
	The quantity of new data depends on the input ratio
	'''

	newX = np.copy(X)

	if replace_labels:
		newY = oracle.predict(X, **params)
		newY = np.reshape(newY, (newY.shape[0],))

	else:
		newY = np.copy(y)

	if ratio > 0:

		size = np.ceil(ratio*X.shape[0]).astype(int)  ### at least one example from each class

		extra_X = np.zeros((size, X.shape[1]))

		idx = np.random.choice(X.shape[0], size=extra_X.shape, replace=True)

		extra_X = X[idx, np.arange(X.shape[1])[None, :]]

		newX = np.vstack((newX, extra_X))

		preds = oracle.predict(extra_X, **params)
		preds = np.reshape(preds, (preds.shape[0],))
		newY = np.concatenate((newY, preds))  ### generate labels for
											  ### the new examples

	idx = np.arange(len(newY))
	np.random.shuffle(idx)  ### shuffle examples
	newX = newX[idx, :]
	newY = newY[idx]

	return(newX, newY)


def DT_extraction(oracle, X, Y, ratio, max_depth, n_trees=1,
				  replace_labels=False, by_class=True,
				  classification=True, **params):

	if hasattr(oracle, "_estimator_type"):
		if oracle._estimator_type == "classifier":
			generateNewData = generateNewDataClassification
			DecisionTree = DecisionTreeClassifier
			RandomForest = RandomForestClassifier
		else:
			generateNewData = generateNewDataRegression
			DecisionTree = DecisionTreeRegressor
			RandomForest = RandomForestRegressor

	else:
		if classification:
			generateNewData = generateNewDataClassification
			DecisionTree = DecisionTreeClassifier
			RandomForest = RandomForestClassifier
		else:
			generateNewData = generateNewDataRegression
			DecisionTree = DecisionTreeRegressor
			RandomForest = RandomForestRegressor

	newX, newY = generateNewData(oracle, X, Y, ratio, replace_labels, by_class, **params)

	if n_trees == 1:
		tree = DecisionTree(max_depth=max_depth, random_state=0)
		tree.fit(newX, newY)

		return tree

	else:
		trees = RandomForest(n_estimators=n_trees, max_depth=max_depth, random_state=0)
		trees.fit(newX, newY)

		return trees
