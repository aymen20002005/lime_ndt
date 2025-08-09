# import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
# from apted import APTED
# from apted.helpers import Tree
# from scipy.stats import hmean
# from sklearn.metrics import (accuracy_score, cohen_kappa_score,
#                              confusion_matrix, precision_recall_fscore_support)
# from sklearn.metrics.pairwise import pairwise_distances
# from sklearn.utils.linear_assignment_ import linear_assignment
# from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
# from tqdm import tqdm

from .common_functions import get_list_split_phi, sameStructure


def group_splits_by_feature(list_splits):
    feature_list_splits = {}
    for id_split, split in list_splits.items():
        feature = split[0]
        if feature not in feature_list_splits:
            feature_list_splits[feature] = OrderedDict()
        feature_list_splits[feature][id_split] = split[1]
    return feature_list_splits


def splits_based_distance(dt_a, dt_b, distance="manhattan", gamma=5, epsilon=None):

    list_splits_a = get_list_split_phi(dt_a)
    list_splits_b = get_list_split_phi(dt_b)

    gb_feature_a = group_splits_by_feature(list_splits_a)
    gb_feature_b = group_splits_by_feature(list_splits_b)

    full_features = list(set(gb_feature_a.keys()).intersection(set(gb_feature_b.keys())))
    intersections = []
    unions = []
    for feature in full_features:
        thresholds_a = np.asarray(list(gb_feature_a[feature].values())).reshape(-1, 1)
        thresholds_b = np.asarray(list(gb_feature_b[feature].values())).reshape(-1, 1)
        if epsilon is None:
            if gamma is None:
                gamma = 0.999
            intra_a_distances = np.triu(pairwise_distances(thresholds_a, thresholds_a, metric=distance), 1)
            intra_b_distances = np.triu(pairwise_distances(thresholds_b, thresholds_b, metric=distance), 1)
            intra_distrib = np.asarray(list(intra_a_distances[intra_a_distances != 0])
                                       + list(intra_b_distances[intra_b_distances != 0]))
            epsilon = np.percentile(intra_distrib, gamma)
            # epsilon = gamma*min(intra_a_distances[intra_a_distances!=0].min(), intra_b_distances[intra_b_distances!=0].min())
        distances = pairwise_distances(thresholds_a, thresholds_b, metric=distance)
        axis = 0
        if thresholds_a.size < thresholds_b.size:
            axis = 1
        inter = (distances.min(axis=axis) <= epsilon).sum()
        intersections.append(inter)
        unions.append(thresholds_a.size + thresholds_b.size - inter)
    return 1-np.asarray(intersections).sum()*1./np.asarray(unions).sum()


def CE(y_A, y_B):
    confusion_matrix = pd.crosstab(y_A, y_B)
    best_A_B_couples = linear_assignment(-confusion_matrix)
    return sum([confusion_matrix.iloc[couple[0], couple[1]] for couple in best_A_B_couples])*1./sum(confusion_matrix.sum(0))


def CE_based_comparison(dt_a, dt_b, X):
    leafs_ids_a = dt_a.apply(X)
    leafs_ids_b = dt_b.apply(X)
    return CE(leafs_ids_a, leafs_ids_b)


def accuracy_based_distance(dt_a, dt_b, X):
    y_a = dt_a.predict(X)
    y_b = dt_b.predict(X)
    return 1-accuracy_score(y_a, y_b)


def kappa_based_distance(dt_a, dt_b, X):
    y_a = dt_a.predict(X)
    y_b = dt_b.predict(X)

    k = cohen_kappa_score(y_a, y_b)
    if k < 0:
        k = 0

    return 1-k


def disagreement(dt_a, dt_b, X):
    y_a = dt_a.predict(X)
    y_b = dt_b.predict(X)

    cm = confusion_matrix(y_a, y_b)
    cm_not_diag = np.copy(cm)
    np.fill_diagonal(cm_not_diag, 0)

    # return 1-1.0*sum(sum(cm_not_diag))/sum(sum(cm))
    return 1.0*sum(sum(cm_not_diag))/sum(sum(cm))


def TMD(dt_a, dt_b):
    '''
    Calculate Tree Matching Diversity (normalized), a structure diversity measure for decision trees
    using the APTED algorithm implementation

    to install apted: pip install apted
    '''
    def to_string(children_left, children_right, feature, current_node=0):
        '''
        Represent the DT structure in the accepted format for apted, the nodes' names are the features
        '''
        if feature[current_node] == -2:
            return "{-2}"
        else:
            return "{"+str(feature[current_node])+to_string(children_left, children_right, feature,
                                                            children_left[current_node]) +\
                           to_string(children_left, children_right, feature,
                                     children_right[current_node])+"}"

    t_a = Tree.from_text(to_string(dt_a.tree_.children_left, dt_a.tree_.children_right, dt_a.tree_.feature))
    t_b = Tree.from_text(to_string(dt_b.tree_.children_left, dt_b.tree_.children_right, dt_b.tree_.feature))

    a = APTED(t_a, t_b)
    return a.compute_edit_distance()/max(dt_a.tree_.node_count, dt_b.tree_.node_count)
    # return 1-a.compute_edit_distance()/max(dt_a.tree_.node_count, dt_b.tree_.node_count)
    # os.popen('python3 -m apted -t '+t_a+' '+t_b).read())/max(tree_a.tree_.node_count, tree_b.tree_.node_count

    # t_a = to_string(dt_a.tree_.children_left, dt_a.tree_.children_right, dt_a.tree_.feature)
    # t_b = to_string(dt_b.tree_.children_left, dt_b.tree_.children_right, dt_b.tree_.feature)

    # return 1-float(os.popen("java -jar apted.jar -t "+t_a+" "+t_b).read())/max(dt_a.tree_.node_count, dt_b.tree_.node_count)


def hmean_similarities(sim_a, sim_b):
    result = np.zeros_like(sim_a)

    for i in range(result.shape[0]):
        for j in range(i+1, result.shape[1]):
            if sim_a[i, j] != 0 and sim_b[i, j] != 0:
                result[i, j] = hmean([sim_a[i, j], sim_b[i, j]])
                result[j, i] = result[i, j]
            else:
                result[i, j] = (sim_a[i, j]+sim_b[i, j])/2
                result[j, i] = result[i, j]

    return result


def compare_trees_from_forest(tree_list, similarity_function, **params):  # list of trees
    similarity_matrix = np.zeros((len(tree_list), len(tree_list)))
    # for i in tqdm(range(len(tree_list)-1)):
    for i in range(len(tree_list)-1):
        # print(i)
        # for j in tqdm(range(i+1, len(tree_list))):
        for j in range(i+1, len(tree_list)):
            # print(j)
            dt_a = tree_list[i]
            dt_b = tree_list[j]
            similarity_matrix[i, j] = similarity_function(dt_a, dt_b, **params)
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return similarity_matrix


def fleiss_score(dt_list, X):

    if len(dt_list) == 1:
        return 1

    else:
        preds = np.zeros((X.shape[0], len(dt_list)))
        for j in range(len(dt_list)):
            preds[:, j] = dt_list[j].predict(X)

        predictions = aggregate_raters(preds)[0]

        return fleiss_kappa(predictions)


def treeDistanceStruct(dt_a, dt_b):

    def nodeDistance(feature_a, threshold_a, feature_b, threshold_b):
        d = 0

        if feature_a == feature_b:
            if threshold_a == threshold_b:
                d = 0
            else:
                d = np.abs(threshold_a-threshold_b)/max(threshold_a, threshold_b)
        else:
            d = 1

        return d

    def leafDistance(value_a, value_b):
        d = np.linalg.norm(value_a-value_b)
        return d

    def rec_distance(dict_a, dict_b, node_a=0, node_b=0):
        if dict_a["children_left"][node_a] == dict_a["children_right"][node_a]:
            return leafDistance(dict_a["value"][node_a], dict_b["value"][node_b])
        else:
            return nodeDistance(dict_a["feature"][node_a], dict_a["threshold"][node_a],
                                dict_b["feature"][node_b], dict_b["threshold"][node_b]) + \
                   1/2*rec_distance(dict_a, dict_b, node_a=dict_a["children_left"][node_a],
                                    node_b=dict_b["children_left"][node_b]) + \
                   1/2*rec_distance(dict_a, dict_b, node_a=dict_a["children_right"][node_a],
                                    node_b=dict_b["children_right"][node_b])

    d_a, d_b = sameStructure(dt_a, dt_b)

    d = rec_distance(d_a, d_b)

    return d


def classPerformanceDistance(dt_a, dt_b, X, y):
    classes, c = np.unique(y, return_counts=True)

    pred_a = np.take(classes, dt_a.predict(X).astype(int))
    pred_b = np.take(classes, dt_b.predict(X).astype(int))

    # print("#############")
    # print(classes)
    # print(np.unique(pred_a))
    # print(np.unique(pred_b))

    perf_a = precision_recall_fscore_support(y, pred_a)[0]  # *c/np.max(c)
    perf_b = precision_recall_fscore_support(y, pred_b)[0]  # *c/np.max(c)

    d = np.linalg.norm(perf_a-perf_b)

    return d
