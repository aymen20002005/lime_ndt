# -*- coding: utf-8 -*-
"""
@author: sergio
"""

# import copy
from collections import OrderedDict

import numpy as np
# from sklearn import tree


"""
children_left

children_right

feature

threshold

value : counts of each class at each node

impurity

n_node_samples
"""


def get_list_split_phi(decision_tree, node=0):

    def rec_get_list_split_phi(decisiontree, splits_phi, node):

        phi = decisiontree.feature[node]
        threshold = decisiontree.threshold[node]
        if phi != -2:
            splits_phi[node] = [phi, threshold]
        children = [decisiontree.children_left[node],
                    decisiontree.children_right[node]]

        for child in children:
            if child != -1:
                rec_get_list_split_phi(decisiontree, splits_phi, child)
        return 0

    splits_phi = OrderedDict()
    rec_get_list_split_phi(decision_tree.tree_, splits_phi, node=node)

    return splits_phi


def get_parent_nodes_leaf(leaf, decision_tree):
    def bottom_up_get_parent(node, parents, tree):
        p, b = find_parent(tree, node)
        if b != 0:
            parents.append((p, b))
            bottom_up_get_parent(p, parents, tree)
    parents = []
    bottom_up_get_parent(leaf, parents, decision_tree.tree_)
    return parents


def get_parents_nodes_leaves_dic(decision_tree):
    leaves = leaves_id(decision_tree.tree_)
    leaves_parents = OrderedDict()
    for leave in leaves:
        parents = get_parent_nodes_leaf(leave, decision_tree)
        leaves_parents[leave] = parents[:]
    return leaves_parents


def get_list_split_phi_forest(random_forest):
    list_split_phi = []
    for i, dtree in enumerate(random_forest.estimators_):
        list_split_phi_tree = get_list_split_phi(dtree)
        list_split_phi += list_split_phi_tree
    return list_split_phi


def get_children_distributions(decisiontree, node_index):
    tree = decisiontree.tree_
    child_l = tree.children_left[node_index]
    child_r = tree.children_right[node_index]
    Q_source_l = tree.value[child_l]
    Q_source_r = tree.value[child_r]
    return [np.asarray(Q_source_l, dtype='double'), np.asarray(Q_source_r, dtype='double')]


def compute_Q_children(X_node, Y_node, phi, threshold, classes):
    # Split parent node target sample using the threshold provided
    # instances <= threshold go to the left
    # instances > threshold go to the right

    decision_l = X_node[:, phi] <= threshold
    decision_r = np.logical_not(decision_l)
    Y_child_l = Y_node[decision_l]
    Y_child_r = Y_node[decision_r]

    Q_l = compute_class_distribution(classes, Y_child_l)
    Q_r = compute_class_distribution(classes, Y_child_r)

    return Q_l, Q_r


def get_node_distribution(decisiontree, node_index):
    tree = decisiontree.tree_
    Q = tree.value[node_index]
    return np.asarray(Q, dtype='double')


def compute_class_distribution(classes, class_membership):

    unique, counts = np.unique(class_membership, return_counts=True)
    classes_counts = dict(zip(unique, counts))
    classes_index = dict(zip(classes, range(len(classes))))
    distribution = np.zeros(len(classes))

    for label, count in classes_counts.items():
        class_index = classes_index[label]
        distribution[class_index] = count
    return distribution


def KL_divergence(class_counts_P, class_counts_Q):
    # KL Divergence to assess the difference between two distributions
    # Definition: $D_{KL}(P||Q) = \sum{i} P(i)ln(\frac{P(i)}{Q(i)})$
    # epsilon to avoid division by 0

    epsilon = 1e-8

    class_counts_P += epsilon
    class_counts_Q += epsilon

    P = class_counts_P * 1./class_counts_P.sum()
    Q = class_counts_Q * 1./class_counts_Q.sum()

    Dkl = (P * np.log(P * 1./Q)).sum()

    return Dkl


def H(class_counts):
    # Entropy
    # Definition: $H(P) = \sum{i} -P(i) ln(P(i))$
    epsilon = 1e-8
    class_counts += epsilon
    P = class_counts * 1./class_counts.sum()
    return - (P * np.log(P)).sum()


def IG(class_counts_parent, class_counts_children):
    # Information Gain

    H_parent = H(class_counts_parent)
    H_children = np.asarray([H(class_counts_child)
                             for class_counts_child in class_counts_children])

    N = class_counts_parent.sum()
    p_children = np.asarray([class_counts_child.sum()*1./N for class_counts_child in class_counts_children])
    information_gain = H_parent - (p_children * H_children).sum()

    return information_gain


def JSD(P, Q):
    M = (P+Q) * 1./2
    Dkl_PM = KL_divergence(P, M)
    Dkl_QM = KL_divergence(Q, M)
    return (Dkl_PM + Dkl_QM) * 1./2


def DG(Q_source_l, Q_source_r, Q_target_l, Q_target_r):
    # compute proportion of instances at left and right

    p_l = Q_target_l.sum()
    p_r = Q_target_r.sum()

    total_counts = p_l + p_r

    p_l /= total_counts
    p_r /= total_counts

    # compute the DG
    return 1. - p_l * JSD(Q_target_l, Q_source_l) - p_r * JSD(Q_target_r, Q_source_r)


def prune_subtree(decisiontree, node_index):

    tree = decisiontree.tree_
    if tree.children_left[node_index] != -1:
        prune_subtree(decisiontree, tree.children_left[node_index])
        tree.children_left[node_index] = -1

    if tree.children_right[node_index] != -1:
        prune_subtree(decisiontree, tree.children_right[node_index])
        tree.children_right[node_index] = -1


def GINI(class_distribution):
    if class_distribution.sum():
        p = class_distribution / class_distribution.sum()
        return 1 - (p**2).sum()
    return 0


def leaves_id(tree):
    return np.asarray(range(tree.children_right.size))[tree.children_left == tree.children_right]


def find_parent(tree, i_node):
    p = -1
    b = 0
    dic = tree.__getstate__()
    if i_node != 0 and i_node != -1:
        if i_node in dic['nodes']['left_child']:
            p = list(dic['nodes']['left_child']).index(i_node)
            b = -1
        elif i_node in dic['nodes']['right_child']:
            p = list(dic['nodes']['right_child']).index(i_node)
            b = 1
    return p, b


def print_decision_path(tree, X, sample_id=0):
    node_indicator = tree.decision_path(X)
    # leave_id = tree.apply(X)
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
    print('Rules used to predict sample %s: ' % sample_id)
    print(node_index)

    for node_id in node_index:
        if (X[sample_id, tree.tree_.feature[node_id]] <= tree.tree_.threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
              % (node_id, sample_id, tree.tree_.feature[node_id],
                 X[sample_id, tree.tree_.feature[node_id]],
                 threshold_sign, tree.tree_.threshold[node_id]))


def sameStructure(dt_a, dt_b):

    def add_children(node, children_add, children_other, feature, threshold, value, n_nodes):
        # print(children_add.shape)

        ch_a = np.copy(children_add)
        ch_o = np.copy(children_other)
        f = np.copy(feature)
        t = np.copy(threshold)
        v = np.copy(value)
        nn = n_nodes

        ch_a[node] = nn
        ch_a = np.append(ch_a, -1)
        f = np.append(f, -2)
        t = np.append(t, -2)
        ch_o = np.append(ch_o, -1)
        x = np.zeros((1, v.shape[1], v.shape[2]))
        v = np.vstack((v, x))

        # print(children_add.shape)

        return ch_a, ch_o, f, t, v, nn+1

    n_nodes_a = np.copy(dt_a.tree_.node_count)
    children_left_a = np.copy(dt_a.tree_.children_left)
    children_right_a = np.copy(dt_a.tree_.children_right)
    feature_a = np.copy(dt_a.tree_.feature)
    threshold_a = np.copy(dt_a.tree_.threshold)
    value_a = np.copy(dt_a.tree_.value)

    stack_a = [0]

    n_nodes_b = np.copy(dt_b.tree_.node_count)
    children_left_b = np.copy(dt_b.tree_.children_left)
    children_right_b = np.copy(dt_b.tree_.children_right)
    feature_b = np.copy(dt_b.tree_.feature)
    threshold_b = np.copy(dt_b.tree_.threshold)
    value_b = np.copy(dt_b.tree_.value)

    stack_b = [0]

    # print(n_nodes_a, n_nodes_b)

    while len(stack_a) > 0 and len(stack_b) > 0:
        node_a = stack_a.pop()
        node_b = stack_b.pop()

        if children_left_a[node_a] == -1 and children_left_b[node_b] != -1:
            children_left_a, children_right_a, feature_a, threshold_a, value_a,\
                n_nodes_a = add_children(node_a, children_left_a, children_right_a,
                                         feature_a, threshold_a, value_a, n_nodes_a)
            stack_a.append(children_left_a[node_a])
            stack_b.append(children_left_b[node_b])

        elif children_left_b[node_b] == -1 and children_left_a[node_a] != -1:
            children_left_b, children_right_b, feature_b, threshold_b, value_b, \
                n_nodes_b = add_children(node_b, children_left_b, children_right_b,
                                         feature_b, threshold_b, value_b, n_nodes_b)
            stack_b.append(children_left_b[node_b])
            stack_a.append(children_left_a[node_a])

        elif children_left_a[node_a] != -1 and children_left_b[node_b] != -1:
            stack_a.append(children_left_a[node_a])
            stack_b.append(children_left_b[node_b])

        if children_right_a[node_a] == -1 and children_right_b[node_b] != -1:
            children_right_a, children_left_a, feature_a, threshold_a, value_a,\
                n_nodes_a = add_children(node_a, children_right_a, children_left_a,
                                         feature_a, threshold_a, value_a, n_nodes_a)
            stack_a.append(children_right_a[node_a])
            stack_b.append(children_right_b[node_b])

        elif children_right_b[node_b] == -1 and children_right_a[node_a] != -1:
            children_right_b, children_left_b, feature_b, threshold_b, value_b,\
                n_nodes_b = add_children(node_b, children_right_b, children_left_b,
                                         feature_b, threshold_b, value_b, n_nodes_b)
            stack_b.append(children_right_b[node_b])
            stack_a.append(children_right_a[node_a])

        elif children_right_a[node_a] != -1 and children_right_b[node_b] != -1:
            stack_a.append(children_right_a[node_a])
            stack_b.append(children_right_b[node_b])

    dict_dt_a = {"n_nodes": n_nodes_a,
                 "children_left": children_left_a,
                 "children_right": children_right_a,
                 "feature": feature_a,
                 "threshold": threshold_a,
                 "value": value_a
                 }

    dict_dt_b = {"n_nodes": n_nodes_b,
                 "children_left": children_left_b,
                 "children_right": children_right_b,
                 "feature": feature_b,
                 "threshold": threshold_b,
                 "value": value_b
                 }

    return dict_dt_a, dict_dt_b
