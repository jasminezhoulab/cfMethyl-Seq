#
# Python 3.x
#
import sys, re, gzip, configparser
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from scipy.sparse import lil_matrix, csc_matrix
import pandas as pd
from datetime import datetime
from itertools import combinations
import collections
from collections import Counter
from operator import itemgetter

def read_configuration_file(file):
    config = configparser.ConfigParser()
    config.read(file)
    config_ = {}

    # Section 1
    config_['number_data_views'] = int(config.get('number_data_views', 'number_data_views'))
    # Section 2
    config_['class_name_of_controls'] = config.get('class_name_of_controls', 'class_name')
    # Section 3
    options_ = config.options('filename_of_data_views_for_training')
    config_['filename_class_labels_for_training'] = config.get('filename_of_data_views_for_training', 'class_labels')
    config_['filename_of_data_views_for_training'] = ['']*config_['number_data_views']
    for i in range(config_['number_data_views']):
        view_id = 'view%d'%(i+1)
        if view_id in options_:
            index_ = options_.index(view_id)
            config_['filename_of_data_views_for_training'][i] = config.get('filename_of_data_views_for_training', options_[index_])
    # Section 4
    options_ = config.options('filename_of_data_views_for_testing')
    config_['filename_class_labels_for_testing'] = config.get('filename_of_data_views_for_testing', 'class_labels')
    config_['filename_of_data_views_for_testing'] = [''] * config_['number_data_views']
    for i in range(config_['number_data_views']):
        view_id = 'view%d' % (i + 1)
        if view_id in options_:
            index_ = options_.index(view_id)
            config_['filename_of_data_views_for_testing'][i] = config.get('filename_of_data_views_for_testing', options_[index_])
    # Section 5
    options_ = config.options('level1_classifiers_of_data_views_for_cancer_detection')
    config_['level1_classifiers_of_data_views_for_cancer_detection'] = [''] * config_['number_data_views']
    for i in range(config_['number_data_views']):
        view_id = 'view%d' % (i + 1)
        if view_id in options_:
            index_ = options_.index(view_id)
            config_['level1_classifiers_of_data_views_for_cancer_detection'][i] = config.get(
                'level1_classifiers_of_data_views_for_cancer_detection',
                options_[index_])
    # Section 6
    options_ = config.options('level1_classifiers_of_data_views_for_TOO')
    config_['level1_classifiers_of_data_views_for_TOO'] = [''] * config_['number_data_views']
    for i in range(config_['number_data_views']):
        view_id = 'view%d' % (i + 1)
        if view_id in options_:
            index_ = options_.index(view_id)
            config_['level1_classifiers_of_data_views_for_TOO'][i] = config.get(
                'level1_classifiers_of_data_views_for_TOO',
                options_[index_])
    # Section 7
    config_['level1_num_folds_of_training_data_for_OOFP'] = int(config.get('level1_num_folds_of_training_data_for_OOFP', 'num_folds'))
    # Section 8
    config_['level2_classifier_for_cancer_detection'] = config.get(
        'level2_classifier_for_cancer_detection', 'classifier')
    # Section 9
    config_['level2_classifier_for_TOO'] = config.get(
        'level2_classifier_for_TOO', 'classifier')
    # Section 10
    config_['file_of_cancer_detection_prediction'] = config.get('output_prediction_scores', 'file_of_cancer_detection_prediction')
    config_['file_of_TOO_prediction'] = config.get('output_prediction_scores', 'file_of_TOO_prediction')
    return(config_)

# In returned 'labels_with_indexes', label 0 is always for control class
def read_class_labels_file_of_samples_for_multiclass(file, control_classname='class1'):
    labels_with_classnames = []
    with open(file) as f:
        for line in f:
            labels_with_classnames.append(line.rstrip())
    unique_classnames = sorted(list(set(labels_with_classnames)))
    if unique_classnames.index(control_classname)!=0:
        # To make unique_classnames[0] be control_classname:
        # we switch classnames of two indexes: [0] and [old_control_index]
        old_control_index = unique_classnames.index(control_classname)
        unique_classnames[old_control_index] = unique_classnames[0]
        unique_classnames[0] = control_classname
    labels_with_indexes = [unique_classnames.index(l) for l in labels_with_classnames] # 0, 1, 2, ...
    return ((np.array(labels_with_indexes), np.array(labels_with_classnames),unique_classnames))

# In returned 'labels_with_indexes', label 0 is always for control class; label 1 for all other classes
def read_class_labels_file_of_samples_for_binary_class(file, control_classname='class1'):
    labels_with_indexes, labels_with_classnames, unique_classnames = read_class_labels_file_of_samples_for_multiclass(file, control_classname)
    if len(unique_classnames)>2: # more than 2 classes
        unique_classnames_for_binary = [unique_classnames[0], '+'.join(unique_classnames[1:])]
        labels_with_indexes_for_binary = [1 if l>0 else 0 for l in labels_with_indexes]
        labels_with_classnames_for_binary = [unique_classnames_for_binary[1] if l!=control_classname else l for l in labels_with_classnames]
    else:
        unique_classnames_for_binary = unique_classnames
        labels_with_indexes_for_binary = labels_with_indexes
        labels_with_classnames_for_binary = labels_with_classnames
    return ((np.array(labels_with_indexes_for_binary), np.array(labels_with_classnames_for_binary),unique_classnames_for_binary))

# return 0-based indexes of those samples whose class labels == query_class. labels could be strings or integers
def find_samples_of_a_specific_classname(sample_class_labels_list, query_class):
    indexes = np.array(sample_class_labels_list) == query_class
    return(np.where(indexes)[0])

def read_input_data_view_file(file):
    data = []
    with open(file) as f:
        for line in f:
            items = line.rstrip().split('\t')
            data.append(np.array(list(map(float, items))))
    return (np.array(data))

# 'indexes_of_samples_to_be_removed' are 0-based 1d numpy array
def remove_data_of_selected_samples(X, indexes_of_samples_to_be_removed, y=None):
    num_samples, num_features = X.shape
    indexes_of_samples_to_be_kept = sorted(np.setdiff1d(np.array(range(num_samples)), indexes_of_samples_to_be_removed))
    if y is None:
        ret = X[indexes_of_samples_to_be_kept,:]
    else:
        ret = (X[indexes_of_samples_to_be_kept,:], y[indexes_of_samples_to_be_kept], indexes_of_samples_to_be_kept)
    return(ret)

def extract_first_number_from_string(str):
    return (re.search(r'\d+[\.\d]*', str)[0])

def classify_multiclass(method, train_x, train_y, test_x, verbose=0):
    nsample, nfeature = train_x.shape
    items = method.split('_')
    if 'ovr_LinearSVC_' in method:
        # 'ovr_LinearSVC_l2_c1'
        penalty = items[2]
        if penalty == 'l1':
            # dual=True is not implemented for penalty='l1' yet.
            dual = False
        else:
            dual = True
        c = float(items[3][1:])
        if verbose>0:
            print('  OvR linear SVM (penalty=%s, c=%g)'%(penalty, c))
        model = LinearSVC(penalty=penalty, C=c, dual=dual, multi_class='ovr', class_weight='balanced', max_iter=20000, random_state=0, verbose=0)

    elif method.startswith('ovr_RF'):
        # 'ovr_RF_ntree1000_mtry1.5'
        ntree = int(extract_first_number_from_string(items[2]))
        mtry = int(float(extract_first_number_from_string(items[3])) * int(np.sqrt(nfeature)))
        if mtry > nfeature:
            mtry = nfeature
        elif mtry <= 0:
            mtry = 1
        if verbose > 0:
            print('  OvR random forest: ntree=%d, mtry=%g, max_features=%d' % (ntree, float(extract_first_number_from_string(items[3])), mtry))
        model = OneVsRestClassifier(
            RandomForestClassifier(n_estimators=ntree, max_features=mtry, class_weight='balanced', oob_score=True,
                                   n_jobs=1, verbose=0))

    fit = model.fit(train_x, train_y)
    if 'SVC' in method:
        # linearSVC do not support predict_proba. Instead, decision_function provides the signed distance of that sample to the hyperplane.
        pred = model.decision_function(test_x)
    else:
        pred = model.predict_proba(test_x)
    return (pred)


def classify_binaryclass(method, train_x, train_y, test_x):
    nsample, nfeature = train_x.shape
    items = method.split('_')
    if 'LinearSVC_' in method:
        # 'LinearSVC_l2_c1'
        penalty = items[1]
        if penalty == 'l1':
            # dual=True is not implemented for penalty='l1' yet.
            dual = False
        else:
            dual = True
        c = float(items[2][1:])
        model = LinearSVC(penalty=penalty, C=c, dual=dual, class_weight='balanced', max_iter=10000, random_state=0,
                          verbose=0)

    elif method.startswith('RF'):
        # 'RF_ntree1000_mtry0.5'
        ntree = int(extract_first_number_from_string(items[1]))
        mtry = int(float(extract_first_number_from_string(items[2])) * int(np.sqrt(nfeature)))
        if mtry > nfeature:
            mtry = nfeature
        elif mtry <= 0:
            mtry = 1
        print('ntree=%d, mtry=%g, max_features=%d' % (ntree, float(extract_first_number_from_string(items[2])), mtry))
        model = RandomForestClassifier(n_estimators=ntree, max_features=mtry, class_weight='balanced', oob_score=True, n_jobs=1, verbose=0)

    fit = model.fit(train_x, train_y)

    if 'SVC' in method:
        # linearSVC do not support predict_proba. Instead, decision_function provides the signed distance of that sample to the hyperplane.
        pred = model.decision_function(test_x)
    else:
        pred = model.predict_proba(test_x)[:,1]  # output only prediction probability for cancer class. But the output is still an array which has only one element.
    return (np.vstack(pred)) # 'pred' is a 1d array. 'np.vstack(pred)' is a 2d array with size N X 1 (N rows and 1 column)

# Take example: original_rows_indexes = [2, 0, 1] indicates
# row 0 of 'mat' is moved to row 2 of 'mat_new'
# row 1 of 'mat' is moved to row 0 of 'mat_new'
# row 2 of 'mat' is moved to row 1 of 'mat_new'
def reorder_rows_of_matrix(mat, original_rows_indexes):
    nrow, ncol = mat.shape
    mat_new = np.empty((nrow, ncol,))
    mat_new[original_rows_indexes,:] = mat
    return(mat_new)

# 'samples_indexes' are 0-based
def write_preds_of_multiclass_with_sampleindex_and_classnames(output_file, preds_mat, samples_indexes, classnames):
    num_samples, num_classes = preds_mat.shape
    classnames_with_prefix = ['prediction_score_of_'+c for c in classnames]
    with open(output_file, 'w') as fout:
        fout.write('sample_index(1-based)\tpredicted_classname\t%s\n'%('\t'.join(classnames_with_prefix)))
        for i in range(num_samples):
            i_max = np.argmax(preds_mat[i,:])
            fout.write('%d\t%s'%(samples_indexes[i]+1, classnames[i_max]))
            for j in range(num_classes):
                fout.write('\t%.5f'%preds_mat[i,j])
            fout.write('\n')

# 'samples_indexes' are 0-based. classnames: [0] for control class, [1] for cancer class. preds is a 1d numpy array.
def write_preds_of_binaryclass_with_sampleindex_and_classnames(output_file, preds, classnames):
    num_samples = len(preds)
    with open(output_file, 'w') as fout:
        fout.write('sample_index(1-based)\tprediction_score(higher_scores_indicate_higher_cancer_risk, %s(control) vs %s(cancer))\n'%(classnames[0],classnames[1]))
        for i in range(num_samples):
            fout.write('%d\t%.5f\n'%(i+1, preds[i]))


#############################
## Functions of Identifying Markers
#############################

# Input:
# alpha_value_distribution file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read    unique_alpha_values read_freq_of_unique_alpha_values    unique_meth_counts  read_freq_of_unique_meth_counts
# 2   7   122 0.429,0.571,0.714,0.857,1   1,2,27,42,50    3,4,5,6,7   1,2,27,42,50
# 27  9   39  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13 4,5,6,7,8   1,2,9,13,14
# 61  12  44  0.75,0.833,0.917,1  2,11,12,19  9,10,11,12  2,11,13,18
# 63  5   100 0.6,0.8,1   4,23,73 3,4,5   4,23,73
# 65  5   83  0,0.2,0.4,0.6,0.8,1 1,3,2,9,26,42   0,1,2,3,4,5 1,3,2,9,26,42
# ...
#
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':read_frequency}, e.g., {'0.7':200, '1.0':41}, where 200 means there are 200 reads with alpha==0.7 and 41 reads with alpha==1.0 among the pooled reads in all samples loaded. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}.
#
def load_one_alpha_value_distribution_file(file, alpha_hists, marker2max_cpg_num):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = int(items[0])
            max_num_cpg = items[1]
            unique_alpha_values_str = items[3]
            read_freq_of_unique_alpha_values_str = items[4]
            unique_alpha_values = unique_alpha_values_str.split(',')
            read_freq = list(map(int, read_freq_of_unique_alpha_values_str.split(',')))
            n = len(unique_alpha_values)
            alpha_hist_dict = {unique_alpha_values[i]: read_freq[i] for i in range(n)}
            if marker_index not in alpha_hists:
                alpha_hists[marker_index] = {}
            # update histgrams
            alpha_hists[marker_index] = mergeDict_by_adding_values_of_common_keys(alpha_hists[marker_index], alpha_hist_dict)
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg

# Input file format is the same as input file of function 'load_one_alpha_value_distribution_file'
# Output:
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':sample_frequency}, e.g., {'0.7':2, '1.0':1}, where the frequency 2 and 1 means there are 2 samples which have reads with alpha==0.7 and 1 sample which have reads with alpha==1.0. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}.
#
def load_one_alpha_value_distribution_file_by_making_read_freq_as_ONE_for_one_sample(file, alpha_hists, marker2max_cpg_num):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = int(items[0])
            max_num_cpg = items[1]
            unique_alpha_values_str = items[3]
            # read_freq_of_unique_alpha_values_str = items[4]
            unique_alpha_values = unique_alpha_values_str.split(',')
            # read_freq = list(map(int, read_freq_of_unique_alpha_values_str.split(',')))
            n = len(unique_alpha_values)
            alpha_hist_dict = {unique_alpha_values[i]: 1 for i in range(n)}
            if marker_index not in alpha_hists:
                alpha_hists[marker_index] = {}
            # update histgrams
            alpha_hists[marker_index] = mergeDict_by_adding_values_of_common_keys(alpha_hists[marker_index], alpha_hist_dict)
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg


# Input:
#   file: format is the same as input file of function 'load_one_alpha_value_distribution_file'
#   sample_index: an integer sample_index of the input file. It will be put to the list of a unique alpha, like {alpha_value:[list of sample_index which have the reads with the same alpha_value]}. To save space, we use 1D numpy.array to replace
#
# Output:
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':sample_frequency}, e.g., {'0.7':2, '1.0':1}, where the frequency 2 and 1 means there are 2 samples which have reads with alpha==0.7 and 1 sample which have reads with alpha==1.0. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}. The input "marker2max_cpg_num" could be an empty dictionary {}.
#
def load_one_alpha_value_distribution_file_by_recording_sample_index_of_this_file(file, sample_index, alpha_hists, marker2max_cpg_num):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = int(items[0])
            max_num_cpg = items[1]
            unique_alpha_values_str = items[3]
            # read_freq_of_unique_alpha_values_str = items[4]
            unique_alpha_values = unique_alpha_values_str.split(',')
            # read_freq = list(map(int, read_freq_of_unique_alpha_values_str.split(',')))
            n = len(unique_alpha_values)
            alpha_hist_dict = {unique_alpha_values[i]: {sample_index} for i in range(n)}
            if marker_index not in alpha_hists:
                alpha_hists[marker_index] = {}
            # update histgrams
            alpha_hists[marker_index] = mergeDict_by_union_valuesets_of_common_keys(alpha_hists[marker_index], alpha_hist_dict)
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg

# Input:
#   file: format is the same as input file of function 'load_one_alpha_value_distribution_file'
#   sample_index: an integer sample_index of the input file. It will be put to the list of a unique alpha, like {alpha_value:[list of sample_index which have the reads with the same alpha_value]}. To save space, we use 1D numpy.array to replace
#
# Output:
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':sample_frequency}, e.g., {'0.7':2, '1.0':1}, where the frequency 2 and 1 means there are 2 samples which have reads with alpha==0.7 and 1 sample which have reads with alpha==1.0. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}. The input "marker2max_cpg_num" could be an empty dictionary {}.
#
def load_one_alpha_value_distribution_file_of_positive_sample_by_recording_sample_index_of_this_file_and_by_excluding_sample_index_which_appear_in_paired_negative_sample(file, sample_index, alpha_hists_positive, alpha_hists_negative, marker2max_cpg_num):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = int(items[0])
            max_num_cpg = items[1]
            unique_alpha_values_str = items[3]
            # read_freq_of_unique_alpha_values_str = items[4]
            unique_alpha_values = unique_alpha_values_str.split(',')
            # read_freq = list(map(int, read_freq_of_unique_alpha_values_str.split(',')))
            n = len(unique_alpha_values)
            if marker_index not in alpha_hists_negative:
                # negative sample does not have this marker, so all unique alpha values can add this sample_index to their sample sets
                # We add this sample_index to the sample_set of each  unique alpha value
                alpha_hist_dict = {unique_alpha_values[i]: {sample_index} for i in range(n)}
            else:
                # negative sample has this marker, we further check each unique alpha value one by one
                # if alpha_value appears in
                alpha_hist_dict = {}
                marker_of_negative = alpha_hists_negative[marker_index]
                for i in range(n):
                    a = unique_alpha_values[i]
                    if a not in marker_of_negative:
                        alpha_hist_dict[a] = {sample_index}
            # update histgrams
            if len(alpha_hist_dict) > 0:
                if marker_index not in alpha_hists_positive:
                    alpha_hists_positive[marker_index] = {}
                alpha_hists_positive[marker_index] = mergeDict_by_union_valuesets_of_common_keys(alpha_hists_positive[marker_index], alpha_hist_dict)
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg



# Input:
#   file: format is the same as the output file of function 'write_alpha_value_distribution_file_with_recorded_sample_index_set'
# marker_index    max_num_cpg num_read  unique_alpha_values read_freq_of_unique_alpha_values
# 3 8   3   0.875,1 14,3_21
# 9 13  6   0.385,0.462,0.538,0.615,0.692,0.769,0.833,0.846,0.923,1 18,8_14_18,8_11_13_14_18,8_11_14_18,8_11_14_18,8_13_14_18_24,8,8_11_13_14_18,8_11_13_14_18,8_11_13_14_18
# 15    3   1   0.667   1
# 20    4   4   0.75,1  18,12_21_23
# 25    7   1   0.857   9
# ...
#   sample_index: an integer sample_index of the input file. It will be put to the list of a unique alpha, like {alpha_value:[list of sample_index which have the reads with the same alpha_value]}. To save space, we use 1D numpy.array to replace
#
# Output:
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':sample_frequency}, e.g., {'0.7':2, '1.0':1}, where the frequency 2 and 1 means there are 2 samples which have reads with alpha==0.7 and 1 sample which have reads with alpha==1.0. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}. The input "marker2max_cpg_num" could be an empty dictionary {}.
#
def load_one_alpha_value_distribution_file_that_has_sample_index_sets(file, alpha_hists, marker2max_cpg_num):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = int(items[0])
            max_num_cpg = items[1]
            unique_alpha_values_str = items[3]
            sample_index_sets_of_unique_alpha_values_str_list = items[4].split(',')
            unique_alpha_values = unique_alpha_values_str.split(',')
            n = len(unique_alpha_values)
            alpha_hist_dict = {unique_alpha_values[i]: set(sample_index_sets_of_unique_alpha_values_str_list[i].split('_')) for i in range(n)}
            if marker_index not in alpha_hists:
                alpha_hists[marker_index] = {}
            # update histgrams
            alpha_hists[marker_index] = mergeDict_by_union_valuesets_of_common_keys(alpha_hists[marker_index], alpha_hist_dict)
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg

# Input:
#   file: format is the same as the output file of function 'write_alpha_value_distribution_file_with_recorded_sample_index_set'
# marker_index    max_num_cpg num_read  unique_alpha_values read_freq_of_unique_alpha_values
# 29    8   5   0.75,0.875,1    5:0.125,10:0.333_25:0.4_5:0.375_6:0.333_8:0.571,10:0.667_25:0.6_5:0.5_6:0.667_8:0.429
# 223   8   4   0,0.125,0.75,0.875,1    14:0.1_8:0.2,8:0.1,14:0.2_25:0.2_3:0.0909,14:0.1_25:0.2_3:0.273_8:0.2,14:0.6_25:0.6_3:0.636_8:0.5
# ...
#   sample_index: an integer sample_index of the input file. It will be put to the list of a unique alpha, like {alpha_value:[list of sample_index which have the reads with the same alpha_value]}. To save space, we use 1D numpy.array to replace
#
# Output:
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':sample_frequency}, e.g., {'0.7':2, '1.0':1}, where the frequency 2 and 1 means there are 2 samples which have reads with alpha==0.7 and 1 sample which have reads with alpha==1.0. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}. The input "marker2max_cpg_num" could be an empty dictionary {}.
#
def load_one_alpha_value_distribution_file_that_has_sample_index_sets_and_read_fractions(file, alpha_hists, marker2max_cpg_num, marker2samplenum):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            marker_index = int(items[0])
            max_num_cpg = items[1]
            sample_num = int(items[2]) # the number of unique samples that appear in 'sample_index_sets_of_unique_alpha_values_str_list'
            unique_alpha_values_str = items[3]
            sample_index_sets_of_unique_alpha_values_str_list = items[4].split(',')
            unique_alpha_values = unique_alpha_values_str.split(',')
            n = len(unique_alpha_values)
            # alpha_hist_dict = {unique_alpha_values[i]: set(sample_index_sets_of_unique_alpha_values_str_list[i].split('_')) for i in range(n)}
            alpha_hist_dict = dict([])
            for i in range(n):
                alpha_hist_dict[unique_alpha_values[i]] = dict([])
                sampleindex_and_readfraction_str_list = sample_index_sets_of_unique_alpha_values_str_list[i].split('_')
                for sampleindex_and_readfraction_str in sampleindex_and_readfraction_str_list:
                    sampleindex, readfraction = sampleindex_and_readfraction_str.split(':')
                    readfraction = float(readfraction)
                    alpha_hist_dict[unique_alpha_values[i]][sampleindex] = readfraction
            if marker_index not in alpha_hists:
                alpha_hists[marker_index] = {}
            # update histgrams
            for alpha in alpha_hist_dict:
                if alpha not in alpha_hists[marker_index]:
                    alpha_hists[marker_index][alpha] = {}
                alpha_hists[marker_index][alpha] = mergeDict_by_adding_values_of_common_keys(alpha_hists[marker_index][alpha], alpha_hist_dict[alpha])
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg
            if marker_index not in marker2samplenum:
                marker2samplenum[marker_index] = sample_num

# Input:
#   file: format is the same as input file of function 'load_one_alpha_value_distribution_file'
#   sample_index: an integer sample_index of the input file. It will be put to the list of a unique alpha, like {alpha_value:[list of sample_index which have the reads with the same alpha_value]}. To save space, we use 1D numpy.array to replace
#
# Output:
# alpha_hists: a dictionary { marker_index:alpha_histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'alpha_value':sample_frequency}, e.g., {'0.7':2, '1.0':1}, where the frequency 2 and 1 means there are 2 samples which have reads with alpha==0.7 and 1 sample which have reads with alpha==1.0. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:'max_cpg_num'}. The input "marker2max_cpg_num" could be an empty dictionary {}.
#
def load_one_alpha_value_distribution_file_by_recording_sample_index_and_read_fraction_of_this_file(file, sample_index, min_read_coverage, alpha_hists, marker2max_cpg_num):
    with gzip.open(file, 'rt') as f:
        next(f) # skip the first header line
        for line in f:
            items = line.rstrip().split()
            total_read_count = float(items[2])
            if total_read_count < min_read_coverage:
                continue
            marker_index = int(items[0])
            max_num_cpg = items[1]
            unique_alpha_values_str = items[3]
            read_freq_of_unique_alpha_values_str = items[4]
            unique_alpha_values = unique_alpha_values_str.split(',')
            read_freq = list(map(int, read_freq_of_unique_alpha_values_str.split(',')))
            n = len(unique_alpha_values)
            alpha_hist_dict = {unique_alpha_values[i]:{'%d:%.3g'%(sample_index,read_freq[i]/total_read_count)} for i in range(n)}
            if marker_index not in alpha_hists:
                alpha_hists[marker_index] = {}
            # update histgrams
            alpha_hists[marker_index] = mergeDict_by_union_valuesets_of_common_keys(alpha_hists[marker_index], alpha_hist_dict)
            if marker_index not in marker2max_cpg_num:
                marker2max_cpg_num[marker_index] = max_num_cpg



# Input:
#   alpha2freq: a dictionary {alpha: read_freq}. For example, {'0.2':20}
# Output:
#   meth_histgram: a dictionary structure {alpha:read_freq}. For example, {0.2:10}.
def filter_alpha2freq_by_alpha(alpha2freq, alpha_cutoff, direction_to_keep_alpha2freq='>='):
    ret_alpha2freq = {}
    if direction_to_keep_alpha2freq=='>=':
        # keep all alpha2freq and their meth_strings if their alpha>=alpha_cutoff
        for alpha, freq in alpha2freq.items():
            if float(alpha)>=alpha_cutoff:
                ret_alpha2freq[alpha] = freq
    elif direction_to_keep_alpha2freq=='>':
        # keep all alpha2freq and their meth_strings if their alpha>alpha_cutoff
        for alpha, freq in alpha2freq.items():
            if float(alpha) > alpha_cutoff:
                ret_alpha2freq[alpha] = freq
    elif direction_to_keep_alpha2freq=='<=':
        # keep all alpha2freq and their meth_strings if their alpha<=alpha_cutoff
        for alpha, freq in alpha2freq.items():
            if float(alpha) <= alpha_cutoff:
                ret_alpha2freq[alpha] = freq
    elif direction_to_keep_alpha2freq=='<':
        # keep all alpha2freq and their meth_strings if their alpha<alpha_cutoff
        for alpha, freq in alpha2freq.items():
            if float(alpha) < alpha_cutoff:
                ret_alpha2freq[alpha] = freq
    return(ret_alpha2freq)

# Input:
#   alpha_list: a list of alpha_values with a specific order (in descending or increasing order)
#   freq_cumsum: a 1D numpy.array with the same length and the same order of alpha_list, and freq_cumsum[i] is the accumulated frequency sum of the alpha_value alpha_list[i]
# Output:
#   ret_alpha2freq: a dictionary {alpha_value:freq_cum_sum}
def filter_by_freq_cumsum_and_create_alpha2freqcumsum(alpha_cutoff, alpha_list, freq_cumsum, direction_to_keep_alpha2freq='>='):
    ret_alpha2freq = {}
    n = len(alpha_list)
    if direction_to_keep_alpha2freq == '>=':
        # keep all alpha2freq and their meth_strings if their alpha>=alpha_cutoff
        for i in range(n):
            alpha = alpha_list[i]
            if float(alpha) >= alpha_cutoff:
                ret_alpha2freq[alpha] = freq_cumsum[i]
    elif direction_to_keep_alpha2freq == '>':
        # keep all alpha2freq and their meth_strings if their alpha>alpha_cutoff
        for i in range(n):
            alpha = alpha_list[i]
            if float(alpha) > alpha_cutoff:
                ret_alpha2freq[alpha] = freq_cumsum[i]
    elif direction_to_keep_alpha2freq == '<=':
        # keep all alpha2freq and their meth_strings if their alpha<=alpha_cutoff
        for i in range(n):
            alpha = alpha_list[i]
            if float(alpha) <= alpha_cutoff:
                ret_alpha2freq[alpha] = freq_cumsum[i]
    elif direction_to_keep_alpha2freq == '<':
        # keep all alpha2freq and their meth_strings if their alpha<alpha_cutoff
        for i in range(n):
            alpha = alpha_list[i]
            if float(alpha) < alpha_cutoff:
                ret_alpha2freq[alpha] = freq_cumsum[i]
    return (ret_alpha2freq)

# arr_neg, arr_pos: two 1D numpy.array with the same size. Find the index on which
#     arr_neg_cumsum[index] <= arr_neg_cumsum_threshold
#     arr_pos_cumsum[index] >= arr_pos_cumsum_threshold
# If there exist multiple index satisfying the above criteria, choose the largest index
def identify_turning_point_of_two_cumsum_array(arr_neg, arr_pos, arr_neg_cumsum_threshold, arr_pos_cumsum_threshold):
    arr_pos_cumsum = np.cumsum(arr_pos) - arr_pos_cumsum_threshold
    arr_neg_cumsum = np.cumsum(arr_neg) - arr_neg_cumsum_threshold
    n = len(arr_pos_cumsum)
    index_list = [i for i in range(n) if ((arr_pos_cumsum[i]>=0) and (arr_neg_cumsum[i]<=0))]
    if len(index_list)==0:
        # No turning point
        return(-2)
    else:
        # if (index_list[-1]+1) < (n-1):
        if (index_list[-1]) < (n - 1):
            return(index_list[-1]+1)
        else:
            if (len(index_list)>=2):
                if (index_list[-2]) < (n - 1):
                    return(index_list[-2]+1)
            # -1 means the index should be the one greater than n
            return(-1)

# Input:
#   alpha2freq_of_pos_class: a dictionary {'alpha_value':frequency} for positive class, e.g. {'0.7':2, '0.9':4}
#   alpha2freq_of_neg_class: a dictionary {'alpha_value':frequency} for negative class, e.g. {'0.7':2, '0.9':4}
#   marker_type: 'hyper' marker to identify alpha threshold which make more frequency whose alpha > threshold in positive class, than those in negative class
#                'hypo' marker to identify alpha threshold which make more frequency whose alpha < threshold in positive class, than those in negative class
def identify_alpha_threshold_by_alpha2freq_of_pos_and_neg_class(alpha2freq_of_neg_class,
                                                                alpha2freq_of_pos_class,
                                                                max_freq_cumsum_of_neg,
                                                                min_freq_cumsum_of_pos,
                                                                marker_type='hyper'):
    if 'hyper' in marker_type:
        if len(alpha2freq_of_pos_class)>0 and len(alpha2freq_of_neg_class)==0:
            alpha_threshold = 0 # any reads with any alpha values should be used. So let alpha_threshold==0 for hyper markers
            return (alpha_threshold)
        if len(alpha2freq_of_pos_class)>0 and len(alpha2freq_of_neg_class)>0:
            alpha_union_list = sorted(list(set(list(alpha2freq_of_pos_class.keys()) + list(alpha2freq_of_neg_class.keys()))), reverse=True) # decreasing order
            freq_array_of_pos = np.array([alpha2freq_of_pos_class[a] if a in alpha2freq_of_pos_class else 0 for a in alpha_union_list])
            freq_array_of_neg = np.array([alpha2freq_of_neg_class[a] if a in alpha2freq_of_neg_class else 0 for a in alpha_union_list])
    elif 'hypo' in marker_type:
        if len(alpha2freq_of_pos_class)>0 and len(alpha2freq_of_neg_class)==0:
            alpha_threshold = 1 # any reads with any alpha values should be used. So let alpha_threshold==1 for hypo markers
            return (alpha_threshold)
        if len(alpha2freq_of_pos_class)>0 and len(alpha2freq_of_neg_class)>0:
            alpha_union_list = sorted(list(set(list(alpha2freq_of_pos_class.keys()) + list(alpha2freq_of_neg_class.keys()))))  # increasing order
            freq_array_of_pos = np.array([alpha2freq_of_pos_class[a] if a in alpha2freq_of_pos_class else 0 for a in alpha_union_list])
            freq_array_of_neg = np.array([alpha2freq_of_neg_class[a] if a in alpha2freq_of_neg_class else 0 for a in alpha_union_list])
    alpha_index = identify_turning_point_of_two_cumsum_array(freq_array_of_neg, freq_array_of_pos, max_freq_cumsum_of_neg, min_freq_cumsum_of_pos)
    if alpha_index==-2:
        alpha_threshold = None
    else:
        if alpha_index==-1:
            alpha_threshold = None
            # if marker_type=='hyper':
            #     alpha_threshold = 0
            # elif marker_type == 'hypo':
            #     alpha_threshold = 1
        else:
            alpha_threshold = float(alpha_union_list[alpha_index])
            # if len(alpha2freq_of_neg_class) == 1:
            #     # Remove the following case which identifies hyper marker:
            #     # positive class: {'0.857':2, '0.286':1, '0.167':2, '0.143':4, '0':8}
            #     # negative class: {'0.167':2}
            #     # alpha_union_list: ['0.857', '0.286', '0.167', '0.143', '0']
            #     # identify_turning_point_of_two_cumsum_array returns alpha_index=4
            #     # But actually if alpha_threshold=='0', then negative class is incorrect for our purpose.
            #     if (('hyper' in marker_type) and (alpha_threshold<float(alpha2freq_of_neg_class.values[0]))) or (('hypo' in marker_type) and (alpha_threshold>float(alpha2freq_of_neg_class.values[0]))):
            #         alpha_threshold = None
    return(alpha_threshold)


# alpha_union_list: the list of alpha values, which are not only union of alpha values of pos_class and neg_class, but also with the order of alpha values, with respect to 'hyper' (in decreasing order) or 'hypo' (in increasing order).
# alpha2sampleindexset_of_neg_class, alpha2sampleindexset_of_pos_class: two dictionaries {'alpha_value':{sample_index}}
# Algorithm:
#    Step 1: accumulate sample_index_set of each alpha_value, by the alpha_value order of alpha_union_list
#    Step 2: calcualte size of accumulated sample_index_set of each alpha_value
#    Step 3: compute the following to determine the turning point. This step is the same as function 'identify_alpha_threshold_by_alpha2freq_of_pos_and_neg_class'
#            arr_neg_cumsum[index] <= arr_neg_cumsum_threshold
#            arr_pos_cumsum[index] >= arr_pos_cumsum_threshold
#            If there exist multiple index satisfying the above criteria, choose the largest index
#    Step 4: return the index of turning point (i.e., the index of alpha threshold in alpha_union_list)
#
def identify_turning_point_of_two_alpha2sampleindexset(alpha_union_list, alpha2sampleindexset_of_neg_class, alpha2sampleindexset_of_pos_class, arr_neg_cumsum_threshold, arr_pos_cumsum_threshold):
    n_alpha = len(alpha_union_list)
    # implement cumsum using set union operator
    alpha2cumset_neg = {a:set([]) for a in alpha_union_list}
    alpha2cumset_pos = dict(alpha2cumset_neg)
    for i in range(n_alpha):
        a = alpha_union_list[i]
        if i == 0:
            alpha2cumset_neg[a] = alpha2sampleindexset_of_neg_class[a] if a in alpha2sampleindexset_of_neg_class else set([])
            alpha2cumset_pos[a] = alpha2sampleindexset_of_pos_class[a] if a in alpha2sampleindexset_of_pos_class else set([])
        else:
            a_prev = alpha_union_list[i - 1]
            alpha2cumset_neg[a] = alpha2sampleindexset_of_neg_class[a].union(alpha2cumset_neg[a_prev]) if a in alpha2sampleindexset_of_neg_class else alpha2cumset_neg[a_prev]
            alpha2cumset_pos[a] = alpha2sampleindexset_of_pos_class[a].union(alpha2cumset_pos[a_prev]) if a in alpha2sampleindexset_of_pos_class else alpha2cumset_pos[a_prev]
    arr_neg_cumsum = np.array([len(alpha2cumset_neg[a]) for a in alpha_union_list])
    arr_pos_cumsum = np.array([len(alpha2cumset_pos[a]) for a in alpha_union_list])
    index_list = [i for i in range(n_alpha) if ((arr_pos_cumsum[i]>=arr_pos_cumsum_threshold) and (arr_neg_cumsum[i]<=arr_neg_cumsum_threshold))]
    if len(index_list)==0:
        # No turning point
        return( (-2, arr_neg_cumsum, arr_pos_cumsum) )
    else:
        # if (index_list[-1]+1) < (n_alpha-1):
        if (index_list[-1]) < (n_alpha - 1):
            return( (index_list[-1]+1, arr_neg_cumsum, arr_pos_cumsum) )
        else:
            if len(index_list)>=2:
                if (index_list[-2]) < (n_alpha - 1):
                    return( (index_list[-2]+1, arr_neg_cumsum, arr_pos_cumsum) )
            # -1 means the index should be the one greater than n
            return( (-1, arr_neg_cumsum, arr_pos_cumsum) )


# Input:
#   alpha2freq_of_pos_class: a dictionary {'alpha_value':set(sample_index)} for positive class, e.g. {'0.7':[1,2], '0.9':[2,4]}
#   alpha2freq_of_neg_class: a dictionary {'alpha_value':set(sample_index)} for negative class, e.g. {'0.7':[1,2], '0.9':[2,4]}
#   marker_type: 'hyper' marker to identify alpha threshold which make more sample frequency whose alpha > threshold in positive class, than those in negative class
#                'hypo' marker to identify alpha threshold which make more sample frequency whose alpha < threshold in positive class, than those in negative class
def identify_alpha_threshold_by_alpha2sampleindexset_of_pos_and_neg_class(alpha2sampleindexset_of_neg_class,
                                                                alpha2sampleindexset_of_pos_class,
                                                                max_freq_cumsum_of_neg,
                                                                min_freq_cumsum_of_pos,
                                                                marker_type='hyper'):
    if 'hyper' in marker_type:
        if len(alpha2sampleindexset_of_pos_class)>0 and len(alpha2sampleindexset_of_neg_class)==0:
            alpha_threshold = 0 # any reads with any alpha values should be used. So let alpha_threshold==0 for hyper markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexset_of_pos_class)>0 and len(alpha2sampleindexset_of_neg_class)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexset_of_pos_class.keys()) + list(alpha2sampleindexset_of_neg_class.keys()))), reverse=True) # decreasing order
    elif 'hypo' in marker_type:
        if len(alpha2sampleindexset_of_pos_class)>0 and len(alpha2sampleindexset_of_neg_class)==0:
            alpha_threshold = 1 # any reads with any alpha values should be used. So let alpha_threshold==1 for hypo markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexset_of_pos_class)>0 and len(alpha2sampleindexset_of_neg_class)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexset_of_pos_class.keys()) + list(alpha2sampleindexset_of_neg_class.keys()))))  # increasing order
    alpha_index, arr_neg_cumsum, arr_pos_cumsum = identify_turning_point_of_two_alpha2sampleindexset(alpha_union_list,
                                                                     alpha2sampleindexset_of_neg_class,
                                                                     alpha2sampleindexset_of_pos_class,
                                                                     max_freq_cumsum_of_neg,
                                                                     min_freq_cumsum_of_pos)
    if alpha_index==-2:
        alpha_threshold = None
    else:
        if alpha_index==-1:
            alpha_threshold = None
            # if marker_type=='hyper':
            #     alpha_threshold = 0
            # elif marker_type == 'hypo':
            #     alpha_threshold = 1
        else:
            alpha_threshold = float(alpha_union_list[alpha_index])
            # if len(alpha2freq_of_neg_class) == 1:
            #     # Remove the following case which identifies hyper marker:
            #     # positive class: {'0.857':2, '0.286':1, '0.167':2, '0.143':4, '0':8}
            #     # negative class: {'0.167':2}
            #     # alpha_union_list: ['0.857', '0.286', '0.167', '0.143', '0']
            #     # identify_turning_point_of_two_cumsum_array returns alpha_index=4
            #     # But actually if alpha_threshold=='0', then negative class is incorrect for our purpose.
            #     if (('hyper' in marker_type) and (alpha_threshold<float(alpha2freq_of_neg_class.values[0]))) or (('hypo' in marker_type) and (alpha_threshold>float(alpha2freq_of_neg_class.values[0]))):
            #         alpha_threshold = None
    return( (alpha_threshold, alpha_union_list, arr_neg_cumsum, arr_pos_cumsum) )



# alpha_union_list: the list of alpha values, which are not only union of alpha values of pos_class and neg_class, but also with the order of alpha values, with respect to 'hyper' (in decreasing order) or 'hypo' (in increasing order).
# alpha2sampleindexsetandreadfraction_of_neg_class, alpha2sampleindexsetandreadfraction_of_pos_class: two dictionaries {'alpha_value':{sample_index:read_fraction}}
# Algorithm:
#    Step 1: accumulate sample_index_set of each alpha_value, by the alpha_value order of alpha_union_list
#    Step 2: calcualte size of accumulated sample_index_set of each alpha_value
#    Step 3: compute the following to determine the turning point. This step is the same as function 'identify_alpha_threshold_by_alpha2freq_of_pos_and_neg_class'
#            arr_neg_cumsum[index] <= arr_neg_cumsum_threshold
#            arr_pos_cumsum[index] >= arr_pos_cumsum_threshold
#            If there exist multiple index satisfying the above criteria, choose the largest index
#    Step 4: return the index of turning point (i.e., the index of alpha threshold in alpha_union_list)
#
def identify_turning_point_of_two_alpha2sampleindexsetandreadfraction(min_read_fraction_neg, min_read_fraction_pos, alpha_union_list, alpha2sampleindexsetandreadfraction_of_neg_class, alpha2sampleindexsetandreadfraction_of_pos_class, arr_neg_cumsum_threshold, arr_pos_cumsum_threshold):
    n_alpha = len(alpha_union_list)
    # implement cumsum using dict merge operator "mergeDict_by_adding_values_of_common_keys"
    alpha2cumdict_neg = {a:dict() for a in alpha_union_list}
    alpha2cumdict_pos = dict(alpha2cumdict_neg)
    for i in range(n_alpha):
        a = alpha_union_list[i]
        if i == 0:
            alpha2cumdict_neg[a] = alpha2sampleindexsetandreadfraction_of_neg_class[a] if a in alpha2sampleindexsetandreadfraction_of_neg_class else dict()
            alpha2cumdict_pos[a] = alpha2sampleindexsetandreadfraction_of_pos_class[a] if a in alpha2sampleindexsetandreadfraction_of_pos_class else dict()
        else:
            a_prev = alpha_union_list[i - 1]
            alpha2cumdict_neg[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_neg_class[a], alpha2cumdict_neg[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_neg_class else alpha2cumdict_neg[a_prev]
            alpha2cumdict_pos[a] = mergeDict_by_adding_values_of_common_keys(alpha2sampleindexsetandreadfraction_of_pos_class[a], alpha2cumdict_pos[a_prev]) if a in alpha2sampleindexsetandreadfraction_of_pos_class else alpha2cumdict_pos[a_prev]
    arr_neg_cumsum = np.array([sum([True if alpha2cumdict_neg[a][sample_index]>=min_read_fraction_neg else False for sample_index in alpha2cumdict_neg[a]]) for a in alpha_union_list])
    arr_pos_cumsum = np.array([sum([True if alpha2cumdict_pos[a][sample_index]>=min_read_fraction_pos else False for sample_index in alpha2cumdict_pos[a]]) for a in alpha_union_list])
    index_list = [i for i in range(n_alpha) if ((arr_pos_cumsum[i]>=arr_pos_cumsum_threshold) and (arr_neg_cumsum[i]<=arr_neg_cumsum_threshold))]
    if len(index_list)==0:
        # No turning point
        return( (-2, arr_neg_cumsum, arr_pos_cumsum) )
    else:
        # if (index_list[-1]+1) < (n_alpha-1):
        if (index_list[-1]) < (n_alpha - 1):
            return( (index_list[-1]+1, arr_neg_cumsum, arr_pos_cumsum) )
        else:
            if len(index_list)>=2:
                if (index_list[-2]) < (n_alpha - 1):
                    return( (index_list[-2]+1, arr_neg_cumsum, arr_pos_cumsum) )
            # -1 means the index should be the one greater than n
            return( (-1, arr_neg_cumsum, arr_pos_cumsum) )


def identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_and_neg_class(unique_alpha_values_of_neg_class, sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_class, unique_alpha_values_of_pos_class, sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_pos_class, max_freq_cumsum_of_neg,
                                                                min_freq_cumsum_of_pos, min_read_fraction_neg, min_read_fraction_pos,
                                                                marker_type='hyper'):
    # build the dict for alpha2sampleindexsetandreadfraction for negative class
    alpha2sampleindexsetandreadfraction_of_neg_class = {}
    for i in range(len(unique_alpha_values_of_neg_class)):
        alpha = unique_alpha_values_of_neg_class[i]
        alpha2sampleindexsetandreadfraction_of_neg_class[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_neg_class[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_neg_class[alpha][sample_index] = read_fraction
    # build the dict for alpha2sampleindexsetandreadfraction for positive class
    alpha2sampleindexsetandreadfraction_of_pos_class = {}
    for i in range(len(unique_alpha_values_of_pos_class)):
        alpha = unique_alpha_values_of_pos_class[i]
        alpha2sampleindexsetandreadfraction_of_pos_class[alpha] = {}
        items = sample_index_sets_and_readfrac_of_unique_alpha_values_str_list_of_pos_class[i].split('_')
        for item in items:
            sample_index, read_fraction = item.split(':')
            read_fraction = float(read_fraction)
            alpha2sampleindexsetandreadfraction_of_pos_class[alpha][sample_index] = read_fraction

    # Process
    if 'hyper' in marker_type:
        if len(alpha2sampleindexsetandreadfraction_of_pos_class)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_class)==0:
            alpha_threshold = 0 # any reads with any alpha values should be used. So let alpha_threshold==0 for hyper markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexsetandreadfraction_of_pos_class)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_class)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexsetandreadfraction_of_pos_class.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_class.keys()))), reverse=True) # decreasing order
    elif 'hypo' in marker_type:
        if len(alpha2sampleindexsetandreadfraction_of_pos_class)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_class)==0:
            alpha_threshold = 1 # any reads with any alpha values should be used. So let alpha_threshold==1 for hypo markers
            return (alpha_threshold, [], [], [])
        if len(alpha2sampleindexsetandreadfraction_of_pos_class)>0 and len(alpha2sampleindexsetandreadfraction_of_neg_class)>0:
            alpha_union_list = sorted(list(set(list(alpha2sampleindexsetandreadfraction_of_pos_class.keys()) + list(alpha2sampleindexsetandreadfraction_of_neg_class.keys()))))  # increasing order
    # identify_turning_point_of_two_alpha2sampleindexsetandreadfraction
    alpha_index, arr_neg_cumsum, arr_pos_cumsum = identify_turning_point_of_two_alpha2sampleindexsetandreadfraction(min_read_fraction_neg,
                                                                                                                    min_read_fraction_pos,
                                                                                                                    alpha_union_list,
                                                                                                     alpha2sampleindexsetandreadfraction_of_neg_class,
                                                                                                     alpha2sampleindexsetandreadfraction_of_pos_class,
                                                                                                     max_freq_cumsum_of_neg,
                                                                                                     min_freq_cumsum_of_pos)
    if alpha_index == -2:
        alpha_threshold = None
    else:
        if alpha_index == -1:
            alpha_threshold = None
            # if marker_type=='hyper':
            #     alpha_threshold = 0
            # elif marker_type == 'hypo':
            #     alpha_threshold = 1
        else:
            alpha_threshold = float(alpha_union_list[alpha_index])
            # if len(alpha2freq_of_neg_class) == 1:
            #     # Remove the following case which identifies hyper marker:
            #     # positive class: {'0.857':2, '0.286':1, '0.167':2, '0.143':4, '0':8}
            #     # negative class: {'0.167':2}
            #     # alpha_union_list: ['0.857', '0.286', '0.167', '0.143', '0']
            #     # identify_turning_point_of_two_cumsum_array returns alpha_index=4
            #     # But actually if alpha_threshold=='0', then negative class is incorrect for our purpose.
            #     if (('hyper' in marker_type) and (alpha_threshold<float(alpha2freq_of_neg_class.values[0]))) or (('hypo' in marker_type) and (alpha_threshold>float(alpha2freq_of_neg_class.values[0]))):
            #         alpha_threshold = None
    return ((alpha_threshold, alpha_union_list, arr_neg_cumsum, arr_pos_cumsum))




# Input:
#   in_file1_background and in_file2_cancer: file format is from function 'write_combined_meth_string_histgram'
# Procedure:
#   We first load each of these two files into a dictionary { 'marker':histgram_dictionary }, for example: {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'0.7':2, '0.9':4}. It is the output of 'load_one_alpha_value_distribution_file' or 'combine_multi_alpha_histgram_files'
#
# Output:
#   ret_marker_2_alpha2freq: a dictionary {'alpha_threshold':alpha_cutoff, 'max_cpg_num':max_cpg_num_of_the_marker, 'alpha2freq':histgram_dictionary}
#
def compare_background_vs_cancer_alpha_value_distribution_files(method, in_file1_background, in_file2_cancer):
    a1_background = {}
    marker2max_cpg_num_1 = {}
    marker2sample_num_1 = {}
    if 'samplesetfreq' in method:
        if 'readfrac' in method:
            print('This part is unused anymore, and is replaced by the corresponding part of function "compare_background_vs_cancer_alpha_value_distribution_files_with_memory_saving_way"')
            # a1_background: a dictionary {'marker_index':{'alpha_value':sample_index_set}}
            load_one_alpha_value_distribution_file_that_has_sample_index_sets_and_read_fractions(in_file1_background, a1_background, marker2max_cpg_num_1, marker2sample_num_1)
        else:
            # a1_background: a dictionary {'marker_index':{'alpha_value':sample_index_set}}
            load_one_alpha_value_distribution_file_that_has_sample_index_sets(in_file1_background, a1_background, marker2max_cpg_num_1)
    else:
        # a1_background: a dictionary {'marker_index':{'alpha_value':frequency_int}}
        load_one_alpha_value_distribution_file(in_file1_background, a1_background, marker2max_cpg_num_1)
    a2_cancer = {}
    marker2max_cpg_num_2 = {}
    if 'samplesetfreq' in method:
        # a2_cancer: a dictionary {'marker_index':{'alpha_value':sample_index_set}}
        load_one_alpha_value_distribution_file_that_has_sample_index_sets(in_file2_cancer, a2_cancer, marker2max_cpg_num_2)
    else:
        # a2_cancer: a dictionary {'marker_index':{'alpha_value':frequency_int}}
        load_one_alpha_value_distribution_file(in_file2_cancer, a2_cancer, marker2max_cpg_num_2)
    marker_index_list1 = a1_background.keys()
    marker_index_list2 = a2_cancer.keys()
    marker_index_common_list = sorted(list(set(marker_index_list1).intersection(marker_index_list2)))
    ret_marker_2_alpha2freq = {}
    try:
        if 'hypo.min.alpha.diff' in method: # 'hypo.min.alpha.diff_0.3' if (min_alpha(m1_background[marker_id]) - min_alpha(m2_cancer[marker_id]))>=0.3, then we accept this marker_id and report those meth_strings in m2_cancer[marker_id] whose alpha values < min_alpha(m1_background[marker_id]).
            min_alpha_diff = float(method.split('_')[1])
            for m in marker_index_common_list:
                if (len(a1_background[m])==0) or (len(a2_cancer[m])==0): continue
                a1_min = min(list(map(float, a1_background[m])))
                a2_min = min(list(map(float, a2_cancer[m])))
                if (a1_min - a2_min) >= min_alpha_diff:
                    ret_marker_2_alpha2freq[m] = {'alpha_threshold':a1_min, 'max_cpg_num':marker2max_cpg_num_2[m], 'alpha2freq':filter_alpha2freq_by_alpha(a2_cancer[m], a1_min, '<')}
        elif 'hyper.max.alpha.diff' in method: # 'hyper.max.alpha.diff_0.3' if max_alpha(m2_cancer[marker_id]) - max_alpha(m1_background[marker_id])>=0.3, then we accept this marker_id and report those meth_strings in m2_cancer[marker_id] whose alpha values > max_alpha(m1_background[marker_id]).
            min_alpha_diff = float(method.split('_')[1])
            for m in marker_index_common_list:
                if (len(a1_background[m])==0) or (len(a2_cancer[m])==0): continue
                a1_max = max(list(map(float, a1_background[m])))
                a2_max = max(list(map(float, a2_cancer[m])))
                if (a2_max - a1_max) >= min_alpha_diff:
                    ret_marker_2_alpha2freq[m] = {'alpha_threshold':a1_max, 'max_cpg_num':marker2max_cpg_num_2[m], 'alpha2freq':filter_alpha2freq_by_alpha(a2_cancer[m], a1_max, '>')}
        elif 'samplesetfreq' in method:
            # 'hyper.alpha.samplesetfreq.thresholds.n2.p10': hyper-methylation markers with alpha's frequency on negative class <2 and alpha's freuqency on positive class > 10. Similar to 'hypo.alpha.samplesetfreq.thresholds.n2.p10'.
            # Negative class: background class
            # Positive class: cancer class
            if 'alpha.samplesetfreq.thresholds' in method:
                marker_type, _, _, _, max_freq_cumsum_of_neg_str, min_freq_cumsum_of_pos_str = method.split('.')
                max_freq_cumsum_of_neg = int(max_freq_cumsum_of_neg_str[1:])
                min_freq_cumsum_of_pos = int(min_freq_cumsum_of_pos_str[1:])
                for m in marker_index_common_list:
                    if (len(a1_background[m]) == 0) or (len(a2_cancer[m]) == 0): continue
                    alpha_threshold, alpha_union_list, a1_freq_cumsum_background, a2_freq_cumsum_cancer = identify_alpha_threshold_by_alpha2sampleindexset_of_pos_and_neg_class(a1_background[m],
                                                                                                  a2_cancer[m],
                                                                                                  max_freq_cumsum_of_neg,
                                                                                                  min_freq_cumsum_of_pos,
                                                                                                  marker_type)
                    if alpha_threshold is not None:
                        if 'hyper' in method:
                            ret_marker_2_alpha2freq[m] = {'alpha_threshold': alpha_threshold, 'max_cpg_num': marker2max_cpg_num_2[m],
                                                          'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(alpha_threshold, alpha_union_list, a2_freq_cumsum_cancer, '>')}
                        elif 'hypo' in method:
                            ret_marker_2_alpha2freq[m] = {'alpha_threshold': alpha_threshold, 'max_cpg_num': marker2max_cpg_num_2[m],
                                                          'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(alpha_threshold, alpha_union_list, a2_freq_cumsum_cancer, '<')}
        else:
            if 'alpha.freq.thresholds' in method:
                # 'hyper.alpha.freq.thresholds.n2.p10': hyper-methylation markers with alpha's frequency on negative class <2 and alpha's freuqency on positive class > 10. Similar to 'hypo.alpha.freq.thresholds.n2.p10', 'hyper.alpha.freq.thresholds.n2.p4.enforce_max_output'
                # Negative class: background class
                # Positive class: cancer class
                items = method.split('.')
                marker_type = items[0]
                max_freq_cumsum_of_neg_str = items[4]
                min_freq_cumsum_of_pos_str = items[5]
                # marker_type, _, _, _, max_freq_cumsum_of_neg_str, min_freq_cumsum_of_pos_str = method.split('.')
                max_freq_cumsum_of_neg = int(max_freq_cumsum_of_neg_str[1:])
                min_freq_cumsum_of_pos = int(min_freq_cumsum_of_pos_str[1:])
                for m in marker_index_common_list:
                    if (len(a1_background[m]) == 0) or (len(a2_cancer[m]) == 0): continue
                    alpha_threshold = identify_alpha_threshold_by_alpha2freq_of_pos_and_neg_class(a1_background[m],
                                                                                                  a2_cancer[m],
                                                                                                  max_freq_cumsum_of_neg,
                                                                                                  min_freq_cumsum_of_pos,
                                                                                                  marker_type)
                    if alpha_threshold is not None:
                        if 'hyper' in method:
                            ret_marker_2_alpha2freq[m] = {'alpha_threshold': alpha_threshold, 'max_cpg_num': marker2max_cpg_num_2[m],
                                                          'alpha2freq': filter_alpha2freq_by_alpha(a2_cancer[m], alpha_threshold,
                                                                                                   '>')}
                        elif 'hypo' in method:
                            ret_marker_2_alpha2freq[m] = {'alpha_threshold': alpha_threshold, 'max_cpg_num': marker2max_cpg_num_2[m],
                                                          'alpha2freq': filter_alpha2freq_by_alpha(a2_cancer[m], alpha_threshold,
                                                                                                   '<')}
    except KeyError:
        # marker_index does not exist
        sys.stderr.write('Error: %d does not exist in one of two meth_strings_histgram_files\n  in_file1_background: %s\n  in_file2_cancer: %s\nExit.'%(m, in_file1_background, in_file2_cancer))
        sys.exit(-1)
    # ret_marker2max_cpg_num = {m: marker2max_cpg_num_2[m] for m in ret_marker_2_alpha2freq}
    return( ret_marker_2_alpha2freq )


def compare_background_vs_cancer_alpha_value_distribution_files_with_memory_saving_way(method, in_file1_background, in_file2_cancer):
    ret_marker_2_alpha2freq = {}
    if 'samplesetfreq' in method:
        if 'readfrac' in method: # 'hyper.alpha.samplesetfreq.thresholds.n2.p5.minreadfrac+0.1-0.01'
            two_read_fractions_str = extract_two_numbers_after_a_substring(method, 'minreadfrac') # extract '+0.1' and '-0.01' from 'hyper.alpha.samplesetfreq.thresholds.n2.p5.minreadfrac+0.1-0.01'
            if two_read_fractions_str is None:
                sys.stderr.write('Error: method (%s) after string minreadfrac does not have two min read fractions with format like +0.2-0.3 or +1-1.\nExit.\n'%method)
                sys.exit(-1)
            min_read_fraction_pos = abs(float(two_read_fractions_str[0]))
            min_read_fraction_neg = abs(float(two_read_fractions_str[1]))
            marker_type, _, _, _, max_freq_cumsum_of_neg_str, min_freq_cumsum_of_pos_str = remove_substring_followed_by_two_floats(
                method, '.minreadfrac').split('.')
            if 'nn' in max_freq_cumsum_of_neg_str:
                max_freq_fraction_of_neg = float(max_freq_cumsum_of_neg_str[1:])
            else:
                max_freq_cumsum_of_neg = int(max_freq_cumsum_of_neg_str[1:])
            min_freq_cumsum_of_pos = int(min_freq_cumsum_of_pos_str[1:])

            if in_file1_background.endswith('gz'):
                fid1_background = gzip.open(in_file1_background, 'rt')
            else:
                fid1_background = open(in_file1_background, 'rt')
            if in_file2_cancer.endswith('gz'):
                fid2_cancer = gzip.open(in_file2_cancer, 'rt')
            else:
                fid2_cancer = open(in_file2_cancer, 'rt')
            ### begin to process two input files and write output file
            fid1_background.readline().rstrip() # skip header line
            background_first_marker_line = fid1_background.readline()
            background_items = background_first_marker_line.rstrip().split('\t')
            background_marker_index = int(background_items[0])

            end_of_background_file = False
            fid2_cancer.readline()  # skip header line
            for cancer_line in fid2_cancer:
                cancer_items = cancer_line.rstrip().split()
                cancer_marker_index = int(cancer_items[0])
                while cancer_marker_index > background_marker_index:
                    background_line = fid1_background.readline()
                    if not background_line:
                        end_of_background_file = True
                        break
                    background_items = background_line.rstrip().split('\t')
                    background_marker_index = int(background_items[0])
                if end_of_background_file:
                    break
                if cancer_marker_index < background_marker_index:
                    continue

                # now we begin to process for cancer_marker_index == background_marker_index
                # if cancer_marker_index == 371737:
                #     print('debug 371737')
                max_cpg_num = background_items[1]
                background_sample_num = int(background_items[2])
                background_unique_alpha_values = background_items[3].split(',')
                background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = background_items[4].split(',')

                cancer_unique_alpha_values = cancer_items[3].split(',')
                cancer_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = cancer_items[4].split(',')

                if 'nn' in max_freq_cumsum_of_neg_str:
                    max_freq_cumsum_of_neg = max_freq_fraction_of_neg * background_sample_num

                alpha_threshold, alpha_union_list, a1_freq_cumsum_background, a2_freq_cumsum_cancer = identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_and_neg_class(background_unique_alpha_values, background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list, cancer_unique_alpha_values, cancer_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list, max_freq_cumsum_of_neg, min_freq_cumsum_of_pos, min_read_fraction_neg, min_read_fraction_pos, marker_type)
                if alpha_threshold is not None:
                    if 'hyper' in method:
                        ret_marker_2_alpha2freq[cancer_marker_index] = {'alpha_threshold': alpha_threshold,
                                                      'max_cpg_num':max_cpg_num,
                                                      'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                          alpha_threshold, alpha_union_list, a2_freq_cumsum_cancer,
                                                          '>')}
                    elif 'hypo' in method:
                        ret_marker_2_alpha2freq[cancer_marker_index] = {'alpha_threshold': alpha_threshold,
                                                      'max_cpg_num':max_cpg_num,
                                                      'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                          alpha_threshold, alpha_union_list, a2_freq_cumsum_cancer,
                                                          '<')}

            fid1_background.close()
            fid2_cancer.close()
    return (ret_marker_2_alpha2freq)


#
# Input:
# alpha_hists: a dictionary { marker_index:{'alpha_threshold':threshold, 'max_cpg_num':cpg_num, 'alpha2freq':alpha_histgram_dictionary} }. for example, alpha_histgram_dictionary is {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'0.7':2, '1.0':4}. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:max_cpg_num}. NOTE: It is from the output of function 'compare_background_vs_cancer_alpha_value_distribution_files'
#
# Output:
# alpha_value_distribution_with_threshold file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read  alpha_threshold  unique_alpha_values read_freq_of_unique_alpha_values
# 2   7   122  0.4  0.429,0.571,0.714,0.857,1   1,2,27,42,50
# 27  9   39  0.5  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13
# 61  12  44  0.7 0.75,0.833,0.917,1  2,11,12,19
# 63  5   100  0.5  0.6,0.8,1   4,23,73
# 65  5   83 0.5 0.6,0.8,1 9,26,42
# ...
#
def write_alpha_value_distribution_file_with_alpha_threshold(fout, alpha_hists, frequency_type='alpha2freqeuncy_is_individual'):
    marker_index_list = sorted(list(set(alpha_hists.keys())))
    fout.write(
        'marker_index\tmax_num_cpg\tnum_read\talpha_threshold\tunique_alpha_values\tread_freq_of_unique_alpha_values\n')
    for marker_index in marker_index_list:
        if 'alpha2freqeuncy_is_individual' in frequency_type:
            num_reads = sum(alpha_hists[marker_index]['alpha2freq'].values())
        elif 'enforce_max_output' in frequency_type:
            num_reads = max(alpha_hists[marker_index]['alpha2freq'].values())
        elif 'alpha2freqeuncy_is_cumsum' in frequency_type:
            num_reads = max(alpha_hists[marker_index]['alpha2freq'].values())
        str_for_unique_alpha_values, read_freq_of_unique_alpha_values_str = convert_str2int_dict_to_str(alpha_hists[marker_index]['alpha2freq'])
        fout.write('%d\t%s\t%d\t%g\t%s\t%s\n' % (marker_index,
                                                 alpha_hists[marker_index]['max_cpg_num'],
                                                 num_reads,
                                                 alpha_hists[marker_index]['alpha_threshold'],
                                                 str_for_unique_alpha_values,
                                                 read_freq_of_unique_alpha_values_str
                                                 ))

#
# Input:
# alpha_hists: a dictionary { marker_index:{'alpha_threshold':threshold, 'max_cpg_num':cpg_num, 'alpha2freq':alpha_histgram_dictionary} }. for example, alpha_histgram_dictionary is {27:hist_dict, 63:hist_dict}, where hist_dict is a dictionary {'0.7':2, '1.0':4}. The input 'alpha_hists' can be an empty dictionary {}.
# marker2max_cpg_num: a dictionary {marker_index:max_cpg_num}. NOTE: It is from the output of function 'compare_background_vs_cancer_alpha_value_distribution_files'
#
# Output:
# alpha_value_distribution_with_threshold file (from the output of the function 'summarize_mary_file_binary_meth_values_for_distribution_file'):
# marker_index    max_num_cpg num_read  alpha_threshold_pos  alpha_threshold_neg  unique_alpha_values read_freq_of_unique_alpha_values
# 2   7   122  0.4  0.1  0.429,0.571,0.714,0.857,1   1,2,27,42,50
# 27  9   39  0.5  0.2  0.5,0.625,0.75,0.778,0.875,0.889,1  1,2,9,1,12,1,13
# 61  12  44  0.7  0.3 0.75,0.833,0.917,1  2,11,12,19
# 63  5   100  0.5  0.1  0.6,0.8,1   4,23,73
# 65  5   83 0.5  0.05 0.6,0.8,1 9,26,42
# ...
#
def write_alpha_value_distribution_file_with_two_alpha_thresholds(fout, alpha_hists, frequency_type='alpha2freqeuncy_is_individual'):
    marker_index_list = sorted(list(set(alpha_hists.keys())))
    fout.write(
        'marker_index\tmax_num_cpg\tnum_read\talpha_threshold_pos\talpha_threshold_neg\tunique_alpha_values\tread_freq_of_unique_alpha_values\n')
    for marker_index in marker_index_list:
        if 'alpha2freqeuncy_is_cumsum' in frequency_type:
            num_reads = max(alpha_hists[marker_index]['alpha2freq'].values())
        str_for_unique_alpha_values, read_freq_of_unique_alpha_values_str = convert_str2int_dict_to_str(alpha_hists[marker_index]['alpha2freq'])
        fout.write('%d\t%s\t%d\t%s\t%s\t%s\t%s\n' % (marker_index,
                                                     alpha_hists[marker_index]['max_cpg_num'],
                                                     num_reads,
                                                     alpha_hists[marker_index]['alpha_threshold_of_pos'],
                                                     alpha_hists[marker_index]['alpha_threshold_of_neg'],
                                                     str_for_unique_alpha_values,
                                                     read_freq_of_unique_alpha_values_str
                                                     ))



# Version 5
# Take three alpha_value_distribution files as inputs
# All three kinds of samples (normal plasma, tumors, matched normal tissues) use two different alpha-thresholds
# 1. Normal plasma use the dynamic alpha threshold 'a_n' or 'an', which must take value in a predefined range
# 2. Tumors & their matched/paired normal tissues use another dynamic alpha threshold 'a_t' or 'at' (could be
#    different from 'a_n'), which has a FIXED difference from 'a_n' or 'an'
# 3. The paired tissue is counted into sample frequency, if the difference of read fractions of the tumor and its
#    adjacent normal tissue >= threshold
#
# Explanation of the parameters in the following method ID: 3 parts that are separated by ','
#   For the example ID: 'hyper.alpha.samplesetfreq.triple_V3.thresholds.nn0.2.p1,arange0.7_1,readfrac+pairedtissuediff0.5-ncplasma0.2'
#   Part1:
#     'nn0.2': sample frequency of negative class (i.e., non-cancer plasma that have tumor signals) <= 20% of all samples in negative class
#     'p1': sample frequency of positive class (i.e., tissue pairs that have tumor signals) >= 1
#   Part2:
#     'anrange0_0.5': the dynamic Alpha threshold of Normal plasma is in RANGE [0, 0.5]
#     'adiff0.5': minimum DIFFerence of the dynamic Alpha threshold of Tissues (applied to tumors & adjacent normal tissues) and the dynamic Alpha threshold of normal plasma is 0.5. For hyper-markers, a_tissue - a_normal >= min_a_diff = , e.g., a_tissue - 0.1 >= 0.5; for hypo-markers, a_normal - a_tissue >= min_a_diff, e.g., 0.8 - a_tissue >= 0.5.
#   Part3:
#     'readfrac+pairedtissuediff0.5': for positive class, difference btw fractions of reads with tumor signals for tumor & matched normal tissue >= threshold 0.5
#     '-ncplasma0.2': for negative class, "Non-Cancer plasma" has the fraction of reads with tumor signal <= 0.2
#
def compare_background_vs_tumor_and_paired_normaltissue_alpha_value_distribution_files_with_memory_saving_way_for_cancer_detection_V5(method,
                                                                                                              in_file1_background,
                                                                                                              in_file2_tumors,
                                                                                                              in_file3_paired_normaltissues):
    # Scan three input files to obtain the markers that appear in all three files
    # get_specific_column_of_tab_file(in_file1_background)
    ret_marker_2_alpha2freq = {}
    if 'samplesetfreq' in method:
        if 'readfrac' in method: # 'hyper.alpha.samplesetfreq.triple_V5.thresholds.nn0.2.p1,anrange0_0.5.adiff0.5,readfrac+pairedtissuediff0.3-ncplasma0.2'
            part1_method, part2_method, part3_method = method.split(',')
            marker_type = part1_method.split('.')[0]
            if 'nn' in part1_method:
                max_freq_fraction_of_neg_background = float(extract_number_after_a_substring(part1_method,'nn'))  # max sample frequency, extract '0.2' from 'hyper.alpha.samplesetfreq.triple.thresholds.nn0.2.p1'
            else:
                max_freq_cumsum_of_neg_background = int(extract_number_after_a_substring(part1_method,'n'))  # max sample frequency, extract '2' from 'hyper.alpha.samplesetfreq.triple.thresholds.n2.p1'
            min_freq_cumsum_of_pos_tumors = int(extract_number_after_a_substring(part1_method,'p'))  # min sample frequency, extract '1' from 'hyper.alpha.samplesetfreq.triple.thresholds.nn0.2.p1'
            alpha_threshold_range_neg_backgroundplasma = extract_range_after_a_substring(part2_method, 'anrange')
            min_alpha_threshold_diff = extract_number_after_a_substring(part2_method, 'adiff')
            diff_read_fraction_pairedtissues = abs(float(extract_number_after_a_substring(part3_method, 'pairedtissuediff'))) # paired tissues are tumor & its matched normal tissue
            min_read_fraction_neg_backgroundplasma = abs(float(extract_number_after_a_substring(part3_method, 'ncplasma'))) # non-cancer plasma are background

            if in_file1_background.endswith('gz'):
                fid1_background = gzip.open(in_file1_background, 'rt')
            else:
                fid1_background = open(in_file1_background, 'rt')
            if in_file2_tumors.endswith('gz'):
                fid2_tumors = gzip.open(in_file2_tumors, 'rt')
            else:
                fid2_tumors = open(in_file2_tumors, 'rt')
            if in_file3_paired_normaltissues.endswith('gz'):
                fid3_paired_normaltissues = gzip.open(in_file3_paired_normaltissues, 'rt')
            else:
                fid3_paired_normaltissues = open(in_file3_paired_normaltissues, 'rt')

            ### begin to process three input files and write output file
            fid1_background.readline().rstrip()  # skip header line
            background_first_marker_line = fid1_background.readline()
            background_items = background_first_marker_line.rstrip().split('\t')
            background_marker_index = int(background_items[0])

            fid3_paired_normaltissues.readline().rstrip()  # skip header line
            paired_normaltissues_first_marker_line = fid3_paired_normaltissues.readline()
            paired_normaltissues_items = paired_normaltissues_first_marker_line.rstrip().split('\t')
            paired_normaltissues_marker_index = int(paired_normaltissues_items[0])

            end_of_background_file = False
            end_of_paired_normaltissues_file = False
            fid2_tumors.readline()  # skip header line
            for tumors_line in fid2_tumors:
                tumors_items = tumors_line.rstrip().split()
                tumors_marker_index = int(tumors_items[0])

                while tumors_marker_index > background_marker_index:
                    background_line = fid1_background.readline()
                    if not background_line:
                        end_of_background_file = True
                        break
                    background_items = background_line.rstrip().split('\t')
                    background_marker_index = int(background_items[0])
                if end_of_background_file:
                    break
                if tumors_marker_index < background_marker_index:
                    continue
                # now we arrive at tumors_marker_index == background_marker_index
                while tumors_marker_index > paired_normaltissues_marker_index:
                    paired_normaltissues_line = fid3_paired_normaltissues.readline()
                    if not paired_normaltissues_line:
                        end_of_paired_normaltissues_file = True
                        break
                    paired_normaltissues_items = paired_normaltissues_line.rstrip().split('\t')
                    paired_normaltissues_marker_index = int(paired_normaltissues_items[0])
                if end_of_paired_normaltissues_file:
                    break
                if tumors_marker_index < paired_normaltissues_marker_index:
                    continue
                # now we arrive at tumors_marker_index == background_marker_index == paired_normaltissues_marker_index

                # now we begin to process for tumors_marker_index == background_marker_index == paired_normaltissues_marker_index
                max_cpg_num = background_items[1]
                background_sample_num = int(background_items[2])
                background_unique_alpha_values = background_items[3].split(',')
                background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = background_items[4].split(',')

                tumors_unique_alpha_values = tumors_items[3].split(',')
                tumors_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = tumors_items[4].split(',')

                paired_normaltissues_unique_alpha_values = paired_normaltissues_items[3].split(',')
                paired_normaltissues_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list = paired_normaltissues_items[4].split(',')

                if 'nn' in part1_method:
                    max_freq_cumsum_of_neg_background = max_freq_fraction_of_neg_background * background_sample_num

                # if tumors_marker_index == 1332:
                #     print('debug %d'%tumors_marker_index)

                alpha_threshold_for_neg_backgroundplasma, alpha_threshold_for_pos_tumors, alpha_union_list, a1_freq_cumsum_background, a2_freq_cumsum_tumors  = identify_alpha_threshold_by_alpha2sampleindexsetandreadfraction_of_pos_tumors_neg_paired_normaltissues_and_neg_background_V5(
                    background_unique_alpha_values,
                    background_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    tumors_unique_alpha_values,
                    tumors_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    paired_normaltissues_unique_alpha_values,
                    paired_normaltissues_sample_index_sets_and_readfrac_of_unique_alpha_values_str_list,
                    max_freq_cumsum_of_neg_background,
                    min_freq_cumsum_of_pos_tumors,
                    min_read_fraction_neg_backgroundplasma,
                    diff_read_fraction_pairedtissues,
                    alpha_threshold_range_neg_backgroundplasma,
                    min_alpha_threshold_diff,
                    marker_type)
                if alpha_threshold_for_pos_tumors is not None:
                    if 'hyper' in method:
                        ret_marker_2_alpha2freq[tumors_marker_index] = {'alpha_threshold_of_neg': alpha_threshold_for_neg_backgroundplasma,
                                                                        'alpha_threshold_of_pos': alpha_threshold_for_pos_tumors,
                                                                        'max_cpg_num': max_cpg_num,
                                                                        'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                                            float(alpha_threshold_for_pos_tumors), alpha_union_list,
                                                                            a2_freq_cumsum_tumors,
                                                                            '>')}
                    elif 'hypo' in method:
                        ret_marker_2_alpha2freq[tumors_marker_index] = {'alpha_threshold_of_neg': alpha_threshold_for_neg_backgroundplasma,
                                                                        'alpha_threshold_of_pos': alpha_threshold_for_pos_tumors,
                                                                        'max_cpg_num': max_cpg_num,
                                                                        'alpha2freq': filter_by_freq_cumsum_and_create_alpha2freqcumsum(
                                                                            float(alpha_threshold_for_pos_tumors), alpha_union_list,
                                                                            a2_freq_cumsum_tumors,
                                                                            '<')}

            fid1_background.close()
            fid2_tumors.close()
            fid3_paired_normaltissues.close()
        return (ret_marker_2_alpha2freq)
