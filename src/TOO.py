import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from library_utils import read_configuration_file, read_input_data_view_file, read_class_labels_file_of_samples_for_multiclass, find_samples_of_a_specific_classname, remove_data_of_selected_samples, classify_multiclass, reorder_rows_of_matrix, write_preds_of_multiclass_with_sampleindex_and_classnames

config_file_for_input_data = sys.argv[1]

config = read_configuration_file(config_file_for_input_data)
print('#Configurations:')
print(config)
print('')

print('Load data')
y_train, y_train_classnames, unique_classnames = read_class_labels_file_of_samples_for_multiclass(config['filename_class_labels_for_training'], config['class_name_of_controls']) # Move "control class" to the first element of 'unique_classnames', to make the index of "control class" always 0
y_test, y_test_classnames, _ = read_class_labels_file_of_samples_for_multiclass(config['filename_class_labels_for_testing'], config['class_name_of_controls'])
# Need to remove "control samples" from both training and testing data, since TOO prediction does not need control samples.
indexes_of_control_samples_in_train = find_samples_of_a_specific_classname(y_train_classnames, config['class_name_of_controls'])
indexes_of_control_samples_in_test = find_samples_of_a_specific_classname(y_test_classnames, config['class_name_of_controls'])
X_train = {}
X_test = {}
for i in range(config['number_data_views']):
    view_id = 'view%d'%(i+1)
    X_train[view_id] = read_input_data_view_file(config['filename_of_data_views_for_training'][i])
    X_test[view_id] = read_input_data_view_file(config['filename_of_data_views_for_testing'][i])
    if view_id=='view1':
        X_train[view_id], y_train, indexes_of_training_samples = remove_data_of_selected_samples(X_train[view_id], indexes_of_control_samples_in_train, y_train)
        X_test[view_id], y_test, indexes_of_testing_samples = remove_data_of_selected_samples(X_test[view_id], indexes_of_control_samples_in_test, y_test)
    else:
        X_train[view_id] = remove_data_of_selected_samples(X_train[view_id], indexes_of_control_samples_in_train)
        X_test[view_id] = remove_data_of_selected_samples(X_test[view_id], indexes_of_control_samples_in_test)

print('Train each view\'s level-1 classifier to')
# Split training samples into K folds for out-of-fold predictions (OOFP), which are used for training stacked model
skf = StratifiedKFold(n_splits=config['level1_num_folds_of_training_data_for_OOFP'])
print('  (1) produce training data from level-1 classifiers for level-2 classifier')
indexes_of_oofp = np.array([], dtype=int)
level2_train_data_using_scores_of_oofp = {}
for i in range(config['number_data_views']):
    view_id = 'view%d' % (i + 1)
    level2_train_data_using_scores_of_oofp[view_id] = np.array([])
j = 0
for indexes_of_K_minus_1_folds, indexes_of_1_fold in skf.split(X_train['view1'], y_train):
    j += 1
    indexes_of_oofp = np.concatenate( (indexes_of_oofp, indexes_of_1_fold) )
    for i in range(config['number_data_views']):
        view_id = 'view%d' % (i + 1)
        classifier_id_of_view = config['level1_classifiers_of_data_views_for_TOO'][i]
        preds = classify_multiclass(classifier_id_of_view, X_train[view_id][indexes_of_K_minus_1_folds,:], y_train[indexes_of_K_minus_1_folds], X_train[view_id][indexes_of_1_fold])
        if j==1:
            level2_train_data_using_scores_of_oofp[view_id] = preds
        else:
            level2_train_data_using_scores_of_oofp[view_id] = np.vstack(
                (level2_train_data_using_scores_of_oofp[view_id], preds))  # Vertically concatenate two 2d numpy arrays
            #level2_train_data_using_scores_of_oofp[view_id] = np.concatenate((level2_train_data_using_scores_of_oofp[view_id], preds)) # Vertically concatenate two 2d numpy arrays

# reorder samples/rows of 2d array of OOPS-scores to make their orders the same as rows of 'X_train[view_id]' and y_train
for i in range(config['number_data_views']):
    view_id = 'view%d' % (i + 1)
    level2_train_data_using_scores_of_oofp[view_id] = reorder_rows_of_matrix(level2_train_data_using_scores_of_oofp[view_id], indexes_of_oofp)

print('  (2) produce testing data from level-1 classifiers for level-2 classifier')
level2_test_data = {}
for i in range(config['number_data_views']):
    view_id = 'view%d' % (i + 1)
    classifier_id_of_view = config['level1_classifiers_of_data_views_for_TOO'][i]
    level2_test_data[view_id] = classify_multiclass(classifier_id_of_view, X_train[view_id], y_train, X_test[view_id])

print('Train level-2 classifier and predict the test samples')
combined_level2_train_data_using_scores_of_oofp = None
combined_level2_test_data = None
for i in range(config['number_data_views']):
    view_id = 'view%d' % (i + 1)
    if view_id=='view1':
        combined_level2_train_data_using_scores_of_oofp = level2_train_data_using_scores_of_oofp[view_id]
        combined_level2_test_data = level2_test_data[view_id]
    else:
        # Horizontally concatenate two matrixes
        combined_level2_train_data_using_scores_of_oofp = np.hstack(
            (combined_level2_train_data_using_scores_of_oofp, level2_train_data_using_scores_of_oofp[view_id]))
        #combined_level2_train_data_using_scores_of_oofp = np.concatenate((combined_level2_train_data_using_scores_of_oofp, level2_train_data_using_scores_of_oofp[view_id]), axis=1)
        combined_level2_test_data = np.hstack((combined_level2_test_data, level2_test_data[view_id]))
        #combined_level2_test_data = np.concatenate((combined_level2_test_data, level2_test_data[view_id]), axis=1)
final_preds = classify_multiclass(config['level2_classifier_for_TOO'], combined_level2_train_data_using_scores_of_oofp, y_train, combined_level2_test_data)

print('Write to output prediction file\n  file: %s'%config['file_of_TOO_prediction'])
write_preds_of_multiclass_with_sampleindex_and_classnames(config['file_of_TOO_prediction'], final_preds,
                                                          indexes_of_testing_samples, unique_classnames[1:])
print('Done.')
