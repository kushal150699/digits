"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import sys
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import data_preprocess, train_model, read_digits, split_train_dev_test, p_and_eval,get_all_h_param_comb_svm,get_all_h_param_comb_tree,tune_hparams
import pdb
from joblib import dump,load
import numpy as np
import skimage
from skimage.transform import resize
import pandas as pd
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

## Split data 
X, y = read_digits()

h_metric = metrics.accuracy_score

classifier_param_dict = {}

# SVM Hyperparameters combination

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
all_combos_svm = get_all_h_param_comb_svm(gamma_list,c_list)
classifier_param_dict['Production_Model_svm'] = all_combos_svm


#Decision Tree Hyperparameters combination

max_depth_list = [5,10,15,20,50,100,150]
all_combos_tree = get_all_h_param_comb_tree(max_depth_list)
classifier_param_dict['Candidate_Model_tree'] = all_combos_tree

parser = argparse.ArgumentParser()

# parser.add_argument("--model_type",choices=["svm","tree","svm,tree"],default="svm",help="Model type")
parser.add_argument("--num_runs",type=int,default=1,help="Number of runs")
parser.add_argument("--test_size", type=float, default=0.2, help="test_size")
parser.add_argument("--dev_size", type=float, default=0.2, help="dev_size")

args = parser.parse_args()


results=[]
num_runs = args.num_runs
test_sizes = [args.test_size]
dev_sizes = [args.dev_size]
model_types = "Production_Model_svm,Candidate_Model_tree"

models=model_types.split(',')

for curr_run in range(num_runs):
    curr_run_results={}
  
    # for test_s in args.test_size:
    #     for dev_s in args.dev_size:
    test_s = args.test_size
    dev_s = args.test_size
    train_size = 1 - test_s - dev_s
    X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=test_s, dev_size=dev_s)

    X_train = data_preprocess(X_train)
    X_dev = data_preprocess(X_dev)
    X_test = data_preprocess(X_test)

    # for model_type in classifier_param_dict:
    for model_type in models:
        all_combos = classifier_param_dict[model_type]
        best_hparams, best_model_path , best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, all_combos ,h_metric,model_type)

        best_model = load(best_model_path)

        if model_type=="Production_Model_svm":
            predictions_svm = best_model.predict(X_test)
            confusion_matrix_svm = confusion_matrix(y_test, predictions_svm)
            f1_svm = f1_score(y_test, predictions_svm, average='macro')
        if model_type=="Candidate_Model_tree":
            predictions_tree = best_model.predict(X_test) 
            confusion_matrix_tree = confusion_matrix(y_test, predictions_tree)
            f1_tree = f1_score(y_test, predictions_tree, average='macro')

        test_acc = p_and_eval(best_model,h_metric,X_test,y_test)
        train_acc = p_and_eval(best_model,h_metric,X_train,y_train)
        dev_acc = best_accuracy
        
        print("{}\ttest_size={:.2f} dev_size={:.2f} train_size={:.2f} train_accuracy={:.2f} dev_accuracy={:.2f} test_accuracy={:.2f}".format(model_type,
                test_s,dev_s,train_size,train_acc,dev_acc,test_acc))
        # if model_type=="svm":
        #     print(f"Best Hyperparameters: ( gamma : {best_hparams[0]} , C : {best_hparams[1]} )")
        # if model_type=="tree":
        #     print(f"Best Hyperparameters: ( max_depth : {best_hparams[0]})")    
        curr_run_results = {'model_type': model_type,'run_index': curr_run,'train_acc':train_acc,'dev_acc': dev_acc , 
                            'test_acc':test_acc}
        results.append(curr_run_results)


# print(pd.DataFrame(results).groupby('model_type')[['train_acc', 'dev_acc','test_acc']].agg(['mean', 'std']).T)     

confusion_matrix_svm_tree = confusion_matrix(predictions_tree,predictions_svm)

print("")
print("Confusion matrix between predictions of production and candidate models")
print(confusion_matrix_svm_tree)
print("")

tp = confusion_matrix_svm[1, 1]
tn = confusion_matrix_tree[0, 0]
fp = confusion_matrix_svm[0, 1]
fn = confusion_matrix_svm[1, 0]

matrix_2x2 = [[tp, fp], [fn, tn]]

print("2x2 Confusion Matrix: btw production and candidate model")
for row in matrix_2x2:
    print(row)

print("")
print("Macro-average F1 Score for Production_Model_svm:", f1_svm)
print("Macro-average F1 Score for Candidate_Model_tree:", f1_tree)
