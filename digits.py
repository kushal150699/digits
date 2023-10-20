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
classifier_param_dict['svm'] = all_combos_svm


#Decision Tree Hyperparameters combination

max_depth_list = [5,10,15,20,50,100,150]
all_combos_tree = get_all_h_param_comb_tree(max_depth_list)
classifier_param_dict['tree'] = all_combos_tree

# images_4X4 = []
# images_6X6 = []
# images_8X8 = []
# # X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.3)

# for image in images:
#     images_4X4.append(resize(image, (image.shape[0] // 2, image.shape[1] // 2),
#                         anti_aliasing=True))
#     images_6X6.append(resize(image, (image.shape[0] // 1.25, image.shape[1] // 1.25),
#                         anti_aliasing=True))
#     images_8X8.append(resize(image, (image.shape[0] // 1, image.shape[1] // 1),
#                         anti_aliasing=True))

# images_4X4 = np.array(images_4X4)
# images_6X6 = np.array(images_6X6)
# images_8X8 = np.array(images_8X8)

# X4_train, X4_test, X4_dev, y_train4, y_test4, y_dev4 = split_train_dev_test(images_4X4, y, test_size=0.2, dev_size=0.7)
# X6_train, X6_test, X6_dev, y_train6, y_test6, y_dev6 = split_train_dev_test(images_6X6, y, test_size=0.2, dev_size=0.7)
# X8_train, X8_test, X8_dev, y_train8, y_test8, y_dev8 = split_train_dev_test(images_8X8, y, test_size=0.2, dev_size=0.7)

# # ## Use the preprocessed datas
# X_train4 = data_preprocess(X4_train)
# X_dev4 = data_preprocess(X4_dev)
# X_test4 = data_preprocess(X4_test)

# X_train6 = data_preprocess(X6_train)
# X_dev6 = data_preprocess(X6_dev)
# X_test6 = data_preprocess(X6_test)

# X_train8 = data_preprocess(X8_train)
# X_dev8 = data_preprocess(X8_dev)
# X_test8 = data_preprocess(X8_test)

# model4 = train_model(X_train4, y_train4, {'gamma': 0.001}, model_type='svm')
# model6 = train_model(X_train6, y_train6, {'gamma': 0.001}, model_type='svm')
# model8 = train_model(X_train8, y_train8, {'gamma': 0.001}, model_type='svm')

# y4_train_pred = model4.predict(X_train4)  
# y6_train_pred = model6.predict(X_train6)  
# y8_train_pred = model8.predict(X_train8)

# y4_valid_pred = model4.predict(X_dev4)  
# y6_valid_pred = model6.predict(X_dev6)  
# y8_valid_pred = model8.predict(X_dev8)  

# y4_test_pred = model4.predict(X_test4)  
# y6_test_pred = model6.predict(X_test6)  
# y8_test_pred = model8.predict(X_test8)  

# accuracy_train4 = h_metric(y_pred=y4_train_pred, y_true=y_train4)
# accuracy_train6 = h_metric(y_pred=y6_train_pred, y_true=y_train6)
# accuracy_train8 = h_metric(y_pred=y8_train_pred, y_true=y_train8)

# accuracy_valid4 = h_metric(y_pred=y4_valid_pred, y_true=y_dev4)
# accuracy_valid6 = h_metric(y_pred=y6_valid_pred, y_true=y_dev6)
# accuracy_valid8 = h_metric(y_pred=y8_valid_pred, y_true=y_dev8)

# accuracy_test4 = h_metric(y_pred=y4_test_pred, y_true=y_test4)
# accuracy_test6 = h_metric(y_pred=y6_test_pred, y_true=y_test6)
# accuracy_test8 = h_metric(y_pred=y8_test_pred, y_true=y_test8)

# print(f"test_size={0.2} dev_size={0.7} train_size={0.1} train_acc={accuracy_train4:.2f} dev_acc={accuracy_valid4:.2f} test_acc={accuracy_test4:.2f}")
# print(f"test_size={0.2} dev_size={0.7} train_size={0.1} train_acc={accuracy_train6:.2f} dev_acc={accuracy_valid6:.2f} test_acc={accuracy_test6:.2f}")
# print(f"test_size={0.2} dev_size={0.7} train_size={0.1} train_acc={accuracy_train8:.2f} dev_acc={accuracy_valid8:.2f} test_acc={accuracy_test8:.2f}")

# Predict the value of the digit on the test subset
# predicted = model.predict(X_test)
# Predict the value of the digit on the test subset
# predicted = p_and_eval(model, X_test, y_test)
###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.
# print(
#     f"Classification report for classifier {model}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )

# ###############################################################################
# # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# # true digit values and the predicted digit values.

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

# plt.show()

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:


# The ground truth and predicted lists
# y_true = []
# y_pred = []
# cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
# for gt in range(len(cm)):
#     for pred in range(len(cm)):
#         y_true += [gt] * cm[gt][pred]
#         y_pred += [pred] * cm[gt][pred]

# print(
#     "Classification report rebuilt from confusion matrix:\n"
#     f"{metrics.classification_report(y_true, y_pred)}\n"
# )

parser = argparse.ArgumentParser()

parser.add_argument("--model_type",choices=["svm","tree","svm,tree"],default="svm",help="Model type")
parser.add_argument("--num_runs",type=int,default=3,help="Number of runs")
parser.add_argument("--test_size", type=float, default=0.2, help="test_size")
parser.add_argument("--dev_size", type=float, default=0.2, help="dev_size")

args = parser.parse_args()

results=[]
num_runs = args.num_runs
test_sizes = [args.test_size]
dev_sizes = [args.dev_size]
model_types = args.model_type

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

print(pd.DataFrame(results).groupby('model_type')[['train_acc', 'dev_acc','test_acc']].agg(['mean', 'std']).T)                