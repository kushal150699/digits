# Import datasets, classifiers and performance metrics
import matplotlib.pyplot as plt
import itertools
from sklearn import datasets, metrics, svm , tree
from sklearn.model_selection import train_test_split
from joblib import dump,load
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def read_digits():
    data = datasets.load_digits()
    X = data.images
    y = data.target
    return X, y

def get_all_h_param_comb_svm(gamma_list,c_list):
    return list(itertools.product(gamma_list, c_list))

def get_all_h_param_comb_tree(depth_list):
    return list(itertools.product(depth_list))


## function for data preprocessing
def data_preprocess(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data 

 
## Function for splitting data
def split_dataset(X, y, test_size, random_state = 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)

    return X_train, X_test, y_train, y_test 


## Function for training model
def train_model(x, y, model_params, model_type='svm'):
    if model_type == 'svm':
        clf = svm.SVC
    if model_type=='tree':
        clf = tree.DecisionTreeClassifier
    model = clf(**model_params)
    # pdb.set_trace()
    model.fit(x, y)
    return model 


def tune_hparams(X_train, y_train, X_dev, y_dev, all_combos,metric,model_type='svm'):
    best_accuracy = -1
    best_model=None
    best_hparams = None
    best_model_path=""

    for param in all_combos:
        if model_type=="Production_Model_svm":
            cur_model = train_model(X_train,y_train,{'gamma':param[0],'C':param[1]},model_type='svm')
        if model_type=="Candidate_Model_tree":
            cur_model = train_model(X_train,y_train,{'max_depth':param[0]},model_type='tree')    
        val_accuracy = p_and_eval(cur_model,metric,X_dev,y_dev)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_hparams=param
            best_model_path = "./models/{}_{}.joblib".format(model_type, param).replace(":", "_")
            best_model = cur_model
        
    dump(best_model,best_model_path) 
    # print("Model save at {}".format(best_model_path))   
    return best_hparams, best_model_path, best_accuracy     

def split_train_dev_test(X, y, test_size, dev_size):
    # Split data into test and temporary (train + dev) sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    
    # Calculate the ratio between dev and temp sizes
    dev_ratio = dev_size / (1 - test_size)
    
    # Split temporary data into train and dev sets
    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=dev_ratio, shuffle=True)
    
    return X_train, X_test, X_dev, y_train, y_test, y_dev

def p_and_eval(model,metric, X_test, y_test):
    # Predict the values using the model
    predicted = model.predict(X_test)

    # Visualize the first 4 test samples and show their predicted digit value in the title.
    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, prediction in zip(axes, X_test[:4], predicted[:4]):
    #     ax.set_axis_off()
    #     image = image.reshape(8, 8)
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #     ax.set_title(f"Prediction: {prediction}")

    # plt.show()

    # # Print the classification report
    # print(f"Classification report for classifier {model}:\n{classification_report(y_test, predicted)}\n")

    # # Plot the confusion matrix
    # disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}\n")

    # Rebuild the classification report from the confusion matrix
    # y_true = []
    # y_pred = []
    # cm = disp.confusion_matrix

    # for gt in range(len(cm)):
    #     for pred in range(len(cm)):
    #         y_true += [gt] * cm[gt][pred]
    #         y_pred += [pred] * cm[gt][pred]

    # print("Classification report rebuilt from confusion matrix:\n"
    #       f"{classification_report(y_true, y_pred)}\n")
    accuracy = metric(y_pred=predicted, y_true=y_test)
    return accuracy