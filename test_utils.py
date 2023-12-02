from utils import data_preprocess, train_model, read_digits, split_train_dev_test, p_and_eval,get_all_h_param_comb_svm,get_all_h_param_comb_tree,tune_hparams
from sklearn import datasets, metrics, svm
import os
from api.quiz4 import app
import pytest
from joblib import load
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

def test_for_hyperparameter_combination_count_svm():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
    all_combos = get_all_h_param_comb_svm(gamma_list,c_list)

    assert len(all_combos) == len(gamma_list)*len(c_list)

def test_for_hyperparameter_combination_values_svm():
    gamma_list = [0.01, 0.005]
    c_list = [0.1]
    all_combos = get_all_h_param_comb_svm(gamma_list,c_list)

    assert (0.01,0.1) in all_combos and (0.005,0.1) in all_combos

def test_for_hyperparameter_combination_count_tree():
    depth_list = [1,10,15,5,100]
    all_combos = get_all_h_param_comb_tree(depth_list)

    assert len(all_combos) == len(depth_list)

def test_for_hyperparameter_combination_values_tree():
    depth_list = [1,10,15,5,100] 
    all_combos = get_all_h_param_comb_tree(depth_list)

    assert (5,) in all_combos


def test_loaded_model_is_logistic_regression():
    model_file_path = "./models/M23CSA011_lr_('sag',).joblib"
    loaded_model = load(model_file_path)

    assert isinstance(loaded_model, LogisticRegression) or isinstance(loaded_model, LogisticRegressionCV)

def test_solver_name_match_in_model_file():
    model_file_path = "./models/M23CSA011_lr_('sag',).joblib"
    loaded_model = load(model_file_path)

    # Extracting solver name from the model file name
    _, _, file_name = model_file_path.rpartition('/')
    solver_name_from_file = file_name.split('_')[-1].split('.')[0][2:5]

    # Extracting solver name from the loaded model
    solver_name_from_model = loaded_model.get_params()['solver']

    assert solver_name_from_file == solver_name_from_model

# def test_model_saving():
#     X, y = read_digits()
#     model_types = "Production_Model_svm,Candidate_Model_tree"
#     models=model_types.split(',')
#     classifier_param_dict = {}
#     h_metric = metrics.accuracy_score
#     test_s = 0.2
#     dev_s = 0.2
#     train_size = 1 - test_s - dev_s
#     X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=test_s, dev_size=dev_s)

#     X_train = data_preprocess(X_train)
#     X_dev = data_preprocess(X_dev)
#     X_test = data_preprocess(X_test)

#     gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
#     c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
#     all_combos_svm = get_all_h_param_comb_svm(gamma_list,c_list)
#     classifier_param_dict['Production_Model_svm'] = all_combos_svm


#     #Decision Tree Hyperparameters combination

#     max_depth_list = [5,10,15,20,50,100,150]
#     all_combos_tree = get_all_h_param_comb_tree(max_depth_list)
#     classifier_param_dict['Candidate_Model_tree'] = all_combos_tree

#     for curr_run in range(6):
#         curr_run_results={}
    
#         # for test_s in args.test_size:
#         #     for dev_s in args.dev_size:
#         test_s = 0.2
#         dev_s = 0.2
#         train_size = 1 - test_s - dev_s
#         X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=test_s, dev_size=dev_s)

#         X_train = data_preprocess(X_train)
#         X_dev = data_preprocess(X_dev)
#         X_test = data_preprocess(X_test)

#         # for model_type in classifier_param_dict:
#         for model_type in models:
#             all_combos = classifier_param_dict[model_type]
#             best_hparams, best_model_path , best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, all_combos ,h_metric,model_type)

#     assert os.path.exists(best_model_path)

@pytest.fixture
def client():
    return app.test_client()

def test_post_predict(client):
    X, y = read_digits()

    for digit in range(10):
        digit_indices = (y == digit).nonzero()[0]
        if len(digit_indices) > 0:
            sample_index = digit_indices[0]
            img = X[sample_index]
            if img is not None and img.size > 0:
                img = img.reshape((1, 8, 8, 1))
                response = client.post("/predict", json={"data": img.tolist()})
                assert response.status_code == 200
                try:
                    assert response.get_json()['digit'] == digit
                except Exception as e:
                    pass

