from utils import get_all_h_param_comb

def test_for_hyperparameter_combination_count():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
    all_combos = get_all_h_param_comb(gamma_list,c_list)

    assert len(all_combos) == len(gamma_list)*len(c_list)

def test_for_hyperparameter_combination_values():
    gamma_list = [0.01, 0.005]
    c_list = [0.1]
    all_combos = get_all_h_param_comb(gamma_list,c_list)

    assert (0.01,0.1) in all_combos and (0.005,0.1) in all_combos