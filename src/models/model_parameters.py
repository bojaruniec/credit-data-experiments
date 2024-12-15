import json
import numpy as np
from itertools import product, chain
from pathlib import Path
def models_and_parameters(n_test=100):

    dir_out_path = Path('data/processed/german-credit-data')
    csv_out_path = dir_out_path / 'german_model_parameters.json'
    
    dic_models = dict()
    dic_models['knn'] = dict()
    dic_models['knn']['model'] = 'KNeighborsClassifier'
    dic_models['knn']['parameters'] = {'leaf_size': range(1,51),
                                        'n_neighbors': range(1, min(101, n_test // 2 +1)), 
                                        'algorithm':('auto','ball_tree','kd_tree','brute'),
                                        'p': range(1,3),
                                        'weights': ('uniform','distance')}
    return dic_models
    
    dic_models['knn_5'] = dict()
    dic_models['knn_5']['model'] = 'KNeighborsClassifier'
    dic_models['knn_5']['parameters'] = {'leaf_size': [30], # wartość domyślna
                                        'n_neighbors': chain(range(1,11), [20,30,50,75,100, n_test // 2 +1]), 
                                        'algorithm':['ball_tree','kd_tree','brute'],
                                        'p':[1,2],
                                        'weights': ['uniform', 'distance']}


    dic_models['decision_tree'] = dict()
    dic_models['decision_tree']['model'] = 'DecisionTreeClassifier'
    dic_models['decision_tree']['parameters'] = {'max_leaf_nodes': range(2,51),
                                                'splitter' : ['best','random'],
                                                'random_state' : [20,21],
                                                'criterion': ['gini','entropy','log_loss'],
                                                'min_impurity_decrease' :[0,0.01,0.1]}

    dic_models['logistic_regression'] = dict()
    dic_models['logistic_regression']['model'] = 'LogisticRegression'
    dic_parameters = {'penalty': ['l1', 'l2', 'elasticnet', None],
                          'dual': [False, True],
                          'tol': [1e-6, 1e-4, 1e-2],
                          'C' : [0.5, 1.0, 1.5, 2],
                          'fit_intercept' : [True, False],
                          'intercept_scaling' : [1],
                          'class_weight' : [None, 'balanced'],
                          'random_state' : [None, 20, 21],
                          'solver' :['lbfgs' ,'liblinear','newton-cg', 'newton-cholesky', 'sag', 'saga'],
                          'warm_start' : [False],
                          'l1_ratio': [None, 0, 0.25, 0.50, 0.75, 1.0]}
    
    def f_lr_kryteria(x):
        if x['dual']:
            if x['penalty'] == 'l2' and x['solver'] == 'liblinear':
                pass
            else:
                return False
        
        if x['penalty'] is None:
            if x['C'] != 1:
                return False
            if x['l1_ratio'] is not None:
                return False
            if x['solver'] == 'liblinear':
                return False
            
        elif x['penalty'] == 'l1':
            if x['l1_ratio'] is not None:
                return False
            
            if x['solver'] not in ['liblinear', 'saga']:
                return False
        
        elif x['penalty'] == 'l2':
            if x['l1_ratio'] is not None:
                return False
        elif x['penalty'] == 'elasticnet':
            if x['l1_ratio'] is None:
                return False
            if x['solver'] not in ['saga']:
                return False

        if x['solver'] in ['sag', 'saga', 'liblinear']:
            if x['random_state'] is None:
                return False
        else:
            if x['random_state'] is not None:
                return False
        return True

    lst_parameters0 = [dict(zip(dic_parameters.keys(), combination)) for combination in product(*dic_parameters.values())]
    dic_models['logistic_regression']['parameters']  = list(filter(f_lr_kryteria, lst_parameters0))

    # RandomForest
    dic_models['random_forest'] = dict()
    dic_models['random_forest']['model'] = 'RandomForestClassifier'
    dic_parameters = {'n_estimators' : chain(range(1,11), range(0,101,10)), 
                      'criterion':['gini', 'entropy', 'log_loss'], 
                      'max_depth': [None], #to jest to, co będzie z automatu i zefiniuje mi liczbę  
                      'min_samples_split':[2], 
                      'min_samples_leaf':range(1,11), 
                      'min_weight_fraction_leaf':[0.0], 
                      'max_features':['sqrt', 'log2', None], 
                      'max_leaf_nodes': [None], #to jest to, co będzie z automatu i zefiniuje mi liczbę  
                      'min_impurity_decrease': [0.0, 0.25, 0.5, 0.75, 1], 
                      'bootstrap':[False, True], 
                      'oob_score':[False], 
                      'n_jobs':[None], 
                      'random_state':[20,21], 
                      'verbose':[0], 
                      'warm_start':[False], 
                      'class_weight':['balanced', 'balanced_subsample', None], 
                      'ccp_alpha':[0.0, 0.25, 0.5, 0.75, 1], 
                      'max_samples': [None] + [float(x) for x in np.arange(0.1, 1.2 ,0.2)], 
                      'monotonic_cst':[None]}

    def f_rf_kryteria(x):
        if not x['bootstrap'] and x['max_samples'] is not None:
            return False
        else:
            pass
        return True

    lst_parameters0 = [dict(zip(dic_parameters.keys(), combination)) for combination in product(*dic_parameters.values())]
    dic_models['random_forest']['parameters']  = list(filter(f_rf_kryteria, lst_parameters0))

    return dic_models

if __name__ == "__main__":
    dic_models_and_parameters = models_and_parameters(n_test=100)
    dic_models_and_parameters.keys()
    dic_models_and_parameters['knn_5']['parameters']
    dic_models_and_parameters['logistic_regression']['parameters']
    dic_models_and_parameters['random_forest']['parameters']
    