import numpy as np
from itertools import product, chain
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import  LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier

def models_and_parameters(model_key='knn', n_test=100):

    dic_model_and_parameters = dict()
    # dic_model_and_parameters['model'] = ''
    # dic_model_and_parameters['parameters'] = dict()
    dic_model_and_parameters['filter'] = None
    match model_key:
        case 'knn':
            dic_model_and_parameters['model'] = KNeighborsClassifier
            dic_model_and_parameters['parameters'] = {'leaf_size': range(1,51),
                                                'n_neighbors': range(1, min(101, n_test // 2 +1)), 
                                                'algorithm':('auto','ball_tree','kd_tree','brute'),
                                                'p': range(1,3),
                                                'weights': ('uniform','distance')}
        case 'knn-5':
            dic_model_and_parameters['model'] = KNeighborsClassifier
            dic_model_and_parameters['parameters'] = {'leaf_size': [30], # wartość domyślna
                                                'n_neighbors': chain(range(1,11), [20,30,50,75,100, n_test // 2 +1]), 
                                                'algorithm':['ball_tree','kd_tree','brute'],
                                                'p':[1,2],
                                                'weights': ['uniform', 'distance']}
        case 'decision-tree':
            dic_model_and_parameters['model'] = DecisionTreeClassifier
            dic_model_and_parameters['parameters'] = {'max_leaf_nodes': range(2,51),
                                                        'splitter' : ['best','random'],
                                                        'random_state' : [20,21],
                                                        'criterion': ['gini','entropy','log_loss'],
                                                        'min_impurity_decrease' :[0,0.01,0.1]}
            
            
    
        case 'logistic-regression':
            dic_model_and_parameters['model'] = LogisticRegression
            dic_model_and_parameters['parameters'] = {'penalty': ['l1', 'l2', 'elasticnet', None],
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
            
            dic_model_and_parameters['filter'] = f_lr_kryteria
            
        case 'random-forest':
            dic_model_and_parameters['model'] = RandomForestClassifier
            dic_model_and_parameters['parameters']  = {'n_estimators' : chain(range(1,10,2), range(50,101,15)), 
                      'criterion':['gini', 'entropy', 'log_loss'], 
                      'max_depth': [None], #to jest to, co będzie z automatu i zefiniuje mi liczbę  
                      'min_samples_split':[2], 
                      'min_samples_leaf': range(1,11,2), 
                      'min_weight_fraction_leaf':[0.0], 
                      'max_features':['sqrt', 'log2', None], 
                      'max_leaf_nodes': [None], #to jest to, co będzie z automatu i zefiniuje mi liczbę  
                      'min_impurity_decrease': np.arange(0, 1, 0.33), 
                      'bootstrap':[False, True], 
                      'oob_score':[False], 
                      'n_jobs':[None], 
                      'random_state':[20, 21], 
                      'verbose':[0], 
                      'warm_start':[False], 
                      'class_weight':['balanced', 'balanced_subsample', None], 
                      'ccp_alpha':[0.0, 0.25, 0.5, 0.75, 1], 
                      'max_samples': [None] + [round(float(x),2) for x in np.arange(0.1, 1, 0.3)], 
                      'monotonic_cst':[None]}
            
            def f_rf_kryteria(x):
                if not x['bootstrap'] and x['max_samples'] is not None:
                    return False
                else:
                    pass
                return True
            
            dic_model_and_parameters['filter'] = f_rf_kryteria
            
        case 'perceptron':
            dic_model_and_parameters['model'] = Perceptron
            dic_model_and_parameters['parameters']  = {
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'alpha' : [0.0001, 0.01, None],
                    'l1_ratio': np.arange(0,1, 0.25), 
                    'max_iter': np.arange(1, 1000, 100), 
                    'tol': [None, 1e-3, 1e-6, 1e-1],
                    'shuffle': [True, False], 
                    'verbose':[0], 
                    'eta0':[0.5, 1, 1.3], 
                    'n_jobs' : [1],
                    'random_state': [20, 21],   
                    'early_stopping': [True, False], 
                    'validation_fraction':[0.1, 0.2], 
                    'n_iter_no_change': [2, 5, 10],
                    'class_weight' : [None, 'balanced'],
                    'warm_start':[True, False],}
            
            def f_perceptron_kryteria(x):
                if x['penalty'] is not None and x['alpha'] is None:
                    return False
                elif x['penalty'] is None and x['alpha'] is not None:
                    return False
                else:
                    pass
                
                if x['penalty'] != 'elasticnet' and x['l1_ratio'] > 0:
                    return False
                else:
                    pass
                
                
                return True
            
            dic_model_and_parameters['filter'] = f_perceptron_kryteria
            

    if dic_model_and_parameters['filter'] is None:
        for combination in product(*dic_model_and_parameters['parameters'].values()):
            dic_combination = dict(zip(dic_model_and_parameters['parameters'].keys(), combination))
            print(dic_model_and_parameters['model'])
            yield dic_model_and_parameters['model'](**dic_combination)
    else:
        for combination in product(*dic_model_and_parameters['parameters'].values()):
            dic_combination = dict(zip(dic_model_and_parameters['parameters'].keys(), combination))
            if dic_model_and_parameters['filter'](dic_combination):
                yield dic_model_and_parameters['model'](**dic_combination)
                

if __name__ == "__main__":
    dic_models_and_parameters = models_and_parameters('knn', n_test=100)
    lst= list(dic_models_and_parameters)
    print(len(lst))
    
    dic_models_and_parameters = models_and_parameters('random-forest', n_test=100)
    lst= list(dic_models_and_parameters)
    next(dic_models_and_parameters)

    