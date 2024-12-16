import pandas as pd
import time
import pickle
import csv
import warnings

from tqdm import tqdm
from multiprocessing import Pool

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from pathlib import Path
from datetime import datetime
from itertools import product, islice
from functools import partial

from src.data.make_dataset import get_list_of_numerical_variables, get_data_by_fold
from src.models.model_parameters import  models_and_parameters

def get_parameters_gen(dic_parameters):
    """ Changes dictionary to generate combination of parameters to list of dictionaries
    """
    if isinstance(dic_parameters, dict):
        for combination in product(*dic_parameters.values()):
            yield dict(zip(dic_parameters.keys(), combination))
    else:
        for combination in dic_parameters:
            yield combination

def get_output_folder(model_key) -> Path:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    dir_output = Path(f'reports/german-credit-data/{model_key}/{timestamp}')
    return dir_output

def write_experiments_by_fold_to_csv(lst_folds:list, model_str_num:str, dir_output:Path):
    if len(lst_folds) == 0:
        return
    
    csv_file = dir_output / f'{model_str_num}.csv'
    fieldnames = lst_folds[0].keys()
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lst_folds)

def experiment_all_folds(lst_train_test, model_num_and_model:tuple) -> tuple:
    """
    Performs the same experiment on the set of folds
    """
    model_num, model = model_num_and_model
    lst_output_folds = []
    lst_experiments = []
    for fold_n, (X_train, y_train, X_test, y_test) in enumerate(lst_train_test):
        dic_experiment = experiment_one_fold(model, X_train, X_test, y_train, y_test)
        lst_experiments.append({'model_num': model_num, 'fold':fold_n,}|dic_experiment)
    dic_means = pd.DataFrame.from_dict(lst_experiments).mean().round(8).to_dict()
    
    dic_means_to_return = {'model_num': int(model_num)} | {'model_class': model.__class__.__name__} | dict(model.get_params()) | dic_means
    lst_experiments_output = [{'model_num': int(model_num), 'fold':fold_n,} | dic_x for fold_n, dic_x in enumerate(lst_experiments)]
    return (dic_means_to_return, lst_experiments_output)

def train_test_folds(df, dic_folds, lst_numeric):
    """
    returs the list of tuples for train test split
    """
    lst_output = []
    for fold_n, (lst_idx_train, lst_idx_test) in dic_folds.items():
        df_train = df.iloc[lst_idx_train,:]
        df_test = df.iloc[lst_idx_test,:]
        y_train = df_train.pop('target')
        y_test = df_test.pop('target')
        scaler = MinMaxScaler()
        df_train.loc[:,lst_numeric] = scaler.fit_transform(df_train[lst_numeric])
        df_test.loc[:,lst_numeric] = scaler.transform(df_test[lst_numeric])
        lst_output.append((df_train, y_train, df_test, y_test))
    return lst_output
    
def experiment_one_fold(model, X_train, X_test, y_train, y_test):
    """ Performs one experiment according to the model and paramentesr
    
    """
    dic_experiment = dict()
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    time_train = end_time - start_time 
    dic_experiment['time_train'] = round(time_train,10)

    serialized_model = pickle.dumps(model)
    dic_experiment['model_size'] = len(serialized_model)

    # mierzymy czas predykcji dla danych treningowych
    start_time = time.time()    
    y_train_pred = model.predict(X_train)
    end_time = time.time()
    time_pred = end_time - start_time
    dic_times = dict()
    dic_experiment['time_pred'] = round(time_pred,10)
    
    y_test_pred = model.predict(X_test)
    dic_variables = dict()
    dic_variables['train'] = (y_train, y_train_pred)
    dic_variables['test'] = (y_test, y_test_pred)
    # wynik eksperymentu zostanie dodany na samym końcu
    dic_metrics_to_add = dict()
    dic_metrics_to_csv = dict()
    lst_metrics = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score]
    for train_test in ['train', 'test']:
        dic_metrics_to_add[train_test] = dict()
        for metric in lst_metrics:

            if metric.__name__ == 'precision_score':
                met_value = round(float(metric(*dic_variables[train_test], zero_division = 0.0)),8)
            else:
                met_value = round(float(metric(*dic_variables[train_test])),8)
            dic_experiment.update({f'{metric.__name__}_{train_test}':met_value})
    return(dic_experiment)

def german_experiment(model_key = 'knn', max_params = 100):
    """ Experiment of German Credit Data with multiprocess
    """
    
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    csv_data = 'data/processed/german-credit-data/german.csv'
    lst_numeric = get_list_of_numerical_variables()
    df = pd.read_csv(csv_data, dtype={x:float for x in lst_numeric})
    
    dic_folds = dict()
    for fold_n in range(1,11):
        dic_folds[fold_n] = get_data_by_fold(f'fold{fold_n:02d}')
        
    lst_train_test = train_test_folds(df, dic_folds, lst_numeric)

    gen_models = models_and_parameters(model_key, n_test = 100)
    if max_params is None:
        lst_models = [(model_num, model) for model_num, model in enumerate(gen_models)]
    else:
        lst_models = [(model_num, model) for model_num, model in enumerate(gen_models) if model_num < max_params]
    experiment_all_folds_partial = partial(experiment_all_folds, lst_train_test)
   
    lst_folds_all = []
    lst_means_all = []
    n_folds_file = 1
    
    dir_output_model = get_output_folder(model_key)
    dir_output_model.mkdir(parents=True, exist_ok=True)
    
    with Pool() as pool:
        results = pool.imap_unordered(experiment_all_folds_partial, lst_models, chunksize=1_000)
        for result_n, (dic_means, lst_experiments) in enumerate(tqdm(results, total=len(lst_models)), start=1):
            lst_means_all.append(dic_means)
            lst_folds_all += lst_experiments

            if result_n % 1_000 == 0:
                write_experiments_by_fold_to_csv(lst_folds_all, f"{n_folds_file:05d}", dir_output_model)
                lst_folds_all = []
                n_folds_file += 1
        write_experiments_by_fold_to_csv(lst_folds_all, f"{n_folds_file:05d}", dir_output_model)
        write_experiments_by_fold_to_csv(lst_means_all, "means", dir_output_model)
        print(dir_output_model)
        
if __name__ == "__main__":
    german_experiment()