import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.metrics import Precision, Recall, F1Score, AUC
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import Callback

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adadelta, Adafactor, Adagrad, Adam, AdamW, Adamax, Ftrl, Lion, Nadam, RMSprop, SGD   
import numpy as np
import os
import warnings

import time
import pickle

from tqdm import tqdm
from multiprocessing import Pool

import pandas as pd
from src.data.make_dataset import get_list_of_numerical_variables, get_data_by_fold
from src.models.model_experiment import train_test_folds
from src.models.model_experiment import get_output_folder

from pathlib import Path
from itertools import product

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, confusion_matrix

class TimeHistoryWithoutMetrics(Callback):
    """
    Callback to measure training time (excluding metrics)
    """
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_train_start = time.time()  # Record the start time of the epoch
        self.epoch_train_batch_time = 0.0  # Reset batch training time for this epoch
        self.batch_start_time = None

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()  # Start batch timer

    def on_train_batch_end(self, batch, logs=None):
        if self.batch_start_time is not None:
            self.epoch_train_batch_time += time.time() - self.batch_start_time
            self.batch_start_time = None  # Reset batch timer

    def on_epoch_end(self, epoch, logs=None):
        training_time_without_metrics = self.epoch_train_batch_time
        logs['time_train'] = training_time_without_metrics

class PredictionTimeAndMetricsHistory(Callback):
    """
    Metrics from skleran are used as metrics from the tensorflow are only estimates
    and are not calculated on the whole dataset but a batch trained. The recommendation
    is to calculate the metrics and times on each epoch end. 
    """
    def __init__(self, train_data, test_data):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        # Time predictions on training data
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        
        start_train_pred = time.time()
        raw_predictions_train = self.model.predict(X_train, verbose = 0)
        y_train_pred = (raw_predictions_train > 0.5).astype(int)
        train_pred_time = time.time() - start_train_pred

        raw_predictions_test = self.model.predict(X_test, verbose = 0)
        y_test_pred = (raw_predictions_test > 0.5).astype(int)
        
        logs['accuracy_score_train']= round(float(accuracy_score(y_train, y_train_pred)),8)
        logs['f1_score_train']= round(float(f1_score(y_train, y_train_pred)),8)
        logs['precision_score_train']= round(float(precision_score(y_train, y_train_pred, zero_division = 0.0)),8)
        logs['recall_score_train']= round(float(recall_score(y_train, y_train_pred)),8)
        logs['roc_auc_score_train']= round(float(roc_auc_score(y_train, y_train_pred)),8)

        logs['accuracy_score_test']= round(float(accuracy_score(y_test, y_test_pred)),8)
        logs['f1_score_test']= round(float(f1_score(y_test, y_test_pred)),8)
        logs['precision_score_test']= round(float(precision_score(y_test, y_test_pred, zero_division = 0.0)),8)
        logs['recall_score_test']= round(float(recall_score(y_test, y_test_pred)),8)
        logs['roc_auc_score_test']= round(float(roc_auc_score(y_test, y_test_pred)),8)
      
        logs['time_pred'] = train_pred_time

class PickleSizeCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Serialize the model to a byte stream using pickle
        serialized_model = pickle.dumps(self.model)
        logs['model_size'] = len(serialized_model)

class ModelSparsityCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        total_weights = sum(weight.numpy().size for weight in self.model.weights)
        non_zero_weights = sum(tf.math.count_nonzero(weight).numpy() for weight in self.model.weights)
        logs['model_layers'] = len(self.model.layers)
        logs['model_weights'] = total_weights
        logs['model_parameters_non_zero'] = non_zero_weights
        logs['model_sparsity'] = round(100 - non_zero_weights / total_weights * 100, 2)
        
def get_lst_train_test():
    """
    Gets the train test list with a correct split
    """
    csv_data = 'data/processed/german-credit-data/german.csv'
    lst_numeric = get_list_of_numerical_variables()
    df = pd.read_csv(csv_data, dtype={x:float for x in lst_numeric})

    dic_folds = dict()
    for fold_n in range(1,11):
        dic_folds[fold_n] = get_data_by_fold(f'fold{fold_n:02d}')
    lst_train_test = train_test_folds(df, dic_folds, lst_numeric)
    
    total_memory = sum(
        df_train.memory_usage(deep=True).sum() + ser_train.memory_usage(deep=True) +
        df_test.memory_usage(deep=True).sum() + ser_test.memory_usage(deep=True)
        for df_train, ser_train, df_test, ser_test in lst_train_test
    )
    print(f"Total memory allocated: {total_memory / (1024 ** 2):.2f} MB")
    return lst_train_test

def train_fold(args):
    X_train, y_train, X_test, y_test, parameters, optimizer, n_model, fold_n = args
    print(f'\n model: {n_model}, fold {fold_n} started...')
    n_dims = X_train.shape[1]
    n_epochs = 10
    print(f'\n model: {n_model}, fold {fold_n} make model...')
    
    tf.keras.backend.clear_session()
    model = tf_make_model(n_dims, parameters, optimizer)
    lst_callbacks = [
            TimeHistoryWithoutMetrics(),
            PredictionTimeAndMetricsHistory((X_train, y_train), (X_test, y_test)),
            PickleSizeCallback(),
            ModelSparsityCallback(),]
    print(f'\n model: {n_model}, fold {fold_n} train model...')
    history = model.fit(X_train, y_train, epochs=n_epochs, 
            validation_data = [X_test, y_test], 
            callbacks = lst_callbacks,               
            verbose = 0,
            batch_size = None)
    df_history = pd.DataFrame({'model_num': n_model, 'fold': fold_n, 'epoch' : list(range(1, n_epochs+1))} | history.history)
    
    print(f'\n model: {n_model}, fold {fold_n} finished.')
    return df_history

def tf_experiment_all_folds(lst_train_test, arguments:tuple) -> tuple:
    """
    Performs the same experiment on the set of folds. As a paramenter
    gets the list of data and a tuple with model num and model compile
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU, use CPU only
    n_model, n_dims, parameters, optimizer = arguments
    lst_experiments = []

    # List of arguments to be executed for folds in parallel
    fold_args = [(X_train, y_train, X_test, y_test, parameters, optimizer, n_model, fold_n) for fold_n, (X_train, y_train, X_test, y_test) in enumerate(lst_train_test)]
    
    print(f'tf_experiment_all_folds {n_model}, fold args: {len(fold_args)}')
   
    with Pool(processes=10) as pool:
        results = pool.imap_unordered(train_fold, fold_args)
        lst_experiments = list(results)

    df = pd.concat(lst_experiments, axis=0)
    dic_means = df.loc[df['epoch'] == 10,:].mean().round(8).to_dict()
    dic_means_out = {'model_num':n_model} | {'optimizer':optimizer} | parameters | dic_means
    return df, dic_means_out
       
def gen_tf_model_parameters():
    """
    Generates list of models to be checked
    """
    lst_optimizers = ['Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Ftrl', 'SGD']   
    lst_actication_functions = ['relu', 'elu', 'sigmoid', 'softmax',  'tanh',]
    lst_regulizers = [None, 'l1', 'l2']

    dic_dense_input = {'units': [2**i for i in range(8, 0, -1)],  'activation':lst_actication_functions, 'kernel_regularizer':lst_regulizers}
    keys = dic_dense_input.keys()
    values = dic_dense_input.values()
    # Create a list of dictionaries for each combination
    lst_combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    # Model defined again
    for combination in lst_combinations:
        for optimizer in lst_optimizers:
            yield combination, optimizer

def tf_make_model_dnn1(n_dims, parameters, optimizer):
    """
    Returns model with parameters and optimizer
    """
    model = Sequential([
            Input(shape=(n_dims,), name='input_layer'),
            Dense(**parameters, name='dense_1'),
            Dense(1, activation='sigmoid', name='output_layer')
        ])
    model.compile(optimizer = optimizer,
                loss='binary_crossentropy',
                )
    return model


def tf_make_model_dnn2(n_dims, parameters, optimizer):
    """
    Returns model with parameters and optimizer
    """
    model = Sequential([
            Input(shape=(n_dims,), name='input_layer'),
            Dense(**parameters, name='dense_1'),
            Dense(**parameters, name='dense_2'),
            Dense(1, activation='sigmoid', name='output_layer')
        ])
    model.compile(optimizer = optimizer,
                loss='binary_crossentropy',
                )
    return model

def tf_make_model_dnn3(n_dims, parameters, optimizer):
    """
    Returns model with parameters and optimizer
    """
    model = Sequential([
            Input(shape=(n_dims,), name='input_layer'),
            Dense(**parameters, name='dense_1'),
            Dense(**parameters, name='dense_2'),
            Dense(**parameters, name='dense_3'),
            Dense(1, activation='sigmoid', name='output_layer')
        ])
    model.compile(optimizer = optimizer,
                loss='binary_crossentropy',
                )
    return model


def tf_run_experiments(dnn='dnn1'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU, use CPU only    
    warnings.filterwarnings("ignore", message=".*CUDA_ERROR_NO_DEVICE.*")
    lst_train_test = get_lst_train_test()
    n_dims = lst_train_test[0][0].shape[1]

    global tf_make_model
    if dnn == 'dnn1':   
        tf_make_model = tf_make_model_dnn1
    elif dnn == 'dnn2':  
        tf_make_model = tf_make_model_dnn2
    else:
        tf_make_model = tf_make_model_dnn3
        
    lst_epochs = []
    lst_models = []
    dir_output = get_output_folder(dnn)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    
    n_folds_file = 1 
    num_csv_file_records = 10
    for n_model, (parameters, optimizer) in enumerate(gen_tf_model_parameters(), start=1):
      
        df_epochs, dic_means = tf_experiment_all_folds(lst_train_test, (n_model, n_dims, parameters, optimizer))
        lst_models.append(dic_means)
        lst_epochs.append(df_epochs)

        if n_model % num_csv_file_records == 0:
            df_models = pd.DataFrame.from_records(lst_models).astype({'model_num':int})
            df_models.to_csv(dir_output / 'means.csv', float_format='%.8f' , index=False)
            
            df_epochs = pd.concat(lst_epochs, axis=0)
            df_epochs.to_csv(dir_output / f'{n_folds_file:05d}.csv', float_format='%.8f' , index=False)
            n_folds_file += 1        
            lst_epochs = []

        print(f'Model number {n_model} finished')

    df_models = pd.DataFrame.from_records(lst_models).astype({'model_num':int})
    df_models.to_csv(dir_output / 'means.csv', float_format='%.8f' , index=False)

if __name__ == "__main__":
    tf_run_experiments('dnn1')
    
