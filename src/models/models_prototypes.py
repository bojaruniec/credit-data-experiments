import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.metrics import Precision, Recall, F1Score, AUC
from tensorflow.keras.regularizers import l1, l2, l1_l2

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adadelta, Adafactor, Adagrad, Adam, AdamW, Adamax, Ftrl, Lion, Nadam, RMSprop, SGD   
import numpy as np

import time
import pickle
from tensorflow.keras.callbacks import Callback

import pandas as pd
from src.data.make_dataset import get_list_of_numerical_variables, get_data_by_fold
from src.models.model_experiment import train_test_folds
from src.models.model_experiment import get_output_folder

from pathlib import Path
from itertools import product

tf.config.set_visible_devices([], 'GPU')
# wyłączenie GPU

# Callback to measure training time (excluding metrics)
class TimeHistoryWithoutMetrics(Callback):
    # def on_train_begin(self, logs=None):
    #     self.epoch_train_times = []  # List to store training times for each epoch

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

class PredictionTimeHistory(Callback):
    def __init__(self, train_data):
        super().__init__()
        self.train_data = train_data

    def on_epoch_end(self, epoch, logs=None):
        # Time predictions on training data
        start_train_pred = time.time()
        self.model.predict(self.train_data, verbose=0)
        train_pred_time = time.time() - start_train_pred

        # Time predictions on validation data
        # start_val_pred = time.time()
        # self.model.predict(self.val_data, verbose=0)
        # val_pred_time = time.time() - start_val_pred

        # Store the prediction times
        logs['time_pred'] = train_pred_time

class PickleSizeCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Serialize the model to a byte stream using pickle
        serialized_model = pickle.dumps(self.model)
        logs['model_size_bytes'] = len(serialized_model)

class ModelSparsityCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        total_weights = sum(weight.numpy().size for weight in self.model.weights)
        non_zero_weights = sum(tf.math.count_nonzero(weight).numpy() for weight in self.model.weights)
        logs['model_parameters_non_zero'] = non_zero_weights
        logs['model_sparsity'] = round(100 - non_zero_weights / total_weights * 100, 2)
        



print(tf.__version__)

csv_data = 'data/processed/german-credit-data/german.csv'
lst_numeric = get_list_of_numerical_variables()
df = pd.read_csv(csv_data, dtype={x:float for x in lst_numeric})


dic_folds = dict()
for fold_n in range(1,11):
    dic_folds[fold_n] = get_data_by_fold(f'fold{fold_n:02d}')
    
lst_train_test = train_test_folds(df, dic_folds, lst_numeric)

train_time_callback = TimeHistoryWithoutMetrics()
prediction_time_callback = PredictionTimeHistory(lst_train_test[0][0])
model_size_callback = PickleSizeCallback()
model_sparsity_callback = ModelSparsityCallback()

n_dims = lst_train_test[0][0].shape[1]

def tf_model_summary(model) -> dict:
    """
    Gets the description of a model based on the model object. This will
    be saved in the model list
    """
    dic_summary  = dict()
    dic_summary['model_class'] = model.__class__.__name__ # e.g. Sequential
    dic_summary['optimalizer'] = model.optimizer.name
    dic_summary['learning_rate'] = float(model.optimizer.learning_rate.numpy())
    dic_summary['layers_count_all'] = len(model.layers)
    dic_summary['layers_count_dense'] = len([layer for layer in model.layers if isinstance(layer, Dense)])
    dic_summary['layers_count_dropout'] = len([layer for layer in model.layers if isinstance(layer, Dropout)])
    dic_summary['layers_count_all'] = len(model.layers)
    dic_summary['parameters_trainable'] = model.count_params()
    dic_summary['parameters_non_trainable'] = sum(K.count_params(w) for w in model.non_trainable_weights)
    dic_summary['layers_types'] = ";".join([layer.__class__.__name__ for layer in model.layers])
    dic_summary['layers_output'] = ";".join([f'{layer.output.shape[1]}' for layer in model.layers])
    dic_summary['layers_activation'] = ";".join([layer.activation.__name__ if isinstance(layer, Dense) else  '' for layer in model.layers])
    dic_summary['layers_regularizer'] = ";".join([layer.kernel_regularizer.__class__.__name__ if (isinstance(layer, Dense) & (layer.kernel_regularizer is not None)) else  '' for layer in model.layers])
    return dic_summary


# What is needed for each fold:
#
# model_num,fold,time_train,
# model_size,time_pred,
# accuracy_score_train,f1_score_train,precision_score_train,recall_score_train,roc_auc_score_train,
# accuracy_score_test,f1_score_test,precision_score_test,recall_score_test,roc_auc_score_test
#
# In the model description:
# model_num,model_class,
# optimizer
# loss
# input,dense1,early_stopping,eta0,fit_intercept,l1_ratio,max_iter,n_iter_no_change,n_jobs,penalty,random_state,shuffle,tol,validation_fraction,verbose,warm_start,fold,time_train,model_size,time_pred,accuracy_score_train,f1_score_train,precision_score_train,recall_score_train,roc_auc_score_train,accuracy_score_test,f1_score_test,precision_score_test,recall_score_test,roc_auc_score_test
# 169000.0,Perceptron,0.0001,,False,1.3,True,0.75,301,5,1,elasticnet,20,True,0.001,0.2,0,False,5.0,0.01904451,1935.0,0.00288757,0.73975309,0.82075023,0.79215502,0.86243386,0.6579659,0.71444444,0.80169944,0.77234057,0.84603175,0.62671958

def tf_experiment_all_folds(lst_train_test, model_num_and_model:tuple) -> tuple:
    """
    Performs the same experiment on the set of folds. As a paramenter
    gets the list of data and a tuple with model num and model compile
    """
    model_num, model = model_num_and_model
    lst_experiments = []
    
    dic_columns_rename = {
    'accuracy':	'accuracy_score_train',
    'auc': 'roc_auc_score_train',
    'f1_score':	'f1_score_train',
    'precision': 'precision_score_train',
    'recall': 'recall_score_train',
    'loss': 'loss_train',
    'val_accuracy': 'accuracy_score_test',
    'val_auc': 'roc_auc_score_test',
    'val_f1_score': 'f1_score_test',
    'val_precision': 'precision_score_test',
    'val_recall': 'recall_score_test',
    'val_loss': 'loss_train'
    }
    
    n_epochs = 10
    for fold_n, (X_train, y_train, X_test, y_test) in enumerate(lst_train_test):
        tf.keras.backend.clear_session()
        tf.random.set_seed(32)
        history = model.fit(X_train, y_train, epochs=n_epochs, 
                validation_data=[X_test, y_test], 
                callbacks=[train_time_callback, prediction_time_callback, model_size_callback, model_sparsity_callback])
        df_history = pd.DataFrame({'model_num': model_num, 'fold': fold_n, 'epoch' : list(range(1, n_epochs+1))} | history.history)
        lst_experiments.append(df_history)
    df = pd.concat(lst_experiments, axis=0)
    df.rename(columns=dic_columns_rename, inplace=True)
    
    dic_means = df.loc[df['epoch'] == 10,:].mean().round(8).to_dict()
    return df, dic_means
       
def gen_tf_model_parameters():
    """
    Generates list of models to be checked
    """
    lst_optimizers = [Adadelta(), Adafactor(), Adagrad(), Adam(), Adamax(), Ftrl(), SGD()]   
    
    lst_actication_functions = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 
                                'selu', 'elu', 'exponential', 'leaky_relu', 'relu6', 'silu', 'hard_silu',
                                'gelu', 'hard_sigmoid', 'mish', 'log_softmax']
    lst_regulizers = [None, 'l1', 'l2']
    dic_dense_input = {'units': [2**i for i in range(1,8)],  'activation':lst_actication_functions, 'kernel_regularizer':lst_regulizers}
    keys = dic_dense_input.keys()
    values = dic_dense_input.values()
    # Create a list of dictionaries for each combination
    lst_combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    # Model defined again
    tf.random.set_seed(32)
    tf.keras.backend.clear_session()
    for combination in lst_combinations:
        tf.keras.backend.clear_session()
        tf.random.set_seed(32)
        model = Sequential([
            Input(shape=(n_dims,), name='input_layer'),
            Dense(**combination, name='dense_1'),
            Dense(1, activation='sigmoid', name='output_layer')
        ])
        for optimizer in lst_optimizers:
            model.compile(optimizer = optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', Precision(), Recall(), F1Score(), AUC()])
            yield model


def tf_run_experiments():
    lst_epochs = []
    lst_models = []
    dir_output = get_output_folder('dnn1')
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    
    n_folds_file = 0 
    num_csv_file_records = 1_000
    for n_model, model in enumerate(gen_tf_model_parameters(), start=1):
        print(n_model)
        df_epochs, dic_means = tf_experiment_all_folds(lst_train_test, (n_model, model))
        dic_model = tf_model_summary(model)
        lst_models.append({'model_num':n_model}|dic_model|dic_means)
        lst_epochs.append(df_epochs)

        if n_model % num_csv_file_records == 0:
            df_epochs = pd.concat(lst_epochs, axis=0)
            df_epochs.to_csv(dir_output / f'{n_folds_file:05d}.csv', float_format='%.8f' , index=False)
            n_folds_file += 1        
            lst_epochs = []

    df_models = pd.DataFrame.from_records(lst_models).astype({'model_num':int})
    df_models.to_csv(dir_output / 'means.csv', float_format='%.8f' , index=False)

if __name__ == "__main__":
    tf_run_experiments()
    
