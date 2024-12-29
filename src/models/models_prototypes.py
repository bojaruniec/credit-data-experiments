import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.metrics import Precision, Recall, F1Score, AUC

import time
import pickle
from tensorflow.keras.callbacks import Callback

import pandas as pd
from src.data.make_dataset import get_list_of_numerical_variables, get_data_by_fold
from src.models.model_experiment import train_test_folds

import time
from tensorflow.keras.callbacks import Callback

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
    def __init__(self, train_data, val_data):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.prediction_times = {
            'train': [],
            'val': []
        }

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


print(tf.__version__)

csv_data = 'data/processed/german-credit-data/german.csv'
lst_numeric = get_list_of_numerical_variables()
df = pd.read_csv(csv_data, dtype={x:float for x in lst_numeric})


dic_folds = dict()
for fold_n in range(1,11):
    dic_folds[fold_n] = get_data_by_fold(f'fold{fold_n:02d}')
    
lst_train_test = train_test_folds(df, dic_folds, lst_numeric)


train_time_callback = TimeHistoryWithoutMetrics()
prediction_time_callback = PredictionTimeHistory(lst_train_test[0][0], lst_train_test[0][2])
model_size_callback = PickleSizeCallback()

n_dims = lst_train_test[0][0].shape[1]
# with tf.device("/GPU:0"):
with tf.device("/CPU:0"):
    tf.random.set_seed(32)
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=(n_dims,), name='input_layer'),
        Dense(128, activation='relu', name='dense_1'),
        Dense(64, activation='relu', name='dense_2'),
        Dense(1, activation='sigmoid', name='dense_3') 
    ])
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', Precision(), Recall(), F1Score(), AUC()])

    history = model.fit(lst_train_test[0][0], lst_train_test[0][1], epochs=10, 
                        validation_data=[lst_train_test[0][2], lst_train_test[0][3]], 
                        callbacks=[train_time_callback, prediction_time_callback, model_size_callback])
    model.evaluate(lst_train_test[0][2], lst_train_test[0][3])

epoch_times = train_time_callback.epoch_train_times

history.history['model_size_bytes']

history.history['epoch_train_times'] = epoch_times
history.history['prediction_time'] = prediction_time_callback.prediction_times['train']
history.history['val_prediction_time'] = prediction_time_callback.prediction_times['val']

pd.DataFrame(history.history)
prediction_time_callback.prediction_times

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

parameters = []
# Add the input layer manually if it exists
if model.input_shape:
    input_layer_name = "input_layer"
    input_shape = model.input_shape  # Retrieve the input shape from the model
    parameters.append([input_layer_name, input_shape, 0, False])  # Input layer has no parameters and is not trainable

for layer in model.layers:
    layer_name = layer.name

    # Check if the layer has weights or parameters
    try:
        num_params = layer.count_params()
    except AttributeError:
        num_params = 0  # Input layers do not have parameters
    activation = layer.activation.__name__ 
    trainable = getattr(layer, 'trainable', False)  # Input layers are not trainable
    parameters.append([layer_name, num_params, trainable, activation])
print(parameters)

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
