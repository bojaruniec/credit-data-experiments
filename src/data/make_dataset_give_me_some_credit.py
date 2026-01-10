# -*- coding: utf-8 -*-
import requests
import zipfile
import logging
import re
import csv
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

def extract_give_me_some_credit():
    """ Downloads `Give Me Some Credit` from the Kaggle reporsitory
        unzipps it and saves it in the `data/external/give-me-some-credit` directory 
        https://www.kaggle.com/competitions/GiveMeSomeCredit/data
    """
    logger = logging.getLogger(__name__)

    # url_data = 'https://archive.ics.uci.edu/static/public/28/japanese+credit+screening.zip'
    # Send a GET request to the URL
    path_out = Path('data/external/give-me-some-credit')
    path_out.mkdir(parents=True, exist_ok=True)
    # path_zip = path_out / url_data.split('/')[-1]
    path_zip = path_out / 'GiveMeSomeCredit.zip'
    
    if path_zip.is_file():
        logger.info('file for Japanese already exists')
    
    # Extract zip file
    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(path_out)

def metadata_give_me_some_credit():
    """ Extract metadata from doc file provided in the UCI documentation. 
        The file defines description of varaibles as well as their type: 
        qualitative or numerical. 
    """
    logger = logging.getLogger(__name__)

    path_inp_dir = Path('data/external/give-me-some-credit') 
    path_inp_xls = path_inp_dir / 'Data Dictionary.xlsx'

    df = pd.read_excel(path_inp_xls, skiprows=1, header=0)
    df['type']= df['Type'].map({'percentage':'numerical', 'real':'numerical', 'integer':'numerical', 'Y/N':'qualitative'})
    df = df.rename(columns={'Variable Name':'variable', 'Description':'description'})
    df = df[['variable', 'type', 'description']]
    path_out_dir = Path('data/processed/give-me-some-credit') 
    path_out_dir.mkdir(parents=True, exist_ok=True)
    path_out_csv = path_out_dir / 'give_me_some_credit_metadata.csv'
    df.to_csv(path_out_csv, index=False)
        
def prepare_give_me_some_credit():
    """ Prepare japanese in csv file. 
        1. creates dummy variables for categorical data
        Normalization of numerical data will depend on the split into training and testing data,
        so it is not done at this stage.
    """
    path_out_dir = Path('data/processed/give-me-some-credit') 
    path_out_dir.mkdir(parents=True, exist_ok=True)
    path_out_csv = path_out_dir / 'give_me_some_credit.csv'
    
    path_metadata_csv = Path('data/processed/give-me-some-credit/give_me_some_credit_metadata.csv')     
      
    path_inp_dir = Path('data/external/give-me-some-credit') 
    path_inp_csv = path_inp_dir / 'cs-training.csv'

    df_metadata = pd.read_csv(path_metadata_csv)
    col_target = 'SeriousDlqin2yrs'
    
    lst_qualitative = df_metadata.loc[df_metadata['type'] == 'qualitative','variable'].to_list()
    lst_qualitative = [x for x in lst_qualitative if x != col_target]
    lst_numerical = df_metadata.loc[df_metadata['type'] == 'numerical','variable'].to_list()
    lst_numerical = [x for x in lst_numerical if x != col_target]
    
    lst_header = df_metadata['variable'].to_list()
    col_target_idx = lst_header.index(col_target)
    lst_header[col_target_idx] = 'target'

    df = pd.read_csv(path_inp_csv, delimiter=',', header=0, names=lst_header, index_col=None, na_values=['?', 'NA'])
    df = df [['target'] + lst_numerical + lst_qualitative]
    df['target'] = df['target'].map({0:1, 1:0})
    
    df = pd.get_dummies(df, columns=lst_qualitative, drop_first=True, dtype=int)
    df.to_csv(path_out_csv, index=False)


def get_list_of_numerical_give_me_some_credit() -> list:
    """
    Returns the list of numerical variable for a given dataset. 
    """
    path_metadata_csv = Path('data/processed/give-me-some-credit/give_me_some_credit_metadata.csv')     

    df_metadata = pd.read_csv(path_metadata_csv)
    lst_numerical = df_metadata.loc[df_metadata['type'] == 'numerical','variable'].to_list()

    return lst_numerical


def stratified_k_folds_give_me_some_credit():
    """
    Prepares a file with the split into stratified 10-folds.
    Saves file japanese_folds.csv, where in each column there are observation
    numbers which will be used in test dataset in each fold. The observations
    for train subset are the remaining ones 
    """
    path_inp_dir = Path('data/processed/give-me-some-credit') 
    path_inp_csv = path_inp_dir / 'give_me_some_credit.csv'
    
    path_out_dir = Path('data/processed/give-me-some-credit') 
    path_out_csv = path_out_dir / 'give_me_some_credit_folds.csv'
    
    path_out_stats = path_out_dir / 'give_me_some_credit_stats.csv'
        
    df = pd.read_csv(path_inp_csv)
    # list of numerical variables:
    lst_numerical = get_list_of_numerical_give_me_some_credit()
    
    # folds
    df_shuffled = df.sample(frac=1, random_state=20)
    y = df_shuffled.pop('target')
    X = df_shuffled

    lst_original_iloc = df.index.to_list() 
    dic_orginal_iloc = {v:n for n,v in enumerate(lst_original_iloc)}
    lst_new_iloc = list(df_shuffled.index)
    
    skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    dic_combined = dict()
    dic_folds = dict()
    lst_stats_df = []
    for n_fold, (idx_train, idx_test) in enumerate(skf.split(X, y),start=1):
        dic_combined.update({x:n_fold for x in idx_test})
        dic_folds.update({f'fold{n_fold:02}':sorted([dic_orginal_iloc[lst_new_iloc[x]] for x in idx_test])})
        
        # Statistics min/max for training dataset 
        df_stats = pd.DataFrame()
        df_stats = X.loc[idx_train, lst_numerical].agg(['min','max'])
        df_stats['fold_num'] = n_fold   
        new_index = pd.MultiIndex.from_arrays([df_stats['fold_num'], df_stats.index], names=['fold_num', 'stats'])
        df_stats.index = new_index
        df_stats.drop('fold_num', axis=1, inplace=True)
        lst_stats_df.append(df_stats.copy())
        
    df_folds = pd.DataFrame.from_dict(dic_folds, orient='index')
    df_folds = df_folds.transpose()
    df_folds.to_csv(path_out_csv, index=False, float_format='%.0f')
    
    df_stats_agg = pd.concat(lst_stats_df)
    df_stats_agg.to_csv(path_out_stats, index=True)


def custom_format_float(value):
    """
    Custom format, so that no unnecessary numbers as saved into the file
    """
    if value == 0:
        return "0"  # Or any other representation you want for 0
    elif abs(value) < 1e-8: # handle very small numbers as 0
        return "0"
    else:
        return f"{value:.8f}".rstrip('0').rstrip('.') # removes trailing zeros and . if nothing left


def folds_to_csv_give_me_some_credit():
    """
    Converts data from the whole CSV dataset to csv split by folds.
    """
    # Read data
    path_inp_dir = Path('data/processed/give-me-some-credit') 
    path_inp_csv = path_inp_dir / 'give_me_some_credit.csv'
    df_data = pd.read_csv(path_inp_csv)
   
    # Reaad metadata    
    lst_numerical = get_list_of_numerical_give_me_some_credit()
        
    # Read folds
    path_inp_folds = path_inp_dir / 'give_me_some_credit_folds.csv'   
    df_folds = pd.read_csv(path_inp_folds)
    
    path_out_folds = Path('data/folds/give-me-some-credit')
    path_out_folds.mkdir(exist_ok=True, parents=True)
    for fold_n, fold_idx in df_folds.items():
        # Split data
        fold_idx_not_none = fold_idx.dropna()
        df_fold_test = df_data.iloc[fold_idx_not_none].copy()
        df_fold_train = df_data[~df_data.index.isin(df_fold_test.index)].copy()
        
        # Normalize data
        scaler = MinMaxScaler()
        df_fold_train[lst_numerical] = scaler.fit_transform(df_fold_train[lst_numerical])
        df_fold_test[lst_numerical] = scaler.transform(df_fold_test[lst_numerical])
        # Create csv files with folds
        df_fold_train.to_csv(path_out_folds.joinpath(f'{fold_n}_train.csv'), index=False, float_format=custom_format_float)
        df_fold_test.to_csv(path_out_folds.joinpath(f'{fold_n}_test.csv'), index=False, float_format=custom_format_float)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    pass    
