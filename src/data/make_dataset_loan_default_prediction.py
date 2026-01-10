# -*- coding: utf-8 -*-
import requests
import zipfile
import logging
import re
import csv
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import time
from lxml import html
from bs4 import BeautifulSoup
from functools import partial
from collections import defaultdict

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from itertools import product

from multiprocessing.pool import ThreadPool
import subprocess

def download_loan_default_prediction():
    """ Downloads `Lending Club` data from the available sources
    https://www.kaggle.com/competitions/loan-default-prediction/data
    """
    logger = logging.getLogger(__name__)
    ...

def extract_loan_default_prediction():
    """ Extracts lending club data
    """
    logger = logging.getLogger(__name__)
    # url_data = 'https://archive.ics.uci.edu/static/public/28/japanese+credit+screening.zip'
    # Send a GET request to the URL
    path_out = Path('data/external/loan-default-prediction')
    path_out.mkdir(parents=True, exist_ok=True)
    for path_zip in path_out.glob('*.zip'):
        with zipfile.ZipFile(path_zip, 'r') as zip_ref:
            zip_ref.extractall(path_out)

def make_sample_loan_default_prediction():
    input_filepath = 'data/external/loan-default-prediction/test_v2.csv'
    output_filepath = 'data/interim/loan-default-prediction/test_v2_sample.csv'
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    
    subprocess.run(['head', '-n', '100', input_filepath], stdout=open(output_filepath, 'w'))
    df = pd.read_csv(output_filepath)
    df.to_excel(output_filepath.replace('.csv', '.xlsx'), index=False)

    input_filepath = 'data/external/loan-default-prediction/train_v2.csv'
    output_filepath = 'data/interim/loan-default-prediction/train_v2_sample.csv'
    subprocess.run(['head', '-n', '100', input_filepath], stdout=open(output_filepath, 'w'))
    df = pd.read_csv(output_filepath)
    df.to_excel(output_filepath.replace('.csv', '.xlsx'), index=False)

def check_columns_same(df, columns):
    """
    Checks if all values are the same across specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame.
        columns (list): A list of column names to check.

    Returns:
        bool: True if all values are the same across the columns, False otherwise.
    """
    if not all(col in df.columns for col in columns):
        print("Error: One or more specified columns not found in DataFrame.")
        return False
    first_col = df[columns[0]]
    for col in columns[1:]:
        if not first_col.equals(df[col]):
            return False
    return True

def split_columns_into_groups_with_exactly_the_same_values(df, columns):
    """
    Splits a list of columns into groups where each group contains columns that have exactly the same values.

    Args:
        df (pd.DataFrame): The DataFrame.
        columns (list): A list of column names to split.

    Returns:
        list: A list of lists, where each inner list contains the column names of a group.
    """
    groups = []
    remaining_columns = set(columns)
    while remaining_columns:
        col = remaining_columns.pop()
        group = [col]
        for other_col in remaining_columns.copy():
            if check_columns_same(df, [col, other_col]):
                group.append(other_col)
                remaining_columns.remove(other_col)
        groups.append(group)
    return groups


def check_continuous_range(variables):
    """
    Checks if a list of strings like ['f85', 'f86', 'f87', 'f88'] represents a continuous numerical range.

    Args:
        variables: A list of strings, where each string is in the format 'f' followed by a number.

    Returns:
        A string representing the range (e.g., 'f85_to_88') if continuous, or None if not.
    """
    if not variables:
        return None

    try:
        numbers = [int(v[1:]) for v in variables]  # Extract numbers after 'f'
        if len(numbers) == 1:
            return f"f{numbers[0]}"

        if all(numbers[i] + 1 == numbers[i + 1] for i in range(len(numbers) - 1)):
            return f"f{numbers[0]}_to_{numbers[-1]}"
        elif len(numbers) == 2:
            return f"f{numbers[0]}_and_{numbers[1]}"
        else:
            return None
    except (ValueError, IndexError):
        return None  # Handle cases where input is not in the expected format

def calculate_correlation_for_combined_train_and_test():
    """
    Loads both train and test data, combines them, and calculates the correlation matrix.
    Saves the correlation matrix of the combined dataframe to an Excel file.

    Returns:
        pd.DataFrame: The correlation matrix.
    """
    input_filepath_train = Path('data/external/loan-default-prediction/train_v2.csv')
    input_filepath_test = Path('data/external/loan-default-prediction/test_v2.csv')
    output_filepath = Path('data/interim/loan-default-prediction/correlations.xlsx')
    
    df_train = pd.read_csv(input_filepath_train, index_col=0, header=0)
    df_test = pd.read_csv(input_filepath_test, index_col=0, header=0)
    df_combined = pd.concat([df_train, df_test], axis=0)
    print(f'Shape of train data: {df_train.shape}')
    print(f'Shape of test data: {df_test.shape}')
    print(f'Shape of combined dataframe: {df_combined.shape}')

    # Calculate the correlation matrix
    lst_cols_all = [x for x in df_combined.columns if x != 'loss']
    lst_cols_obj = df_combined.dtypes[df_combined.dtypes == 'object'].index.to_list()

    df_corr_train =  df_train[[x for x in lst_cols_all if x not in lst_cols_obj]].corr()
    df_corr_test =  df_test[[x for x in lst_cols_all if x not in lst_cols_obj]].corr()  
    df_corr_combined = df_combined[[x for x in lst_cols_all if x not in lst_cols_obj]].corr()
    
    with pd.ExcelWriter(output_filepath) as writer:
        df_corr_train.to_excel(writer, sheet_name='train', index=True, float_format=f"%.8f")
        df_corr_test.to_excel(writer, sheet_name='test', index=True, float_format=f"%.8f")
        df_corr_combined.to_excel(writer, sheet_name='combined', index=True, float_format=f"%.8f")
    
    dic_perfect_correlations_train = get_perfect_correlations(df_corr_train)
    dic_perfect_correlations_test = get_perfect_correlations(df_corr_test)
    dic_perfect_correlations_combined = get_perfect_correlations(df_corr_combined)    

    return df_corr_combined

def get_perfect_correlations(df_corr):
    # get variables with perfect correlation
    dic_perfect_correlations = {}
    for col in df_corr.columns:
        correlated_vars = df_corr.index[df_corr[col] == 1].tolist()
        if len(correlated_vars) > 1:
            dic_perfect_correlations[col] = [var for var in correlated_vars if var != col] # exclude self correlation
    return dic_perfect_correlations

def explore_loan_default_prediction():
    input_filepath_train = Path('data/external/loan-default-prediction/train_v2.csv')
    input_filepath_test = Path('data/external/loan-default-prediction/test_v2.csv')
    
    df = pd.read_csv(input_filepath_train, index_col=0, header=0)
    df_test = pd.read_csv(input_filepath_test, index_col=0, header=0)
    
    # (135,204,274,417)
    df.head()
    df.iloc[1,135]
    lst_col_string_iloc = [135,204, 274, 417]
    lst_col_string_loc = df.columns[lst_col_string_iloc].to_list()
    df.iloc[1, lst_col_string_iloc]
    df['f420'].astype('float').describe()
    df['f420'].astype('float').plot.hist(bins=100)

    s_var = 'f138'
    dic_count_zeros = {'0': int((df[s_var] == '0').sum()), 'non-0': int((df[s_var] != '0').sum())}
    dic_count_zeros
    df.loc[df[s_var] != '0', s_var].is_unique
    df.loc[df[s_var] != '0', s_var].value_counts()
    df.loc[df[s_var] == '47800000000000000', ['f136', 'f137', 'f138']].value_counts()
    df[s_var].apply(lambda x: log10(float(x))).value_counts()

    s_var = 'f207'
    dic_count_zeros = {'0': int((df[s_var] == '0').sum()), 'non-0': int((df[s_var] != '0').sum())}
    dic_count_zeros
    df.loc[df[s_var] != '0', s_var].is_unique
    df2 = df.loc[df[s_var] != '0', s_var].value_counts().sort_index()
    df2.to_csv('data/interim/loan-default-prediction/f207.csv')
    df[s_var]
    df.loc[:,['f125','f126', 'f127', 'f128']].corr()
    
    df.loc[df['f125'] == 2730].to_excel('data/interim/loan-default-prediction/f125_2730.xlsx')
    df.loc[:,['f85','f86', 'f87', 'f88']].corr()
    df.loc[:,['f95','f96', 'f97', 'f98']].corr()
    df.loc[:,['f105','f106', 'f107','f108']].corr()
    df.loc[:,['f115','f116', 'f117','f118']].corr()
    df.loc[:,['f125','f126', 'f127','f128']].corr()
    df.loc[:,['f154','f155', 'f156','f157']].corr()
    df.loc[:,['f164','f165', 'f166','f167']].corr()
    
    check_columns_same(df, ['f85','f86', 'f87', 'f88'])
    check_columns_same(df, ['f95','f96', 'f97', 'f98'])
    check_columns_same(df, ['f105','f106', 'f107','f108'])
    check_columns_same(df, ['f115','f116', 'f117','f118'])
    check_columns_same(df, ['f125', 'f126', 'f127', 'f128'])
    check_columns_same(df, ['f154','f155', 'f156','f157'])
    check_columns_same(df, ['f164','f165', 'f166','f167'])
    check_columns_same(df, ['f174','f175', 'f176','f177'])
    check_columns_same(df, ['f184','f185', 'f186','f187'])
    check_columns_same(df, ['f194','f195', 'f196','f197'])
    check_columns_same(df, ['f224','f225', 'f226','f227'])
    check_columns_same(df, ['f234','f235','f236', 'f237'])
    check_columns_same(df, ['f244','f245','f246', 'f247'])
    check_columns_same(df, ['f254','f255','f256', 'f257'])
    check_columns_same(df, ['f264','f265','f266', 'f267'])
    check_columns_same(df, ['f293','f294','f295', 'f296'])
    check_columns_same(df, ['f323','f315',])
    
    check_columns_same(df, ['f205', 'f186','f187'])
    df.dtypes.to_csv('data/interim/loan-default-prediction/dtypes.csv')
    
    
    check_continuous_range(['f85', 'f86', 'f87', 'f88', 'f89'])
    
    
    lst_cols_obj = df.dtypes[df.dtypes == 'object'].index.to_list()
    lst_cols_all = df.columns.to_list()
    df_corr = df[[x for x in lst_cols_all if x not in lst_cols_obj]].corr()
    
    # save correlation matrix to Excel
    df_corr.to_excel('data/interim/loan-default-prediction/corr.xlsx')
    

    # get the set of variables with perfect correlation
    set_perfect_correlations = set()
    for k, v in dic_perfect_correlations.items():
        tup_correlation = tuple(sorted([k] + v, key=lambda x: int(x[1:])))
        set_perfect_correlations.add(tup_correlation)
    set_perfect_correlations
    lst_perfect_correlations = sorted(set_perfect_correlations, key=lambda x: int(x[0][1:]))
    print(len(lst_perfect_correlations))
    
    check_columns_same_fixed_df_train = partial(check_columns_same, df)
    check_columns_same_fixed_df_test = partial(check_columns_same, df_test)
    
    lst_variables_the_same_train = list(map(check_columns_same_fixed_df_train, lst_perfect_correlations))
    lst_variables_the_same_test = list(map(check_columns_same_fixed_df_test, lst_perfect_correlations))

    split_columns_into_groups_with_exactly_the_same_values(df, lst_perfect_correlations[0])
    split_columns_into_groups_with_exactly_the_same_values(df_test, lst_perfect_correlations[0])
    split_columns_into_groups_with_exactly_the_same_values(df_test, lst_perfect_correlations[3])
    
    
    
    # Assuming 'df' is your existing DataFrame
    # Verify all columns exist in the DataFrame
    lst_grouped_columns = [col for group in lst_perfect_correlations for col in group]
    print(len(lst_perfect_correlations))
    print(len(lst_grouped_columns))

    lst_group_names = list(map(check_continuous_range, lst_perfect_correlations))
    df_cols_the_same_summary = pd.DataFrame({'group': lst_perfect_correlations, 'group_name': lst_group_names, 
                  'all_equal_train': lst_variables_the_same_train, 
                  'all_equal_test': lst_variables_the_same_test})
    df_cols_the_same_summary.to_excel('data/interim/loan-default-prediction/perfect_correlations.xlsx')

    # Check if all grouped columns exist in the DataFrame
    set_missing_cols = set(lst_grouped_columns) - set(lst_cols_all)
    if set_missing_cols:
        print(f"Warning: These columns are in your groups but not in the DataFrame: {set_missing_cols}")

    lst_extra_cols = sorted(set(lst_cols_all) - set(lst_grouped_columns+['loss']), key=lambda x: int(x[1:]))
    lst_groups_all = lst_perfect_correlations
    if lst_extra_cols:
        print(f"Warning: These columns are in the DataFrame but not in your groups: {lst_extra_cols}, {len(lst_extra_cols)}")
        # If you want to include extra columns in a separate group:
        lst_groups_all.append(tuple(lst_extra_cols))
        multi_index = pd.MultiIndex.from_tuples(
            [(f'Group_{i+1}', col) for i, group in enumerate(lst_groups_all) for col in group],
            names=['Group', 'Column']
        )
    lst_perfect_correlations
    # Reassign the columns with the MultiIndex
    df_train_grouped = df[lst_grouped_columns+lst_extra_cols].head(100).copy(deep=True)
    df_train_grouped.shape
    df_train_grouped.columns = multi_index
    df_train_grouped.to_excel('data/interim/loan-default-prediction/df_train_grouped.xlsx')
    
    
    df_test_grouped = df_test[lst_grouped_columns+lst_extra_cols].head(100).copy(deep=True)
    df_test_grouped.shape
    df_test_grouped.columns = multi_index
    df_test_grouped.to_excel('data/interim/loan-default-prediction/df_test_grouped.xlsx')
    
    multi_index = pd.MultiIndex.from_tuples(column_tuples, names=['Category', 'Subcategory'
                                                                  
                                                                  ])
    
    df.loc[:,['f115','f116', 'f117','f118']].corr()
    
    s_var = 'f277'
    dic_count_zeros = {'0': int((df[s_var] == '0').sum()), 'non-0': int((df[s_var] != '0').sum())}
    dic_count_zeros
    df.loc[df[s_var] != '0', s_var].is_unique
    df.loc[df[s_var] != '0', s_var].value_counts()
    df[s_var]

    s_var = 'f420'
    dic_count_zeros = {'0': int((df[s_var] == '0').sum()), 'non-0': int((df[s_var] != '0').sum())}
    df.loc[df[s_var] != '0', s_var].is_unique
    df.loc[df[s_var] != '0', s_var].value_counts()
    dic_count_zeros
    df[s_var]

    df['f420'].astype('float').plot.hist(bins=100)   

def get_dic_mean_std_col(df, head=10_000) -> dict:
    """
    """
    df_numeric_head = df.head(head).select_dtypes(include=np.number)
    dic_m_s_col = defaultdict(list)
    ser_mean = df_numeric_head.mean()
    ser_std = df_numeric_head.std()
    for col, m, s in zip(ser_mean.index, ser_mean, ser_std):
        dic_m_s_col[(m,s)].append(col)
    return dic_m_s_col

def get_list_potentially_the_same_cols_on_mean_col(dic_m_s_col:dict) -> list:
    return [v for v in dic_m_s_col.values() if len(v) > 1]

def remove_cols_the_same(df, lst_the_same):
    for lst_col in lst_the_same:
        df.drop(columns=lst_col[1:], inplace=True)

def values_in_cols_the_same(df, lst_cols):
    """
    """
    first_col = df[lst_cols[0]]
    for col in lst_cols[1:]:
        if not first_col.equals(df[col]):
            return False    
    return True
        
def simplyfiy_loan_default_prediction():
    input_filepath_train = Path('data/external/loan-default-prediction/train_v2.csv')
    df_train = pd.read_csv(input_filepath_train, index_col=0, header=0)

    n_rows = 10_000
    dic_m_s_col = get_dic_mean_std_col(df_train, n_rows)
    lst_potential = get_list_potentially_the_same_cols_on_mean_col(dic_m_s_col)
    
    df_train_head = df_train.head(n_rows)
    lst_potential_the_same_head = [lst for lst in lst_potential if values_in_cols_the_same(df_train_head, lst)]
    lst_the_same = [lst for lst in lst_potential_the_same_head if values_in_cols_the_same(df_train, lst)]
    remove_cols_the_same(df_train, lst_the_same)
    df_train.to_csv('data/interim/loan-default-prediction/train_v2_simplified.csv')

def loan_defualt_prediction_categorical():
    lst_categorical = ['f776', 'f77', 'f778', 'f2', 'f4', 'f5']
    # https://www.kaggle.com/competitions/loan-default-prediction/discussion/6978
    # df_train [lst_categorical].nunique()
    # df_train ['f4'].value_counts().sort_index()
    # df_train['f778'] #geography
    # df_train['f6'].value_counts().sort_index()
    # df_train.columns
    # df_train['f776'].value_counts().sort_index()
    # df_train['f777'].value_counts().sort_index()
    # df_train['f778'].value_counts().sort_index()
    
def metadata_loan_default_prediction():
    """
    Procedure to change columns to numeric data, 
    so that it could be used in machine learning algorithms.
    
    In addition, it creates metadata file with information about the columns.
    """
    path_inp = Path('data/interim/loan-default-prediction')
    path_out= Path('data/processed/loan-default-prediction')
    path_out.mkdir(parents=True, exist_ok=True)
    file = path_inp / 'train_v2_simplified.csv'
    df = pd.read_csv(file, header=0, index_col=0)   
    df['target'] = df['loss'].apply(lambda x: 1 if x == 0 else 0)
    lst_cols_target = ['target']
    lst_cols_categorical = sorted(['f776', 'f77', 'f778', 'f2', 'f4', 'f5'])
    lst_cols_numerical = [x for x in df.columns if x not in (lst_cols_target + lst_cols_categorical + ['loss'])]
    lst_cols_ordinal = []
    lst_cols_text = []
    lst_cols = lst_cols_target + lst_cols_numerical + lst_cols_categorical + lst_cols_text
    df = df[lst_cols]
    lst_cols_before = df.columns.to_list()
    df2 = pd.get_dummies(df, columns=lst_cols_categorical, dummy_na=True, drop_first=False, dtype='Int64')
    lst_cols_after = df2.columns.to_list()
    lst_cols_diff = [x for x in lst_cols_after if x not in lst_cols_before]
    df2[lst_cols_diff] = df2[lst_cols_diff].replace(0, np.nan)
    df2.to_csv(path_out / 'loan_default_prediction.csv', index=True)
    # prepraring for lending_club_metadata.csv
    dic_metadata = dict()
    dic_metadata = dic_metadata | {x:'numerical' for x in lst_cols_numerical}
    dic_metadata = dic_metadata | {x:'ordinal' for x in lst_cols_ordinal}      
    dic_metadata = dic_metadata | {x:'categorical' for x in lst_cols_categorical}      
    dic_metadata = dic_metadata| {x:'text' for x in lst_cols_text}   
    df_metadata = pd.DataFrame.from_dict(dic_metadata, orient='index', columns=['type'])
    df_metadata.index.name = 'variable'
    df_metadata['description'] = ''
    df_metadata.to_csv(path_out / 'loan_default_prediction_metadata.csv', index=True)

     
def get_list_of_numerical(path_metadata_csv) -> list:
    """
    Returns the list of numerical variable for a given dataset. 
    """
    path_metadata_csv = Path(path_metadata_csv)     
    df_metadata = pd.read_csv(path_metadata_csv)
    lst_numerical = df_metadata.loc[df_metadata['type'] == 'numerical','variable'].to_list()
    return lst_numerical

get_list_of_numerical_loan_default_prediction= partial(get_list_of_numerical, 'data/processed/loan-default-prediction/loan_default_prediction_metadata.csv')

def stratified_k_folds_loan_default_prediction():
    """
    Prepares a file with the split into stratified 10-folds.
    Saves file japanese_folds.csv, where in each column there are observation
    numbers which will be used in test dataset in each fold. The observations
    for train subset are the remaining ones 
    """
    path_inp_dir = Path('data/processed/loan-default-prediction') 
    path_inp_csv = path_inp_dir / 'loan_default_prediction.csv'
    
    path_out_dir = Path('data/processed/loan-default-prediction') 
    path_out_csv = path_out_dir / 'loan_default_prediction_folds.csv'
    path_out_stats = path_out_dir / 'loan_default_prediction_stats.csv'
        
    df = pd.read_csv(path_inp_csv, low_memory=False)
    # list of numerical variables:
    lst_numerical = get_list_of_numerical_loan_default_prediction()
    lst_object_cols = df.select_dtypes(include=['object']).columns
    for col in lst_object_cols:
        df[col] = df[col].astype(float)
    # folds
    df_shuffled = df.sample(frac=1, random_state=20)
    y = df_shuffled.pop('target')
    X = df_shuffled

    lst_original_iloc = df.index.to_list() 
    dic_orginal_iloc = {v:n for n,v in enumerate(lst_original_iloc)}
    lst_new_iloc = list(df_shuffled.index)
    
    skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    # StratifiedGroupKFold to be used here
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


def folds_to_csv_loan_default_prediction():
    """
    Converts data from the whole CSV dataset to csv split by folds.
    """
    # Read data
    path_inp_dir = Path('data/processed/loan-default-prediction') 
    path_inp_csv = path_inp_dir / 'loan_default_prediction.csv'
    df_data = pd.read_csv(path_inp_csv)
    for col in df_data.columns:
        df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
   
    # Reaad metadata    
    lst_numerical = get_list_of_numerical_loan_default_prediction()
        
    # Read folds
    path_inp_folds = path_inp_dir / 'loan_default_prediction_folds.csv'   
    df_folds = pd.read_csv(path_inp_folds)
    
    
    path_out_folds = Path('data/folds/loan-default-prediction')
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
    # logging.basicConfig(level=logging.INFO, format=log_fmt)
    # download_sales_lending_club()
    # df = read_lending_club()
    # extract_sales_repors_zip()
    # sales_report_from_folder()

    # merge_lending_club_only_selected_columns()
    # read_merged_file()
    # clean_merged_file()
    # draw_stratfied_sample()
    # metadata_loan_default_prediction()
    # stratified_k_folds_loan_default_prediction()
    folds_to_csv_loan_default_prediction()