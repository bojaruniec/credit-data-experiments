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

def download_japanese():
    """ Downloads `Japanese` from the UCI reporsitory
        unzipps it and saves it in the `data/external/japanese` directory 
    """
    logger = logging.getLogger(__name__)

    url_data = 'https://archive.ics.uci.edu/static/public/28/japanese+credit+screening.zip'
    # Send a GET request to the URL
    path_out = Path('data/external/japanese')
    path_out.mkdir(parents=True, exist_ok=True)
    path_zip = path_out / url_data.split('/')[-1]
    
    if path_zip.is_file():
        logger.info('file for Japanese already exists')
        return # Early exit
    
    logger.info('downloading Japanese')
    response = requests.get(url_data)
    response.raise_for_status()
    
    # Save zip file
    with open(path_zip, 'wb') as f:
        f.write(response.content)
    # Extract zip file
    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(path_out)


def extract_content_between_brackets(text, open_bracket='(', close_bracket=')'):
    """
    Extracts the content between matching open and close brackets in a string.
    Handles nested brackets correctly by ensuring the matching bracket is found.

    Args:
        text: The string to search within.
        open_bracket: The character representing the opening bracket (default: '(').
        close_bracket: The character representing the closing bracket (default: ')').

    Returns:
        A list of strings, where each string is the content between a matching pair of brackets.
        Returns an empty list if no matching brackets are found.
    """

    results = []
    start = -1
    count = 0  # Track nesting level

    for i, char in enumerate(text):
        if char == open_bracket:
            if count == 0:
                start = i + 1  # Start position after the opening bracket
            count += 1
        elif char == close_bracket:
            count -= 1
            if count == 0:
                if start != -1:
                    results.append(text[start:i])
                    start = -1  # Reset for the next pair

    return results

def lisp_to_csv():
    """
    Gets data from lips format in japanese to csv format
    """
    def person_number(lst_content_values):
        dic_var = dict()
        for val in lst_content_values:
            person, value = val.split(' ')
            dic_var[person.lower()] = int(value)
        return dic_var
    
    def person_atom(lst_content_values):
        dic_var = dict()
        for val in lst_content_values:
            person, value = val.split(' ')
            dic_var[person.lower()] = str(value)
        return dic_var
    
    def person_only(lst_content_values, val=1):
        dic_var = dict()
        for person in lst_content_values:
            dic_var[person.lower()] = val
        return dic_var
        
    path_inp_dir = Path('data/external/japanese') 
    path_inp_csv = path_inp_dir / 'credit.lisp'
    with open(path_inp_csv) as f:
        text_lisp = f.read()
    lst_lisp = extract_content_between_brackets(text_lisp)
    len_def_pred = len('def-pred')
    dic_out = dict()
    lst_numerical = []
    for line in lst_lisp:
        if not line.startswith('def-pred'):
            continue
        pos_type = line.index('type')
        var_name = line[len_def_pred+1: pos_type-1].strip()
        lst_line_content = extract_content_between_brackets(line)
        
        lst_content_values = extract_content_between_brackets(lst_line_content[1])
        if lst_line_content[0] == ':person :number':
            dic_var = person_number(lst_content_values)
            lst_numerical.append(var_name)
        elif lst_line_content[0] == ':person :atom':
            dic_var = person_atom(lst_content_values)
        elif lst_line_content[0] == ':person':
            dic_var = person_only(lst_content_values, 1)
            lst_numerical.append(var_name)

        if len(lst_line_content) == 3:
            lst_content_values_neg = extract_content_between_brackets(lst_line_content[2])
            dic_var = dic_var | person_only(lst_content_values_neg, 0 )

        dic_out[var_name] = dic_var      
    df = pd.DataFrame(dic_out).fillna(0)
    df[lst_numerical] = df[lst_numerical].astype(int)
    df = df.drop(columns='female', axis=0)
    lst_columns = df.columns
    
    df.index.name = 'person'

    # Write data
    path_out_dir = Path('data/processed/japanese') 
    path_out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_out_dir / 'japanese_from_lisp.csv', index=True)
    
def metadata_japanese():
    """ Extract metadata from doc file provided in the UCI documentation. 
        The file defines description of varaibles as well as their type: 
        qualitative or numerical. 
        The script simply opens the .doc file as it is a simple text file
        and then writes the output to csv
    """
    logger = logging.getLogger(__name__)

    path_inp_dir = Path('data/external/japanese') 
    path_inp_csv = path_inp_dir / 'credit.lisp'
    with open(path_inp_csv) as f:
        text_lisp = f.read()
    lst_lisp = extract_content_between_brackets(text_lisp)
    len_def_pred = len('def-pred')
    dic_type = dict()
    for line in lst_lisp:
        if not line.startswith('def-pred'):
            continue
        pos_type = line.index('type')
        var_name = line[len_def_pred+1: pos_type-1].strip()
        
        if var_name == 'female':
            continue
        
        lst_line_content = extract_content_between_brackets(line)
        
        lst_content_values = extract_content_between_brackets(lst_line_content[1])
        if lst_line_content[0] == ':person :number':
            dic_type[var_name] = 'numerical'
        elif lst_line_content[0] == ':person :atom':
            dic_type[var_name] = 'qualitative'
        elif lst_line_content[0] == ':person':
            dic_type[var_name] = 'qualitative'
    path_out_dir = Path('data/processed/japanese') 
    path_out_dir.mkdir(parents=True, exist_ok=True)
    path_out_csv = path_out_dir / 'japanese_metadata.csv'
    with open(path_out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header first
        writer.writerow(['variable', 'type', 'description'])
        # Write the data rows
        writer.writerows(dic_type.items())

        
def prepare_japanese():
    """ Prepare japanese in csv file. 
        1. creates dummy variables for categorical data
        2. assigns 1 to good credit scoring and 0 to bad credit scoring instead of + and -

        Normalization of numerical data will depend on the split into training and testing data,
        so it is not done at this stage.
    """
    path_out_dir = Path('data/processed/japanese') 
    path_out_dir.mkdir(parents=True, exist_ok=True)
    path_out_csv = path_out_dir / 'japanese.csv'
    
    path_metadata_csv = Path('data/processed/japanese/japanese_metadata.csv')     
      
    path_inp_dir = Path('data/processed/japanese') 
    path_inp_csv = path_inp_dir / 'japanese_from_lisp.csv'

    df_metadata = pd.read_csv(path_metadata_csv)
    col_target = 'credit_screening'
    
    lst_qualitative = df_metadata.loc[df_metadata['type'] == 'qualitative','variable'].to_list()
    lst_qualitative = [x for x in lst_qualitative if x != col_target]
    lst_numerical = df_metadata.loc[df_metadata['type'] == 'numerical','variable'].to_list()
    lst_numerical = [x for x in lst_numerical if x != col_target]
    
    lst_header = df_metadata['variable'].to_list()
    col_target_idx = lst_header.index(col_target)
    lst_header[col_target_idx] = 'target'

    df = pd.read_csv(path_inp_csv, delimiter=',', header=0, names=lst_header, index_col=None, na_values=['?'])
    df = df [['target'] + lst_numerical + lst_qualitative]
    df = pd.get_dummies(df, columns=lst_qualitative, drop_first=True, dtype=int)
    df.to_csv(path_out_csv, index=False)


def get_list_of_numerical_japanese() -> list:
    """
    Returns the list of numerical variable for a given dataset. 
    """
    path_metadata_csv = Path('data/processed/japanese/japanese_metadata.csv')     

    df_metadata = pd.read_csv(path_metadata_csv)
    lst_numerical = df_metadata.loc[df_metadata['type'] == 'numerical','variable'].to_list()

    return lst_numerical


def stratified_k_folds_japanese():
    """
    Prepares a file with the split into stratified 10-folds.
    Saves file japanese_folds.csv, where in each column there are observation
    numbers which will be used in test dataset in each fold. The observations
    for train subset are the remaining ones 
    """
    path_inp_dir = Path('data/processed/japanese') 
    path_inp_csv = path_inp_dir / 'japanese.csv'
    
    path_out_dir = Path('data/processed/japanese') 
    path_out_csv = path_out_dir / 'japanese_folds.csv'
    
    path_out_stats = path_out_dir / 'japanese_folds_stats.csv'
        
    df = pd.read_csv(path_inp_csv)
    # list of numerical variables:
    lst_numerical = get_list_of_numerical_japanese()
    
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


def folds_to_csv_japanese():
    """
    Converts data from the whole CSV dataset to csv split by folds.
    """
    # Read data
    path_inp_dir = Path('data/processed/japanese') 
    path_inp_csv = path_inp_dir / 'japanese.csv'
    df_data = pd.read_csv(path_inp_csv)
   
    # Reaad metadata    
    lst_numerical = get_list_of_numerical_japanese()
        
    # Read folds
    path_inp_folds = path_inp_dir / 'japanese_folds.csv'   
    df_folds = pd.read_csv(path_inp_folds)
    
    path_out_folds = Path('data/folds/japanese')
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
        df_fold_train.to_csv(path_out_folds.joinpath(f'{fold_n}_train.csv'), index=False, float_format='%.8f')
        df_fold_test.to_csv(path_out_folds.joinpath(f'{fold_n}_test.csv'), index=False, float_format='%.8f')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    pass    
