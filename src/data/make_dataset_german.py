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

def download_german_credit_data():
    """ Downloads `German Credit Data` from the UCI reporsitory
        unzipps it and saves it in the `data/external/german-credit-data` directory 
    """
    logger = logging.getLogger(__name__)

    url_german = 'https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip'
    # Send a GET request to the URL
    path_out = Path('data/external/german-credit-data')
    path_out.mkdir(parents=True, exist_ok=True)
    path_zip = path_out / url_german.split('/')[-1]
    
    if path_zip.is_file():
        logger.info('file for German Credit Data already exists')
        return # Early exit
    
    logger.info('downloading German Credit Data')
    response = requests.get(url_german)
    response.raise_for_status()
    
    # Save zip file
    with open(path_zip, 'wb') as f:
        f.write(response.content)
    # Extract zip file
    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(path_out)

def metadata_german_credit_data():
    """ Extract metadata from doc file provided in the UCI documentation. 
        The file defines description of varaibles as well as their type: 
        qualitative or numerical. 
        The script simply opens the .doc file as it is a simple text file
        and then writes the output to data/processed/german-credit-data/geramn_metadata.csv
    """
    logger = logging.getLogger(__name__)

    path_out_dir = Path('data/processed/german-credit-data') 
    path_out_dir.mkdir(parents=True, exist_ok=True)
    path_out_csv = path_out_dir / 'german_metadata.csv'    
    if path_out_csv.is_file():
        logger.info('metadata file for German Credit Data already exists')
        return # Early exit
    logger.info('reading the content of metadata for German Credit Data')
    path_doc = Path('data/external/german-credit-data/german.doc')
    
    assert path_doc.is_file(), f"The file {path_doc} needs to exist. Please run download_german_credit_data()."
    
    within_found_attribute = False
    lst_metadata = []
    with open(path_doc, 'r') as f:
        line = f.readline()
        while line:
            if within_found_attribute:
                match = re.search(r'Att[a-z]+\s+(\d+):\s*\(([^)]+)\)', line)
                if match:
                    attr_number = int(match.group(1))  # The digits after 'Attribute '
                    attr_type = match.group(2)      # The text inside the parentheses
                    
                    line = f.readline()
                    attr_desc = line.strip()
                    lst_metadata.append((f'A{attr_number:02d}', attr_type, attr_desc))
            if line.startswith('7.  Attribute description for german'):
                within_found_attribute = True
            elif line.startswith('8.  Cost Matrix'):
                break
            line = f.readline()
            
    logger.info('writting the content of metadata for German Credit Data to csv file')
    with open(path_out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header first
        writer.writerow(['variable', 'type', 'description'])
        # Write the data rows
        writer.writerows(lst_metadata)
        
def prepare_german_credit_data():
    """ Prepare german credit data in csv file. 
        1. creates dummy variables for categorical data
        2. assigns 1 to good credit scoring and 0 to bad credit scoring instead of 2 for bad

        Normalization of numerical data will depend on the split into training and testing data,
        so it is not done at this stage.
    """
    path_out_dir = Path('data/processed/german-credit-data') 
    path_out_dir.mkdir(parents=True, exist_ok=True)
    path_out_csv = path_out_dir / 'german.csv'
    
    path_metadata_csv = Path('data/processed/german-credit-data/german_metadata.csv')     
      
    path_inp_dir = Path('data/external/german-credit-data') 
    path_inp_csv = path_inp_dir / 'german.data'

    df_metadata = pd.read_csv(path_metadata_csv)
    lst_qualitative = df_metadata.loc[df_metadata['type'] == 'qualitative','variable'].to_list()
    lst_numerical = df_metadata.loc[df_metadata['type'] == 'numerical','variable'].to_list()
    
    lst_header = df_metadata['variable'].to_list() + ['target']
    df = pd.read_csv(path_inp_csv, delimiter=' ', header=None, names=lst_header, index_col=None)
    df = df [['target'] + lst_numerical + lst_qualitative]
    df['target'] = df['target'].map({2:0, 1:1})
    df = pd.get_dummies(df, columns=lst_qualitative, drop_first=True, dtype=int)
    df.to_csv(path_out_csv, index=False)


def get_list_of_numerical_variables_german() -> list:
    """
    Returns the list of numerical variable for a given dataset. 
    So far the dataset is fixed and it is german. 
    """
    path_metadata_csv = Path('data/processed/german-credit-data/german_metadata.csv')     

    df_metadata = pd.read_csv(path_metadata_csv)
    lst_numerical = df_metadata.loc[df_metadata['type'] == 'numerical','variable'].to_list()

    return lst_numerical


def stratified_k_folds_german_credit_data():
    """
    Prepares a file with the split into stratified 10-folds.
    Saves file german_folds.csv, where in each column there are observation
    numbers which will be used in test dataset in each fold. The observations
    for train subset are the remaining ones 
    """
    path_inp_dir = Path('data/processed/german-credit-data') 
    path_inp_csv = path_inp_dir / 'german.csv'
    
    path_out_dir = Path('data/processed/german-credit-data') 
    path_out_csv = path_out_dir / 'german_folds.csv'
    
    path_out_stats = path_out_dir / 'german_folds_stats.csv'
        
    df = pd.read_csv(path_inp_csv)
    # list of numerical variables:
    lst_numerical = get_list_of_numerical_variables_german()
    
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

def folds_to_csv_german():
    """
    Converts data from the whole CSV dataset to csv and TensorFlow Dataset split by folds.
    """
    # Read data
    path_inp_dir = Path('data/processed/german-credit-data') 
    path_inp_csv = path_inp_dir / 'german.csv'
    df_data = pd.read_csv(path_inp_csv)
   
    # Reaad metadata    
    lst_numerical = get_list_of_numerical_variables_german()
        
    # Read folds
    path_inp_folds = path_inp_dir / 'german_folds.csv'   
    df_folds = pd.read_csv(path_inp_folds)
    
    path_out_folds = Path('data/folds/german-credit-data')
    path_out_folds.mkdir(exist_ok=True, parents=True)
    for fold_n, fold_idx in df_folds.items():
        # Split data
        df_fold_test = df_data.iloc[fold_idx].copy()
        df_fold_train = df_data[~df_data.index.isin(df_fold_test.index)].copy()
        
        # Normalize data
        scaler = MinMaxScaler()
        df_fold_train[lst_numerical] = scaler.fit_transform(df_fold_train[lst_numerical])
        df_fold_test[lst_numerical] = scaler.transform(df_fold_test[lst_numerical])
        # Create csv files with folds
        df_fold_train.to_csv(path_out_folds.joinpath(f'{fold_n}_train.csv'), index=False, float_format='%.8f')
        df_fold_test.to_csv(path_out_folds.joinpath(f'{fold_n}_test.csv'), index=False, float_format='%.8f')



def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    download_german_credit_data()
    metadata_german_credit_data()
    prepare_german_credit_data()
    stratified_k_folds_german_credit_data()
    
