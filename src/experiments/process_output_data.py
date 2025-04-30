import polars as pl
from xlsxwriter import Workbook
from collections import defaultdict
from pathlib import Path


def wybierz_najnowsze_pliki(dir_input):
    """
    Wyniki eksperymentów mogą odnosić się do kilku prób. Wybieram zawsze najnowsz
    """
    def extract_prefix_and_timestamp(path):
        # Rozdziela nazwę na prefix i timestamp
        stem = path.stem  # bez rozszerzenia
        *prefix_parts, timestamp = stem.split('_')
        prefix = '_'.join(prefix_parts)
        return prefix, timestamp 
    
    gen_pliki = Path(dir_input).rglob('*.csv')
    files_by_prefix = defaultdict(list)
    for p in gen_pliki:
        # check if p is a file:
        if not p.is_file():
            continue
        prefix, timestamp = extract_prefix_and_timestamp(p)
        files_by_prefix[prefix].append((timestamp, p))
    latest_files = []
    for prefix, files in files_by_prefix.items():
        # Sortowanie po timestampie (stringowo, bo format YYYYMMDDHHMMSS)
        latest = max(files, key=lambda x: x[0])
        latest_files.append(latest[1])
    return latest_files

def folder_structure():
    """
    1. Sprawdzam strukturę folderów - wyniki eksperymentów 
    """
    dir_experiments = Path('experiments/output')
    lst_datasets = ['japanese', 'credit-approval', 'german-credit-data', 'lending-club', 'give-me-some-credit']
    lst_balanced = ['balanced', 'unbalanced']
    lst_cpu_gpu = ['cpu', 'gpu']
    for dataset in lst_datasets:
        for balanced in lst_balanced:
            for cpu_gpu in lst_cpu_gpu:
                folder_path = dir_experiments.joinpath(f'{dataset}/{cpu_gpu}/{balanced}/')
                print(folder_path, len(list(folder_path.rglob('*.csv'))), len(wybierz_najnowsze_pliki(folder_path)))
                

def read_into_polars_and_save_parquet():
    dir_experiments = Path('experiments/output')
    lst_datasets = ['japanese', 'credit-approval', 'german-credit-data', 'lending-club', 'give-me-some-credit']
    lst_balanced = ['balanced', 'unbalanced']
    lst_cpu_gpu = ['cpu', 'gpu']
    lst_dfs0 = []   
    for dataset in lst_datasets:
        for balanced in lst_balanced:
            for cpu_gpu in lst_cpu_gpu:
                print(dataset, balanced, cpu_gpu)
                folder_path = dir_experiments.joinpath(f'{dataset}/{cpu_gpu}/{balanced}/')
                lst_files = wybierz_najnowsze_pliki(folder_path)
                pl_data = pl.scan_csv(lst_files).with_columns(
                    pl.lit(dataset).alias('dataset'),
                    pl.lit(balanced).alias('balanced'),
                    pl.lit(cpu_gpu).alias('cpu_gpu')
                )
                lst_dfs0.append(pl_data)
    cols_to_move = ['dataset', 'cpu_gpu', 'balanced',]
    pl_data_all = pl.concat(lst_dfs0)
    all_cols = pl_data_all.collect_schema().names()
    remaining_cols = [col for col in all_cols if col not in cols_to_move]
    new_col_order = cols_to_move + remaining_cols
    pl_data_all = pl_data_all.select(new_col_order).collect()
    pl_data_all.write_parquet('experiments/output/experiments_data.parquet')

def read_from_parquet_file():
    pl_data = pl.read_parquet('experiments/output/experiments_data.parquet')
    pl_data.shape
    pl_data.estimated_size("gb")

#                   cpu-balanced, cpu-unbalanced, gpu-balanced, gpu-unbalanced
# credit-approval:  x              x                not             not
# give-me-some credit

# arranging the output data - credit approval
def arrange_credit_approval_data():
    
    # write to experimetns / output / experiments_data.xlsx
    # Datasets
    ## CPU
    # balanced
    dir_input= 'experiments/output/credit-approval/cpu/balanced/**/*.csv'
    pl_data1 = pl.read_csv(dir_input)
    pl_data1.get_column('optimizer').value_counts()
    pl_data1.get_column('model_config_num').unique().len()
    pl_data1 = pl_data1.with_columns(cpu_gpu = pl.lit('cpu')).with_columns(balnaced = pl.lit('balanced'))
    
    # unbalanced 
    dir_input= 'experiments/output/credit-approval/cpu/unbalanced/**/*.csv'
    pl_data2 = pl.read_csv(dir_input)
    pl_data2.get_column('optimizer').value_counts()
    pl_data2.get_column('model_config_num').unique().len()
    pl_data2 = pl_data2.with_columns(cpu_gpu = pl.lit('cpu')).with_columns(balnaced = pl.lit('unbalanced'))

    # concatenate two polars dataframe
    pl_data = pl.concat([pl_data1, pl_data2]) 
    cols_to_move = ['cpu_gpu', 'balnaced',]
    all_cols = pl_data.columns
    remaining_cols = [col for col in all_cols if col not in cols_to_move]
    new_col_order = cols_to_move + remaining_cols
    pl_data_reordered = pl_data.select(new_col_order)

    ## GPU
    dir_input= 'experiments/output/credit-approval/gpu/balanced/**/*.csv'
    pl_data1 = pl.read_csv(dir_input)
    pl_data1.get_column('optimizer').value_counts()
    pl_data1.get_column('model_config_num').unique().len()
    pl_data1 = pl_data1.with_columns(cpu_gpu = pl.lit('gpu')).with_columns(balnaced = pl.lit('balanced'))
    
    # unbalanced 
    dir_input= 'experiments/output/credit-approval/gpu/unbalanced/**/*.csv'
    pl_data2 = pl.read_csv(dir_input)
    pl_data2.get_column('optimizer').value_counts()
    pl_data2.get_column('model_config_num').unique().len()
    pl_data2 = pl_data2.with_columns(cpu_gpu = pl.lit('gpu')).with_columns(balnaced = pl.lit('unbalanced'))

    # concatenate two polars dataframe
    pl_data = pl.concat([pl_data1, pl_data2]) 
    cols_to_move = ['cpu_gpu', 'balnaced',]
    all_cols = pl_data.columns
    remaining_cols = [col for col in all_cols if col not in cols_to_move]
    new_col_order = cols_to_move + remaining_cols
    pl_data_reordered = pl_data.select(new_col_order)
    
    print("\nReordered DataFrame:")
    print(pl_data_reordered)
    
    


        
def arrange_give_me_some_credit():
    # write to experimetns / output / experiments_data.xlsx
    # Datasets
    dataset = 'give-me-some-credit'
    ## CPU
    # balanced
    dir_input= 'experiments/output/give-me-some-credit/cpu/balanced/**/*.csv'
    lst_files = wybierz_najnowsze_pliki(dir_input)
    dfs = [pl.read_csv(str(p)) for p in lst_files]
    pl_data1 = pl.concat(dfs)
    pl_data1.get_column('optimizer').value_counts()
    pl_data1.get_column('model_config_num').unique().len()
    pl_data1 = pl_data1.with_columns(cpu_gpu = pl.lit('cpu')).with_columns(balnaced = pl.lit('balanced'))
    
    # unbalanced 
    dir_input= 'experiments/output/credit-approval/cpu/unbalanced/**/*.csv'
    pl_data2 = pl.read_csv(dir_input)
    pl_data2.get_column('optimizer').value_counts()
    pl_data2.get_column('model_config_num').unique().len()
    pl_data2 = pl_data2.with_columns(cpu_gpu = pl.lit('cpu')).with_columns(balnaced = pl.lit('unbalanced'))

    # concatenate two polars dataframe
    pl_data = pl.concat([pl_data1, pl_data2]) 
    cols_to_move = ['cpu_gpu', 'balnaced',]
    all_cols = pl_data.columns
    remaining_cols = [col for col in all_cols if col not in cols_to_move]
    new_col_order = cols_to_move + remaining_cols
    pl_data_reordered = pl_data.select(new_col_order)

    ## GPU
    dir_input= Path(f'experiments/output/{dataset}/gpu/balanced/')
    lst_files = wybierz_najnowsze_pliki(dir_input)
    dfs = [pl.read_csv(str(p)) for p in lst_files]
    pl_data1 = pl.concat(dfs)
    pl_data1.get_column('optimizer').value_counts()
    pl_data1.get_column('model_config_num').unique().len()
    pl_data1 = pl_data1.with_columns(cpu_gpu = pl.lit('gpu')).with_columns(balnaced = pl.lit('balanced'))
    
    # unbalanced 
    dir_input= Path(f'experiments/output/{dataset}/gpu/unbalanced/')
    lst_files = wybierz_najnowsze_pliki(dir_input)
    dfs = [pl.read_csv(str(p)) for p in lst_files]
    pl_data2 = pl.concat(dfs)
    pl_data2.get_column('optimizer').value_counts()
    pl_data2.get_column('model_config_num').unique().len()
    pl_data2 = pl_data2.with_columns(cpu_gpu = pl.lit('gpu')).with_columns(balnaced = pl.lit('unbalanced'))

    # concatenate two polars dataframe
    pl_data = pl.concat([pl_data1, pl_data2]) 
    cols_to_move = ['cpu_gpu', 'balnaced',]
    all_cols = pl_data.columns
    remaining_cols = [col for col in all_cols if col not in cols_to_move]
    new_col_order = cols_to_move + remaining_cols
    pl_data_reordered = pl_data.select(new_col_order)
    
    pl_data_reordered
    
    print("\nReordered DataFrame:")
    print(pl_data_reordered)
    
    # Problem jest taki, że nie mamy 
    pl_data_reordered

    
    
def load_experiments_data_with_pandas_library():    
    # With pandas library
    dir_input= 'experiments/output/credit-approval/balanced/'
    df = pd.DataFrame()
    for file in Path(dir_input).rglob('*.csv'):
        df = pd.concat([df, pd.read_csv(file)])
    print(df.shape)
    df.size
    
if __name__ == "__main__":
    read_into_polars_and_save_parquet()
    #folder_structure()