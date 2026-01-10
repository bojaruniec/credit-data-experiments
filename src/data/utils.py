import pandas as pd
from pathlib import Path

def get_list_of_numerical(inp_folder:str) -> list:
    """
    Returns the list of numerical variable for a given dataset. 
    """
    inp_folder = 'japanese'
    path_metadata_csv = next(Path(f'data/processed/{inp_folder}').glob('*_metadata.csv'))

    df_metadata = pd.read_csv(path_metadata_csv)
    lst_numerical = df_metadata.loc[df_metadata['type'] == 'numerical','variable'].to_list()
    return lst_numerical