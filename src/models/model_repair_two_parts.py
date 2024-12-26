import pandas as pd
import csv
from pathlib import Path
from src.data.make_dataset import get_list_of_numerical_variables, get_data_by_fold
from src.models.model_parameters import  models_and_parameters
from src.models.model_experiment import write_experiments_by_fold_to_csv

def check_experiment_copletness():
    model_key = 'perceptron'
    set_model_num = set()
    dir_report = Path('reports/german-credit-data/perceptron/202412162133')
    for csv_file in dir_report.glob('[0-9]*.csv'):
        with open(csv_file, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                set_model_num.add(int(line.split(',',1)[0]))
            print(len(set_model_num))
    print(max(set_model_num))
    
    
def calclulate_means_from_existing():
    model_key = 'perceptron'
    set_model_num = set()
    dir_report = Path('reports/german-credit-data/perceptron/202412162133')
    n_model_num = 0
    lst_models = list(models_and_parameters(model_key, n_test=100))
    lst_means_all = []
    for csv_file in dir_report.glob('[0-9]*.csv'):
        print(csv_file)
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = []
            model_num_prev = 0
            model_num_curr = 1
            for row in reader:
                model_num_curr = int(row['model_num'])
                if model_num_prev == model_num_curr:
                    data.append(row)
                elif len(data) > 0:
                    n_model_num += 1
                    dic_means = pd.DataFrame.from_dict(data).astype(float).mean().round(8).to_dict()
                    model_num = model_num_curr
                    model = lst_models[model_num]
                    dic_means_to_return = {'model_num': int(model_num)} | {'model_class': model.__class__.__name__} | dict(model.get_params()) | dic_means
                    lst_means_all.append(dic_means_to_return)
                    data = []
                model_num_prev = model_num_curr
                # if n_model_num > 10:
                #     break
    write_experiments_by_fold_to_csv(lst_means_all, "means", dir_report)
    
def move_to_one_folder():
    csv_path1 = Path('reports/german-credit-data/perceptron/202412162133/means.csv')
    csv_path2 = Path('reports/german-credit-data/perceptron/202412191348/means.csv')
    df1 = pd.read_csv(csv_path1, index_col=0, dtype={'model_num':int}).sort_index()
    df2 = pd.read_csv(csv_path2, index_col=0, dtype={'model_num':int}).sort_index()


def join_two_dataframes():
    csv_path1 = Path('reports/german-credit-data/perceptron/202412162133/means.csv')
    csv_path2 = Path('reports/german-credit-data/perceptron/202412191348/means.csv')
    df1 = pd.read_csv(csv_path1, index_col=0, dtype={'model_num':int}).sort_index()
    df2 = pd.read_csv(csv_path2, index_col=0, dtype={'model_num':int}).sort_index()

if __name__ == "__main__":
    # german_experiment()
    calclulate_means_from_existing()
