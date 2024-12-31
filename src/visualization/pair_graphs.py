import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from pathlib import Path

def pair_graphs(df, model_key):
    num_vars = df.shape[1]
    lst_columns = df.columns
    num_plots = (num_vars * (num_vars - 1)) // 2  # Liczba par (combinations)
    grid_size = int(num_plots**0.5) + 1  # Liczba wierszy i kolumn w siatce

    # Tworzenie siatki subplotów
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(50, 50))
    axes = axes.flatten()

    # Tworzenie scatter plotów dla każdej pary zmiennych
    for i, (col1, col2) in enumerate(combinations(lst_columns, 2)):
        axes[i].scatter(df[col1], df[col2], alpha=0.7, marker =',')
        axes[i].set_title(f"{i}. {col1.replace('_score','')} vs {col2.replace('_score','')}")
        axes[i].set_xlabel(col1)
        axes[i].set_ylabel(col2)
        axes[i].grid(True)

    # Ukrywanie pustych osi
    for ax in axes[num_plots:]:
        ax.axis('off')

    # Dodawanie tytułu do całej siatki
    fig.suptitle(model_key, fontsize=20, fontweight='bold')
    plt.tight_layout()
        
    # Zapis do pliku PDF
    dir_output = Path(r'reports/figures/german-credit-data/')
    dir_output.mkdir(parents=True, exist_ok=True)
    
    png_output = dir_output / f'{model_key}.png'
    fig.savefig(png_output, format='png')
    print(png_output)

def pair_graphs_german():
    csv_experiments = Path('reports/german-credit-data/knn/202412152158/means.csv')
    lst_metrics = ['model_num', 'time_train', 'model_size', 'time_pred', 'accuracy_score_train',
                   'f1_score_train', 'precision_score_train', 'recall_score_train',
                   'roc_auc_score_train', 'accuracy_score_test', 'f1_score_test', 
                   'precision_score_test','recall_score_test','roc_auc_score_test']
    # KNN
    lst_graphs1 = [
        ('knn', Path('reports/german-credit-data/knn/202412152158/means.csv')),
        ('decision-tree', Path('reports/german-credit-data/decision-tree/202412160820/means.csv')),
        ('logistic-regression', Path('reports/german-credit-data/logistic-regression/202412160832/means.csv')),
        ('random-forest', Path('reports/german-credit-data/random-forest/202412161052/means.csv')),
    ]
    
    lst_graphs2 = [
        ('perceptron', Path('reports/german-credit-data/perceptron/202412162133/means.csv')),
        ('perceptron', Path('reports/german-credit-data/perceptron/202412162133/means.csv')),
    ]
    
    lst_graphs3 = [
        ('dnn1', Path('reports/german-credit-data/dnn1/202412301742/means.csv')),
        ('dnn2', Path('reports/german-credit-data/dnn2/202412162133/means.csv')),
        ('dnn3', Path('reports/german-credit-data/dnn3/202412310239/means.csv')),
    ]
    
    lst_graphs = lst_graphs1 + lst_graphs2
    for model_key, csv_experiments in lst_graphs3:
        df = pd.read_csv(csv_experiments, usecols=lst_metrics, index_col='model_num')
        pair_graphs(df, model_key)
    
if __name__ == "__main__":
    pair_graphs_german()