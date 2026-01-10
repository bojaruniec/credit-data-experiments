import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps


def analyze():
    """
    Analiza danych
    """
    pl_data = pl.read_parquet('experiments/output/experiments_data.parquet')
    pl_data.shape
    
    pl_datat1 = pl_data.filter(pl.col('balanced')=='balanced', 
               pl.col('cpu_gpu')=='cpu', 
               pl.col('dataset') == 'give-me-some-credit',
               pl.col('fold') == 1)
    
    pl_data1.shape
    
    lst_metrics = ['dataset',
    'cpu_gpu',
    'balanced',
    'model_config_num',
    'fold',
    'epoch',
    'layers',
    'optimizer',
    'lr_mili',
    'auc',
    'binary_accuracy',
    'false_negatives',
    'false_positives',
    'loss',
    'precision',
    'recall',
    'true_negatives',
    'true_positives',
    'val_auc',
    'val_binary_accuracy',
    'val_false_negatives',
    'val_false_positives',
    'val_loss',
    'val_precision',
    'val_recall',
    'val_true_negatives',
    'val_true_positives',
    'train_time',
    'validation_time',
    'model_weights',
    'model_parameters_non_zero',
    'model_sparsity',
    'ignored_variables_count',
    'train_pred_time',
    'test_pred_time']
    show_scatter_plot_epochs(pl_data1, 'val_precision', 'val_recall')
    show_scatter_plot_epochs(pl_data1, 'val_loss', 'val_binary_accuracy')
    show_scatter_plot_epochs(pl_data1, 'val_loss', 'val_precision')
    show_scatter_plot_epochs(pl_data1, 'val_loss', 'val_recall')
    show_scatter_plot_epochs(pl_data1, 'val_binary_accuracy', 'train_time')

    pl_front_pareto = znajdz_pareto_max(pl_data1, 'val_precision', 'val_recall')
    pl_front_pareto.group_by(pl.col('epoch')).len().sort('len', descending=True)
    pl_front_pareto.group_by(pl.col('optimizer')).len().sort('len', descending=True)
    pl_front_pareto.group_by(pl.col('model_config_num')).len().sort('len', descending=True)
    pl_front_pareto.group_by(pl.col('layers')).len().sort('len', descending=True)
    

def znajdz_pareto_max(df, x_col, y_col):
    pareto_df = (
        df.sort([pl.col(x_col).reverse(), pl.col(y_col).reverse()])
        .with_columns(pl.col(y_col).cum_max().alias("max_y"))
        .filter(pl.col(y_col) == pl.col("max_y"))
        .drop("max_y")
        .unique(subset=[x_col, y_col])
    )
    return pareto_df


def znajdz_pareto_max_numpy(pl_data, lst_cols):
    points = pl_data.select(lst_cols).to_numpy()
    n = points.shape[0]
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        # Check if any other point dominates point i
        dominating = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        dominating[i] = False  # Don't compare with itself
        if np.any(dominating):
            is_dominated[i] = True
    # non_dominated_points = points[~is_dominated]
    dominated_indices_np = np.where(is_dominated)[0]
    non_dominated_indices_np = np.where(~is_dominated)[0]
    return dominated_indices_np, non_dominated_indices_np

def analiza_test():
    # the more criteria, the more non-dominated solutions
    dom, non_dom = znajdz_pareto_max_numpy(pl_data1, ['val_precision', 'val_recall'])
    print(len(non_dom))
    dom, non_dom = znajdz_pareto_max_numpy(pl_data1, ['val_precision', 'val_recall', 'auc'])
    print(len(non_dom))
    dom, non_dom = znajdz_pareto_max_numpy(pl_data1, ['val_precision', 'val_recall', 'auc', 'train_time'])
    print(len(non_dom))

    dom_series = pl.Series(name="is_dominated", values=dom, dtype=pl.Boolean)
    dominated_df = pl_data1.filter(dom_series)
    pl_data1[non_dom].shape


def show_scatter_plot_epochs(pl_data, x_metric, y_metric):
    x = pl_data[x_metric]
    y = pl_data[y_metric]
    c = pl_data['epoch']
    cmap = colormaps["coolwarm"].resampled(c.max())
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, s=5, c=c, cmap=cmap, alpha=0.7, edgecolor='w', linewidth=0.3)  # s=10 makes dots smaller than the default
    
    # Add colorbar (legend for color mapping) on the right
    cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(1, 31))
    cbar.set_label('Epoch Number', fontsize=12)
    cbar.set_ticks(np.arange(1, 31, 2))  # Show every 2nd epoch for clarity

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title('Scatter plot with smaller dots')
    plt.show()
    
if __name__ == "__main__":
    pass