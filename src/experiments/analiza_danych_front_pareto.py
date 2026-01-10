import polars as pl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colormaps

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


pl_data = pl.read_parquet('experiments/output/experiments_data.parquet')
lst_cat_columns = ['dataset', 'balanced', 'cpu_gpu', 'optimizer', 'lr_mili']
pl_data = pl_data.with_columns([
    pl.col(c).cast(pl.Categorical) for c in lst_cat_columns
])
lst_conditions = [pl.col('dataset') == 'lending-club', pl.col('balanced') == 'balanced', pl.col('cpu_gpu') == 'cpu']
lst_grouping = [pl.col('model_config_num'), pl.col('optimizer'), pl.col('epoch'), pl.col('layers')]

filter_columns = []
for cond in lst_conditions:
    filter_columns.extend(cond.meta.root_names())

lst_model_parameters = ['model_config_num',	'fold',	 
                        'epoch', 'layers', 'optimizer']

lst_metrics = [ 
 'val_loss',
 'val_binary_accuracy',
 'val_auc',
 'val_precision',
 'val_recall',
 'train_time',
 'validation_time',
 'train_pred_time',
 'test_pred_time']
pl_data_filtered = pl_data.filter(lst_conditions).select(lst_model_parameters+lst_metrics)
pl_data_agg = pl_data_filtered.group_by(lst_grouping).agg([
    pl.all().mean().name.suffix("_mean"),
    pl.all().max().name.suffix("_max"),
    pl.all().min().name.suffix("_min"),
])
pl_data_agg.shape

# Wszystkie razem
fig = plt.figure(figsize=(8.27, 11.69))
fig.subplots_adjust(hspace=0.1)
# plt.axis('off')
# fig.text(0.5, 0.5, 'Strona A4 - miejsce na wykresy', 
#          ha='center', va='center', fontsize=16)
spec = fig.add_gridspec(2, 1)
ax = fig.add_subplot(spec[0])
np_data = pl_data_agg.select('val_precision_mean', 'val_recall_mean').to_numpy()
ax.scatter(np_data[:,0], np_data[:,1], alpha=0.5, s=10, marker = '.')
ax.set_xlim(0, 1.01)
ax.set_ylim(0, 1.01)
# fig.savefig('reports/figures/strona.pdf')


# każda epoka z oddzielnym kolorem
fig = plt.figure(figsize=(8.27, 11.69))
fig.subplots_adjust(hspace=0.1)
# plt.axis('off')
# fig.text(0.5, 0.5, 'Strona A4 - miejsce na wykresy', 
#          ha='center', va='center', fontsize=16)
spec = fig.add_gridspec(2, 2)
ax = fig.add_subplot(spec[0])
ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
lst_epochs = pl_data_agg.select('epoch').unique().sort('epoch').to_numpy().flatten()
cmap = colormaps['coolwarm'].resampled(len(lst_epochs))
lst_colors = cmap(np.linspace(0, 1, len(lst_epochs)))
lst_columns = ['val_precision_mean', 'val_recall_mean']
for epoch in lst_epochs:
    np_data = pl_data_agg.filter(pl.col('epoch') == epoch).select(lst_columns).to_numpy()
    ax.scatter(np_data[:,0], np_data[:,1], alpha=0.7, s=10, marker = '.', color=lst_colors[epoch-1])
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    
# fig.savefig('reports/figures/strona.pdf')
ax = fig.add_subplot(spec[1])
np_domiated_indices, np_non_dominated = znajdz_pareto_max_numpy(pl_data_agg, lst_columns)
np_data = pl_data_agg[np_non_dominated].select(lst_columns)
ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
ax.scatter(np_data[:,0], np_data[:,1], alpha=0.7, s=10, marker = '.', color='blue')
ax.set_xlim(0, 1.01)
ax.set_ylim(0, 1.01)

np_non_dominated.shape
lst_grouping_reordered = ['model_config_num','layers', 'optimizer','epoch']
pl.Config.set_tbl_rows(-1)
pl_data_agg_params = pl_data_agg[np_non_dominated].select(lst_grouping_reordered).sort(lst_grouping_reordered)
pl_data_agg_params.select('model_config_num').group_by('model_config_num').len().sort('model_config_num')
pl_data_agg_params.select('layers').group_by('layers').len().sort('layers')
pl_data_agg_params.select('optimizer').group_by('optimizer').len().sort('optimizer')
pl_data_agg_params.select('epoch').group_by('epoch').len().sort('epoch')
plt.close(fig)

