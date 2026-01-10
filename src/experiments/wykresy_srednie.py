import matplotlib.pyplot as plt
import polars as pl

# metriki do analizy
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

# wczytywanie danych
pl_data = pl.read_parquet('experiments/output/experiments_data.parquet')
lst_conditions = [pl.col('dataset') == 'lending-club', pl.col('balanced') == 'balanced', 
                  pl.col('model_config_num')== 20, pl.col('layers') == 3, pl.col('optimizer') == 'adam',
                  pl.col('cpu_gpu') == 'cpu']
lst_columns = ['fold','epoch',] + lst_metrics
pl_data_filtered = pl_data.filter(lst_conditions).select(lst_columns).sort('fold', 'epoch')
np_epochs = pl_data_filtered.select('epoch').unique().sort('epoch').to_numpy().flatten()

pl_data_agg = pl_data_filtered.group_by("epoch").agg([
    pl.all().mean().name.suffix("_mean"),
    pl.all().max().name.suffix("_max"),
    pl.all().min().name.suffix("_min"),
]).sort('epoch')

# pl_data_filtered.select('epoch', 'fold', 'val_binary_accuracy')
# mean over folds
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

fig = plt.figure(figsize=(8.27, 11.69))

fig.subplots_adjust(hspace=0.3)
# plt.axis('off')
# fig.text(0.5, 0.5, 'Strona A4 - miejsce na wykresy', 
#          ha='center', va='center', fontsize=16)

spec = fig.add_gridspec(5, 2)
for metric, ax_spec in zip(lst_metrics, spec):
    ax = fig.add_subplot(ax_spec)    
    # obszar min / max
    np_data = pl_data_agg.select('epoch', f'{metric}_min', f'{metric}_max').to_numpy()
    ax.fill_between(np_data[:,0], np_data[:,1], np_data[:,2], alpha=0.2)
    
    # średnia
    np_data = pl_data_agg.select('epoch', f'{metric}_mean').to_numpy()
    ax.plot(np_data[:,0], np_data[:,1])
    ax.set_xlim(np_epochs.min(), )
    ax.set_title(metric)

fig.savefig('reports/figures/strona.pdf')
plt.close(fig)
