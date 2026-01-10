import matplotlib.pyplot as plt
import polars as pl

# metriki do analizy
lst_metrics = ['val_auc',
 'val_binary_accuracy',
 'val_loss',
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
spec = fig.add_gridspec(5, 2)

# plt.axis('off')
fig.text(0.5, 0.5, 'Strona A4 - miejsce na wykresy', 
         ha='center', va='center', fontsize=16)

# WIERSZ 1
# wykres wszystkich foldów
ax11 = fig.add_subplot(spec[0, 0])
ax11.tick_params(labelleft=True, left=True, right=True, labelright=False)
np_folds = pl_data_filtered.select('fold').unique().sort('fold').to_numpy().flatten()
for fold in np_folds:
    np_data = pl_data_filtered.filter(pl.col('fold') == fold).select('epoch', 'val_binary_accuracy').to_numpy()
    ax11.plot(np_data[:,0], np_data[:,1], label=f'Fold {fold}')

# tylko średnich ze śladem 
ax12 = fig.add_subplot(spec[0, 1], sharey=ax11)
ax12.tick_params(labelleft=False, left=True, right=True, labelright=True)

# obszar zakresu
np_data = pl_data_agg.select('epoch', 'val_binary_accuracy_min', 'val_binary_accuracy_max').to_numpy()
ax12.fill_between(np_data[:,0], np_data[:,1], np_data[:,2], alpha=0.2)

# wykres średniej
np_data = pl_data_agg.select('epoch', 'val_binary_accuracy_mean').to_numpy()
ax12.plot(np_data[:,0], np_data[:,1])


fig.savefig('reports/figures/strona.pdf')
plt.close(fig)
