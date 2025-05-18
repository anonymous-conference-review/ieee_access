from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import seaborn as sns
import torch
import wandb
from joblib import delayed, Parallel
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from data import get_dataset
from jumpstart.metrics import JRMetricManager
from util import to_latex, to_boxplot, download_metrics


# pio.kaleido.scope.mathjax = None


def generate_key_metrics_table(summaries, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else Path('tables')
    summaries = summaries.reset_index()
    summaries.loc[summaries['skip_connections'] == 'None', 'skip_connections'] = pd.NA
    summaries['is_jumpstart'] = summaries['lambda_'] > 0
    summaries['is_skip'] = summaries['skip_connections'] > 0
    summaries['is_gelu'] = summaries['activation'] == 'gelu'
    cols = {}
    names = pd.Series(['Jumpstart', 'Skip conn.', 'BatchNorm', 'GELU'])
    min_acc = pd.Series({name: group['val/acc'].min() for name, group in summaries.groupby('dataset')}).sort_index()
    max_acc = pd.Series({name: group['val/acc'].max() for name, group in summaries.groupby('dataset')}).sort_index()
    for name, group in summaries.groupby(['is_jumpstart', 'is_skip', 'use_batchnorm', 'is_gelu']):
        name = names[np.array(name)]
        if len(name) == 0:
            name = 'Baseline'
        else:
            name = ' + '.join(name)
        # more than 95% of the units have a 20% > dead units
        has_dead = (group['train/jr/unit_dead'] > 0.20).mean()
        has_linear = (group['train/jr/unit_linear'] > 0.20).mean()
        acc = group[['dataset', 'val/acc']].sort_values('val/acc', ascending=False).drop_duplicates('dataset')
        acc = acc.set_index('dataset').squeeze().sort_index()
        has_trainability = (acc > min_acc[acc.index]).mean()
        has_convergence = (max_acc[acc.index] - acc).mean()
        cols[name] = pd.Series({'Dead unit': has_dead,
                                'Linear unit': has_linear,
                                'Trainability': has_trainability,
                                'Convergence': has_convergence})
    table = pd.DataFrame(cols)
    table = table[['Baseline', 'BatchNorm', 'Skip conn.', 'Jumpstart', 'GELU']]
    caption = 'Comparison of our proposal versus common techniques to improve the training of deep neural networks. ' \
              'We compare our proposal to the baseline, jumpstart, skip connections, batch normalization and GELU activation. ' \
              'We train a model using each technique on a selection of UCI datasets and report a summary of the results. ' \
              'The first row reports the fraction of datasets in which the best model for that technique has a dead unit ratio of more than 20\%. ' \
              'The second row reports the fraction of datasets in which the best model for that technique has a linear unit ratio of more than 20\%. ' \
              'The third row reports the fraction of datasets in which the best model has a higher accuracy than the global worst accuracy attained over all techniques. ' \
              'The fourth row reports the average difference between the best model for each technique and the global best accuracy attained over all techniques. '

    gmap = table.copy()
    gmap.loc['Trainability'] = 1 - gmap.loc['Trainability']
    to_latex(table, caption, 'tab:key_metrics', output_dir / 'key_metrics.tex', cmap='Blues', gmap=gmap,
             clines=None, font_size=10)


def generate_win_eq_losses_table(summaries, skip_combinations=True, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else Path('tables')
    summaries = summaries.reset_index()
    summaries.loc[summaries['skip_connections'] == 'None', 'skip_connections'] = pd.NA
    summaries['is_jumpstart'] = summaries['lambda_'] > 0
    summaries['is_skip'] = summaries['skip_connections'] > 0
    summaries['is_gelu'] = summaries['activation'] == 'gelu'
    names = pd.Series(['Jumpstart', 'Skip conn.', 'BatchNorm', 'GELU'])
    accuracies = {}
    for name, group in summaries.groupby(['is_jumpstart', 'is_skip', 'use_batchnorm', 'is_gelu']):
        name = names[np.array(name)]
        if skip_combinations and len(name) > 1:
            pass
        else:
            if len(name) == 0:
                name = 'Baseline'
            else:
                name = ' + '.join(name)
            acc = group[['dataset', 'val/acc']].sort_values('val/acc', ascending=False).drop_duplicates('dataset')
            acc = acc.set_index('dataset').squeeze().sort_index()
            accuracies[name] = acc
    accuracies = pd.DataFrame(accuracies)
    n_datasets = len(accuracies)
    nan_accuracies = accuracies.index[accuracies.isna().any(axis=1)].to_list()
    accuracies = accuracies.dropna(axis=0)
    if len(accuracies) < n_datasets:
        print(f'WARNING: Dropped {n_datasets - len(accuracies)} datasets\n\t')
        print('\n\t'.join(nan_accuracies))
    # Example list of methods and datasets
    methods = accuracies.columns
    datasets = accuracies.index

    # Create a dataframe to store the matrix
    win_eq_loss = pd.DataFrame(index=methods, columns=methods)
    win_eq_loss = win_eq_loss.fillna(0)

    gmap = pd.DataFrame(index=methods, columns=methods)
    gmap = gmap.fillna(0)

    # Evaluate the performance of each method on each dataset
    for i, method_i in enumerate(methods):
        for j, method_j in enumerate(methods):
            if i == j:
                win_eq_loss.iloc[i, j] = '-'
            else:
                wins = 0
                draws = 0
                losses = 0
                for dataset in datasets:
                    if accuracies.loc[dataset, method_i] > accuracies.loc[dataset, method_j]:
                        wins += 1
                    elif accuracies.loc[dataset, method_i] == accuracies.loc[dataset, method_j]:
                        draws += 1
                    else:
                        losses += 1
                win_eq_loss.iloc[i, j] = f'{wins}-{draws}-{losses}'
                gmap.iloc[i, j] = wins

    to_latex(win_eq_loss,
             'Comparison of the number of wins, losses and equalities for each technique on each dataset.',
             'tab:win_eq_losses', output_dir / 'win_eq_losses.tex', gmap=gmap.astype(int), clines=None)


def generate_lambda_dataset_table(df):
    df = df.sort_values('val/acc', ascending=False).drop_duplicates(subset=['dataset', 'lambda_'], keep='first')
    df = df[['dataset', 'lambda_', 'train/acc', 'train/jr/unit_nonlinear', 'train/jr/point_nonlinear',
             'train/jr/point_nonlinearity', 'val/acc', 'val/jr/unit_nonlinear', 'val/jr/point_nonlinear',
             'val/jr/point_nonlinearity']]
    df['dataset'] = df['dataset'].str.replace('_', ' ')
    df = df.rename(columns={'lambda_': 'lambda'})
    df = df.sort_values(['dataset', 'lambda'])
    df = df.set_index(['dataset', 'lambda'])
    df.columns = df.columns.str.replace('jr/', '').str.replace('point_nonlinearity', 'global')
    df.columns = df.columns.str.replace('_nonlinear', '')
    df.columns = df.columns.str.split('/', expand=True)
    caption = ('Accuracy, ratio of nonlinear units, and nonlinear points for training and validation across a chosen '
               'set of UCI datasets.')
    to_latex(df, caption, 'tab:dataset_table', 'tables/dataset_table_lambda.tex')


def get_result_uci_table(summaries, metrics=None, labels=None, skip_combinations=True):
    print('Generating results table')
    summaries = summaries.reset_index()
    summaries.loc[summaries['skip_connections'] == 'None', 'skip_connections'] = pd.NA
    summaries['is_jumpstart'] = summaries['lambda_'] > 0
    summaries['is_skip'] = summaries['skip_connections'] > 0
    summaries['is_gelu'] = summaries['activation'] == 'gelu'
    names = pd.Series(['Jumpstart', 'Skip connections', 'BatchNorm', 'GELU'])
    table = {}
    metrics = ['val/acc',
               'train/jr/unit_dead',
               'train/jr/unit_linear',
               'train/jr/unit_nonlinear',
               'train/jr/point_dead',
               'train/jr/point_nonlinear',
               'train/jr/point_deathness',
               'train/jr/point_linearity',
               'train/jr/point_nonlinearity'] if metrics is None else metrics

    for name, group in summaries.groupby(['is_jumpstart', 'is_skip', 'use_batchnorm', 'is_gelu']):
        name = names[np.array(name)]
        if skip_combinations and len(name) > 1:
            pass
        else:
            if len(name) == 0:
                name = 'Baseline'
            else:
                name = ' + '.join(name)

            dataset_metrics = group[['dataset'] + metrics].sort_values('val/acc', ascending=False).drop_duplicates(
                'dataset')
            dataset_metrics = dataset_metrics.set_index('dataset').squeeze().sort_index()
            table[name] = dataset_metrics

    table = pd.concat(table, names=['Technique'])
    table.index = table.index.set_names(['Technique', 'Dataset'])

    return table


def get_result_imagenet_table(summaries, metrics=None, labels=None, skip_combinations=True, metric_epoch=100):
    print('Generating results table')
    summaries = summaries.reset_index()
    summaries['type'] = 'Jumpstart'
    summaries.loc[summaries['use_skip_connections'] > 0, 'type'] = 'ResNet'
    table = {}

    metrics = ['train/acc',
               'val/acc',
               'width_multiplier',
               'train/jr/unit_dead_ratio',
               'train/jr/unit_linear_ratio',
               'train/jr/unit_nonlinear_ratio',
               'train/jr/point_dead_ratio',
               'train/jr/point_nonlinear_ratio',
               'train/jr/deathness_ratio',
               'train/jr/linearity_ratio',
               'train/jr/nonlinearity_ratio',
               ] if metrics is None else metrics
    greater_is_better = ['train/acc',
                         'val/acc',
                         'train/jr/unit_nonlinear_ratio',
                         'train/jr/point_nonlinear_ratio',
                         'train/jr/nonlinearity_ratio']
    lesser_is_better = ['train/jr/unit_dead_ratio',
                        'train/jr/unit_linear_ratio',
                        'train/jr/point_dead_ratio',
                        'train/jr/deathness_ratio',
                        'train/jr/linearity_ratio']

    summaries['width_multiplier'] = summaries['width_multiplier'].fillna(2)

    for name, group in summaries.groupby(['type', 'width_multiplier']):
        maxes = group[metrics].max()
        mins = group[metrics].min()
        maxes = maxes[greater_is_better]
        mins = mins[lesser_is_better]
        dataset_metrics = pd.concat([maxes, mins])
        table[name] = dataset_metrics
    table = pd.DataFrame(table)
    table = table.T
    table.index = table.index.set_names(['Technique', 'Width multiplier'])

    return table


def generate_result_table(summaries, metrics=None, labels=None, skip_combinations=True, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else Path('tables')
    table = get_result_uci_table(summaries, metrics, labels, skip_combinations)
    gmap = table.copy()
    reverse = ['Accuracy', 'Nonlinear']
    for item, metric in gmap:
        if metric in reverse:
            gmap[(item, metric)] = 1 - gmap[(item, metric)]
    # gmap = gmap / (gmap.max(axis=0) - gmap.min(axis=0))
    # output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    to_latex(table,
             'Accuracy, ratio of nonlinear units and nonlinear points for train and val in a selection of UCI datasets.',
             'tab:results', output_dir / 'uci_results.tex', cmap='Blues', gmap=gmap, environment='table*')


def export_final_uci_table(summaries, project, entity='jumpstart', metrics=None, labels=None, skip_combinations=True,
                           output_dir=None, num_workers=-1, verbose=10, skip_last=True):
    output_dir = Path(output_dir) if output_dir is not None else Path('tables')
    table = get_result_uci_table(summaries, metrics, labels, skip_combinations)

    table['val/acc'] = table['val/acc'].round(2)
    table['val/acc/rank'] = table['val/acc'].groupby('Dataset').rank(ascending=False, method='min').astype(int)
    table.index = pd.MultiIndex.from_tuples([(x[0], x[1].replace('_', ' ').capitalize()) for x in table.index])
    labels = {'val/acc': ('', 'Accuracy'),
              'val/acc/rank': ('', 'Rank'),
              'train/jr/unit_dead': ('Unit', 'U_d'),
              'train/jr/unit_linear': ('Unit', 'U_l'),
              'train/jr/unit_nonlinear': ('Unit', 'U_n'),
              'train/jr/point_dead': ('Network', 'D^{network}_d'),
              'train/jr/point_nonlinear': ('Network', 'D^{network}_n'),
              'train/jr/point_deathness': ('Layer', 'D^{layer}_d'),
              'train/jr/point_linearity': ('Layer', 'D^{layer}_l'),
              'train/jr/point_nonlinearity': ('Layer', 'D^{layer}_n'),
              'train/jr/point_efficiency': ('Network', 'E'),

              }

    table = table.rename(columns=labels)

    columns = [
        ('', 'Rank'),
        ('', 'Accuracy'),
        ('Unit', 'U_d'),
        ('Unit', 'U_l'),
        ('Unit', 'U_n'),
        ('Layer', 'D^{layer}_d'),
        ('Layer', 'D^{layer}_l'),
        ('Layer', 'D^{layer}_n'),
        ('Network', 'D^{network}_d'),
        ('Network', 'D^{network}_n'),
        ('Network', 'E'),

    ]

    table = table[columns]
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    # Rename Early stage diabetes risk to Early stage
    table = table.rename(index={'Early stage diabetes risk': 'Early stage'})
    # Rename Jumpstart to Jumpstart (Ours)
    table = table.rename(index={'Jumpstart': 'Jumpstart (Ours)'})

    gmap = table.copy()
    reverse = ['Accuracy', 'U_n', 'D^{layer}_n' , 'D^{network}_n', 'E']
    for item, metric in gmap:
        if metric in reverse:
            gmap[(item, metric)] = 1 - gmap[(item, metric)]
    gmap = (gmap - gmap.min(axis=0)) / (gmap.max(axis=0) - gmap.min(axis=0))
    # gmap = np.log(gmap + 1)
    # gmap = gmap / (gmap.max(axis=0) - gmap.min(axis=0))
    # output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    to_latex(table,
             'The table is organized into four distinct parts: accuracy, unit, point with respect to the layer, '
             'and point with respect to the network. The first section includes accuracy of the network on each '
             'dataset along with the rankings across all the compared methods, showing that our method (Jumpstart) '
             'outperforms all other methods, closely followed by skip connections. The second part involves the '
             'ratios of dead, linear, and nonlinear units across the entire network, where our approach surpasses all '
             'other contestants, with batch normalization being the next best. Interestingly, skip connections, '
             'despite being the second in accuracy, significantly drops in this metric, indicating that most of its '
             'units are left unused. The third section shows the ratios of dead, linear, and nonlinear points across '
             'all the layers of the network. Once again, our proposal surpasses our competitors, although this metric '
             'is easier to achieve, so the difference is not as pronounced. The fourth part assesses points across '
             'the entire network, considering a point dead if it dies in any layer, even if it is linear or nonlinear '
             'in the rest. Therefore, we only report dead and nonlinear points across all layers. This more difficult '
             'metric shows our proposal as the only one able to score. This is not necessarily a fatal flaw, '
             'as a dead point may still be correctly classified, but it might hinder gradient propagation. Finally, '
             'we introduce a metric named Efficiency, which measures the average layer number in which points die, '
             'quantifying the number of layers in which the gradient flow will be impacted, and under this metric our '
             'approach greatly surpasses our competitors.',
             'tab:uci_results', output_dir / 'final_uci_results.tex', cmap='Blues', gmap=gmap, environment='table*',
             normalize='global', escape_columns=False)


def export_final_imagenet_table(summaries, project, entity='jumpstart', metrics=None, labels=None,
                                skip_combinations=True,
                                output_dir=None, num_workers=-1, verbose=10, skip_last=True, precision=2):
    output_dir = Path(output_dir) if output_dir is not None else Path('tables')
    table = get_result_imagenet_table(summaries, metrics, labels, skip_combinations)

    table['val/acc'] = table['val/acc'].round(3)
    table['train/acc'] = table['train/acc'].round(3)

    labels = {
        'train/acc': ('Accuracy', 'Train'),
        'val/acc': ('Accuracy', 'Validation'),
        'train/jr/unit_dead_ratio': ('Unit', r'\( U_d \)'),
        'train/jr/unit_linear_ratio': ('Unit', r'\( U_l \)'),
        'train/jr/unit_nonlinear_ratio': ('Unit', r'\( U_n \)'),
        'train/jr/point_dead_ratio': ('Network', r'\( D^{network}_d \)'),
        'train/jr/point_linear_ratio': ('Network', r'\( D^{network}_l \)'),
        'train/jr/point_nonlinear_ratio': ('Network', r'\( D^{network}_n \)'),
        'train/jr/deathness_ratio': ('Layer', r'\( D^{layer}_d \)'),
        'train/jr/linearity_ratio': ('Layer', r'\( D^{layer}_l \)'),
        'train/jr/nonlinearity_ratio': ('Layer', r'\( D^{layer}_n \)'),
    }
    table = table.rename(columns=labels)



    columns = [
        ('Accuracy', 'Train'),
        ('Accuracy', 'Validation'),
        ('Unit', r'\( U_d \)'),
        ('Unit', r'\( U_l \)'),
        ('Unit', r'\( U_n \)'),
        ('Layer', r'\( D^{layer}_d \)'),
        ('Layer', r'\( D^{layer}_l \)'),
        ('Layer', r'\( D^{layer}_n \)'),
        ('Network', r'\( D^{network}_d \)'),
        ('Network', r'\( D^{network}_n \)'),
    ]

    table = table[columns]
    table.columns = pd.MultiIndex.from_tuples(table.columns)

    gmap = table.copy()
    reverse = ['Accuracy', 'Nonlinear', 'Efficiency', 'Train', 'Validation']
    for item, metric in gmap:
        if metric in reverse:
            gmap[(item, metric)] = 1 - gmap[(item, metric)]

    output_dir.mkdir(exist_ok=True)

    to_latex(table,
             'This table presents a comparison between our proposed method - Jumpstart - and ResNet50. '
             'ResNet50 is explicitly tailored to mitigate the issues analyzed in this paper, with features such as '
             'progressively wider layers, skip connections, and batch normalization. These techniques '
             'result in much lower scores for dead units and dead points when compared to the 50x4 archtecture from  '
             'Table \\ref{tab:uci_results}, demonstrating its effectiveness. Our '
             'experiment shows that Jumpstart achieves similar train performance without relying on skip '
             'connections or batch normalization, even when ResNet50 base architecture is used. '
             'We trained a reduced version of ResNet50 with the width multiplier reduced to 1.5, '
             'instead of the default 2.0, in order to tackle the overfitting displayed by the full-width, '
             'with satisfactory results.',
             'tab:imagenet_results', output_dir / 'final_imagenet_results.tex', cmap='Blues', gmap=gmap,
             environment='table*',
             normalize='global',
             escape_columns=False,
             precision=precision)

def generate_accuracy_boxplots(summaries, skip_combinations=True, metrics=None, labels=None, output_dir=None,
                               columns=1):
    output_dir = Path(output_dir) if output_dir is not None else Path('plots')
    summaries = summaries.reset_index()
    summaries.loc[summaries['skip_connections'] == 'None', 'skip_connections'] = pd.NA
    summaries['is_jumpstart'] = summaries['lambda_'] > 0
    summaries['is_skip'] = summaries['skip_connections'] > 0
    summaries['is_gelu'] = summaries['activation'] == 'gelu'
    names = pd.Series(['Jumpstart', 'Skip conn.', 'BatchNorm', 'GELU'])
    metric_data = {}
    metrics = ['val/acc', 'train/jr/unit_dead', 'train/jr/unit_linear', 'train/jr/unit_nonlinear',
               'train/jr/point_dead', 'train/jr/point_linear',
               'train/jr/point_nonlinear'] if metrics is None else metrics

    labels = {'val/acc': 'Accuracy', 'train/jr/unit_dead': 'Dead units', 'train/jr/unit_linear': 'Linear units',
              'train/jr/unit_nonlinear': 'Nonlinear units', 'train/jr/point_dead': 'Dead points',
              'train/jr/point_linear': 'Linear points',
              'train/jr/point_nonlinear': 'Nonlinear points'} if labels is None else labels
    for name, group in summaries.groupby(['is_jumpstart', 'is_skip', 'use_batchnorm', 'is_gelu']):
        name = names[np.array(name)]
        if skip_combinations and len(name) > 1:
            pass
        else:
            if len(name) == 0:
                name = 'Baseline'
            else:
                name = ' + '.join(name)

            dataset_metrics = group[['dataset'] + metrics].sort_values('val/acc', ascending=False).drop_duplicates(
                'dataset')
            dataset_metrics = dataset_metrics.set_index('dataset').squeeze().sort_index()
            metric_data[name] = dataset_metrics

    metric_data = pd.concat(metric_data, names=['technique'])
    metric_data = metric_data.pivot_table(index='technique', columns='dataset', values=metrics).T
    for metric_name in metrics:
        ylabel = labels[metric_name]
        metric_name_str = metric_name.replace('jr/', '').split('/')[1]

        to_boxplot(ylabel, metric_data.loc[metric_name], output_dir / f'boxplot_{metric_name_str}.pdf', columns=columns)
        # to_violinplot(metric_name, metric_data.loc[metric_name], output_dir / f'violinplot_{metric_name_str}.pdf')


def generate_dataset_metadata_table(datasets, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else Path('tables')
    datasets = datasets.reset_index()['dataset'].unique()
    data = {}
    for dataset_name in datasets:
        config = {'dataset': dataset_name, 'batch_size': 128, 'num_workers': 0, 'seed': 0}
        data_module, n_classes, n_features = get_dataset(config)
        dataset_metadata = [n_features, n_classes, len(data_module.train_dataset), len(data_module.val_dataset)]
        if data_module.test_dataset is not None:
            dataset_metadata += [len(data_module.test_dataset)]
        data[dataset_name] = dataset_metadata
    index = ['n_features', 'n_classes', 'n_train', 'n_val', ]
    if data_module.test_dataset is not None:
        index += ['n_test']
    data = pd.DataFrame(data, index=index).T
    data = data.sort_index()
    data = data.astype(int)
    data = data.rename(
        columns={'n_features': 'Features', 'n_classes': 'Classes', ' n_train': 'Train', 'n_val': 'Validation',
                 'n_test': 'Test'})
    data.index = data.index.str.replace('_', ' ').str.capitalize()
    data = data.rename_axis('Dataset')
    to_latex(data, caption='UCI datasets used in the experiments.', label='tab:dataset_metadata', cmap=None,
             output_path=output_dir / 'dataset_metadata.tex',
             escape='latex', clines=None)


def generate_accuracy_correlation_scatter(df, yaxis='val/acc', metrics=None, labels=None, output_dir='plots'):
    print(f'Exporting accuracy correlation plots to {output_dir}')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    metrics = [
        'train/jr/unit_dead',
        'train/jr/unit_linear',
        'train/jr/unit_nonlinear',
        'train/jr/point_dead',
        'train/jr/point_linear',
        'train/jr/point_nonlinear',
        'train/jr/point_deathness',
        'train/jr/point_linearity',
        'train/jr/point_nonlinearity'] if metrics is None else metrics

    labels = {'val/acc': 'Accuracy',
              'train/jr/unit_dead': 'Dead unit',
              'train/jr/unit_linear': 'Linear unit',
              'train/jr/unit_nonlinear': 'Nonlinear unit',
              'train/jr/point_dead': 'Dead point',
              'train/jr/point_linear': 'Linear point',
              'train/jr/point_nonlinear': 'Nonlinear point',
              'train/jr/point_deathness': 'Dead global',
              'train/jr/point_linearity': 'Linear global',
              'train/jr/point_nonlinearity': 'Nonlinear global'} if labels is None else labels

    for metric in metrics:
        sns.scatterplot(data=df, y=yaxis, x=metric, legend=False)

        plt.ylabel(labels[yaxis])
        plt.xlabel(labels[metric])
        plt.tight_layout()
        metric_str = metric.replace('/', '_')
        yaxis_str = yaxis.replace('/', '_')
        filepath = output_dir / f'{yaxis_str}-{metric_str}.pdf'
        plt.savefig(filepath)
        plt.close()


def generate_accuracy_correlation_heatmap(df, yaxis='val/acc', metrics=None, labels=None, output_dir='plots',
                                          normalize=True, normalization_mode='minmax', split_legend=True):
    print(f'Exporting accuracy heatmap plots to {output_dir}')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    metrics = [
        'train/jr/unit_nonlinear',
        'train/jr/point_nonlinear',
    ] if metrics is None else metrics
    df = df.copy()

    df['is_jumpstart'] = df['lambda_'] > 0
    df['is_skip'] = df['skip_connections'] > 0
    df['is_gelu'] = df['activation'] == 'gelu'
    df['is_baseline'] = ~df[['is_jumpstart', 'is_skip', 'use_batchnorm', 'is_gelu']].any(axis=1)
    names = pd.Series(['Jumpstart', 'Skip connections', 'BatchNorm', 'GELU'])
    # Drop if they use more than a single trick
    df = df[df[['is_jumpstart', 'is_skip', 'use_batchnorm', 'is_gelu']].sum(axis=1) <= 1].copy()
    # pick only the best accuracies across lambda values
    jumpstart = df.loc[df['is_jumpstart']]
    jumpstart = jumpstart.sort_values('val/acc', ascending=False).drop_duplicates(['lr', 'dataset'])
    df = pd.concat([df.loc[~df['is_jumpstart']], jumpstart])
    method = df[['is_baseline', 'is_jumpstart', 'is_skip', 'use_batchnorm', 'is_gelu']].idxmax(axis=1)
    df['method'] = method.map({'is_baseline': 'Baseline', 'is_jumpstart': 'Jumpstart', 'is_skip': 'Skip connections',
                               'use_batchnorm': 'BatchNorm', 'is_gelu': 'GELU'})

    if normalize:
        if yaxis != 'val/acc':
            raise ValueError('yaxis must be val/acc when normalizing')
        # Divide yaxis by the trivial accuracy
        datasets = df['dataset'].unique()
        for dataset_name in datasets:
            config = {
                'dataset': dataset_name,
                'batch_size': 128,
                'num_workers': 0,
                'seed': 0
            }
            datamodule, _, _ = get_dataset(config)
            targets = pd.Series(datamodule.val_dataset.Y)
            trivial_accuracy = targets.value_counts().max() / len(targets)
            best_accuracy = df.loc[df['dataset'] == dataset_name, 'val/acc'].max()
            if normalization_mode == 'max_trivial':
                df.loc[df['dataset'] == dataset_name, 'val/acc'] = (df.loc[df[
                                                                               'dataset'] == dataset_name, 'val/acc'] - trivial_accuracy) / (
                                                                           best_accuracy - trivial_accuracy)
            elif normalization_mode == 'trivial':
                df.loc[df['dataset'] == dataset_name, 'val/acc'] /= trivial_accuracy
            elif normalization_mode == 'max':
                df.loc[df['dataset'] == dataset_name, 'val/acc'] /= best_accuracy
            elif normalization_mode == 'minmax':
                df.loc[df['dataset'] == dataset_name, 'val/acc'] = (df.loc[df['dataset'] == dataset_name, 'val/acc'] -
                                                                    df.loc[df[
                                                                               'dataset'] == dataset_name, 'val/acc'].min()) / (
                                                                           df.loc[df[
                                                                                      'dataset'] == dataset_name, 'val/acc'].max() -
                                                                           df.loc[df[
                                                                                      'dataset'] == dataset_name, 'val/acc'].min())
            else:
                raise ValueError(f'Unknown normalization mode {normalization_mode}')

    labels = {'val/acc': 'Accuracy' if not normalize else 'Normalized accuracy',
              'train/jr/unit_dead': 'Dead unit',
              'train/jr/unit_linear': 'Linear unit',
              'train/jr/unit_nonlinear': 'Nonlinear unit',
              'train/jr/point_dead': 'Dead point',
              'train/jr/point_linear': 'Linear point',
              'train/jr/point_nonlinear': 'Nonlinear point',
              'train/jr/point_deathness': 'Dead global',
              'train/jr/point_linearity': 'Linear global',
              'train/jr/point_nonlinearity': 'Nonlinear global'} if labels is None else labels

    for metric in metrics:
        metric_str = metric.replace('/', '_')
        yaxis_str = yaxis.replace('/', '_')
        # sns.kdeplot(data=df, y=yaxis, x=metric, bw_adjust=0.5, fill=True)
        with sns.plotting_context("paper", font_scale=2):
            fig, ax = plt.subplots()
            sns.kdeplot(ax=ax, data=df, y=yaxis, x=metric, bw_adjust=0.5, thresh=0.1, levels=3)
            sns.scatterplot(ax=ax, data=df, y=yaxis, x=metric, hue='method', legend=True)
            h, l = ax.get_legend_handles_labels()

            if not split_legend:
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title='Method')
            else:
                ax.get_legend().remove()
            ax.set_ylabel(labels[yaxis])
            ax.set_xlabel(labels[metric])
            fig.tight_layout()
            fig.savefig(output_dir / f'{yaxis_str}-{metric_str}-heatmap.pdf')

            if split_legend:
                # if split legend show the legend in another figure
                fig, legend_ax = plt.subplots()
                legend_ax.legend(h, l, loc='center', title='Method')
                legend_ax.axis('off')
                fig.tight_layout()
                # legend_ax.get_legend().set_bbox_to_anchor((0.5, 0.5))
                fig.savefig(output_dir / f'{yaxis_str}-{metric_str}-legend.pdf')
            plt.close()

def generate_imagenet_result_table(df, labels=None, output_dir='tables'):
    output_dir = Path(output_dir)
    params = ['is_resnet', 'lr', 'width_multiplier']
    df = df.sort_values('val/acc', ascending=False).reset_index()
    df = df.set_index('name')
    df = df.drop(index=['lambda-cosine', 'gradient-clip-0.5', 'lambda-clipping-good', 'baseline-3'])
    df['is_resnet'] = df['use_batchnorm'] & df['use_skip_connections']
    df['is_resnet'] = df['is_resnet'].replace({True: 'ResNet', False: 'Jumpstart'})
    df['width_multiplier'] = df['width_multiplier'].fillna(2)
    # pick only lambda 0.0010 or 0
    df = df[df['lambda_'].isin([0.001, 0])]
    # pick lr = 0.00003
    df = df[df['lr'] == 0.00003]
    # Drop lambda

    metrics = ['train/acc', 'val/acc',
               # 'train/jr/unit_nonlinear', 'train/jr/point_nonlinear',
               # 'val/jr/unit_nonlinear', 'val/jr/point_nonlinear',
               # # 'train/jr/unit_dead', 'train/jr/point_dead',
               # 'val/jr/unit_dead', 'val/jr/point_dead'
               ]
    labels = {'val/acc': 'Validation Accuracy',
              'train/acc': 'Train Accuracy',
              'lr': 'Learning rate',
              'batch_size': 'Batch size',
              'is_resnet': 'Model',
              'width_multiplier': 'Width multiplier',
              'val/jr/unit_dead': 'Dead unit',
              'val/jr/unit_linear': 'Linear unit',
              'val/jr/unit_nonlinear': 'Nonlinear unit',
              'val/jr/point_dead': 'Dead point',
              'val/jr/point_linear': 'Linear point',
              'val/jr/point_nonlinear': 'Nonlinear point',
              'val/jr/point_deathness': 'Dead global',
              'val/jr/point_linearity': 'Linear global',
              'val/jr/point_nonlinearity': 'Nonlinear global',
              'lambda_': 'Lambda'} if labels is None else labels
    df = df[params + metrics]
    # df = df.sort_values('val/acc').drop_duplicates(['lr', 'batch_size', 'lambda_'])
    df = df.sort_values(params)
    df = df.set_index(params)
    df.index = df.index.rename(labels)
    df = df.rename(labels, axis=1)

    print(df)
    caption = """Results of experiments conducted on the ImageNet dataset comparing the performance of Jumpstart and 
    ResNet models across various hyperparameters. Jumpstart is able to match ResNet accuracy at 0.67. Notably, the top-performing Jumpstart 
    model uses a width multiplier of 1.5 to reduce overfitting, suggesting that enforcing non-linearity in all units and points could 
    potentially reduce the necessity for additional regularization techniques such as dropout or weight decay."""
    return to_latex(df,
                    caption=caption,
                    label='tab:imagenet_results', cmap=None, environment='table*', index_precision=5, precision=2,
                    clines='all;index',
                    output_path=output_dir / 'imagenet_results.tex', )


def generate_status_plots(summaries, metrics=None, entity='jumpstart', project='uci', skip_combinations=True,
                          output_dir='plots/status', n_jobs=12, verbose=10):
    output_dir = Path(output_dir)
    summaries = summaries.reset_index()
    summaries.loc[summaries['skip_connections'] == 'None', 'skip_connections'] = pd.NA
    summaries['is_jumpstart'] = summaries['lambda_'] > 0
    summaries['is_skip'] = summaries['skip_connections'] > 0
    summaries['is_gelu'] = summaries['activation'] == 'gelu'
    names = pd.Series(['Jumpstart', 'Skip connections', 'BatchNorm', 'GELU'])
    table = {}
    metrics = ['val/acc'] if metrics is None else metrics

    for name, group in summaries.groupby(['is_jumpstart', 'is_skip', 'use_batchnorm', 'is_gelu']):
        name = names[np.array(name)]
        if skip_combinations and len(name) > 1:
            pass
        else:
            if len(name) == 0:
                name = 'Baseline'
            else:
                name = ' + '.join(name)

            dataset_metrics = group[['dataset', 'epoch', 'run_id'] + metrics]
            dataset_metrics = dataset_metrics.sort_values('val/acc', ascending=False).drop_duplicates('dataset')
            dataset_metrics = dataset_metrics.set_index('dataset').squeeze().sort_index()
            table[name] = dataset_metrics

    table = pd.concat(table)
    print('Fetching for')
    print(table)

    jobs = []
    keys = []
    for (technique, dataset), run_data in tqdm(table.iterrows(), total=len(table)):
        keys.append((technique, dataset))
        jobs.append(delayed(_download_available_plots)(entity, project, run_data["run_id"]))

    all_plots = Parallel(n_jobs=n_jobs, backend='threading', verbose=verbose)(jobs)
    all_plots = dict(zip(keys, all_plots))

    jobs = []
    for (technique, dataset), plots in all_plots.items():
        for axis in {'unit', 'point'}:
            for partition in {'train', 'val'}:
                df = plots[plots['name'].str.contains(f'{partition}/status/{axis}')]
                if not df.empty and run_data['epoch'] > 0:
                    jobs.append(delayed(_download_status)(df, run_data['epoch'], technique, dataset, axis, partition,
                                                          output_dir))
    Parallel(n_jobs=n_jobs, backend="threading", verbose=verbose)(jobs)


def _download_available_plots(entity, project, run_id):
    try:
        api = wandb.Api()
        run = api.run(f'{entity}/{project}/{run_id}')
        plots = pd.DataFrame([(int(f.name.split('_')[1]), f.name, f)
                              for f in run.files() if 'point' in f.name or 'unit' in f.name],
                             columns=['epoch', 'name', 'file'])
        return plots

    except Exception as ex:
        print(f'Failed to fetch run {run_id} with {ex}')


def _download_status(df, epoch, technique, dataset, axis, partition, output_dir):
    yaxis_title = 'Units' if axis == 'unit' else 'Data points'
    best_epoch = abs(df.index - epoch).argmin()
    best_epoch = df.index[best_epoch]
    file = df.loc[best_epoch]['file']
    download_path = file.download(replace=True).name
    technique = technique.replace(' ', '_').lower()
    output_dir.mkdir(exist_ok=True, parents=True)
    Path(download_path).rename(output_dir / f'{technique}.plotly.json')
    try:
        with open(output_dir / f'{technique}.plotly.json') as f:
            figure = plotly.io.from_json(f.read().strip(), engine='json')
        x_ticks = [f'Layer {x.split("-")[-1]}' for x in figure['data'][0]['x']]
        figure.update_layout(
            xaxis_title=f'Layer',
            yaxis_title=yaxis_title,
            xaxis_tickmode='array',
            xaxis_tickvals=list(range(len(x_ticks))),
            xaxis_ticktext=x_ticks,
        )
        if axis == 'unit':
            # remove fractional ticks
            y_ticks = list(range(0, len(figure['data'][0]['z'])))
            figure.update_layout(
                yaxis_tickvals=y_ticks,
            )
            figure.update_traces(xgap=1, ygap=1, hovertemplate=None)
            figure.update_layout(
                autosize=False,
                height=300,
            )

        figure.update_layout(
            autosize=False,
            width=1000,
        )

        output_path = output_dir / f'{dataset}' / f'{axis}_{partition}_{technique}.pdf'
        # output_path = output_dir / dataset / axis / partition / f'{technique}.pdf'
        output_path.parent.mkdir(exist_ok=True, parents=True)

        figure.write_image(str(output_path))
    except Exception as ex:
        print(f'Failed to generate {technique} for {dataset} {axis} {partition} with {ex} at file {download_path}')


def download_project(project, entity='jumpstart', num_workers=12, verbose=10, compute=True, cache_dir=None,
                     force_download=False, include_crashed=False):
    if cache_dir:
        print(f'Using cache dir {cache_dir}')
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)
        cache_file = cache_dir / f'{project}.csv'
        if cache_file.exists() and not force_download:
            print(f'Loading from cache {cache_file}')
            df = pd.read_csv(cache_file)
        else:
            print(f'Downloading {project} from {entity} and saving to {cache_file}')
            df = download_metrics(project, entity=entity, compute=compute, num_workers=num_workers, verbose=verbose,
                                  include_crashed=include_crashed)
            df.to_csv(cache_file)
    else:
        print(f'No cache dir provided, downloading {project} from {entity}')
        df = download_metrics(project, entity=entity, compute=compute, num_workers=num_workers, verbose=verbose)
    df.columns = df.columns.str.replace('jr.', 'jr/').to_list()
    return df


def generate_uci_results(project, output_dir='./', entity='jumpstart', num_workers=12, verbose=10, compute=True,
                         cache_dir=None):
    output_dir = Path(output_dir)
    df = download_project(project, entity=entity, num_workers=num_workers, verbose=verbose, compute=compute,
                          cache_dir=cache_dir)

    plot_dir = output_dir / 'img'
    table_dir = output_dir / 'tables'
    status_dir = plot_dir / 'status'

    plot_dir.mkdir(exist_ok=True, parents=True)
    table_dir.mkdir(exist_ok=True, parents=True)
    status_dir.mkdir(exist_ok=True, parents=True)

    generate_key_metrics_table(df, output_dir=table_dir)
    generate_win_eq_losses_table(df, output_dir=table_dir)
    generate_result_table(df, output_dir=table_dir)
    export_final_uci_table(df, project, output_dir=table_dir)
    generate_accuracy_boxplots(df, output_dir=plot_dir, columns=2)
    generate_dataset_metadata_table(df, output_dir=table_dir)
    generate_accuracy_correlation_heatmap(df, output_dir=plot_dir)
    generate_status_plots(df, project=project, n_jobs=num_workers, output_dir=status_dir, verbose=verbose)


def generate_imagenet_results(project, output_dir='./', num_workers=-2, verbose=10, compute=True,
                              cache_dir=None, force_download=False):
    output_dir = Path(output_dir)
    table_dir = output_dir / 'tables'
    df = download_project(project, num_workers=num_workers, verbose=verbose, compute=compute,
                          cache_dir=cache_dir, force_download=force_download, include_crashed=True)

    export_final_imagenet_table(df, project, output_dir=table_dir, precision=3)

