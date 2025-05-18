import functools
import inspect
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import seaborn as sns
import typeguard
import wandb
from joblib import Parallel, delayed
from matplotlib import pyplot as plt


def to_violinplot(name, data, filepath):
    # sort by mean
    data = data[data.mean().sort_values().index]
    sns.violinplot(data=data, orient='v', palette='Blues_r')
    plt.ylabel(name.split('/')[1].capitalize())
    plt.xlabel('Technique')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def to_boxplot(name, data, filepath, columns=1):
    # sort by mean
    palette = 'Blues_r' if name == 'Accuracy' or 'Nonlinear' in name else 'Blues'
    data = data[data.mean().sort_values().index]
    figure = plt.figure(figsize=(6, 4 * columns))
    sns.boxplot(data=data, orient='v', palette=palette, showfliers=True)
    sns.stripplot(data=data, orient='v', edgecolor='black', alpha=0.7, palette='dark:#FFFFFF', linewidth=1)
    if columns > 1:
        # increase text size
        plt.rcParams.update({'font.size': 16})
        plt.xticks(rotation=90)

    plt.ylabel(name)
    plt.xlabel('Technique')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def to_latex(table, caption, label, output_path, cmap="Blues_r", gmap=None, escape_index=True, escape_columns=True,
             escape='latex', environment=None, multicol_align="|c|",
             index_precision=2, precision=2, clines="all;index", font_size=12, normalize='global',):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    styler = table.style
    if cmap is not None:
        cmap = sns.color_palette(cmap, as_cmap=True)
        if normalize == 'global':
            try:
                if gmap is not None:
                    vmax = gmap.max().max() if gmap.max().max() > 1 else 1
                    vmin = 0
                else:
                    vmax = table.max().max() if table.max().max() > 1 else 1
                    vmin = 0
            except TypeError:
                vmax = 1
                vmin = 0
            styler = styler.background_gradient(cmap=cmap,
                                                axis=None,
                                                vmin=vmin,
                                                vmax=vmax,
                                                gmap=gmap)

        elif normalize == 'column':
            styler = styler.background_gradient(cmap=cmap,
                                                axis='columns',
                                                gmap=gmap)

        else:
            vmax = 1
            vmin = 0
            styler = styler.background_gradient(cmap=cmap,
                                                axis=None,
                                                vmin=vmin,
                                                vmax=vmax,
                                                gmap=gmap)
    if escape_index:
        styler = styler.format_index(escape=escape, axis=0)
    if escape_columns:
        styler = styler.format_index(escape=escape, axis=1)
    # set font size
    styler = styler.set_properties(**{'font-size': f'{font_size}px'})
    styler = styler.format(precision=precision).format_index(precision=index_precision)

    styler.to_latex(output_path,
                    label=label,
                    hrules=True,
                    caption=caption,
                    clines=clines,
                    position_float="centering",
                    multicol_align=multicol_align,
                    environment=environment,
                    convert_css=True)
    styler.to_html(output_path.with_suffix('.html'))


def clean_history(df):
    df = df.select_dtypes(exclude=['object'])
    # Remove _step columns that are not global_step
    df = df.drop([col for col in df.columns if col.endswith('_step') and col != 'trainer/global_step'], axis=1)
    # Rename _epoch columns to just the name
    df = df.rename(columns={col: col.replace('_epoch', '') for col in df.columns if col.endswith('_epoch')})

    # summaries = summaries.dropna(axis=1, how='all')

    return df


def patch_config(name, run_id, config, metrics):
    config = pd.Series(config).astype(str)
    try:
        config = pd.to_numeric(config)
    except Exception as ex:
        if not 'balanced' in str(ex):
            print(f'Error in {name} with {ex}')
    config = config.replace({'True': True, 'False': False})
    config['name'] = name
    config['run_id'] = run_id
    # patched = pd.concat((config, metrics), keys=['config', 'metrics'])
    patched = pd.concat((config, metrics))
    patched = patched.to_frame().T.set_index(config.index.to_list())

    return patched


def find_best(run, fast=False, metric='val/acc', mode='auto'):
    raw_history = run.history() if fast else pd.DataFrame([run for run in run.scan_history()])
    raw_history = raw_history.replace({'True': True, 'False': False, 'NaN': None})
    for column in raw_history:
        try:
            raw_history[column] = pd.to_numeric(raw_history[column])
        except Exception as ex:
            if 'status' in column or 'surface' in column or 'losses' in column:
                pass
            else:
                print(f'Error in {column} with {ex}')
    # Groupby over epoch
    history = {}
    for epoch, group in raw_history.groupby('epoch'):
        epoch_data = {}
        for column in group:
            if column.startswith('train') or column.startswith('val') or column.startswith('test'):
                if 'status' in column or 'surface' in column or 'losses' in column:
                    epoch_data[column] = group[column].iloc[group[column].astype(bool).argmax()]
                else:
                    try:
                        epoch_data[column] = group[column].mean()
                    except Exception as ex:
                        print(f'Error in {column} with {ex}')
            else:
                epoch_data[column] = group[column].iloc[-1]
        history[epoch] = epoch_data

    history = pd.DataFrame.from_dict(history, orient='index')

    if history.shape[0] == 0:
        raise ValueError(f'Empty dataframe for run {run.name}')
    history = clean_history(history)
    if mode == 'auto':
        if metric.endswith('acc'):
            mode = 'max'
        elif metric.endswith('loss'):
            mode = 'min'
        else:
            raise ValueError(f'Unable to infer summary mode for metric {metric}')
    if metric in history.columns:
        if mode == 'max':
            best_idx = history[metric].argmax()
        elif mode == 'min':
            best_idx = history[metric].argmin()
        else:
            raise ValueError(f'Unknown mode {mode}')

        best = history.iloc[best_idx].copy()
        best['epoch'] = history.index[best_idx]
        # find closest non nan epoch if nan
        for column in best.index:
            if pd.isna(best[column]):
                non_nans = history[column].dropna()
                if len(non_nans) > 0:
                    best[column] = non_nans.iloc[np.abs(non_nans.index - best_idx).argmin()]
        best = patch_config(run.name, run.id, run.config, best)
    else:
        print(f'No metric {metric} in run {run.name}')
        raise RuntimeError(f'No metric {metric} in run {run.name}')

    if not best.shape[0] == 1:
        raise RuntimeError(f'Multiple rows for best in run {run.name}, {best}')
    if best.isna().any().any():
        print(f'Nans in {run.name} {run.url} {best.columns[best.iloc[0].isna()].to_list()}')
    return best


def find_and_upload_best(run, fast=False, metric='val/acc', summary='auto'):
    try:
        run_summary = find_best(run, fast=fast, metric=metric, mode=summary)
        for key, val in run_summary.iloc[0].items():
            run.summary[key] = val
        run.summary.update()
        return run_summary
    except Exception as ex:
        print(f'Run {run.name} failed with: {ex}')
        return None


def download_best(run):
    run_config = pd.Series(run.config)
    run_config = run_config.astype(str).apply(lambda x: pd.to_numeric(x, errors='ignore'))
    run_summary = pd.DataFrame.from_dict({k: [v] for k, v in run.summary.items()})
    run_summary = pd.concat((run_config.to_frame().T, run_summary), axis=1)
    run_summary = run_summary.set_index(run_config.index.to_list())
    run_summary = run_summary.select_dtypes(exclude=['object'])
    return run_summary

def download_metrics(project, entity='jumpstart', fast=False, metric='val/acc', summary='auto', compute=False,
                     num_workers=1, verbose=10, include_crashed=True):
    api = wandb.Api()
    jobs = []
    print(f'Fetching metrics for {project}')
    for run in api.runs(entity + "/" + project):
        try:
            if run.state != 'finished':
                raise ValueError(f'Run state {run.state}')

            if compute:
                jobs.append(delayed(find_and_upload_best)(run, fast=fast, metric=metric, summary=summary))
            else:
                jobs.append(delayed(download_best)(run))

        except ValueError as ex:
            print(f'Run {run.name} failed with: {ex}')
            if include_crashed:
                print('Including crashed run')
                if compute:
                    jobs.append(delayed(find_and_upload_best)(run, fast=fast, metric=metric, summary=summary))
                else:
                    jobs.append(delayed(download_best)(run))
    # jobs = [j for j in jobs if j[1][0].name == 'pleasant-sweep-86']
    print(f'Running {len(jobs)} jobs')
    run_metrics = Parallel(n_jobs=num_workers, backend="threading", verbose=verbose)(jobs)
    print(f'Finished fetching metrics for {project}')
    configs = pd.concat([run.index.to_frame() for run in run_metrics if run is not None])
    configs = configs.reset_index(drop=True)
    run_metrics = [run for run in run_metrics if run is not None]
    run_metrics = pd.concat(run_metrics)
    run_metrics = run_metrics.reset_index(drop=True)
    run_metrics = pd.concat([configs, run_metrics], axis=1)
    if 'width_multiplier' in run_metrics:
        run_metrics['width_multiplier'] = run_metrics['width_multiplier'].fillna(2)
    run_metrics = run_metrics.set_index(configs.columns.to_list())

    return run_metrics










def typechecked(func):
    __annotations__ = getattr(func, '__annotations__', None)
    if __annotations__:
        __signature__ = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return typeguard.typechecked(func)(*args, **kwargs)
            except TypeError as exc:
                err = str(exc)
                found = False
                for param in __signature__.parameters.values():
                    if f'type of argument "{param.name}" ' not in err:
                        continue
                    found = True
                    if param.kind == param.POSITIONAL_ONLY or param.default == param.empty:
                        err = err.replace(param.name, param.name.upper())
                    elif param.kind in {param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY}:
                        err = err.replace(param.name, f"--{param.name.replace('_', '-')}")
                if found:
                    raise fire.core.FireError(err)
                raise exc

        return wrapper
    return func
