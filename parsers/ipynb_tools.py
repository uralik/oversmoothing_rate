import pandas as pd
import numpy as np
from IPython.core.display import display
import itertools
import matplotlib
from matplotlib import pyplot as plt
import os
import pickle

def filter_models_df(dfs, params):
    selected_ids = {}
    for experiment, df in dfs.items():
        selected_ids[experiment] = []
        for i in range(df.index.size):
            eq = 0
            for j in range(df.columns.size):
                for param in params:
                    if param == df.columns[j] and df.iloc[i, j] in params[param]:
                        eq += 1
            if eq == len(params):
                selected_ids[experiment] += [i]
    return selected_ids

def avg_by_seeds(result_df, params, metrics, precision=4):
    unique_values = {}
    for param in params:
        unique_values[param] = result_df[param].unique().tolist()

    grid = (dict(zip(unique_values, x)) for x in itertools.product(*unique_values.values()))
    result = []
    for setup in grid:
        setup_unwrapped = {}
        for param in setup:
            setup_unwrapped[param] = {setup[param]}
        selected_ids = filter_models_df({'res': result_df}, setup_unwrapped)['res']
        result += [setup]
        data = result_df.iloc[selected_ids, :].drop_duplicates('seed')
        for metric in metrics:
            if data[metric].size == 1:
                result[-1][metric] = f'{round(data[metric].mean(), precision)}'
            else:
                result[-1][metric] = f'{round(data[metric].mean(), precision)} +- {round(data[metric].std(), precision)}'
    return pd.DataFrame(result)

def draw_plot_wrt_beam(axis, metric, model_name, exp_name, base_path, metric_name=None):
    values, stds = {}, {}
    for neos in [1, 8, 32, 128, 512, 1024, 2048]:
        values[neos] = {}
        stds[neos] = {}

    for beam in [1, 5, 100, 250, 500, 750, 1000]:
        res_path = f'{base_path}/{model_name}/metrics_{model_name}_beam{beam}.csv'
        if not os.path.exists(res_path):
            continue
        metrics = pd.read_csv(res_path, index_col=0)
        res = avg_by_seeds(metrics, params={'EOS#'}, metrics={metric})
        if len(res) == 0:
            continue
        for neos, value in zip(res['EOS#'], res[metric]):
            value, std = value.split('+-')
            value, std = float(value), float(std)
            values[neos][beam] = value
            stds[neos][beam] = std

    to_return = {}
    for neos in values:
        stds_down = [values[neos][beam] - stds[neos][beam] for beam in values[neos]]
        stds_up = [values[neos][beam] + stds[neos][beam] for beam in values[neos]]
        axis.plot(values[neos].keys(), values[neos].values(), marker='o', label=f'EOS:{neos}', linewidth=2)
        axis.fill_between(values[neos].keys(), stds_down, stds_up, alpha=0.2)
        to_return[neos] = (values[neos].keys(), values[neos].values(), stds_down, stds_up)

    axis.legend()
    axis.grid()

    if metric_name is None:
        metric_name = metric
    axis.set_xlabel('Beam', fontsize=16)
    axis.set_ylabel(metric_name, fontsize=16)
    axis.set_title(f'Task: {exp_name}', fontsize=16)
    return to_return

def draw_plot_wrt_beam_oversmoothing(axis, metric, model_name, exp_name, base_path, metric_name=None):
    values, stds = {}, {}
    for weight in [0.05 * i for i in range(20)]:
        values[weight] = {}
        stds[weight] = {}

    for beam in [1, 5, 100, 250, 500, 750, 1000]:
        res_path = f'{base_path}/{model_name}/metrics_{model_name}_beam{beam}.csv'
        if not os.path.exists(res_path):
            continue
        metrics = pd.read_csv(res_path, index_col=0)
        res = avg_by_seeds(metrics, params={'OSL weight'}, metrics={metric})
        if len(res) == 0:
            continue
        for weight, value in zip(res['OSL weight'], res[metric]):
            value, std = value.split('+-')
            value, std = float(value), float(std)
            values[weight][beam] = value
            stds[weight][beam] = std

    to_return = {}
    for neos in values:
        stds_down = [values[neos][beam] - stds[neos][beam] for beam in values[neos]]
        stds_up = [values[neos][beam] + stds[neos][beam] for beam in values[neos]]
        axis.plot(values[neos].keys(), values[neos].values(), marker='o', label=f'Weight:{neos}', linewidth=2)
        axis.fill_between(values[neos].keys(), stds_down, stds_up, alpha=0.2)
        to_return[neos] = (values[neos].keys(), values[neos].values(), stds_down, stds_up)

    axis.legend()
    axis.grid()

    if metric_name is None:
        metric_name = metric
    axis.set_xlabel('Beam', fontsize=16)
    axis.set_ylabel(metric_name, fontsize=16)
    axis.set_title(f'Task: {exp_name}', fontsize=16)
    return to_return

def load_experiments_results(experiments, results_dir='/scratch/mae9785/results', beam_for_table=1000, target_beam=2):
    results = {}
    for exp, path in experiments.items():
        #if 'wmt19' in path:
        #    target_beam = 5
        #    beam_for_table = max(beam_for_table, 5)
        
        #metrics_target = pd.read_csv(f'{results_dir}/{path}/metrics_{path}.csv', index_col=0)
        metrics = pd.read_csv(f'{results_dir}/{path}/metrics_{path}_beam{beam_for_table}.csv', index_col=0)
        metrics['OSL weight'] = metrics['OSL weight'].round(2)
        #winrates = pd.read_csv(f'{results_dir}/{path}/winrates_{path}.csv', index_col=0)
        #nupdates = pd.read_csv(f'{results_dir}/{path}/nupdates_{path}.csv', index_col=0)
        #endings = pd.read_csv(f'{results_dir}/{path}/endings_{path}_beam{beam_for_table}.csv', index_col=0)
        #oversmoothing = pd.read_csv(f'{results_dir}/{path}/oversmoothing_{path}.csv', index_col=0)
        #prob_diff = pd.read_csv(f'{results_dir}/{path}/probdiff_{path}.csv', index_col=0)
        results[exp] = {
            #'metrics_target': metrics_target,
            'metrics': metrics,
        #    'endings': endings, 
        #    'improvement': pd.DataFrame(winrates['improvement ratio']),
        #    'non-degradation': pd.DataFrame(winrates['non-degradation ratio']),
        #    'nupdates': nupdates.sort_values('EOS#').set_index('EOS#'),
        #    'oversmoothing': oversmoothing,
        #    'terminal-prob-diff': pd.DataFrame(prob_diff['teminal probs']),
        #    'non-terminal-prob-diff': pd.DataFrame(prob_diff['non-terminal probs']),
        }
        #results[exp]['nupdates'].index.name = None
        #if not results[exp]['endings'].empty:
        #    results[exp]['endings'] = avg_by_seeds(results[exp]['endings'],
        #                                           params={'EOS#'},
        #                                           metrics={'not E'}).sort_values('EOS#').set_index('EOS#') 
        #    
        #    results[exp]['endings'].index.name = None
        #if not results[exp]['oversmoothing'].empty:
        #    results[exp]['oversmoothing'] = avg_by_seeds(results[exp]['oversmoothing'],
        #                                              params={'EOS#'},
        #                                              metrics={'OS'}).sort_values('EOS#').set_index('EOS#')
        #    results[exp]['oversmoothing'].index.name = None
    return results

def print_table(experiments_results, metric, external=False, maximize=True, highlight=True, return_idxs=False, target=False):
    output = pd.DataFrame()
    columns = []
    for exp in experiments_results:
        columns += [exp]
        if external:
            df = experiments_results[exp][metric]
            output = pd.concat([output, df], axis=1)
        else:
            if target:
                metrics_name = 'metrics_target'
            else:
                metrics_name = 'metrics'
            df = avg_by_seeds(experiments_results[exp][metrics_name],
                              params={'OSL weight'},
                              metrics={metric}).sort_values('OSL weight').set_index('OSL weight')
            df.index.name = None
            output = pd.concat([output, df], axis=1)
    output.columns = columns
    if 'OSL weight' in output.columns:
        output = output.sort_values('OSL weight')
    best_idxs, overlapping_idxs = select_best(output, maximize=maximize)
    if highlight:
        output = output.style.apply(highlight_best(output, best_idxs, overlapping_idxs), axis=1)
    if return_idxs:
        return output, best_idxs, overlapping_idxs
    return output

def select_best(df, maximize=True):
    def is_better(a, b):
        if maximize:
            return a > b
        else:
            return b > a

    invert = 1
    if not maximize:
        invert = -1
    best_mean = [invert * -np.inf] * df.columns.size
    std_for_best_mean = [0] * df.columns.size
    idx_for_best_mean = [''] * df.columns.size
    for i in range(1):
        for j in range(df.columns.size):
            if isinstance(df.iloc[i, j], str) and '+-' in df.iloc[i, j]:
                mean_str, std_str = df.iloc[i, j].split(' +- ')
                mean, std = float(mean_str), float(std_str)
            else:
                mean = float(df.iloc[i, j])
                std = 0
            if is_better(mean, best_mean[j]):
                best_mean[j] = mean
                std_for_best_mean[j] = std
                idx_for_best_mean[j] = i

    lower_values = [best_mean[k] +  std_for_best_mean[k] for k in range(df.columns.size)]

    overlapping_idxs = [set() for i in range(df.columns.size)]
    for i in range(df.index.size):
        for j in range(df.columns.size):
            if i == idx_for_best_mean[j]:
                continue
            if isinstance(df.iloc[i, j], str) and '+-' in df.iloc[i, j]:
                mean_str, std_str = df.iloc[i, j].split(' +- ')
                mean, std = float(mean_str), float(std_str)
            else:
                mean = float(df.iloc[i, j])
                std = 0
            if mean - invert * std > lower_values[j]:
                overlapping_idxs[j].add(i)

    return idx_for_best_mean, overlapping_idxs

def highlight_best(df, best_idx, overlapping_idxs):
    standard_style = 'border: 1px  black solid; text-align:center !important; color: black !important;'
    row_number = 0
    def color_to_highlight(row):
        nonlocal row_number
        style = []
        for i in range(df.columns.size):
            if best_idx[i] == row_number:
                style += [f'background-color: lightgreen; {standard_style}']
            elif row_number in overlapping_idxs[i]:
                style += [f'background-color: lightyellow; {standard_style}']
            else:
                style += [standard_style]
        row_number += 1
        return style
    return color_to_highlight

def draw_plots(experiments, metric, base_path, separately=False, metric_name=None):
    to_return = {}
    if separately: 
        for i, exp in enumerate(experiments):
            fig, axis = plt.subplots(1, 1, figsize=(7, 7))
            output = draw_plot_wrt_beam(axis, exp_name=exp, metric=metric, model_name=experiments[exp], base_path=base_path, metric_name=metric_name)
            to_return[exp] = output
        plt.show()
        return to_return
    subplots = np.ceil(len(experiments) / 2)
    fig, axes = plt.subplots(int(subplots), 2, figsize=(15, subplots * 7))
    axes = axes.flatten()
    for i, exp in enumerate(experiments):
        output = draw_plot_wrt_beam(axes[i], exp_name=exp, metric=metric, model_name=experiments[exp], base_path=base_path, metric_name=metric_name)
        to_return[exp] = output
    plt.show()
    return to_return

def draw_plots_oversmoothing(experiments, metric, base_path, separately=False, metric_name=None):
    to_return = {}
    if separately:
        for i, exp in enumerate(experiments):
            fig, axis = plt.subplots(1, 1, figsize=(7, 7))
            output = draw_plot_wrt_beam_oversmoothing(axis, exp_name=exp, metric=metric, model_name=experiments[exp], base_path=base_path, metric_name=metric_name)
            to_return[exp] = output
        plt.show()
        return to_return
    subplots = np.ceil(len(experiments) / 2)
    fig, axes = plt.subplots(int(subplots), 2, figsize=(15, subplots * 7))
    axes = axes.flatten()
    for i, exp in enumerate(experiments):
        output = draw_plot_wrt_beam_oversmoothing(axes[i], exp_name=exp, metric=metric, model_name=experiments[exp], base_path=base_path, metric_name=metric_name)
        to_return[exp] = output
    plt.show()
    return to_return

def draw_ranks_distribution(experiments, base_path):
    subplots = np.ceil(len(experiments) / 2)
    fig, axes = plt.subplots(int(subplots), 2, figsize=(15, subplots * 7))
    axes = axes.flatten()
    for i, experiment in enumerate(experiments):
        bins_eos = pickle.load(open(f'{base_path}/{experiments[experiment]}/dranks_{experiments[experiment]}_beam5.pkl', 'rb'))
        for neos in list(bins_eos.keys())[::-1]:
            axes[i].bar(list(bins_eos[neos].keys()), list(bins_eos[neos].values()), width=50, label=f'EOS:{neos}', alpha=0.15)
        axes[i].set_title(f'Non-terminal EOS ranks, {experiment}', fontsize=16)
        axes[i].set_xlabel('NT EOS rank', fontsize=14)
        axes[i].legend()
        axes[i].set_yscale('symlog')
        axes[i].grid()

def errorfill(x, y, yerrplus, yerrminus, color=None, alpha_fill=0.2, ax=None, label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    if np.isscalar(yerrplus) or len(yerrplus) == len(y):
        ymin = yerrminus
        ymax = yerrplus
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill, label=label)
    
def get_data(suffix_dict, quantiles):
    xs = []
    yss = {q:[] for q in quantiles}
    for suffix_length in sorted(suffix_dict.keys()):
        xs.append(suffix_length)
        current_ys = []
        for q in quantiles:
            yss[q].append(np.quantile(suffix_dict[suffix_length], q=q))
    for k,v in yss.items():
        yss[k] = np.array(v)
    return xs, yss

def get_data_mean(suffix_dict, max_suffix_length=1000):
    xs = []
    ys = []
    for suffix_length in sorted(suffix_dict.keys()):
        if suffix_length > max_suffix_length:
            continue
        xs.append(suffix_length)
        ys.append(np.array(suffix_dict[suffix_length]).mean())
    return xs, ys

def get_suffixlen_distr(suffix_dict, max_suffix_length=1000):
    suffix_length = sorted(suffix_dict.keys())
    counts = [len(suffix_dict[l]) for l in suffix_length if l <= max_suffix_length]    
    return counts

def draw_oversmoothing_rate(experiments, neos_list=[1], plot_on_one_ax=True,
                    exp_names=['FT IWSLT17 DE-EN NLL ES', 'FT IWSLT17 FR-EN NLL ES', 
                              'FT IWSLT17 ZH-EN NLL ES', 'FT WMT16 EN-DE NLL ES', 
                              'FT WMT19 RU-EN'], max_suffix_length=1000, plot_width=25, legend=True):
    number_of_exprs = len(exp_names)*len(neos_list)
    numfigs = 1 if plot_on_one_ax else number_of_exprs

    # oversmoothing rate figs
    figs = []
    axs = []
    for i in range(numfigs):
        _fig, _ax = plt.subplots(figsize=(plot_width,5))
        figs.append(_fig)
        axs.append(_ax)

    # counts of prefixes figs
    counts_figs = []
    counts_axs = []
    for _ in range(numfigs):
        _fig, _ax = plt.subplots(figsize=(plot_width,5))
        counts_figs.append(_fig)
        counts_axs.append(_ax)

    to_return = {}

    xs_max = 0
    for expi, experiment in enumerate(exp_names):
        # if we plot on the same ax, then we always use axs[0] or so, otherwise we map every experiment/neos to its own ax
        axi = 0 if plot_on_one_ax else expi
        if not plot_on_one_ax:
            xs_max = 0
        to_return[experiment] = {}
        mode = 'hard'
        suffix_oversmoothing = pickle.load(open(f'/scratch/mae9785/new_results/{experiments[experiment]}/{mode}oversmoothing_{experiments[experiment]}.pkl', 'rb'))
        
        for neosi, neos in enumerate(neos_list):
            axi = axi if plot_on_one_ax else expi*len(neos_list)+neosi
            plot_label = f'#EOS:{neos}, {experiment}' if neos > 1 else f'{experiment}'
            for seed, suffix_dict in suffix_oversmoothing[neos].items():
                xs, ys = get_data_mean(suffix_dict, max_suffix_length)
                suffix_counts = get_suffixlen_distr(suffix_dict, max_suffix_length)
                line = axs[axi].plot(xs, ys, label=plot_label, linewidth=1.5)
                counts_axs[axi].plot(xs, suffix_counts, c=line[-1].get_color(), label=plot_label, linewidth=1.5)
                to_return[experiment][neos] = {
                    'suffix_lengths': xs,
                    'hard_os': ys,
                    'suffix_counts': suffix_counts,
                    'suffix_dict': suffix_dict,
                }
                if max(xs) > xs_max:
                    xs_max = max(xs)
                break
            minor_suffix_length_ticks = np.arange(0, xs_max, 5)
            major_suffix_length_ticks = np.arange(0, xs_max, 10)
            axs[axi].set_xticks(major_suffix_length_ticks)
            axs[axi].set_xticks(minor_suffix_length_ticks, minor=True)
            axs[axi].grid(which='both', alpha=0.5)
            if legend:
                axs[axi].legend()
            axs[axi].set_ylabel('Oversmoothing rate')
            axs[axi].set_xlabel('Suffix length')
            
            counts_axs[axi].set_xticks(major_suffix_length_ticks)
            counts_axs[axi].set_xticks(minor_suffix_length_ticks, minor=True)
            counts_axs[axi].set_yscale('log')
            counts_axs[axi].grid(which='both', alpha=0.5)
            if legend:
                counts_axs[axi].legend()
            counts_axs[axi].set_ylabel('Number of prefixes')
            counts_axs[axi].set_xlabel('Suffix length')
    
    for fig in figs:
        fig.show()
    for fig in counts_figs:
        fig.show()

    return (figs, axs), (counts_figs, counts_axs), to_return

def draw_oversmoothing_residual(experiments, neos_list=[1], plot_on_one_ax=True, quantiles=(.05,.5,.95),
                    exp_names=['FT IWSLT17 DE-EN NLL ES', 'FT IWSLT17 FR-EN NLL ES', 
                              'FT IWSLT17 ZH-EN NLL ES', 'FT WMT16 EN-DE NLL ES', 
                              'FT WMT19 RU-EN']):
    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 22}

    matplotlib.rc('font', **font)

    number_of_exprs = len(exp_names)*len(neos_list)
    numfigs = 1 if plot_on_one_ax else number_of_exprs

    # oversmoothing rate figs
    figs = []
    axs = []
    for i in range(numfigs):
        _fig, _ax = plt.subplots(figsize=(25,5))
        figs.append(_fig)
        axs.append(_ax)

    to_return = {}

    xs_max = 0
    ys_max = []
    for expi, experiment in enumerate(exp_names):
        # if we plot on the same ax, then we always use axs[0] or so, otherwise we map every experiment/neos to its own ax
        axi = 0 if plot_on_one_ax else expi
        if not plot_on_one_ax:
            xs_max = 0
        to_return[experiment] = {}
        mode = 'soft'
        suffix_oversmoothing = pickle.load(open(f'/scratch/mae9785/new_results/{experiments[experiment]}/{mode}oversmoothing_{experiments[experiment]}.pkl', 'rb'))
        
        for neosi, neos in enumerate(neos_list):
            axi = axi if plot_on_one_ax else expi*len(neos_list)+neosi
            plot_label = f'#EOS:{neos}, {experiment}' if neos > 1 else f'{experiment}'
            for seed, suffix_dict in suffix_oversmoothing[neos].items():
                xs, yss = get_data(suffix_dict, quantiles=quantiles)
                errorfill(xs, yss[quantiles[1]], yerrplus=yss[quantiles[2]], yerrminus=yss[quantiles[0]], ax=axs[axi], label=plot_label)
                ys_max.append(max(yss[quantiles[2]]))
                to_return[experiment][neos] = {
                    'suffix_lengths': xs,
                    'prob_diff': yss,
                    'suffix_dict': suffix_dict,
                }
                if max(xs) > xs_max:
                    xs_max = max(xs)
                break
            minor_suffix_length_ticks = np.arange(0, xs_max, 5)
            major_suffix_length_ticks = np.arange(0, xs_max, 10)
            axs[axi].set_xticks(major_suffix_length_ticks)
            axs[axi].set_xticks(minor_suffix_length_ticks, minor=True)
            axs[axi].grid(which='both', alpha=0.5)
            axs[axi].legend()
            axs[axi].set_ylabel('Oversmoothing residual')
            axs[axi].set_xlabel('Suffix length')
            axs[axi].set_ylim(-0.00001, np.median(ys_max))
    
    for fig in figs:
        fig.show()

    return (figs, axs), to_return

        
def draw_smoothing(experiments, mode='hard', quantiles=(.05,.5,.95), beam=2, neos_list=[1], use_one_ax=True, use_one_fig=False,
                   exp_names=['FT IWSLT17 DE-EN NLL ES', 'FT IWSLT17 FR-EN NLL ES', 
                              'FT IWSLT17 ZH-EN NLL ES', 'FT WMT16 EN-DE NLL ES', 
                              'FT WMT19 RU-EN']):
    import matplotlib.ticker as plticker
    import numpy

        
    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 22}

    matplotlib.rc('font', **font)
        

    number_exps = []
    fig, ax = plt.subplots(2,1,figsize=(25,10))

    xs_max = 0
    to_return = {}

    for i, experiment in enumerate(exp_names):
        to_return[experiment] = {}
        suffix_oversmoothing = pickle.load(open(f'/scratch/mae9785/new_results/{experiments[experiment]}/{mode}oversmoothing_{experiments[experiment]}.pkl', 'rb'))
        if mode == 'soft':
            fig, ax = plt.subplots(figsize=(25,5))
            for neos in neos_list:
                for seed, suffix_dict in suffix_oversmoothing[neos].items():
                    xs, yss = get_data(suffix_dict, quantiles=quantiles)
                    errorfill(xs, yss[quantiles[1]], yerrplus=yss[quantiles[2]], yerrminus=yss[quantiles[0]], ax=ax, label=f'#EOS:{neos} med+-45pct')
                    to_return[experiment][neos] = (xs, yss, suffix_dict)
                    break
            ax.set_ylim(-0.00001, max(yss[quantiles[2]]))
            ax.set_title(f'{experiment}')
            ax.set_xticks(xs)
            ax.set_ylabel('Soft degree of over-smoothing\n p(eos|pre) - p(suffix|pre)')
            for i, label in enumerate(ax.xaxis.get_ticklabels()):
                if i % 7 == 0:
                    label.set_visible(True)
                else:
                    label.set_visible(False)

            #plt.axhline(y=0.0, color='r', linestyle='-', lw=0.5)
            ax.legend()
        else:
            for neos in neos_list:
                plot_label = f'#EOS:{neos}, {experiment}' if neos > 1 else f'{experiment}'
                for seed, suffix_dict in suffix_oversmoothing[neos].items():
                    xs, ys = get_data_mean(suffix_dict)
                    suffix_counts = get_suffixlen_distr(suffix_dict)
                    line = ax[0].plot(xs, ys, label=plot_label, linewidth=3.0)
                    ax[1].plot(xs, suffix_counts, c=line[-1].get_color(), label=plot_label, linewidth=3.0)
                    to_return[experiment][neos] = {
                        'suffix_lengths': xs,
                        'hard_os': ys,
                        'suffix_counts': suffix_counts,
                        'suffix_dict': suffix_dict,
                    }
                    if max(xs) > xs_max:
                        xs_max = max(xs)
                    break

            ax[0].set_ylabel('Oversmoothing rate')
            ax[1].set_ylabel('Number of prefixes')
            ax[1].set_xlabel('Suffix length')
            ax[1].set_yscale('log')

            ax[0].legend()
            ax[1].legend()
    
    minor_suffix_length_ticks = np.arange(0, xs_max, 5)
    major_suffix_length_ticks = np.arange(0, xs_max, 10)

    try:
        for _ax in ax:
            _ax.set_xticks(major_suffix_length_ticks)
            _ax.set_xticks(minor_suffix_length_ticks, minor=True)
            _ax.grid(which='both', alpha=0.5)
    except:
        ax.set_xticks(major_suffix_length_ticks)
        ax.set_xticks(minor_suffix_length_ticks, minor=True)
        ax.grid(which='both', alpha=0.5)

    fig.tight_layout()
        
    return to_return

def create_internal_metric_latex(results, metric, columns, maximize=False, highlight=False, 
                                 dest_dir=None, external=False, precision=2):
    df, best_idxs, overlapping_idxs = print_table(results, metric, external=external, 
                                                  maximize=maximize, highlight=False, 
                                                  return_idxs=True) 
    column_row = ' &'.join(columns) + '\\\\ \\midrule'
    
    table_header = """
    \\begin{{table*}}[]
    \\begin{{tabular}}{{{}}}
    \\toprule

    {}
    """.format(('l' * (len(columns) + 1))[:-1], column_row)
    
    for i, row_name in enumerate(df.index):
        current_row = f'{row_name} '
        for j, column_name in enumerate(df.columns):
            value = df.iloc[i, j]
            if isinstance(value, str) and '+-' in value:
                mean, std = value.split(' +- ')
            else:
                mean = value
                std = '0'
            mean = round(float(mean), precision)
            std = round(float(std), precision)
            if highlight and (best_idxs[j] == i or i in overlapping_idxs[j]):
                mean = '\\textbf{{{}}}'.format(mean)
            if std > 0:
                value_latex = '{} \\scriptsize{{$\pm$ {}}}'.format(mean, std)
            else:
                value_latex = '{}'.format(mean)
            current_row += f" & {value_latex}"
        table_header += f' {current_row} \\\\ \n'
    
    table_header += """
    \\bottomrule
    \\end{tabular}
    \\caption{}
    \\end{table*}
    """
    
    if dest_dir is not None:
        metric = metric.replace('/', '_')
        metric = metric.replace('|', '')
        metric = metric.lower()
        metric = metric.replace(' ', '_')
        save_path = f'{dest_dir}/{metric}.tex'
        with open(save_path, 'w') as f:
            f.write(table_header)
    
    return table_header

