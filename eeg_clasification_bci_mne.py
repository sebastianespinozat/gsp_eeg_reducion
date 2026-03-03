# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 00:03:50 2025

@author: sebas
"""
import time
import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scipy.interpolate import interp1d
from mne.preprocessing import ICA
# %% FUNCIONTS


def time_to_samples(mrk, fs, offset):
    events_time = (mrk['time'].item()[0])/1000
    sample_idx = (events_time*fs).astype(int) + offset
    # 16 = left hand = 1, 32 = right hand = 0
    class_labels = mrk['y'].item()[0]
    return sample_idx, class_labels

# %% pre procesamiento test mne


'''
    =========  ===================================
    run        task
    =========  ===================================
    1          Baseline, eyes open
    2          Baseline, eyes closed
    3, 7, 11   Motor execution: left vs right hand
    4, 8, 12   Motor imagery: left vs right hand
    5, 9, 13   Motor execution: hands vs feet
    6, 10, 14  Motor imagery: hands vs feet
    =========  ===================================

T0 corresponds to rest
T1 corresponds to onset of motion (real or imagined) of
the left fist (in runs 3, 4, 7, 8, 11, and 12)
both fists (in runs 5, 6, 9, 10, 13, and 14)
T2 corresponds to onset of motion (real or imagined) of
the right fist (in runs 3, 4, 7, 8, 11, and 12)
both feet (in runs 5, 6, 9, 10, 13, and 14)

'''


def process_subjects(task):

    n_subjects = 109
    subjects = np.arange(1, n_subjects+1)

    runs = {0: [1], 1: [2], 2: [3, 7, 11], 3: [
        4, 8, 12], 4: [5, 9, 13], 5: [6, 10, 14]}
    PATH = 'C:/Users/sebas/Desktop/Universidad/Tesis/codigo/'
    for s in subjects:
        raw_fnames = [mne.datasets.eegbci.load_data(
            s, runs[task], path=PATH)]
        raw_fnames = np.reshape(raw_fnames, -1)
        raw = mne.io.concatenate_raws(
            [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
        mne.datasets.eegbci.standardize(raw)

        raw.annotations.rename(dict(T1="hands", T2="feet"))
        raw.set_eeg_reference(projection=True)
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage)

        eeg_pos = np.array(
            [pos for _, pos in raw.get_montage().get_positions()['ch_pos'].items()])
        channels = [ch for ch in raw.get_montage().get_positions()
                    ['ch_pos'].keys()]

        # if there are bad channels
        # picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

        # filtering
        lowF = 8
        highF = 30
        order = 3
        projection = False
        iir_params = dict(order=order, ftype='butter')
        filt = raw.filter(lowF, highF, method='iir',
                          iir_params=iir_params, skip_by_annotation='edge')
        filt.set_eeg_reference('average', projection=projection)

        events, events_id = mne.events_from_annotations(raw)

        TMIN, TMAX = -1.0, 4.0
        picks = mne.pick_types(raw.info, meg=False, eeg=True,
                               stim=False, eog=False, exclude="bads")
        epochs = mne.Epochs(raw, events, events_id,
                            picks=picks, tmin=TMIN,
                            tmax=TMAX, baseline=None,
                            preload=True,
                            detrend=1)
        useful_events = epochs.events
        epochs_data = epochs.get_data(copy=True)

        mask_events = useful_events[:, 2] != 1

        np.savez(f'data/task_{task}/mne_dataset_{s}_{lowF}-{highF}.npz',
                 data=epochs_data[mask_events],
                 time=epochs.times,
                 labels=useful_events[mask_events][:, 2]-2,
                 coordinates=eeg_pos,
                 ch_names=channels)
        print(f'############## subject {s} ##############')
        break


# %%
start = time.time()
for t in range(2, 6):
    process_subjects(t)
end = time.time()
elapsed = end-start

print(elapsed)

# %%
# Extrae un canal (por ejemplo el 0)
channel = 29
data_before, times = raw_original[channel]
data_after, _ = filt[channel]

plt.figure(figsize=(15, 4))
# plt.plot(times, data_before[0] * 1e6, label='Antes de ICA')  # volt -> µV
plt.plot(times, data_after[0] * 1e6, label='Después de ICA')  # volt -> µV
plt.title(f'Comparación canal {channel}')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (µV)')
plt.legend()
plt.tight_layout()
plt.show()

# %%


# %%


def compute_cross_validation(X, y, pipeline, subject, fold_num):

    y = np.asarray(y)
    name = pipeline["name"]
    clf = Pipeline(pipeline["pipeline"])

    # Split data
    cv = ShuffleSplit(n_splits=fold_num, test_size=0.2, random_state=42)

    results = {}
    results["name"] = name
    info = []
    scores = []
    preds = []
    folds = []
    y_test_out = []

    for i, (train_index, test_index) in enumerate(cv.split(X)):
        X_test = X[test_index]
        X_train = X[train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        if name == "CSP":
            info.append({"graphs": None, "Ws": None})
        clf.fit(X_train, y_train)
        if name.startswith("GSP"):
            clf_gsp = pipeline["pipeline"][0][-1]
            graphs = clf_gsp.get_graphs()
            Ws = clf_gsp.get_Ws()
            info.append({"graphs": graphs, "Ws": Ws})
        y_pred = clf.predict(X_test)
        preds.append(y_pred)
        scores.append([np.mean(y_pred == y_test)])
        folds.append(i)
        y_test_out.append(y_test)

    results["info"] = info
    results["scores"] = scores
    results["y"] = y_test_out
    results["y_pred"] = preds
    results["fold"] = folds

    print(
        f"""Subject: {subject} | pipeline: {name} | Classification accuracy: {
        np.mean(scores)}"""
    )

    return results
# %%


def process_subject(subject, pipelines, DATASET_FOLDER, fold_num, channels, task):
    lowF = 8
    highF = 30
    df_subject = []
    data = np.load(
        f'./{DATASET_FOLDER}/task_{task}/mne_dataset_{subject}_{lowF}-{highF}.npz')

    # load labels
    trials = data["data"]
    trials = trials[:, channels, :]
    labels = data["labels"] - 1
    data["time"]

    for pipeline in pipelines:
        results = compute_cross_validation(
            X=trials,  # [:, :, S1_train:S2_train],
            y=labels,
            pipeline=pipeline,
            subject=subject,
            fold_num=fold_num,
        )
        scores = results["scores"]
        # name = results.get("name")
        folds = results.get("fold")
        y_pred = results.get("y_pred")
        y = results.get("y")
        for fold in folds:
            df_subject.append(
                {
                    "subject": subject,
                    "N_sensors": len(channels),
                    # "name": name,
                    "accuracy": float(
                        scores[fold][0]
                    ),  # El 0 va porque es una lista de un valor
                    "connectivity": pipeline["connectivity"],
                    "transformation": pipeline["transformation"],
                    #  "info": info,
                    "fold": fold,
                    "y_pred": y_pred[fold],
                    "y": y[fold],
                }
            )

        df_subject_avg = {
            "subject": subject,
            "N_sensors": len(channels),
            "Average accuracy": np.mean([dfs['accuracy'] for dfs in df_subject]),
            "std": np.std([dfs['accuracy'] for dfs in df_subject]),
            "connectivity": pipeline["connectivity"],
            "transformation": pipeline["transformation"],
        }

    return df_subject_avg


# %% Analysis per component

def analysis_per_component(stop, step, task):
    componentes = np.arange(6, stop, step)
    N_channels = 64
    N_subjects = 109

    acc_per_comp = []
    std_per_comp = []
    for i in range(len(componentes)):
        n_components = int(componentes[i])
        print(f'COMPONENTE {n_components} DE {len(componentes)}')
        # Initialize classifier
        lda = LDA()

        # Init decoders
        reg = None
        # if task == 1:
        #     reg = 'ledoit_wolf'

        csp = CSP(n_components=n_components, reg=reg,
                  log=True, norm_trace=False)

        # Allocate variables
        pipelines = [
            {
                "name": "CSP",
                "pipeline": [
                    ("decoder", csp),
                    ("Scaler", StandardScaler()),
                    ("classifier", lda),
                ],
                "connectivity": "CSP",
                "transformation": "None",
            },
        ]

        subjects_all = []
        for sub in range(1, N_subjects+1):
            subject_ = process_subject(
                sub, pipelines, "data", 10, np.arange(0, N_channels), task)
            # se puede graficar el promedio de las acccuracy o el promedio y como caja la desviacion estandar.
            subjects_all.append(subject_)
            print(f'############### SUBJECT {sub} ###############')

        acc_per_comp.append([d.get('Average accuracy')
                             for d in subjects_all if 'Average accuracy' in d])
        std_per_comp.append([d.get('std') for d in subjects_all if 'std' in d])

    average_accuracy_per_comp = np.round(
        [[np.mean(lst) for lst in acc_per_comp]], 3)

    return average_accuracy_per_comp, acc_per_comp, std_per_comp, componentes

# %% PLOTS ACCURACY VS N_COMPONENTS


def plot_acc_vs_components(average_accuracy_per_comp, componentes, task, save=True):

    idx_max = np.argmax(average_accuracy_per_comp)

    f = interp1d(componentes, average_accuracy_per_comp.ravel(), kind='cubic')
    x_new = np.linspace(componentes[0], componentes[-1], 100)

    plt.figure()
    plt.plot(componentes, average_accuracy_per_comp.ravel(),
             '*-', label='original')
    plt.plot(x_new, f(x_new), '-', label='interpolado')
    plt.plot([componentes[idx_max]],
             [average_accuracy_per_comp.ravel()[idx_max]], 'r*', label='Max accuracy')
    plt.title(f'Average accuracy per N_components. Task {task}.')
    plt.xlabel('N_components')
    plt.ylabel('Avg acc')

    if save:
        plt.savefig('./data/acc_vs_components_mne_eegbci_task_{task}.png')

    plt.legend()
    plt.show()


def get_best_component(average_accuracy_per_comp, componentes):
    idx_max = np.argmax(average_accuracy_per_comp)
    return componentes[idx_max]


# %% CREATES PLOTS OF EACH N_COMPONENT ACCURACY PER SUBJECT

def plot_each_subject_per_component(acc_per_comp, std_per_comp, componentes, task, save=True):

    N_subjects = 109
    num_subplots = len(acc_per_comp)
    cols = 3
    rows = int(np.ceil(num_subplots / cols))

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(
        5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()  # Aplanar la matriz de ejes a una lista para iterar fácilmente

    # Crear cada subplot
    for i in range(num_subplots):
        avg = round(np.mean(acc_per_comp[i]), 3)
        std = np.round(std_per_comp[i], 3)
        axes[i].bar(np.arange(1, N_subjects+1),
                    acc_per_comp[i], color='skyblue')
        axes[i].axhline(y=avg, color='black', linestyle='--')
        # axes[i].errorbar(np.arange(1, N_subjects+1), acc_per_comp[i],
        #                  yerr=std, fmt="o", color="r")
        axes[i].set_title(
            f"N_components = {componentes[i]}. Avg Acc = {avg}. Task {task}")
        # Escala del eje Y (opcional, ajusta si es necesario)
        axes[i].set_ylim(0, 1)
        axes[i].set_xticklabels(np.arange(1, N_subjects+1), rotation=45)

    plt.tight_layout()
    if save:
        plt.savefig('./data/avg_acc_mne_eegbci_task_{task}.png')

    plt.show()


# %%
times_per_task = []
best_components = []
for t in range(2, 6):
    inicio = time.time()
    avg_comp, acc_comp, std_comp, comp = analysis_per_component(10, 1, t)
    final = time.time()
    times_per_task.append(final-inicio)
    best_comp_t = get_best_component(avg_comp, comp)
    best_components.append(best_comp_t)
    plot_acc_vs_components(avg_comp, comp, t, True)
    plot_each_subject_per_component(acc_comp, std_comp, comp, t, True)
    if t == 3:
        if best_components[0] == best_components[1]:
            print("EL MEJOR COMPONENTE SE REPITE JJIJUJUJIJI")
            break

# %%


# %%
n_components = 4
lda = LDA()

# Init decoders
csp = CSP(n_components=n_components, reg='ledoit_wolf',
          log=True, norm_trace=False)

# Allocate variables
pipelines = [
    {
        "name": "CSP",
        "pipeline": [
            ("decoder", csp),
            ("Scaler", StandardScaler()),
            ("classifier", lda),
        ],
        "connectivity": "CSP",
        "transformation": "None",
    },
]

# for i in range(1, 4):
#     sets_grafos = np.load(
#         f'./dataset_bbci/subgrafos_{i}.npy', allow_pickle=True).item()
#     for m in sets_grafos:
#         sets_grafos[m].append(list(range(1, 31)))
# %%


def run_graph_analysis(task, componente):
    n_grafos = 3
    N_subjects = 109
    N_channels = 64
    all_results = []

    lda = LDA()
    reg = None

    # Init decoders
    csp = CSP(n_components=int(componente), reg=reg,
              log=True, norm_trace=False)

    # Allocate variables
    pipelines = [
        {
            "name": "CSP",
            "pipeline": [
                ("decoder", csp),
                ("Scaler", StandardScaler()),
                ("classifier", lda),
            ],
            "connectivity": "CSP",
            "transformation": "None",
        },
    ]

    for i in range(1, n_grafos+1):
        grafo = np.load(
            f'./data/subgrafos_{i}.npy', allow_pickle=True).item()
        total_methods = 3
        for m in grafo:
            grafo[m].append(list(range(0, N_channels)))
        for sub in range(1, N_subjects+1):
            print(f'-------------SUBJECT {sub}-------------')
            for method in range(1, total_methods+1):
                sets = grafo[f'm{method}']
                for sensor_set in sets:
                    print(f'########## size{len(sensor_set)}')
                    subject_ = process_subject(
                        sub, pipelines, "data", 10, sensor_set, task)
                    row = {
                        'subject': sub,
                        'graph': i,
                        'method': method,
                        'n_channels': len(sensor_set),
                        'accuracy': subject_['Average accuracy']
                    }
                    all_results.append(row)

    df_results = pd.DataFrame(all_results)

    df_results.to_csv(f'./data/results_mne_eegbci_task_{task}.csv',
                      index=False, encoding='utf-8')

    return df_results


# %% CHECK IF RESULTS EXIST

def load_results(task):

    df_results = pd.read_csv(f'./data/results_mne_eegbci_task_{task}.csv')
    return df_results
# %%


df_results = load_results(5)

# %%
best_components = [8, 9, 6, 6]
results = []
total_start = time.time()
for t in range(0, 4):
    results_t = run_graph_analysis(t+2, best_components[t])
    results.append(results_t)
total_final = time.time()
total_time = total_final - total_start

# %%


# %%
df_ejemplo_1 = df_results[(df_results['subject'] == 'aa') & (
    df_results['graph'] == 1) & (df_results['method'] == 1)]
df_ejemplo_2 = df_results[(df_results['subject'] == 'aa') & (
    df_results['graph'] == 1) & (df_results['method'] == 2)]
df_ejemplo_3 = df_results[(df_results['subject'] == 'aa') & (
    df_results['graph'] == 1) & (df_results['method'] == 3)]
plt.figure()
plt.plot(df_ejemplo_1['n_channels'],
         df_ejemplo_1['accuracy'], '-*', label='method 1')
plt.plot(df_ejemplo_2['n_channels'],
         df_ejemplo_2['accuracy'], '-*', label='method 2')
plt.plot(df_ejemplo_3['n_channels'],
         df_ejemplo_3['accuracy'], '-*', label='method 3')

plt.xlabel('n channels')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# %%
def plot_graph_results(graph, df_results, task, save=True):

    df_g_m1 = df_results[(df_results['graph'] == graph)
                         & (df_results['method'] == 1)]
    df_avg_g_m1 = df_g_m1.groupby('n_channels')[
        'accuracy'].mean().reset_index()

    df_g_m2 = df_results[(df_results['graph'] == graph)
                         & (df_results['method'] == 2)]
    df_avg_g_m2 = df_g_m2.groupby('n_channels')[
        'accuracy'].mean().reset_index()

    df_g_m3 = df_results[(df_results['graph'] == graph)
                         & (df_results['method'] == 3)]
    df_avg_g_m3 = df_g_m3.groupby('n_channels')[
        'accuracy'].mean().reset_index()

    plt.figure()
    plt.plot(df_avg_g_m1['n_channels'],
             df_avg_g_m1['accuracy'], '-*', label=f'g{graph}_m1')
    plt.plot(df_avg_g_m2['n_channels'],
             df_avg_g_m2['accuracy'], '-*', label=f'g{graph}_m2')
    plt.plot(df_avg_g_m3['n_channels'],
             df_avg_g_m3['accuracy'], '-*', label=f'g{graph}_m3')
    plt.xlabel('n channels')
    plt.ylabel('accuracy')
    plt.title(
        f'Average accuracy by n_channels with graph {graph}. Task {task}')
    plt.legend()

    if save:
        plt.savefig(f'./data/eeg_bci_graph_{graph}_task_{task}.png')
    plt.show()


def plot_all_graphs(df_results, task, save=True):
    n_graphs = 3
    t = task-1
    plt.figure()
    for g in range(1, n_graphs+1):

        df_g_m1 = df_results[(df_results['graph'] == g)
                             & (df_results['method'] == 1)]
        df_avg_g_m1 = df_g_m1.groupby('n_channels')[
            'accuracy'].mean().reset_index()

        df_g_m2 = df_results[(df_results['graph'] == g)
                             & (df_results['method'] == 2)]
        df_avg_g_m2 = df_g_m2.groupby('n_channels')[
            'accuracy'].mean().reset_index()

        df_g_m3 = df_results[(df_results['graph'] == g)
                             & (df_results['method'] == 3)]
        df_avg_g_m3 = df_g_m3.groupby('n_channels')[
            'accuracy'].mean().reset_index()

        plt.plot(df_avg_g_m1['n_channels'],
                 df_avg_g_m1['accuracy'], '-o', label=f'g{g}_m1')
        plt.plot(df_avg_g_m2['n_channels'],
                 df_avg_g_m2['accuracy'], '-o', label=f'g{g}_m2')
        plt.plot(df_avg_g_m3['n_channels'],
                 df_avg_g_m3['accuracy'], '-o', label=f'g{g}_m3')
        plt.xlabel('n channels')
        plt.ylabel('accuracy')
    plt.title(
        f'Average accuracy by n_channels and graphs. Task 3')
    plt.legend()

    if save:
        plt.savefig(f'./data/eeg_bci_graphs_task_{t}.png')
    plt.show()


def plot_all_average(df_results, save=True):
    n_graphs = 3

    plt.figure()
    for g in range(1, n_graphs+1):

        df_g_m1 = df_results[(df_results['graph'] == g)
                             & (df_results['method'] == 1)]
        df_avg_g_m1 = df_g_m1.groupby('n_channels')[
            'accuracy'].mean().reset_index()

        df_g_m2 = df_results[(df_results['graph'] == g)
                             & (df_results['method'] == 2)]
        df_avg_g_m2 = df_g_m2.groupby('n_channels')[
            'accuracy'].mean().reset_index()

        df_g_m3 = df_results[(df_results['graph'] == g)
                             & (df_results['method'] == 3)]
        df_avg_g_m3 = df_g_m3.groupby('n_channels')[
            'accuracy'].mean().reset_index()

        plt.plot(df_avg_g_m1['n_channels'],
                 df_avg_g_m1['accuracy'], '-o', label=f'g{g}_m1')
        plt.plot(df_avg_g_m2['n_channels'],
                 df_avg_g_m2['accuracy'], '-o', label=f'g{g}_m2')
        plt.plot(df_avg_g_m3['n_channels'],
                 df_avg_g_m3['accuracy'], '-o', label=f'g{g}_m3')
        plt.xlabel('n channels')
        plt.ylabel('accuracy')
    plt.title(
        f'Average accuracy for all tasks by n_channels.')
    plt.legend()

    if save:
        plt.savefig(f'./data/eeg_bci_graphs.png')
    plt.show()


# %% plot motor imaginery task
t = 5

plot_graph_results(1, df_results, t, False)
plot_graph_results(2, df_results, t, False)
plot_graph_results(3, df_results, t, False)


# %% plot all graphs and methods for task

plot_all_graphs(results[0], 2)

plot_all_graphs(results[1], 3)

plot_all_graphs(results[2], 4)

plot_all_graphs(results[3], 5)
# %%
df1 = results[0].groupby(['graph', 'method', 'n_channels'])[
    'accuracy'].mean().reset_index()
df1['task'] = 2

df2 = results[1].groupby(['graph', 'method', 'n_channels'])[
    'accuracy'].mean().reset_index()
df2['task'] = 3

df3 = results[2].groupby(['graph', 'method', 'n_channels'])[
    'accuracy'].mean().reset_index()
df3['task'] = 4

df4 = results[3].groupby(['graph', 'method', 'n_channels'])[
    'accuracy'].mean().reset_index()
df4['task'] = 5

df_all = pd.concat([df1, df2, df3, df4], ignore_index=True)

df_all = df_all.groupby(['task', 'graph', 'method', 'n_channels'])[
    'accuracy'].mean().reset_index()
# %%
plot_all_average(df_all, save=True)
# %%

# %%

# %%
n_grafos = 3
ite = 0
SUBJECT = ['al']

# fig, axes = plt.subplots(4, 6, figsize=(30, 3.5))

for i in range(1, n_grafos+1):
    grafo = np.load(f'./data/subgrafos_{i}.npy', allow_pickle=True).item()
    total_methods = 3
    for m in grafo:
        grafo[m].append(list(range(1, 64)))
    # for sub in SUBJECT:
    sub = 1
    print(f'-------------SUBJECT {sub}-------------')
    for method in range(1, total_methods+1):
        # fig, axes = plt.subplots(1, 10, figsize=(30, 3.5))
        sets = grafo[f'm{method}']
        fig, axes = plt.subplots(4, 6, figsize=(35, 40))
        # fig_topo, axes_topo = plt.subplots(4, 6, figsize=(30, 3.5))
        axes = axes.reshape(-1)
        # axes_topo = axes_topo.reshape(-1)
        fig.suptitle(f'graph {i}, metodo {method}')
        # fig_topo.suptitle(f'graph {i}, metodo {method}')
        print(len(sets))
        for s in range(len(sets)):
            sensor_set = sets[s]
            info = filt.info
            info_new = mne.pick_info(info, sel=sensor_set)
            info_new.plot_sensors(
                axes=axes[s], sphere=0.12, show_names=False)
            axes[s].set_title(f'{len(sensor_set)} electrodes')

            # topo_data = np.zeros(118)
            # topo_data[sensor_set] = 1
            # mne.viz.plot_topomap(
            #     topo_data, info, sphere=1, image_interp="nearest", axes=axes_topo[s], vlim=(0, 1))
        plt.savefig(
            f'./data/mnebci_topomap_g{i}_m{method}.png')
        # break
    # break


print('done.')

# %%

# %%

# %%

# %%

# %%
