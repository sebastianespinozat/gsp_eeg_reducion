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


# %% PRE PROCESAMIENTo

def preprocess_bbci(task, n_components=30, tmax=3.0):
    N_subjects = 29
    # task: 0 if imaginery, 1 if arithmetic

    for subject in range(1, N_subjects+1):
        if task == 0:
            SAVE_FILE_PATH = f'./dataset_bbci/{subject}_t0.npz'
        else:
            SAVE_FILE_PATH = f'./dataset_bbci/{subject}_t1.npz'
        if subject < 10:
            cnt_location = f'./dataset_bbci/subject 0{subject}/with occular artifact/cnt.mat'
            mrk_location = f'./dataset_bbci/subject 0{subject}/with occular artifact/mrk.mat'
            mnt_location = f'./dataset_bbci/subject 0{subject}/with occular artifact/mnt.mat'
        else:
            cnt_location = f'./dataset_bbci/subject {subject}/with occular artifact/cnt.mat'
            mrk_location = f'./dataset_bbci/subject {subject}/with occular artifact/mrk.mat'
            mnt_location = f'./dataset_bbci/subject {subject}/with occular artifact/mnt.mat'

        cnt = loadmat(cnt_location)
        cnt = cnt['cnt']
        mrk = loadmat(mrk_location)
        mrk = mrk['mrk']
        mnt = loadmat(mnt_location)
        mnt = mnt['mnt']

        clab_mnt = [c[0] for c in mnt['clab'].item()[0]]
        x_pos = mnt['x'].item()
        y_pos = mnt['y'].item()
        eeg_pos = mnt['pos_3d'].item()

        clab = cnt[0, 0]['clab']
        ch_names = [c.item() for c in clab[0, 0][0]]
        fs = cnt[0, 0]['fs'].item().item()
        # imaginery task
        if task == 0:
            x1 = cnt[0, 0]['x'].item()
            x2 = cnt[0, 2]['x'].item()
            x3 = cnt[0, 4]['x'].item()
            mrk1 = mrk[0, 0]
            mrk2 = mrk[0, 2]
            mrk3 = mrk[0, 4]
            event_description = {0: 'right', 1: 'left'}
        # arithmetic task
        else:
            x1 = cnt[0, 1]['x'].item()
            x2 = cnt[0, 3]['x'].item()
            x3 = cnt[0, 5]['x'].item()
            mrk1 = mrk[0, 1]
            mrk2 = mrk[0, 3]
            mrk3 = mrk[0, 5]
            event_description = {0: 'baseline', 1: 'mental arithmetic'}

        X_total = np.concatenate([x1, x2, x3], axis=0)
        # X_total /= 1e6

        samples1, labels1 = time_to_samples(mrk1, fs, offset=0)
        samples2, labels2 = time_to_samples(mrk2, fs, offset=x1.shape[0])
        samples3, labels3 = time_to_samples(
            mrk3, fs, offset=x1.shape[0] + x2.shape[0])

        samples_all = np.concatenate([samples1, samples2, samples3])
        labels_all = np.concatenate([labels1, labels2, labels3])

        # mne object
        # 16 = left hand = 1, 32 = right hand = 0, cnt{1,3,5} o cnt[0,2,4]
        # 16 = left hand = 1, 32 = right hand = 0, cnt{2,4,6} o cnt[1,3,5]
        ch_types = ['eeg']*30+['eog']*2
        montage = mne.channels.make_dig_montage(
            ch_pos=dict(zip(ch_names[:30], eeg_pos[:, :30].T)), coord_frame='head')
        info_mne = mne.create_info(ch_names, fs, ch_types)
        data = X_total.T
        raw = mne.io.RawArray(data * 1e-6, info_mne)
        raw.set_montage(montage)

        # raw_original = raw.copy()
        raw = raw.filter(0.5, 45)

        # ARTIFACT REMOVAL
        ica = ICA(n_components=n_components, random_state=97)
        ica.fit(raw)

        eog_inds, scores = ica.find_bads_eog(raw)
        ica.exclude = eog_inds
        ica.apply(raw)

        # useful ICA plot functions
        # ica.plot_components()
        # ica.plot_scores(scores, eog_inds)
        # ica.exclude = eog_inds
        # ica.plot_components()
        # ica.plot_overlay(raw, exclude=eog_inds)
        # ica.plot_overlay(raw, exclude=eog_inds, show=True,
        #                  picks='eeg')
        # ica.plot_sources(raw)
        # ica.plot_properties(raw, picks=eog_inds, log_scale=True)

        # filtering
        lowF = 8
        highF = 30
        order = 3
        projection = False
        iir_params = dict(order=order, ftype='butter')
        filt = raw.filter(lowF, highF, method='iir',
                          iir_params=iir_params, skip_by_annotation='edge')
        filt.set_eeg_reference('average', projection=projection)

        # epoch data
        events = np.zeros((len(samples_all), 3))
        events[:, 0] = samples_all
        events[:, 2] = labels_all
        annotations = mne.annotations_from_events(
            events, fs, event_description)
        filt.set_annotations(annotations)
        events, _ = mne.events_from_annotations(filt)

        tmin = 0.5
        # tmax = 3.0
        epochs = mne.Epochs(filt, events, tmin=tmin, tmax=tmax, reject=dict(eeg=100e-6),
                            baseline=None, preload=True)

        epochs_data = epochs.get_data(copy=True)
        epochs_data = np.delete(epochs_data, [-1, -2], axis=1)
        # shape: (trial, channel, data)

    #   saving data
        print(epochs_data.shape)
        np.savez(SAVE_FILE_PATH,
                 data=epochs_data,
                 time=epochs.times,
                 labels=labels_all,
                 coordinates=eeg_pos,
                 ch_names=ch_names)
        print(f'############################ N{subject}')
        break


# %%
preprocess_bbci(0)
preprocess_bbci(1)
# %%


plt.figure()
plt.plot(epochs_data[0, 14, :])

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
    df_subject = []
    if task == 0:
        data = np.load(f'./{DATASET_FOLDER}/{subject}_t0.npz')
    else:
        data = np.load(f'./{DATASET_FOLDER}/{subject}_t1.npz')

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
    componentes = np.arange(2, stop, step)
    N_channels = 30
    N_subjects = 29

    acc_per_comp = []
    std_per_comp = []
    for i in range(len(componentes)):
        n_components = int(componentes[i])
        print(f'COMPONENTE {n_components} DE {len(componentes)}')
        # Initialize classifier
        lda = LDA()

        # Init decoders
        reg = None
        if task == 1:
            reg = 'ledoit_wolf'

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
                sub, pipelines, "dataset_bbci", 10, np.arange(0, N_channels), task)
            # se puede graficar el promedio de las acccuracy o el promedio y como caja la desviacion estandar.
            subjects_all.append(subject_)

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
        if task == 0:
            plt.savefig('./dataset_bbci/acc_vs_components_bbci_imaginery.png')
        else:
            plt.savefig('./dataset_bbci/acc_vs_components_bbci_arithmetic.png')
    plt.legend()
    plt.show()


def get_best_component(average_accuracy_per_comp, componentes):
    idx_max = np.argmax(average_accuracy_per_comp)
    return componentes[idx_max]


# %% CREATES PLOTS OF EACH N_COMPONENT ACCURACY PER SUBJECT

def plot_each_subject_per_component(acc_per_comp, std_per_comp, componentes, task, save=True):

    N_subjects = 29
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
        axes[i].errorbar(np.arange(1, N_subjects+1), acc_per_comp[i],
                         yerr=std, fmt="o", color="r")
        axes[i].set_title(f"N_components = {componentes[i]}. Avg Acc = {avg}.")
        # Escala del eje Y (opcional, ajusta si es necesario)
        axes[i].set_ylim(0, 1)
        axes[i].set_xticklabels(np.arange(1, N_subjects+1), rotation=45)

    plt.tight_layout()
    if save:
        if task == 0:
            plt.savefig('./dataset_bbci/avg_acc_bbci_imaginery.png')
        else:
            plt.savefig('./dataset_bbci/avg_acc_bbci_arithmetic.png')
    plt.show()


# %%

avg_comp, acc_comp, std_comp, comp = analysis_per_component(10, 2, 0)

plot_acc_vs_components(avg_comp, comp, 0, False)

plot_each_subject_per_component(acc_comp, std_comp, comp, 0, False)

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
    N_subjects = 29
    N_channels = 30
    all_results = []

    lda = LDA()
    reg = None
    if task == 1:
        reg = 'ledoit_wolf'
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
            f'./dataset_bbci/subgrafos_{i}.npy', allow_pickle=True).item()
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
                        sub, pipelines, "dataset_bbci", 10, sensor_set, task)
                    row = {
                        'subject': sub,
                        'graph': i,
                        'method': method,
                        'n_channels': len(sensor_set),
                        'accuracy': subject_['Average accuracy']
                    }
                    all_results.append(row)

    df_results = pd.DataFrame(all_results)

    if task == 0:
        df_results.to_csv('./dataset_bbci/results_BBCI_motor.csv',
                          index=False, encoding='utf-8')
    else:
        df_results.to_csv('./dataset_bbci/results_BBCI_arithmetic.csv',
                          index=False, encoding='utf-8')

    return df_results


# %% CHECK IF RESULTS EXIST

def load_results(task):

    if task == 0:
        df_results = pd.read_csv('./dataset_bbci/results_BBCI_motor.csv')
    else:
        df_results = pd.read_csv('./dataset_bbci/results_BBCI_arithmetic.csv')
    return df_results


# %%
start = time.time()

avg_acc_t0, acc_comp_t0, std_comp_t0, comp_t0 = analysis_per_component(
    30, 1, 0)
avg_acc_t1, acc_comp_t1, std_comp_t1, comp_t1 = analysis_per_component(
    30, 1, 1)

best_comp_t0 = get_best_component(avg_acc_t0, comp_t0)
best_comp_t1 = get_best_component(avg_acc_t1, comp_t1)

end = time.time()
time_cmp = end-start
# %%


start = time.time()

results_t0 = run_graph_analysis(0, best_comp_t0)
results_t1 = run_graph_analysis(1, best_comp_t1)

end = time.time()
time_graph_analysis = end-start


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
        f'Average accuracy by n_channels with graph {graph}. Task {task+1}')
    plt.legend()

    if save:
        if task == 0:
            plt.savefig(f'./dataset_bbci/bbci_motor_graph_{graph}.png')
        else:
            plt.savefig(f'./dataset_bbci/bbci_arithmetic_graph_{graph}.png')
    plt.show()


def plot_all_graphs(df_results, task, save=True):
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
        f'Average accuracy by n_channels and graphs. Task {task}')
    plt.legend()

    if save:
        if task == 0:
            plt.savefig(f'./dataset_bbci/bbci_motor_graphs.png')
        else:
            plt.savefig(f'./dataset_bbci/bbci_arithmetic_graphs.png')
    plt.show()

# %% plot motor imaginery task


results_t0 = load_results(0)
results_t1 = load_results(1)
# %%


plot_graph_results(1, results_t0, 0, True)
plot_graph_results(2, results_t0, 0, True)
plot_graph_results(3, results_t0, 0, True)

# %% plot arithmetic task

plot_graph_results(1, results_t1, 1, True)
plot_graph_results(2, results_t1, 1, True)
plot_graph_results(3, results_t1, 1, True)
# %% plot all graphs and methods for task

plot_all_graphs(results_t0, 0)

plot_all_graphs(results_t1, 1)
# %%

# %%

n_grafos = 3
ite = 0
SUBJECT = ['al']

# fig, axes = plt.subplots(4, 6, figsize=(30, 3.5))

for i in range(1, n_grafos+1):
    grafo = np.load(
        f'./dataset_bbci/subgrafos_{i}.npy', allow_pickle=True).item()
    total_methods = 3
    for m in grafo:
        grafo[m].append(list(range(1, 30)))
    # for sub in SUBJECT:
    sub = 1
    print(f'-------------SUBJECT {sub}-------------')
    for method in range(1, total_methods+1):
        # fig, axes = plt.subplots(1, 10, figsize=(30, 3.5))
        sets = grafo[f'm{method}']
        fig, axes = plt.subplots(3, 6, figsize=(35, 40))
        # fig_topo, axes_topo = plt.subplots(4, 6, figsize=(30, 3.5))
        axes = axes.reshape(-1)
        # axes_topo = axes_topo.reshape(-1)
        fig.suptitle(f'graph {i}, metodo {method}')
        # fig_topo.suptitle(f'graph {i}, metodo {method}')
        print(len(sets))
        for s in range(len(sets)):
            sensor_set = sets[s]
            info = filt.info
            # sphere = mne.make_sphere_model(
            #     r0="auto", head_radius="auto", info=info)
            info_new = mne.pick_info(info, sel=sensor_set)
            info_new.plot_sensors(
                axes=axes[s], sphere=1.0, show_names=False)
            axes[s].set_title(f'{len(sensor_set)} electrodes')

            # topo_data = np.zeros(118)
            # topo_data[sensor_set] = 1
            # mne.viz.plot_topomap(
            #     topo_data, info, sphere=1, image_interp="nearest", axes=axes_topo[s], vlim=(0, 1))
        plt.savefig(
            f'./dataset_bbci/bbci_topomap_g{i}_m{method}.png')
        # break
    # break


print('done.')
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
