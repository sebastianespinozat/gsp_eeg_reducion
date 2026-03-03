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
# %%


def get_dropped_channels(ch_names, sensor_sets):
    dropped_channels = []
    for selected in sensor_sets:
        dropped = [ch_names[i]
                   for i in range(len(ch_names)) if i not in selected]
        dropped_channels.append(dropped)
    return dropped_channels

# %%


SAVE_FILE_PATH = "./dataset_comp_IVb/al.npz"
train_data = loadmat('./dataset_comp_IVb/data_set_IVb_al_train.mat')
test_data = loadmat('./dataset_comp_IVb/data_set_IVb_al_test.mat')
test_labels = loadmat('./dataset_comp_IVb/true_labels.mat')

labels = test_labels["true_y"].squeeze()
template = test_labels["template"].squeeze()

mrk = train_data["mrk"][0, 0]
events_samples = mrk[0][0, :]
events_labels = mrk[1][0, :]
events_labels[events_labels == -1] = 0
event_description = {0: 'left', 1: 'foot'}
# event_description = {key + 1: value[0]
#                      for key, value in enumerate(mrk[1][0])}

info = train_data["nfo"][0, 0]
cnt = (train_data["cnt"].T)*0.1
FS = int(info[1][0, :])
ch_names = [ch_name[0] for ch_name in info[2][0]]
pos = np.array([info[3][:, 0], info[4][:, 0]])
zeros = np.zeros((1, pos.shape[1]))
pos = np.vstack((pos, zeros))


# Create mne structure
info_mne = mne.create_info(ch_names, FS, "eeg")
raw = mne.io.RawArray(cnt, info_mne)
montage = mne.channels.make_dig_montage(
    ch_pos=dict(zip(ch_names, pos.T)), coord_frame='head')
raw.set_montage(montage)
# filt = raw.filter(8, 30)

raw_clean = raw.copy()
raw = raw.filter(0.5, 45)

# ARTIFACT REMOVAL
n_components = 30
ica = ICA(n_components=n_components, random_state=97)
ica.fit(raw)
ica.plot_components()

ica.exclude = [0, 2, 11]

ica.apply(raw)

# filtering

lowF = 8
highF = 30
order = 3
projection = False
iir_params = dict(order=order, ftype='butter')
filt = raw.filter(lowF, highF, method='iir',
                  iir_params=iir_params, skip_by_annotation='edge')

filt.set_eeg_reference("average", projection=projection)
# epoch data
events = np.zeros((len(events_samples), 3))
events[:, 0] = events_samples
events[:, 2] = events_labels
annotations = mne.annotations_from_events(events, FS, event_description)
filt.set_annotations(annotations)
events, _ = mne.events_from_annotations(filt)

epochs = mne.Epochs(filt, events, tmin=0.5, tmax=2.5,
                    baseline=None, preload=True)

epochs_data = epochs.get_data(copy=True)

np.savez(SAVE_FILE_PATH,
         data=epochs_data,
         time=epochs.times,
         labels=events_labels,
         coordinates=pos,
         ch_names=ch_names
         )

# %%


# %% test data
SAVE_FILE_PATH_test = "./dataset_comp_IVb/al_test.npz"
cnt_test = (test_data["cnt"].T)*0.1
mask = np.isin(labels, [-1, 1])
events_samples_test = np.where(mask)[0]
events_labels_test = labels[mask]

onsets = events_samples_test[np.insert(
    np.diff(events_samples_test) > 1, 0, True)]
event_clases = labels[onsets]

event_clases[event_clases == -1] = 0
event_description = {0: 'left', 1: 'foot'}

# Create mne structure
info_mne_test = mne.create_info(ch_names, FS, "eeg")
raw_test = mne.io.RawArray(cnt_test, info_mne_test)
montage = mne.channels.make_dig_montage(
    ch_pos=dict(zip(ch_names, pos.T)), coord_frame='head')
raw_test.set_montage(montage)
# filt = raw.filter(8, 30)

raw_test_raw = raw_test.copy()
raw_test = raw_test.filter(0.5, 45)

# ARTIFACT REMOVAL
n_components = 30
ica = ICA(n_components=n_components, random_state=97)
ica.fit(raw)

ica.plot_components()
# %%
ica.exclude = [13, 20]

ica.apply(raw_test)

# filtering

lowF = 8
highF = 30
order = 3
projection = False
iir_params = dict(order=order, ftype='butter')
filt_test = raw_test.filter(lowF, highF, method='iir',
                            iir_params=iir_params, skip_by_annotation='edge')

filt_test.set_eeg_reference("average", projection=projection)
# epoch data
events_test = np.zeros((len(onsets), 3))
events_test[:, 0] = onsets
events_test[:, 2] = event_clases
annotations = mne.annotations_from_events(events_test, FS, event_description)
filt_test.set_annotations(annotations)
events_test, _ = mne.events_from_annotations(filt_test)

epochs_test = mne.Epochs(filt_test, events_test, tmin=0, tmax=1.0,
                         baseline=None, preload=True)

epochs_data_test = epochs_test.get_data(copy=True)

np.savez(SAVE_FILE_PATH_test,
         data=epochs_data_test,
         time=epochs_test.times,
         labels=event_clases,
         coordinates=pos,
         ch_names=ch_names
         )

# %%
channel = 0
data_raw, time_raw = raw_clean[channel, :]
data, time = filt[channel, :]
plt.figure()
plt.plot(data_raw[0], label='raw signal')
plt.plot(data[0], label='preprocessed signal')
plt.xlabel('Muesta N')
plt.ylabel('Amplitud (µV)')
plt.legend()
plt.tight_layout()
plt.show()


# %%
def compute_cross_validation(X, y, pipeline, subject, fold_num, X_test, y_test):

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

    # now test
    if X_test is not None and y_test is not None:
        clf.fit(X, y)
        y_test_pred = clf.predict(X_test)
        test_acc = np.mean(y_test_pred == y_test)
        results["test_accuracy"] = test_acc
        results["test_y_pred"] = y_test_pred
        results["test_y"] = y_test
        print(
            f"Subject: {subject} | pipeline: {name} | TEST Accuracy: {test_acc:.3f}")

    return results


# %%


def process_subject(subject, pipelines, DATASET_FOLDER, fold_num, channels, X_test, y_test):
    df_subject = []
    data = np.load(f'./{DATASET_FOLDER}/{subject}.npz')

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
            X_test=X_test,
            y_test=y_test
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

    # return df_subject_avg, results
    return df_subject_avg


# %% Analysis per component
X_test = epochs_data_test
y_test = event_clases

SUBJECT = ['al']
componentes = np.arange(16, 17)
acc_per_comp = []
std_per_comp = []
for i in range(len(componentes)):
    n_components = int(componentes[i])
    print(f'COMPONENTE {n_components} DE {len(componentes)}')
    # Initialize classifier
    lda = LDA()

    # Init decoders
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)

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
    for sub in SUBJECT:
        subject_, results_test = process_subject(
            sub, pipelines, "dataset_comp_IVb", 10, np.arange(0, 118), X_test, y_test)
        # se puede graficar el promedio de las acccuracy o el promedio y como caja la desviacion estandar.
        subjects_all.append(subject_)

    acc_per_comp.append([d.get('Average accuracy')
                         for d in subjects_all if 'Average accuracy' in d])
    std_per_comp.append([d.get('std') for d in subjects_all if 'std' in d])

average_accuracy_per_comp = np.round(
    [[np.mean(lst) for lst in acc_per_comp]], 3)

# %%


def get_best_component(average_accuracy_per_comp, componentes):
    idx_max = np.argmax(average_accuracy_per_comp)
    return componentes[idx_max]

# %%


best_comp = get_best_component(average_accuracy_per_comp, componentes)

# %%
f = interp1d(componentes, average_accuracy_per_comp.ravel(), kind='cubic')
x_new = np.linspace(componentes[0], componentes[-1], 100)

plt.figure()
plt.plot(componentes, average_accuracy_per_comp.ravel(), '*-', label='original')
plt.plot(x_new, f(x_new), '-', label='interpolado')
plt.title('Average accuracy per N_components')
plt.xlabel('N_components')
plt.ylabel('Avg acc')
# plt.savefig('acc_vs_components.png')
plt.legend()
plt.show()


# %%
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
    axes[i].bar(SUBJECT, acc_per_comp[i], color='skyblue')
    axes[i].axhline(y=avg, color='black', linestyle='--')
    axes[i].errorbar(SUBJECT, acc_per_comp[i], yerr=std, fmt="o", color="r")
    axes[i].set_title(f"N_components = {componentes[i]}. Avg Acc = {avg}.")
    # Escala del eje Y (opcional, ajusta si es necesario)
    axes[i].set_ylim(0, 1)
    axes[i].set_xticklabels(SUBJECT, rotation=45)

plt.tight_layout()
# plt.savefig('./dataset_comp_IVb/avg_acc_comp_iva.png')
plt.show()
# %%


def run_graph_analysis(componente):

    SUBJECT = ["al"]
    n_grafos = 3
    all_results = []

    lda = LDA()
    # Init decoders
    csp = CSP(n_components=int(componente), reg=None,
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
        grafo = np.load(f'subgrafos_{i}.npy', allow_pickle=True).item()
        total_methods = 3
        for m in grafo:
            grafo[m].append(list(range(1, 118)))
        for sub in SUBJECT:
            print(f'-------------SUBJECT {sub}-------------')
            for method in range(1, total_methods+1):
                sets = grafo[f'm{method}']
                for sensor_set in sets:
                    subject_ = process_subject(
                        sub, pipelines, "dataset_comp_IVb", 10, sensor_set, X_test, y_test)
                    row = {
                        'subject': sub,
                        'graph': i,
                        'method': method,
                        'n_channels': len(sensor_set),
                        'accuracy': subject_['Average accuracy']
                    }
                    all_results.append(row)
    print('done.')

    df_results = pd.DataFrame(all_results)
    df_results.to_csv('results_BCI_COMP_IVb.csv',
                      index=False, encoding='utf-8')
    return df_results


# %% CHECK IF RESULTS EXIST
try:
    df_results
except NameError:
    df_results = pd.read_csv('results_BCI_COMP_IVb.csv')
# %%


def plot_graph_results(graph, df_results, save=True):

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
             df_avg_g_m1['accuracy'], '-o', label=f'g{graph}_m1')
    plt.plot(df_avg_g_m2['n_channels'],
             df_avg_g_m2['accuracy'], '-o', label=f'g{graph}_m2')
    plt.plot(df_avg_g_m3['n_channels'],
             df_avg_g_m3['accuracy'], '-o', label=f'g{graph}_m3')
    plt.xlabel('n channels')
    plt.ylabel('accuracy')
    plt.title(
        f'Average accuracy by n_channels with graph {graph}')
    plt.legend()

    if save:
        plt.savefig(f'./dataset_comp_IVb/comp_IVb_graph_{graph}.png')
    plt.show()


def plot_all_graphs(df_results, save=True):
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
    plt.title('Average accuracy by n_channels and graphs.')
    plt.legend()

    if save:
        plt.savefig(f'./dataset_comp_IVb/comp_IVb_graphs.png')
    plt.show()
# %%


g = 1
df_g_m1 = df_results[(df_results['graph'] == g) & (df_results['method'] == 1)]
df_avg_g_m1 = df_g_m1.groupby('n_channels')['accuracy'].mean().reset_index()
df_g_m2 = df_results[(df_results['graph'] == g) & (df_results['method'] == 2)]
df_avg_g_m2 = df_g_m2.groupby('n_channels')['accuracy'].mean().reset_index()

df_g_m3 = df_results[(df_results['graph'] == g) & (df_results['method'] == 3)]
df_avg_g_m3 = df_g_m3.groupby('n_channels')['accuracy'].mean().reset_index()

# %%


# results = run_graph_analysis(best_comp)
results = run_graph_analysis(16)
# %%
# %%
plot_graph_results(1, results, save=True)
plot_graph_results(2, results, save=True)
plot_graph_results(3, results, save=True)
# %%


plot_all_graphs(results, save=False)
# %% plots eegs

montage.plot(sphere=0.95, kind='topomap')

ch_borrar = [0, 1, 2, 3, 44, 66, 12, 23, 45, 65, 20, 21, 23, 35, 64]
info = raw.info
nombres_borrar = [info['ch_names'][i] for i in ch_borrar]
print(nombres_borrar)
info_new = mne.pick_info(info)  # , exclude=nombres_borrar)
info_new.plot_sensors()
# info.plot_sensors()


# %%
n_grafos = 3
ite = 0
SUBJECT = ['al']

# fig, axes = plt.subplots(4, 6, figsize=(30, 3.5))

for i in range(1, n_grafos+1):
    grafo = np.load(f'subgrafos_{i}.npy', allow_pickle=True).item()
    total_methods = 3
    for m in grafo:
        grafo[m].append(list(range(1, 118)))
    for sub in SUBJECT:
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
                info = raw.info
                info_new = mne.pick_info(info, sel=sensor_set)
                info_new.plot_sensors(
                    axes=axes[s], sphere=0.95, show_names=False)
                axes[s].set_title(f'{len(sensor_set)} electrodes')

                # topo_data = np.zeros(118)
                # topo_data[sensor_set] = 1
                # mne.viz.plot_topomap(
                #     topo_data, info, sphere=1, image_interp="nearest", axes=axes_topo[s], vlim=(0, 1))
            # plt.savefig(
                # f'./dataset_comp_IVb/comp_IVb_topomap_g{i}_m{method}.png')
        # break
    # break


print('done.')
# %%
montage = mne.channels.make_dig_montage(
    ch_pos=dict(zip(ch_names, pos.T)), coord_frame='head')

# %%
# data = np
data_shape = epochs.average().get_data().shape
times = np.arange(0.5, 2.5, 0.5)
epochs.average().plot_topomap(times, ch_type="eeg",
                              sphere=1)  # , exclude=list(range(21)))
# %%
topo_data = epochs.average().get_data()[:, 0]
mne.viz.plot_topomap(topo_data, info, sphere=1, image_interp="cubic")
mne.viz.plot_topomap(topo_data, info, sphere=1, image_interp="nearest")
mne.viz.plot_topomap(topo_data, info, sphere=1, image_interp="linear")
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
