
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

# %%


def get_dropped_channels(ch_names, sensor_sets):
    dropped_channels = []
    for selected in sensor_sets:
        dropped = [ch_names[i]
                   for i in range(len(ch_names)) if i not in selected]
        dropped_channels.append(dropped)
    return dropped_channels

# %%


SUBJECT = ["aa", "al", "av", "aw", "ay"]

for sub in SUBJECT:
    SAVE_FILE_PATH = f"./dataset_comp/{sub}.npz"
    data = loadmat(f'./dataset_comp/data_set_IVa_{sub}.mat')
    labels_dict = loadmat(f'./dataset_comp/true_labels_{sub}.mat')

    labels = labels_dict["true_y"].squeeze()
    test_idx = labels_dict["test_idx"].squeeze()

    mrk = data["mrk"][0, 0]
    events_samples = mrk[0][0, :]
    event_description = {}
    event_description = {key + 1: value[0]
                         for key, value in enumerate(mrk[2][0])}

    info = data["nfo"][0, 0]
    cnt = (data["cnt"].T)*0.1
    FS = int(info[1][0, :])
    ch_names = [ch_name[0] for ch_name in info[2][0]]
    pos = np.array([info[3][:, 0], info[4][:, 0]])

    # Create mne structure
    info_mne = mne.create_info(ch_names, FS, "eeg")
    raw = mne.io.RawArray(cnt, info_mne)

    # filt = raw.filter(8, 30)
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
    events[:, 2] = labels
    annotations = mne.annotations_from_events(events, FS, event_description)
    filt.set_annotations(annotations)
    events, _ = mne.events_from_annotations(filt)

    epochs = mne.Epochs(filt, events, tmin=0.5, tmax=2.5,
                        baseline=None, preload=True)

    epochs_data = epochs.get_data(copy=True)

    np.savez(SAVE_FILE_PATH,
             data=epochs_data,
             time=epochs.times,
             labels=labels,
             test_idx=test_idx,
             coordinates=pos,
             ch_names=ch_names
             )


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


def process_subject(subject, pipelines, DATASET_FOLDER, fold_num, channels):
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
    # return df_subject


# %% Analysis per component
componentes = np.arange(2, 50, 5)
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
        subject_ = process_subject(
            sub, pipelines, "dataset_comp", 10, np.arange(0, 118))
        # se puede graficar el promedio de las acccuracy o el promedio y como caja la desviacion estandar.
        subjects_all.append(subject_)

    acc_per_comp.append([d.get('Average accuracy')
                         for d in subjects_all if 'Average accuracy' in d])
    std_per_comp.append([d.get('std') for d in subjects_all if 'std' in d])

average_accuracy_per_comp = np.round(
    [[np.mean(lst) for lst in acc_per_comp]], 3)
# %% PLOTS ACCURACY VS N_COMPONENTS

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

# %% CREATES PLOTS OF EACH N_COMPONENT ACCURACY PER SUBJECT


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
plt.savefig('avg_acc_comp_iva.png')
plt.show()


# %%
n_components = 12
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

for i in range(1, 4):
    sets_grafos = np.load(f'subgrafos_{i}.npy', allow_pickle=True).item()
    for m in sets_grafos:
        sets_grafos[m].append(list(range(1, 118)))
# %%
SUBJECT = ["aa", "al", "av", "aw", "ay"]
n_grafos = 3
all_results = []
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
                    sub, pipelines, "dataset_comp", 10, sensor_set)
                row = {
                    'subject': sub,
                    'graph': i,
                    'method': method,
                    'n_channels': len(sensor_set),
                    'accuracy': subject_['Average accuracy']
                }
                all_results.append(row)
print('done.')
# print('inicio: 21:57')
df_results = pd.DataFrame(all_results)
# df_results.to_csv('results_BCI_COMP_IVa.csv', index=False, encoding='utf-8')
# %% CHECK IF RESULTS EXIST
try:
    df_results
except NameError:
    df_results = pd.read_csv('results_BCI_COMP_IVa.csv')


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
def plot_graph_results(graph, df_results):

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
    plt.title(f'Average accuracy by n_channels with graph {graph}')
    plt.legend()
    # plt.ylim((0.5, 1))
    # plt.savefig(f'BCI_COMP_IVa_graph_{graph}.png')
    plt.show()

# %%


plot_graph_results(1, df_results)

plot_graph_results(2, df_results)

plot_graph_results(3, df_results)

# %%


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
        plt.savefig(f'BCI_COMP_IVa_graphs.png')
    plt.show()


# %%
plot_all_graphs(df_results)

# %% PLOT EEG SIGNALS WITH MNE
sub = 'aa'
SAVE_FILE_PATH = f"./dataset_comp/{sub}.npz"
data = loadmat(f'./dataset_comp/data_set_IVa_{sub}.mat')
labels_dict = loadmat(f'./dataset_comp/true_labels_{sub}.mat')

labels = labels_dict["true_y"].squeeze()
test_idx = labels_dict["test_idx"].squeeze()

mrk = data["mrk"][0, 0]
events_samples = mrk[0][0, :]
event_description = {}
event_description = {key + 1: value[0]
                     for key, value in enumerate(mrk[2][0])}

info = data["nfo"][0, 0]
cnt = (data["cnt"].T)*0.1
FS = int(info[1][0, :])
ch_names = [ch_name[0] for ch_name in info[2][0]]
pos = np.array([info[3][:, 0], info[4][:, 0]])

# Create mne structure
info_mne = mne.create_info(ch_names, FS, "eeg")
raw = mne.io.RawArray(cnt, info_mne)

# filt = raw.filter(8, 30)
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
events[:, 2] = labels
annotations = mne.annotations_from_events(events, FS, event_description)
filt.set_annotations(annotations)
events, _ = mne.events_from_annotations(filt)

epochs = mne.Epochs(filt, events, tmin=0.5, tmax=2.5,
                    baseline=None, preload=True)

epochs_data = epochs.get_data(copy=True)


# %% https://mne.tools/stable/auto_examples/visualization/evoked_topomap.html#sphx-glr-auto-examples-visualization-evoked-topomap-py
i = 1
grafo = np.load(f'subgrafos_{i}.npy', allow_pickle=True).item()
method = 1
sets = grafo[f'm{method}']
dropped = get_dropped_channels(ch_names, sets)
cols = 4
rows = math.ceil(len(dropped)/cols)
fig, axes = plt.subplots(rows, cols)
axes = axes.flatten()
sample = 30
data_plot = epochs.average().get_data()[
    :, sample]

# image_interp='nearest', 'cubic', 'linear'
# contours = 4
for i in range(len(sets)):
    img, _ = mne.viz.plot_topomap(data_plot[sets[i]], pos=pos.T[sets[i], :],
                                  axes=axes[i], ch_type='eeg', sphere=1, size=5, res=32, show=True, image_interp='linear')
    axes[i].set_title(f"n_channels={len(sets[i])}")
    cbar = plt.colorbar(ax=axes[i], shrink=0.75,
                        orientation='vertical', mappable=img)
    cbar.set_label('$\mu$V')


# %% SUBJECT DEPENDENT MODEL


# %%

# %%

# %%

# %%
