# -*- coding: utf-8 -*-

# from nodeSelection_algorith import *
import numpy as np
import pygsp2
import eegrasp
import mne
import matplotlib.pyplot as plt
# %% LOAD DATA

DATASET_FOLDER = "dataset_comp_IVb"
data = np.load(f'./{DATASET_FOLDER}/al.npz')
coord = data['coordinates'].T[:, :2]
ch_names = data['ch_names']

# %%

# set 64 uses 'Iz' and 'AFz' electrodes but this montage doesnt use it
set_64 = ['Cz', 'CPz', 'P1', 'P3', 'PO7', 'O1', 'PO3', 'Pz', 'POz', 'Oz', 'O2', 'PO4',
          'P2', 'CP2', 'P4', 'PO8', 'P10',
          'P8', 'P6', 'TP8', 'CP6', 'CP4', 'C2', 'C4', 'C6', 'T8', 'FT8', 'FC6', 'FC4', 'F4', 'F6', 'F8',
          'AF8', 'FC2', 'F2', 'AF4', 'Fp2', 'Fpz', 'Fz', 'FCz', 'FC1', 'F1', 'AF3', 'Fp1', 'AF7',
          'F3', 'F5', 'F7', 'FT7',
          'FC5', 'FC3', 'C1', 'CP1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'P5', 'P7', 'P9']

set_32 = ['Cz', 'P3', 'O1', 'PO3', 'Pz', 'Oz', 'O2', 'PO4',
          'CP2', 'P4',
          'P8', 'CP6', 'C4', 'T8', 'FC6', 'F4', 'F8',
          'FC2', 'AF4', 'Fp2', 'Fz', 'FC1', 'AF3', 'Fp1',
          'F3', 'F7',
          'FC5', 'CP1', 'C3', 'T7', 'CP5',  'P7']

set_19 = ['Cz', 'P3', 'O1', 'Pz', 'O2', 'P4',
          'P8', 'C4', 'T8', 'F4', 'F8',
          'Fp2', 'Fz', 'Fp1',
          'F3', 'F7', 'C3', 'T7', 'P7']

# sort elements in each set
set_64 = sorted(set_64, key=lambda x: list(ch_names).index(x))
set_32 = sorted(set_32, key=lambda x: list(ch_names).index(x))
set_19 = sorted(set_19, key=lambda x: list(ch_names).index(x))

# get index of elements in ch_names
index_set_64 = [list(ch_names).index(x) for x in set_64]
index_set_32 = [list(ch_names).index(x) for x in set_32]
index_set_19 = [list(ch_names).index(x) for x in set_19]

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 fila, 3 columnas

# Ejemplo de gráficas
axes[0].scatter(coord[:, 0], coord[:, 1], label='all')
axes[0].scatter(coord[index_set_64, 0], coord[index_set_64, 1], label='set_64')
axes[0].legend()
axes[0].set_title("Subplot 64")

axes[1].scatter(coord[:, 0], coord[:, 1], label='all')
axes[1].scatter(coord[index_set_32, 0], coord[index_set_32, 1], label='set_32')
axes[1].legend()
axes[1].set_title("Subplot 32")

axes[2].scatter(coord[:, 0], coord[:, 1], label='all')
axes[2].scatter(coord[index_set_19, 0], coord[index_set_19, 1], label='set_19')
axes[2].legend()
axes[2].set_title("Subplot con 19")

plt.tight_layout()  # Ajusta espacios para que no se encimen
plt.show()


# %%
eeg_pos = coord
comp_eeg = eegrasp.EEGrasp(coordinates=coord, labels=ch_names)
distances = comp_eeg.compute_distance(eeg_pos, method='Euclidean')
# %% BASE GRAPH CREATION
grafo_1 = comp_eeg.compute_graph(
    distances=distances, epsilon=0.1, sigma=0.1, coordinates=eeg_pos)
grafo_1.plot()
grafo_2 = comp_eeg.compute_graph(
    distances=distances, epsilon=0.15, sigma=0.1, coordinates=eeg_pos)
grafo_2.plot()
grafo_3 = comp_eeg.compute_graph(
    distances=distances, epsilon=0.2, sigma=0.1, coordinates=eeg_pos)
grafo_3.plot()
# %% POR SI SE ENCUENTRA OTRA OPCION, PERO CON 3 FUE SUFICIENTE

# grafo_4 = comp_eeg.compute_graph(
#     distances=distances, epsilon=0.2, sigma=0.007, coordinates=eeg_pos)
# grafo_4.plot()


# %% GRAFO 1 METODO 1

plotEigenValues(grafo_1)

Wc = np.linspace(0.1, 1.7, 30)
k = 12
Fz, Ss = [], []
for wc in Wc:
    f, S = estimate_Sopt(grafo_1, wc, k)
    Fz.append(f)
    Ss.append(S)

plt.figure()
for i in range(len(Fz)):
    plt.plot(Fz[i], '*-', label='wc:{}'.format(Wc[i]))
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title(f"Estimated frequency vs number of selected nodes, k ={k}")
plt.show()
# %% STORES GRAPHS

m1_grafo_1 = []
n_channels = []
for s in Ss:
    if len(s) not in n_channels:
        n_channels.append(len(s))
        m1_grafo_1.append(reducedGraph(grafo_1, s))
    else:
        continue

# %% plot just in case
grafo_1.plot()
m1_grafo_1[0].plot()
m1_grafo_1[-14].plot()

# %% GETS SELECTED CHANNELS

selected_g1_m1 = []
for g in m1_grafo_1:
    selected_ = selected_channels(grafo_1, g)
    selected_g1_m1.append(selected_)

# %% GRAFO 1 METODO 2 (sampling algorithm)

m2_grafo_1 = []
selected_g1_m2 = []
for i in range(10, 118, 5):
    auxGraph = SamplingAlgorithm(grafo_1, i)
    m2_grafo_1.append(auxGraph)
    selected_ = selected_channels(grafo_1, auxGraph)
    selected_g1_m2.append(selected_)

# %% GRAFO 1 METODO 3
# fast_gsss(L_n, F, bw, nu, U, E)

m3_grafo_1 = []
selected_g1_m3 = []

grafo_1.compute_laplacian(lap_type='normalized')
grafo_1.compute_fourier_basis()
L = grafo_1.L.toarray()
bw = 1.5
nu = 1.5

for i in range(10, 118, 5):
    nods, tt = fast_gsss(L, i, bw, nu, grafo_1.U, grafo_1.e)
    auxGraph = grafo_1.subgraph(nods)
    m3_grafo_1.append(auxGraph)
    selected_ = selected_channels(grafo_1, auxGraph)
    selected_g1_m3.append(selected_)


# %%
subgrafos_1 = {}
subgrafos_1['m1'] = selected_g1_m1
subgrafos_1['m2'] = selected_g1_m2
subgrafos_1['m3'] = selected_g1_m3

np.save(f'./{DATASET_FOLDER}/subgrafos_1.npy', subgrafos_1, allow_pickle=True)


# %% GRAFO 2 METODO 1

plotEigenValues(grafo_2)

Wc = np.linspace(0.1, 1.7, 30)
k = 12
Fz, Ss = [], []
for wc in Wc:
    f, S = estimate_Sopt(grafo_2, wc, k)
    Fz.append(f)
    Ss.append(S)

plt.figure()
for i in range(len(Fz)):
    plt.plot(Fz[i], '*-', label='wc:{}'.format(Wc[i]))
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title(f"Estimated frequency vs number of selected nodes, k ={k}")
plt.show()
# %% STORES GRAPHS

m1_grafo_2 = []
n_channels = []
for s in Ss:
    if len(s) not in n_channels:
        n_channels.append(len(s))
        m1_grafo_2.append(reducedGraph(grafo_2, s))
    else:
        continue

# %% plot just in case
grafo_2.plot()
m1_grafo_2[0].plot()
m1_grafo_2[-14].plot()

# %% GETS SELECTED CHANNELS

selected_g2_m1 = []
for g in m1_grafo_2:
    selected_ = selected_channels(grafo_2, g)
    selected_g2_m1.append(selected_)

# %% GRAFO 2 METODO 2 (sampling algorithm)

m2_grafo_2 = []
selected_g2_m2 = []
for i in range(10, 118, 5):
    auxGraph = SamplingAlgorithm(grafo_2, i)
    m2_grafo_2.append(auxGraph)
    selected_ = selected_channels(grafo_2, auxGraph)
    selected_g2_m2.append(selected_)

# %% GRAFO 2 METODO 3
# fast_gsss(L_n, F, bw, nu, U, E)

m3_grafo_2 = []
selected_g2_m3 = []

grafo_2.compute_laplacian(lap_type='normalized')
grafo_2.compute_fourier_basis()
L = grafo_2.L.toarray()

bw = 1.5
nu = 1.5
for i in range(10, 118, 5):
    nods, tt = fast_gsss(L, i, bw, nu, grafo_2.U, grafo_2.e)
    auxGraph = grafo_2.subgraph(nods)
    m3_grafo_2.append(auxGraph)
    selected_ = selected_channels(grafo_2, auxGraph)
    selected_g2_m3.append(selected_)


# %%
subgrafos_2 = {}
subgrafos_2['m1'] = selected_g2_m1
subgrafos_2['m2'] = selected_g2_m2
subgrafos_2['m3'] = selected_g2_m3

np.save(f'./{DATASET_FOLDER}/subgrafos_2.npy', subgrafos_2, allow_pickle=True)


# %% GRAFO 3 METODO 1

plotEigenValues(grafo_3)

Wc = np.linspace(0.1, 1.7, 30)
k = 12
Fz, Ss = [], []
for wc in Wc:
    f, S = estimate_Sopt(grafo_3, wc, k)
    Fz.append(f)
    Ss.append(S)

plt.figure()
for i in range(len(Fz)):
    plt.plot(Fz[i], '*-', label='wc:{}'.format(Wc[i]))
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title(f"Estimated frequency vs number of selected nodes, k ={k}")
plt.show()
# %% STORES GRAPHS

m1_grafo_3 = []
n_channels = []
for s in Ss:
    if len(s) not in n_channels:
        n_channels.append(len(s))
        m1_grafo_3.append(reducedGraph(grafo_3, s))
    else:
        continue

# %% plot just in case
grafo_3.plot()
m1_grafo_3[0].plot()
m1_grafo_3[-14].plot()

# %% GETS SELECTED CHANNELS

selected_g3_m1 = []
for g in m1_grafo_3:
    selected_ = selected_channels(grafo_3, g)
    selected_g3_m1.append(selected_)

# %% GRAFO 3 METODO 2 (sampling algorithm)

m2_grafo_3 = []
selected_g3_m2 = []
for i in range(10, 118, 5):
    auxGraph = SamplingAlgorithm(grafo_3, i)
    m2_grafo_3.append(auxGraph)
    selected_ = selected_channels(grafo_3, auxGraph)
    selected_g3_m2.append(selected_)

# %% GRAFO 3 METODO 3
# fast_gsss(L_n, F, bw, nu, U, E)

m3_grafo_3 = []
selected_g3_m3 = []

grafo_3.compute_laplacian(lap_type='normalized')
grafo_3.compute_fourier_basis()
L = grafo_3.L.toarray()

for i in range(10, 118, 5):
    nods, tt = fast_gsss(L, i, bw, nu, grafo_3.U, grafo_3.e)
    auxGraph = grafo_3.subgraph(nods)
    m3_grafo_3.append(auxGraph)
    selected_ = selected_channels(grafo_3, auxGraph)
    selected_g3_m3.append(selected_)


# %%
subgrafos_3 = {}
subgrafos_3['m1'] = selected_g3_m1
subgrafos_3['m2'] = selected_g3_m2
subgrafos_3['m3'] = selected_g3_m3

np.save(f'./{DATASET_FOLDER}/subgrafos_3.npy', subgrafos_3, allow_pickle=True)
