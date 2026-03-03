# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 23:56:13 2025

@author: sebas
"""
# from nodeSelection_algorith import *
import numpy as np
import pygsp2
import eegrasp
import mne
import matplotlib.pyplot as plt
# %%
DATASET_FOLDER = "dataset_bbci"
data = np.load(f'./{DATASET_FOLDER}/1_t0.npz')
coord = data['coordinates'].T
coord = np.delete(coord, [-1, -2], axis=0)
ch_names = data['ch_names']
ch_names = ch_names[:-2]

# %%
eeg_pos = coord
comp_eeg = eegrasp.EEGrasp(coordinates=coord, labels=ch_names)
distances = comp_eeg.compute_distance(eeg_pos, method='Euclidean')
# %% BASE GRAPH CREATION
grafo_1 = comp_eeg.compute_graph(
    distances=distances, epsilon=0.4, sigma=0.4, coordinates=eeg_pos)
grafo_1.plot()
grafo_2 = comp_eeg.compute_graph(
    distances=distances, epsilon=0.45, sigma=0.1, coordinates=eeg_pos)
grafo_2.plot()
grafo_3 = comp_eeg.compute_graph(
    distances=distances, epsilon=0.5, sigma=0.1, coordinates=eeg_pos)
grafo_3.plot()
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
for i in range(3, 31, 3):
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

for i in range(3, 31, 3):
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
for i in range(3, 31, 3):
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
for i in range(3, 31, 3):
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
# grafo_3.plot()
# m1_grafo_3[0].plot()
# m1_grafo_3[-14].plot()

# %% GETS SELECTED CHANNELS

selected_g3_m1 = []
for g in m1_grafo_3:
    selected_ = selected_channels(grafo_3, g)
    selected_g3_m1.append(selected_)

# %% GRAFO 3 METODO 2 (sampling algorithm)

m2_grafo_3 = []
selected_g3_m2 = []
for i in range(3, 31, 3):
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

for i in range(3, 31, 3):
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
# %%


# %%
