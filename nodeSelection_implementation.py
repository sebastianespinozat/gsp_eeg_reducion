# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:51:23 2024

@author: sebas
"""

from nodeSelection_algorithm import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pygsp2
import scipy
import eegrasp
import mne
from mne.datasets import eegbci
# %matplotlib qt


# %% GRAPH SENSOR NETWORK
graph = pygsp2.graphs.Sensor(100)
graph.plot()
plotEigenValues(graph)
# print(Kc(graph, WC))

# %% HOW |S| AND W ESTIMATION CHANGES OVER WC, FOR K = 2


# for plot purposes, use odd number of estimations
Wc_SN = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.3]  # , 1.4, 1.5]
K = 2
Wc_SN.reverse()
# Poner valor de Kc para cada Wc
freqz_sn = []
Ssn = []
for freq in Wc_SN:
    F, S = estimate_Sopt(graph, freq, k=K)
    freqz_sn.append(F)
    Ssn.append(S)

plt.figure()
for i in range(len(freqz_sn)):
    plt.plot(freqz_sn[i], '*-', label='wc:{}'.format(Wc_SN[i]))
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title("Estimated frequency vs number of selected nodes, k = 2")
plt.show()


# For the same k, the increase of wc results in the increase of |S|
# why the same line tho? CAUSE IT'S THE SAME VALUE OF K.

# %%
fig, axs = plt.subplots(2, (len(Wc_SN)+1)//2, figsize=(12, 5))
graph.plot(ax=axs[0, 0], title='Original')
for i in range(1, len(Wc_SN)+1):
    reduced = reducedGraph(graph, Ssn[i-1])
    if i < ((len(Wc_SN)+1)//2):
        reduced.plot(ax=axs[0, i])
    elif i >= ((len(Wc_SN)+1)//2):
        reduced.plot(ax=axs[1, i - ((len(Wc_SN)+1)//2)])


# %% HOW |S| AND W ESTIMATION CHANGES OVER WC, FOR K = 12


# for plot purposes, use odd number of estimations
Wc_SN = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.3]  # , 1.4, 1.5]
K = 12
Wc_SN.reverse()
# Poner valor de Kc para cada Wc
freqz_sn = []
Ssn = []
for freq in Wc_SN:
    F, S = estimate_Sopt(graph, freq, k=K)
    freqz_sn.append(F)
    Ssn.append(S)

plt.figure()
for i in range(len(freqz_sn)):
    plt.plot(freqz_sn[i], '*-', label='wc:{}'.format(Wc_SN[i]))
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title("Estimated frequency vs number of selected nodes, k = 12")
plt.show()

# %%
fig, axs = plt.subplots(2, (len(Wc_SN)+1)//2, figsize=(12, 5))
graph.plot(ax=axs[0, 0], title='Original')
for i in range(1, len(Wc_SN)+1):
    reduced = reducedGraph(graph, Ssn[i-1])
    if i < ((len(Wc_SN)+1)//2):
        reduced.plot(ax=axs[0, i])
    elif i >= ((len(Wc_SN)+1)//2):
        reduced.plot(ax=axs[1, i - ((len(Wc_SN)+1)//2)])


# %% HOW |S| changes over k, for Wc = 0.8
WC = 0.8

intento_1, S1 = estimate_Sopt(graph, wc=WC, k=2)
intento_2, S2 = estimate_Sopt(graph, wc=WC, k=6)
intento_3, S3 = estimate_Sopt(graph, wc=WC, k=12)
intento_4, S4 = estimate_Sopt(graph, wc=WC, k=18)


plt.figure()
plt.plot(intento_1, '*-', label="k=2")
plt.plot(intento_2, '*-', label="k=6")
plt.plot(intento_3, '*-', label="k=12")
plt.plot(intento_4, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title("|S| vs \omega")
plt.show()

# For the same wc, increasing k should result in |S| more similar to Kc


# %% HOW |S| changes over k, for Wc = 1.0
WC = 1.0

intento_1, S1 = estimate_Sopt(graph, wc=WC, k=2)
intento_2, S2 = estimate_Sopt(graph, wc=WC, k=6)
intento_3, S3 = estimate_Sopt(graph, wc=WC, k=12)
intento_4, S4 = estimate_Sopt(graph, wc=WC, k=18)


plt.figure()
plt.plot(intento_1, '*-', label="k=2")
plt.plot(intento_2, '*-', label="k=6")
plt.plot(intento_3, '*-', label="k=12")
plt.plot(intento_4, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title("|S| vs \omega")
plt.show()


# %% HOW |S| changes over k, for Wc = 1.2
WC = 0.6

intento_1, S1 = estimate_Sopt(graph, wc=WC, k=2)
intento_2, S2 = estimate_Sopt(graph, wc=WC, k=6)
intento_3, S3 = estimate_Sopt(graph, wc=WC, k=12)
intento_4, S4 = estimate_Sopt(graph, wc=WC, k=18)


plt.figure()
plt.plot(intento_1, '*-', label="k=2")
plt.plot(intento_2, '*-', label="k=6")
plt.plot(intento_3, '*-', label="k=12")
plt.plot(intento_4, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title("|S| vs $\omega$")
plt.show()


# %% Regular cyclic graph with N = 100, KNN = 8, W = 1/distance

G2 = pygsp2.graphs.Ring(100, 4)
distancias = scipy.spatial.distance_matrix(G2.coords, G2.coords)


def calcular_matriz_peso(D):
    W = np.zeros_like(D)
    nonzero_indices = D != 0
    W[nonzero_indices] = 1.0 / (1.0*D[nonzero_indices])
    return W


W = calcular_matriz_peso(distancias)
W = np.multiply(W, G2.W.toarray())
G_ring = pygsp2.graphs.Graph(W)
G_ring.set_coordinates(G2.coords)
# G_ring.W = W

# WC = 1.0
plotEigenValues(G_ring)
print(Kc(G_ring, WC))

# %%

# G3 = pygsp2.graphs.ErdosRenyi(100, 0.2)
WC = 0.8

g2_try_1, SG2_1 = estimate_Sopt(G_ring, wc=WC, k=2)
g2_try_2, SG2_2 = estimate_Sopt(G_ring, wc=WC, k=6)
g2_try_3, SG2_3 = estimate_Sopt(G_ring, wc=WC, k=12)
g2_try_4, SG2_4 = estimate_Sopt(G_ring, wc=WC, k=18)


# plotEigenValues(G_ring)


plt.figure()
plt.plot(g2_try_1, '*-', label="k=2")
plt.plot(g2_try_2, '*-', label="k=6")
plt.plot(g2_try_3, '*-', label="k=12")
plt.plot(g2_try_4, '*-', label="k=18")
plt.legend()
plt.show()


# %%


red_ring_2 = reducedGraph(G_ring, SG2_2)
red_ring_3 = reducedGraph(G_ring, SG2_3)
red_ring_4 = reducedGraph(G_ring, SG2_4)

fig, axs = plt.subplots(2, 2)
G_ring.plot(ax=axs[0, 0])
red_ring_2.plot(ax=axs[0, 1])
red_ring_3.plot(ax=axs[1, 0])
red_ring_4.plot(ax=axs[1, 1])

# %%
graph_rr = pygsp2.graphs.DavidSensorNet(200)
# graph_rr = pygsp2.graphs.BarabasiAlbert()
# graph_rr.set_coordinates()
graph_rr.plot()
plotEigenValues(graph_rr)

# %%
WC = 0.5
frr1, Srr1 = estimate_Sopt(graph_rr, wc=WC, k=2)
frr2, Srr2 = estimate_Sopt(graph_rr, wc=WC, k=6)
frr3, Srr3 = estimate_Sopt(graph_rr, wc=WC, k=12)
frr4, Srr4 = estimate_Sopt(graph_rr, wc=WC, k=18)


plt.figure()
plt.plot(frr1, '*-', label="k=2")
plt.plot(frr2, '*-', label="k=6")
plt.plot(frr3, '*-', label="k=12")
plt.plot(frr4, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title("|S| vs \omega")
plt.show()


# %%

red_rr1 = reducedGraph(graph_rr, Srr1)
red_rr2 = reducedGraph(graph_rr, Srr2)
red_rr3 = reducedGraph(graph_rr, Srr3)
red_rr4 = reducedGraph(graph_rr, Srr4)
fig, axs = plt.subplots(2, 2)

red_rr1.plot(ax=axs[0, 0])
red_rr2.plot(ax=axs[0, 1])
red_rr3.plot(ax=axs[1, 0])
red_rr4.plot(ax=axs[1, 1])


# %% EEG

subject = 1
runs = np.arange(1, 15)
raw_fnames = eegbci.load_data(
    subject, runs, path="C:/Users/sebas/Desktop/Universidad/Tesis/codigo/")
raw = mne.io.concatenate_raws(
    [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)

# raw.annotations.rename(dict(T1="hands", T2="feet"))
raw.set_eeg_reference(projection=True)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)

eeg_pos = np.array(
    [pos for _, pos in raw.get_montage().get_positions()['ch_pos'].items()])
channels = [ch for ch in raw.get_montage().get_positions()['ch_pos'].keys()]

# %%

# raw, ref_data = mne.set_eeg_reference(raw)
# events, events_id = mne.events_from_annotations(raw)

L_FREQ = 7  # Hz
H_FREQ = 30  # Hz
raw.filter(L_FREQ, H_FREQ, fir_design='firwin', skip_by_annotation='edge')
raw, ref_data = mne.set_eeg_reference(raw)

events, events_id = mne.events_from_annotations(raw)

TMIN, TMAX = -1.0, 3.0
picks = mne.pick_types(raw.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude="bads")
epochs = mne.Epochs(raw, events, events_id,
                    picks=picks, tmin=TMIN,
                    tmax=TMAX, baseline=(-1, 0),
                    preload=True,
                    detrend=1)

# rest = epochs['T0']
# erp_rest = rest.average()

# hands = epochs['hands']
# erp_hands = hands.average()

# feet = epochs['feet']
# erp_feet = feet.average()

epochs = epochs.crop(0, 3.0)
epochs_data = epochs.get_data(copy=False)
# %%
# montage = mne.channels.make_standard_montage('biosemi64')
# ch_names = montage.ch_names
# eeg_pos = montage.get_positions()['ch_pos']
# eeg_pos = np.array([pos for _, pos in eeg_pos.items()])

# %%

raw.plot_sensors("3d")
plt.show()

raw.plot_sensors(show_names="true")
plt.show()

# %%
eegsp = eegrasp.EEGrasp(epochs_data, eeg_pos, channels)
distances = eegsp.compute_distance(eeg_pos, method='Euclidean')
grafo_1 = eegsp.compute_graph(
    distances=distances, epsilon=0.3, sigma=0.1, coordinates=eeg_pos)
grafo_2 = eegsp.compute_graph(
    distances=distances, epsilon=0.2, sigma=0.1, coordinates=eeg_pos)


# %%
W_learning, Z = eegsp.learn_graph(a=0.34, b=0.4)
grafo_3 = eegsp.compute_graph(W_learning)


# %%
fig, ax = plt.subplots(1, 3)
im0 = ax[0].imshow(grafo_1.W.toarray(), cmap='jet')
im1 = ax[1].imshow(grafo_2.W.toarray(), cmap='jet')
im2 = ax[2].imshow(W_learning, cmap='jet')
ax[0].title.set_text("Graph 1")  # "Graph GK. $\epsilon = 0.3, \sigma = 0.1$")
ax[1].title.set_text("Graph 2")  # "Graph GK. $\epsilon = 0.2, \sigma = 0.1$")
ax[2].title.set_text("Graph 3")  # "Graph learning")
fig.colorbar(im0, ax=ax[0], orientation='horizontal')
fig.colorbar(im1, ax=ax[1], orientation='horizontal')
fig.colorbar(im2, ax=ax[2], orientation='horizontal')
fig.set_size_inches(18, 10)
# plt.savefig("original_weights.pdf", dpi=400)

# %%

eeg_pos = eeg_pos / np.amax(eeg_pos)
eegsp.graph.set_coordinates(eeg_pos)
eegsp.coordinates = eeg_pos
eegsp.graph.plot()
plt.show()

grafo_1.coords = eeg_pos
grafo_2.coords = eeg_pos
# %%

grafo_1.plot()
grafo_2.plot()
grafo_3.plot()

eegsp.plot(kind='3d')

# %% EEG reduction

# plotEigenValues(grafo)
WC = 0.8
print(Kc(grafo_1, WC))
print(Kc(grafo_2, WC))
# %%

WC = 0.8

# grafo 1
freq1, Seeg1 = estimate_Sopt(grafo_1, wc=WC, k=2)
freq2, Seeg2 = estimate_Sopt(grafo_1, wc=WC, k=6)
freq3, Seeg3 = estimate_Sopt(grafo_1, wc=WC, k=12)
freq4, Seeg4 = estimate_Sopt(grafo_1, wc=WC, k=18)
print("###############")
plt.figure()
plt.plot(freq1, '*-', label="k=2")
plt.plot(freq2, '*-', label="k=6")
plt.plot(freq3, '*-', label="k=12")
plt.plot(freq4, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title(
    "Estimated frequency vs number of selected nodes. $\omega_c= {}$".format(WC))
plt.show()


freq11, Seeg11 = estimate_Sopt(grafo_2, wc=WC, k=2)
freq22, Seeg22 = estimate_Sopt(grafo_2, wc=WC, k=6)
freq33, Seeg33 = estimate_Sopt(grafo_2, wc=WC, k=12)
freq44, Seeg44 = estimate_Sopt(grafo_2, wc=WC, k=18)
print("###############")
# grafo 2
plt.figure()
plt.plot(freq11, '*-', label="k=2")
plt.plot(freq22, '*-', label="k=6")
plt.plot(freq33, '*-', label="k=12")
plt.plot(freq44, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title(
    "Estimated frequency vs number of selected nodes. $\omega_c= {}$".format(WC))
plt.show()

# grafo 3
freq111, Seeg111 = estimate_Sopt(grafo_3, wc=WC, k=2)
freq222, Seeg222 = estimate_Sopt(grafo_3, wc=WC, k=6)
freq333, Seeg333 = estimate_Sopt(grafo_3, wc=WC, k=12)
freq444, Seeg444 = estimate_Sopt(grafo_3, wc=WC, k=18)
print("###############")
plt.figure()
plt.plot(freq111, '*-', label="k=2")
plt.plot(freq222, '*-', label="k=6")
plt.plot(freq333, '*-', label="k=12")
plt.plot(freq444, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title(
    "Estimated frequency vs number of selected nodes. $\omega_c= {}$".format(WC))
plt.show()

# %%

eeg_grafo_1 = reducedGraph(grafo_1, Seeg1)
eeg_grafo_2 = reducedGraph(grafo_2, Seeg11)
eeg_grafo_3 = reducedGraph(grafo_3, Seeg111)

eeg_grafo_1.plot()
eeg_grafo_2.plot()
eeg_grafo_3.plot()

# %%
eeg_grafo_1.compute_laplacian('normalized')
eeg_grafo_2.compute_laplacian('normalized')
eeg_grafo_3.compute_laplacian('normalized')
eeg_grafo_1.compute_fourier_basis()
eeg_grafo_2.compute_fourier_basis()
eeg_grafo_3.compute_fourier_basis()

# %%
eeg_grafo_1.plot(eeg_grafo_1.U[:, 25])
eeg_grafo_2.plot(eeg_grafo_2.U[:, 25])
eeg_grafo_3.plot(eeg_grafo_3.U[:, 25])

# %%
WC = 0.6

# grafo 1
freq1, Seeg1 = estimate_Sopt(grafo_1, wc=WC, k=2)
freq2, Seeg2 = estimate_Sopt(grafo_1, wc=WC, k=6)
freq3, Seeg3 = estimate_Sopt(grafo_1, wc=WC, k=12)
freq4, Seeg4 = estimate_Sopt(grafo_1, wc=WC, k=18)

plt.figure()
plt.plot(freq1, '*-', label="k=2")
plt.plot(freq2, '*-', label="k=6")
plt.plot(freq3, '*-', label="k=12")
plt.plot(freq4, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title(
    "Estimated frequency vs number of selected nodes. $\omega_c= {}$".format(WC))
plt.show()
print("###############")

freq11, Seeg11 = estimate_Sopt(grafo_2, wc=WC, k=2)
freq22, Seeg22 = estimate_Sopt(grafo_2, wc=WC, k=6)
freq33, Seeg33 = estimate_Sopt(grafo_2, wc=WC, k=12)
freq44, Seeg44 = estimate_Sopt(grafo_2, wc=WC, k=18)

# grafo 2
plt.figure()
plt.plot(freq11, '*-', label="k=2")
plt.plot(freq22, '*-', label="k=6")
plt.plot(freq33, '*-', label="k=12")
plt.plot(freq44, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title(
    "Estimated frequency vs number of selected nodes. $\omega_c= {}$".format(WC))
plt.show()
print("###############")
# grafo 3
freq111, Seeg111 = estimate_Sopt(grafo_3, wc=WC, k=2)
freq222, Seeg222 = estimate_Sopt(grafo_3, wc=WC, k=6)
freq333, Seeg333 = estimate_Sopt(grafo_3, wc=WC, k=12)
freq444, Seeg444 = estimate_Sopt(grafo_3, wc=WC, k=18)

plt.figure()
plt.plot(freq111, '*-', label="k=2")
plt.plot(freq222, '*-', label="k=6")
plt.plot(freq333, '*-', label="k=12")
plt.plot(freq444, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title(
    "Estimated frequency vs number of selected nodes. $\omega_c= {}$".format(WC))
plt.show()

# %%

eeg_grafo_1 = reducedGraph(grafo_1, Seeg1)
eeg_grafo_2 = reducedGraph(grafo_2, Seeg11)
eeg_grafo_3 = reducedGraph(grafo_3, Seeg333)

eeg_grafo_1.plot()
eeg_grafo_2.plot()
eeg_grafo_3.plot()

# %%

WC = 0.5

# grafo 1
freq1, Seeg1 = estimate_Sopt(grafo_1, wc=WC, k=2)
freq2, Seeg2 = estimate_Sopt(grafo_1, wc=WC, k=6)
freq3, Seeg3 = estimate_Sopt(grafo_1, wc=WC, k=12)
freq4, Seeg4 = estimate_Sopt(grafo_1, wc=WC, k=18)

plt.figure()
plt.plot(freq1, '*-', label="k=2")
plt.plot(freq2, '*-', label="k=6")
plt.plot(freq3, '*-', label="k=12")
plt.plot(freq4, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title(
    "Estimated frequency vs number of selected nodes. $\omega_c= {}$".format(WC))
plt.show()
print("###############")

freq11, Seeg11 = estimate_Sopt(grafo_2, wc=WC, k=2)
freq22, Seeg22 = estimate_Sopt(grafo_2, wc=WC, k=6)
freq33, Seeg33 = estimate_Sopt(grafo_2, wc=WC, k=12)
freq44, Seeg44 = estimate_Sopt(grafo_2, wc=WC, k=18)

# grafo 2
plt.figure()
plt.plot(freq11, '*-', label="k=2")
plt.plot(freq22, '*-', label="k=6")
plt.plot(freq33, '*-', label="k=12")
plt.plot(freq44, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title(
    "Estimated frequency vs number of selected nodes. $\omega_c= {}$".format(WC))
plt.show()

print("###############")
# grafo 3
freq111, Seeg111 = estimate_Sopt(grafo_3, wc=WC, k=2)
freq222, Seeg222 = estimate_Sopt(grafo_3, wc=WC, k=6)
freq333, Seeg333 = estimate_Sopt(grafo_3, wc=WC, k=12)
freq444, Seeg444 = estimate_Sopt(grafo_3, wc=WC, k=18)

plt.figure()
plt.plot(freq111, '*-', label="k=2")
plt.plot(freq222, '*-', label="k=6")
plt.plot(freq333, '*-', label="k=12")
plt.plot(freq444, '*-', label="k=18")
plt.legend()
plt.xlabel("$|S|$")
plt.ylabel("$\omega$")
plt.title(
    "Estimated frequency vs number of selected nodes. $\omega_c= {}$".format(WC))
plt.show()

# %%
#
eeg_grafo_1 = reducedGraph(grafo_1, Seeg1)
eeg_grafo_2 = reducedGraph(grafo_2, Seeg11)
eeg_grafo_3 = reducedGraph(grafo_3, Seeg111)

eeg_grafo_1.plot()
eeg_grafo_2.plot()
eeg_grafo_3.plot()


# %%

dropped_1 = dropped_channels(grafo_1, eeg_grafo_1, channels)
dropped_2 = dropped_channels(grafo_2, eeg_grafo_2, channels)
dropped_3 = dropped_channels(grafo_3, eeg_grafo_3, channels)
# es posible aplicar .drop_channels() a dato tipo Epochs o Evoked

# %%

epochs_reduced = epochs.copy()

reduced_1 = epochs_reduced.average().drop_channels(dropped_1)
reduced_2 = epochs_reduced.average().drop_channels(dropped_2)
reduced_3 = epochs_reduced.average().drop_channels(dropped_3)

# %%
fig, axs = plt.subplots(4)
mne.viz.plot_topomap(epochs_reduced.average().get_data()[
                     :, 30], eeg_pos[:, :2], sphere=1, axes=axs[0])
mne.viz.plot_topomap(reduced_1.get_data()[
                     :, 30], eeg_grafo_1.coords[:, :2], sphere=1, axes=axs[1])
mne.viz.plot_topomap(reduced_2.get_data()[
                     :, 30], eeg_grafo_2.coords[:, :2], sphere=1, axes=axs[2])
mne.viz.plot_topomap(reduced_3.get_data()[
                     :, 30], eeg_grafo_3.coords[:, :2], sphere=1, axes=axs[3])


# %%

times = np.arange(0, 3.0, 0.5)

epochs_reduced.average().plot_topomap(times, ncols=8, nrows="auto")
plt.suptitle("Original")
plt.savefig("original_time.pdf", dpi=500)
plt.show()

reduced_1.plot_topomap(times, ncols=8, nrows="auto")
plt.suptitle("Graph 1 reduced, $\omega_c = 0.6$")
plt.savefig("graph1_wc06.pdf", dpi=500)
plt.show()

reduced_2.plot_topomap(times, ncols=8, nrows="auto")
plt.suptitle("Graph 2 reduced, $\omega_c = 0.6$")
plt.savefig("graph2_wc06.pdf", dpi=500)
plt.show()

reduced_3.plot_topomap(times, ncols=8, nrows="auto")
plt.suptitle("Graph 3 reduced, $\omega_c = 0.6$")
plt.savefig("graph3_wc06.pdf", dpi=500)
plt.show()
# %% Plot eigenvectors per eigenvalues

# %% Sample reconstuction
index = 100
f_real = epochs.average().get_data()[:, index]

fs = f_real.copy()
for i in range(len(f_real)):
    if i not in dropped_channel_index(grafo_1, eeg_grafo_1):
        fs[i] = 0

f_selected_1 = reduced_1.get_data()[:, index]
f_estimated_1 = estimate_signal(grafo_1, eeg_grafo_1, 0.9, f_selected_1)
f_selected_2 = reduced_2.get_data()[:, index]
f_estimated_2 = estimate_signal(grafo_2, eeg_grafo_2, 0.9, f_selected_2)
f_selected_3 = reduced_3.get_data()[:, index]
f_estimated_3 = estimate_signal(grafo_3, eeg_grafo_3, 0.9, f_selected_3)
# print(MSE(f_real, f_estimated))
# print(SNR(f_real, f_estimated))

plt.figure()
plt.plot(f_real*1e6, '-*', label='real')
# plt.plot(fs, '-*', label='used')
plt.plot(f_estimated_1*1e6, '-*', label='Graph 1 reconstruction')
plt.plot(f_estimated_2*1e6, '-*', label='Graph 2 reconstruction')
plt.plot(f_estimated_3*1e6, '-*', label='Graph 3 reconstruction')
plt.title("Real and estimated signal")
plt.ylabel("Voltage [$\mu V$]")
plt.xlabel("Electrode")
plt.legend()
plt.savefig("reconstruction.pdf")
plt.show()

# fig, axs = plt.subplots(1,3)
# mne.viz.plot_topomap(f_real, eeg_pos[:,:2], sphere=1,axes=axs[0])
# mne.viz.plot_topomap(f_selected, eeg_grafo_1.coords[:,:2], sphere=1,axes=axs[1])
# mne.viz.plot_topomap(f_estimated_1, eeg_pos[:,:2], sphere=1,axes=axs[2])

# %%
n = 100
snr_grafo_1 = []
mse_grafo_1 = []
# grafo 1
for i in range(epochs.average().get_data().shape[1]):
    f_real = epochs.average().get_data()[:, i]
    f_selected = reduced_1.get_data()[:, i]
    f_estimated = estimate_signal(grafo_1, eeg_grafo_1, 0.8, f_selected)
    snr_grafo_1.append(SNR(f_real, f_estimated))
    mse_grafo_1.append(MSE(f_real, f_estimated))
    print(i)

# %%
snr_grafo_2 = []
mse_grafo_2 = []
# grafo 2
for i in range(epochs.average().get_data().shape[1]):
    f_real = epochs.average().get_data()[:, i]
    f_selected = reduced_2.get_data()[:, i]
    f_estimated = estimate_signal(grafo_2, eeg_grafo_2, 0.9, f_selected)
    snr_grafo_2.append(SNR(f_real, f_estimated))
    mse_grafo_2.append(MSE(f_real, f_estimated))
    print(i)

# %%
snr_grafo_3 = []
mse_grafo_3 = []
# grafo 3
for i in range(epochs.average().get_data().shape[1]):
    f_real = epochs.average().get_data()[:, i]
    f_selected = reduced_3.get_data()[:, i]
    f_estimated = estimate_signal(grafo_3, eeg_grafo_3, 0.9, f_selected)
    snr_grafo_3.append(SNR(f_real, f_estimated))
    mse_grafo_3.append(MSE(f_real, f_estimated))
    print(i)

# %%

print("mean SNR graph 1: {}".format(np.mean(snr_grafo_1)))
print("mean SNR graph 2: {}".format(np.mean(snr_grafo_2)))
print("mean SNR graph 3: {}".format(np.mean(snr_grafo_3)))

print("MSE graph 1: {}".format(np.mean(mse_grafo_1)))
print("MSE graph 2: {}".format(np.mean(mse_grafo_2)))
print("MSE graph 3: {}".format(np.mean(mse_grafo_3)))


# %%


fig, ax = plt.subplots(1, 3)
im0 = ax[0].imshow(eeg_grafo_1.W.toarray(), cmap='jet')
im1 = ax[1].imshow(eeg_grafo_2.W.toarray(), cmap='jet')
im2 = ax[2].imshow(eeg_grafo_3.W.toarray(), cmap='jet')
ax[0].title.set_text("Graph GK. $\epsilon = 0.3, \sigma = 0.1$")
ax[1].title.set_text("Graph GK. $\epsilon = 0.2, \sigma = 0.1$")
ax[2].title.set_text("Graph learning")
fig.colorbar(im0, ax=ax[0], orientation='horizontal')
fig.colorbar(im1, ax=ax[1], orientation='horizontal')
fig.colorbar(im2, ax=ax[2], orientation='hoWrizontal')
fig.set_size_inches(18, 10)
# plt.savefig("original_weights.pdf", dpi=400)

# %%


rec = estimateCompleteSignal(
    grafo_1, eeg_grafo_1, 0.8, epochs.average().get_data(), reduced_1.get_data())


# %%


# plt.close()
# plt.figure()
electrode = 41

plt.clf()
# plt.figure()
plt.plot(epochs.average().get_data()[electrode, :], label="orignal")
plt.plot(rec[electrode, :], label="reconstruction")
plt.legend()
# plt.xlim((0, 100))


# %%

graph = pygsp2.graphs.Sensor(100)
graph.plot()

# %%

newGraph = OSP(grafo_1, 20)
newGraph.plot()
# %%
plt.figure()
for i in range(5, 64, 3):
    newGraph = SamplingAlgorithm(grafo_1, i)
    newGraph.plot()
    plt.savefig("./plotsSamplingAlgorithm/sensorNum_" + str(i) + ".jpg")

# plt.figure()
# plt.imshow(newGraph, cmap='jet')
# plt.colorbar()
# plt.show()
# newGraph.plot()
# plt.show()


# %%

graph = pygsp2.graphs.Community(100)
graph.plot()

# %%
graph.compute_laplacian(lap_type='normalized')
graph.compute_fourier_basis()
L = graph.L.toarray()
nods, tt = fast_gsss(L, 50, 0.8, 0.8, graph.U, graph.e)
# %%
nuevoCom = graph.subgraph(nods)
nuevoCom.plot()

# %%
newGraph_0 = SamplingAlgorithm(graph, 80)
newGraph_1 = SamplingAlgorithm(graph, 60)
newGraph_2 = SamplingAlgorithm(graph, 50)
newGraph_3 = SamplingAlgorithm(graph, 40)
newGraph_4 = SamplingAlgorithm(graph, 25)

fig, axs = plt.subplots(2, 3)
graph.plot(ax=axs[0, 0])
newGraph_0.plot(ax=axs[0, 1])
newGraph_1.plot(ax=axs[0, 2])
newGraph_1.plot(ax=axs[1, 0])
newGraph_3.plot(ax=axs[1, 1])
newGraph_4.plot(ax=axs[1, 2])

# %%
newCom = SamplingAlgorithm(graph, 10)
newCom.plot()

# %%

# %%
newEEG_0 = SamplingAlgorithm(grafo_1, 50)
newEEG_1 = SamplingAlgorithm(grafo_1, 40)
newEEG_2 = SamplingAlgorithm(grafo_1, 30)
newEEG_3 = SamplingAlgorithm(grafo_1, 25)
newEEG_4 = SamplingAlgorithm(grafo_1, 20)

# fig, axs = plt.subplots(2,3)
# grafo_1.plot(ax=axs[0,0])
newEEG_0.plot()
newEEG_1.plot()
newEEG_2.plot()
newEEG_3.plot()
newEEG_4.plot()


# %%
G = pygsp2.graphs.Minnesota()
G.plot()
# %%

G.compute_laplacian(lap_type='normalized')
G.compute_fourier_basis()
L = graph.L.toarray()
nods, tt = fast_gsss(L, 50, 0.8, 0.8, G.U, G.e)
# %%
nuevoG = G.subgraph(nods)
nuevoG.plot()
# %%

newCom = SamplingAlgorithm(G, 25)
newCom.plot()

# %%


plt.figure()
for i in range(5, 64, 3):
    grafo_1.compute_laplacian(lap_type='normalized')
    grafo_1.compute_fourier_basis()
    L = graph.L.toarray()
    nods, tt = fast_gsss(L, i, 0.8, 0.8, grafo_1.U, grafo_1.e)
    newGraph = grafo_1.subgraph(nods)
    newGraph.plot()
    plt.savefig("./plotFastAlgorithm/sensorNum_" + str(i) + ".jpg")


# %%
oneSubjectSet = []
for i in range(15, 60, 4):
    print(i)
    auxGrafo = SamplingAlgorithm(grafo_1, i)
    oneSubjectSet.append(selected_channels(grafo_1, auxGrafo))

# np.save('oneSubjectSensorSet', oneSubjectSet)
