# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:00:53 2025

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


# %%

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
eegsp = eegrasp.EEGrasp(epochs_data, eeg_pos, channels)
distances = eegsp.compute_distance(eeg_pos, method='Euclidean')
grafo_1 = eegsp.compute_graph(
    distances=distances, epsilon=0.3, sigma=0.1, coordinates=eeg_pos)
grafo_2 = eegsp.compute_graph(
    distances=distances, epsilon=0.2, sigma=0.1, coordinates=eeg_pos)
W_learning, Z = eegsp.learn_graph(a=0.34, b=0.4)
grafo_3 = eegsp.compute_graph(W_learning)

eeg_pos = eeg_pos / np.amax(eeg_pos)
eegsp.graph.set_coordinates(eeg_pos)
eegsp.coordinates = eeg_pos

grafo_1.coords = eeg_pos
grafo_2.coords = eeg_pos

# %%
WC = 0.9999999
_, Seeg_1_01 = estimate_Sopt(grafo_1, wc=WC, k=12)
_, Seeg_2_01 = estimate_Sopt(grafo_2, wc=WC, k=12)
_, Seeg_3_01 = estimate_Sopt(grafo_3, wc=WC, k=12)

eeg_grafo_1_01 = reducedGraph(grafo_1, Seeg_1_01)
eeg_grafo_2_01 = reducedGraph(grafo_2, Seeg_2_01)
eeg_grafo_3_01 = reducedGraph(grafo_3, Seeg_3_01)

selected_1_01 = selected_channels(grafo_1, eeg_grafo_1_01)
selected_2_01 = selected_channels(grafo_2, eeg_grafo_2_01)
selected_3_01 = selected_channels(grafo_3, eeg_grafo_3_01)

eeg_grafo_1_01.plot()
# %%
WC = 0.6
_, Seeg_1_02 = estimate_Sopt(grafo_1, wc=WC, k=12)
_, Seeg_2_02 = estimate_Sopt(grafo_2, wc=WC, k=12)
_, Seeg_3_02 = estimate_Sopt(grafo_3, wc=WC, k=12)

eeg_grafo_1_02 = reducedGraph(grafo_1, Seeg_1_02)
eeg_grafo_2_02 = reducedGraph(grafo_2, Seeg_2_02)
eeg_grafo_3_02 = reducedGraph(grafo_3, Seeg_3_02)

selected_1_02 = selected_channels(grafo_1, eeg_grafo_1_02)
selected_2_02 = selected_channels(grafo_2, eeg_grafo_2_02)
selected_3_02 = selected_channels(grafo_3, eeg_grafo_3_02)

eeg_grafo_1_02.plot()

# %%
WC = 0.5
_, Seeg_1_03 = estimate_Sopt(grafo_1, wc=WC, k=12)
_, Seeg_2_03 = estimate_Sopt(grafo_2, wc=WC, k=12)
_, Seeg_3_03 = estimate_Sopt(grafo_3, wc=WC, k=12)

eeg_grafo_1_03 = reducedGraph(grafo_1, Seeg_1_03)
eeg_grafo_2_03 = reducedGraph(grafo_2, Seeg_2_03)
eeg_grafo_3_03 = reducedGraph(grafo_3, Seeg_3_03)

selected_1_03 = selected_channels(grafo_1, eeg_grafo_1_03)
selected_2_03 = selected_channels(grafo_2, eeg_grafo_2_03)
selected_3_03 = selected_channels(grafo_3, eeg_grafo_3_03)

eeg_grafo_1_03.plot()

# %%
WC = 0.94
_, Seeg_1_03 = estimate_Sopt(grafo_1, wc=WC, k=1)
_, Seeg_2_03 = estimate_Sopt(grafo_2, wc=WC, k=12)
_, Seeg_3_03 = estimate_Sopt(grafo_3, wc=WC, k=12)

eeg_grafo_1_03 = reducedGraph(grafo_1, Seeg_1_03)
eeg_grafo_2_03 = reducedGraph(grafo_2, Seeg_2_03)
eeg_grafo_3_03 = reducedGraph(grafo_3, Seeg_3_03)

selected_1_03 = selected_channels(grafo_1, eeg_grafo_1_03)
selected_2_03 = selected_channels(grafo_2, eeg_grafo_2_03)
selected_3_03 = selected_channels(grafo_3, eeg_grafo_3_03)


eeg_grafo_1_03.plot()
