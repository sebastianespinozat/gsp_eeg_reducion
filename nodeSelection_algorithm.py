# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:47:10 2024

@author: sebas
"""

# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pygsp2
import scipy
import eegrasp
import mne
import random
from mne.datasets import eegbci

# %% RESPALDO
# estimates S opt. Returns list of estimated frequencies and the selected nodes


def estimate_Sopt(graph, wc, k):

    graph.compute_laplacian(lap_type='normalized')
    graph.compute_fourier_basis()
    KC = Kc(graph, wc)
    L = graph.L.toarray()
    auxGraph = pygsp2.graphs.Graph(graph.W, coords=graph.coords)
    auxGraph.compute_laplacian(lap_type='normalized')
    auxGraph.compute_fourier_basis()
    S = []  # coords of selected sensors: S
    # sensorsList = [] # coordinates of Sopt
    w = 0
    freqz = []
    while w <= wc:
        # auxGraph.compute_laplacian(lap_type = 'normalized')
        # L = auxGraph.L.toarray()
        # np.linalg.eig() not sorted
        # np.linalg.eigh()
        L_k = np.linalg.matrix_power(L, k)
        e, u = np.linalg.eig(L_k)  # gets eigen values and vectors of L^ke
        e_sorted = np.sort(e)
        # print(e_sorted[:5])
        # e[e<1e-17] = 0
        idx = np.argsort(e)  # [:2]
        idx = idx[0]
        e_1k = e[idx]  # smallest eigenvalue
        u_1k = u[:, idx]  # eigenvector of smallest eigenvalue
        # w = np.abs(np.power(e[1], 1/k)) # estimates frequency
        w = np.abs(np.power(np.abs(e_1k), 1/k))  # estimates frequency

        if w > wc:
            break
        else:
            U_1 = u_1k
            smoothest = np.argmax(np.power(U_1, 2))
            # smoothest = np.argmax(np.abs(U_1))
            S.append(auxGraph.coords[smoothest, :])  # selected sensors
            L_0 = np.delete(L, smoothest, 0)  # delete column
            L = np.delete(L_0, smoothest, 1)  # delete row
            nRows, _ = L.shape

            auxGraph = auxGraph.subgraph(
                [i for i in range(auxGraph.N) if i != smoothest])

            freqz.append(w)

        if nRows > 1:
            continue
        else:
            break
    print("w: {}, wc: {}. |Sest|:{}, Kc:{}".format(
        freqz[-1], wc, len(freqz), KC))

    # S = pygsp2.graphs.Sensor(len(sensorsList))
    # S.set_coordinates(sensorsList)
    return freqz, S


# %% Code from the "Towards a sampling theorem for signals on arbitrary graphs" algorithm Anis et al

# return number of eigenvalues of below wc
def Kc(graph, wc):
    graph.compute_laplacian(lap_type='normalized')
    graph.compute_fourier_basis()
    Kc = len(graph.e[graph.e < wc])
    return Kc

# plot eigenvalues


def plotEigenValues(graph):
    graph.compute_laplacian(lap_type='normalized')
    graph.compute_fourier_basis()
    plt.figure()
    plt.plot(graph.e, '*-')
    plt.show()

# plot eigenvalues and Kc given wc


def plotKc(graph, wc):
    kc = Kc(graph, wc)
    graph.compute_laplacian(lap_type='normalized')
    graph.compute_fourier_basis()
    plt.figure()
    plt.plot(graph.e, '*-')
    plt.axhline(y=wc, color='r', linestyle='-')
    plt.title("Eigenvalues. $w_c = {}$ and $Kc = {}$".format(wc, kc))
    plt.show()


# estimates S opt. Returns list of estimated frequencies and the selected nodes
def estimate_Sopt(graph, wc, k):

    graph.compute_laplacian(lap_type='normalized')
    graph.compute_fourier_basis()
    KC = Kc(graph, wc)
    L = graph.L.toarray()
    auxGraph = pygsp2.graphs.Graph(graph.W, coords=graph.coords)
    auxGraph.compute_laplacian(lap_type='normalized')
    auxGraph.compute_fourier_basis()
    S = []  # coords of selected sensors: S
    # sensorsList = [] # coordinates of Sopt
    w = 0
    freqz = []
    while w <= wc:
        auxGraph.compute_laplacian(lap_type='normalized')
        L = auxGraph.L.toarray()
        # np.linalg.eig() not sorted
        # np.linalg.eigh()
        L_k = np.linalg.matrix_power(L, k)
        e, u = np.linalg.eig(L_k)  # gets eigen values and vectors of L^ke
        e_sorted = np.sort(e)
        # print(e_sorted[:5])
        # e[e<1e-17] = 0
        idx = np.argsort(e)  # [:2]
        idx = idx[0]
        e_1k = e[idx]  # smallest eigenvalue
        u_1k = u[:, idx]  # eigenvector of smallest eigenvalue
        # w = np.abs(np.power(e[1], 1/k)) # estimates frequency
        w = np.abs(np.power(np.abs(e_1k), 1/k))  # estimates frequency

        if w > wc:
            break
        else:
            U_1 = u_1k
            smoothest = np.argmax(np.power(U_1, 2))
            # smoothest = np.argmax(np.abs(U_1))
            S.append(auxGraph.coords[smoothest, :])  # selected sensors
            L_0 = np.delete(L, smoothest, 0)  # delete column
            L = np.delete(L_0, smoothest, 1)  # delete row
            nRows, _ = L.shape

            nodes = list(range(auxGraph.N))
            nodes.pop(smoothest)
            auxGraph = auxGraph.subgraph(nodes)

            # auxGraph = auxGraph.subgraph(
            #     [i for i in range(auxGraph.N) if i != smoothest])

            freqz.append(w)

        if nRows > 1:
            continue
        else:
            break
    print("w: {}, wc: {}. |Sest|:{}, Kc:{}".format(
        freqz[-1], wc, len(freqz), KC))

    # S = pygsp2.graphs.Sensor(len(sensorsList))
    # S.set_coordinates(sensorsList)
    return freqz, S


# graph: original graph
# S: coords of selected nodes after using estimate_Sopt
def reducedGraph(graph, S):
    idx = []
    for i in range(len(graph.coords)):
        for j in S:
            if (graph.coords[i] == j).all():
                idx.append(i)

    reducedGraph = graph.subgraph(idx)

    return reducedGraph


# %%% Code from the "A novel method for sampling bandlimited graph signals" Tzamarias et al

def SamplingAlgorithm(graph, n):
    graph.compute_laplacian()
    graph.compute_fourier_basis()
    Qn = graph.U[:, :n]
    firstEval = graph.e[0:n-1]
    L = graph.L
    Vi = (graph.U[0, :]).nonzero()[0][0]
    S = []
    # S.append(graph.coords[Vi,:])
    S.append(Vi)
    Sc = np.arange(0, Qn.shape[0])
    Sc = [s for s in Sc if s not in S]
    for m in range(1, n):
        # Create Qm(S)
        Qm = Qn[:, :m]
        Qm_Sc = Qn[:, :m]
        Qm = Qm[S, :]
        Qm_Sc = Qm_Sc[Sc]
        # compute x=null(Qm(S))
        _, _, Vt = np.linalg.svd(Qm)  # Descomposición SVD
        x = Vt[-1, :]  # El último vector de Vt corresponde al vector nulo
        # compute b=Qm(S^c)x
        b = Qm_Sc @ x
        # i = argmax_i |b(i)|
        i = np.argmax(np.abs(b))
        nuevo_nodo = Sc[i]  # Nodo a agregar al conjunto S
        # S = S U S^c(i)
        S.append(nuevo_nodo)
        Sc = np.setdiff1d(Sc, [nuevo_nodo])

    Sopt = 0
    Sopt = graph.subgraph(S)
    return Sopt

# %% Code from the "Optimal sensor placement for leak location in water distribution networks:
# A feature selection method combined with graph signal processing" Cheng et all


def OSP(graph, sensorNum):
    graph.compute_laplacian()
    graph.compute_fourier_basis()
    N = len(graph.e)
    S = []
    C_selected = np.zeros(N)
    c = len(C_selected)
    a = 0
    t = 0
    L = graph.L
    E = np.diag(graph.e)
    U = graph.U
    edges, _, _ = graph.get_edge_list()
    B = 8
    print(edges.shape)
    Psi_g = np.zeros((N, N))
    Psi_vi = np.zeros(N)
    epsilon = 1e-15
    for i in range(N):
        Psi_vi = np.zeros(N)
        for j in range(N):
            Psi_vi_j = 0
            s = 5*edges.shape[0]
            tau = (B*(N*0.5)*s)/(np.power(N, 3)*graph.e[-1])

            g = pygsp2.filters.Heat(graph)  # , tau)
            for k in range(N):
                Psi_vi_j += g(graph.e[k])*np.conjugate(U[i, k])*U[j, k]
            Psi_vi[j] = Psi_vi_j
        Psi_g[i, :] = Psi_vi

    plt.figure()
    plt.imshow(Psi_g, cmap='bwr')
    plt.gca().invert_yaxis()
    plt.colorbar()

    # return Psi_g
    V_nodenum = -1e3
    newGraph = pygsp2.graphs.Graph(graph.W)
    newGraph.coords = graph.coords

    # sensorNum = 20
    for s in range(sensorNum):
        a = 0
        notSelected = np.where(C_selected == 0)[0]
        selected = np.where(C_selected == 1)[0]
        for v in notSelected:
            t_num = np.mean(Psi_g)*np.linalg.norm(Psi_g[v, :], 1)
            sumatoria = 0

            # for j in notSelected:
            for j in range(N):
                sumatoria += np.linalg.norm(
                    np.multiply(Psi_g[v, :], Psi_g[j, :]), 1)

            t_den = epsilon + (sumatoria/len(notSelected))
            t = t_num/t_den

            if t > a:
                print(t)
                a = t
                V_nodenum = v
        if (V_nodenum == -1e3):
            continue
        if (C_selected[V_nodenum] == 1):
            continue
        else:
            print("El sensor seleccionado es: ", V_nodenum)
            C_selected[V_nodenum] = 1
            S.append(V_nodenum)

    sampled_graph = graph.subgraph(S)

    print(np.sort(S))
    print("Cantidad de sensores finales: ", len(S))
    return sampled_graph

# check reduction method in pygsp
# %% Code for the "Recovery of bandlimited graph signals on the reproducing kernel Hilbert space"


def BGSRP_recon(G, x0, y0, wc, gamma):
    G.compute_fourier_basis()
    G.compute_laplacian()

    ell = len(x0)
    n = Kc(G, wc)
    Un = G.U[:, 1:n]
    mu = G.e

    Phin = np.diag(1.0 / mu[1:n])
    A = np.eye(ell) - np.ones((ell, ell)) / ell
    B = Un[x0, :]

    GG = B @ (gamma * Phin + Phin @ B.T @ A @ B @ Phin) @ B.T
    d = B @ Phin @ B.T @ A @ y0

    xi = np.linalg.pinv(GG) @ d
    g = Un @ Phin @ B.T @ xi
    z = -np.ones(ell) @ (g[x0] - y0) * np.sqrt(G.N) / ell
    y = z * np.ones(G.N) / np.sqrt(G.N) + g

    return y

# %% Code for "Eigendecomposition-Free Sampling Set Selection for Graph Signals"


def fast_gsss(L_n, F, bw, nu, U, E):
    """

    Parameters:
    L_n: np.ndarray
        Symmetric normalized graph Laplacian.
    F: int
        Number of vertices to select.
    bw: float
        Bandwidth of the graph signals.
    nu: float
        Parameter controlling the width of the filter kernel.
    U: np.ndarray
        Eigenvector matrix of L_n.
    E: np.ndarray
        Eigenvalue matrix of L_n.

    Returns:
    selected_nodes: list
        Indices of selected vertices.
    T: np.ndarray
        Basis for reconstruction.
    """

    N = L_n.shape[0]
    A_tmp = (L_n - np.diag(np.diag(L_n))) != 0
    num_edges = np.sum(A_tmp) / 2
    p = num_edges / N  # Edge probability
    n = F / N          # Sampling ratio
    k = bw / N         # Bandwidth

    # Preparing T
    lmax = np.max(E)
    g_E = np.exp(-nu * p * n * k * E / lmax)
    T_g_tmp1 = U @ np.diag(g_E) @ U.T

    # Sampling Set Selection (SSS)
    T_g_tmp = np.abs(T_g_tmp1)
    selected_nodes = []

    def selection(selected, T_g_tmp):
        if len(selected) != 0:
            T = np.sum(T_g_tmp[:, selected], axis=1)
            T2 = np.mean(T) - T
            T2[T2 < 0] = 0
            T_g = T_g_tmp @ T2
        else:
            T_g = np.sum(T_g_tmp, axis=0)
        T_g[selected] = 0
        sensor = np.argmax(T_g)
        return sensor

    for _ in range(F):
        sensor = selection(selected_nodes, T_g_tmp)
        selected_nodes.append(sensor)

    T = T_g_tmp1

    return selected_nodes, T


# %% useful functions

def dropped_channels(graph, reduction, ch_list):
    dropped = []
    for i in range(len(graph.coords)):
        if not np.any(np.all(graph.coords[i] == reduction.coords, axis=1)):
            dropped.append(ch_list[i])
    return dropped


def selected_channels(graph, reduction):
    idx = []
    for i in range(len(graph.coords)):
        if np.any(np.all(graph.coords[i] == reduction.coords, axis=1)):
            idx.append(i)
    return idx


def dropped_channel_index(graph, reduction):
    idx = []
    for i in range(len(graph.coords)):
        if not np.any(np.all(graph.coords[i] == reduction.coords, axis=1)):
            idx.append(i)
    return idx


def SNR(signal, estimation):
    P_signal = np.mean(np.square(signal))
    P_noise = np.mean(np.square(np.subtract(signal, estimation)))
    SNR = 10*np.log10(P_signal/P_noise)
    return SNR


def MSE(signal, estimation):
    return np.square(np.subtract(signal, estimation)).mean()


def estimate_signal(graph, reduced_graph, wc, f_selected):
    graph.compute_laplacian(lap_type='normalized')
    e, U = np.linalg.eig(graph.L.toarray())

    idxR = np.argwhere(e >= wc)
    idxR = idxR.reshape(len(idxR))
    Uvr = np.delete(U, idxR, 1)

    idxS = dropped_channel_index(graph, reduced_graph)
    Usr = np.delete(Uvr, idxS, 0)

    f = Uvr @ np.linalg.pinv(Usr) @ f_selected

    return f


def estimateCompleteSignal(graph, reduced_graph, wc, original_signal, reduced_signal):
    reconstructions = np.zeros(
        (original_signal.shape[0], original_signal.shape[1]))
    for i in range(original_signal.shape[1]):
        # print(i)
        # f_real = original_signal[:,i]
        f_selected = reduced_signal[:, i]
        f_estimated = estimate_signal(graph, reduced_graph, wc, f_selected)
        reconstructions[:, i] = f_estimated

    return reconstructions


# %%
def create_weighted_ring(N=100, K=4):
    base_graph = pygsp2.graphs.Ring(N, K)
    A = base_graph.W.toarray()

    # Crear nueva matriz de pesos inversamente proporcional a la distancia
    new_weights = np.zeros_like(A)
    for i in range(N):
        for j in range(N):
            if A[i, j] != 0:
                dist = min(abs(i - j), N - abs(i - j))
                new_weights[i, j] = 1.0 / dist

    # Asegurarse que sea simétrica
    new_weights = (new_weights + new_weights.T) / 2

    # Crear nuevo grafo con los pesos ajustados
    G = pygsp2.graphs.Graph(new_weights)
    G.set_coordinates('ring2D')
    G.compute_laplacian()  # (lap_type='normalized')
    G.compute_fourier_basis()
    return G

# %%


graph = create_weighted_ring(100, 4)
graph.plot()
plt.figure()
plt.imshow(graph.W.toarray())
plt.colorbar()
Kc(graph, 0.8)
# %%


graph = pygsp2.graphs.Ring(100, 4)
graph.plot()

# plotEigenValues(graph)

# %%

K = [6, 12, 18]
wc = 0.3
freqz_sn = []
Ssn = []

for k in K:
    print(f'k = {k}')
    F, S = estimate_Sopt(graph, wc, k=k)
    freqz_sn.append(F)
    Ssn.append(S)


# plt.figure()
# for i in range(len(freqz_sn)):
#     plt.plot(freqz_sn[i], '*-', label='k:{}'.format(K[i]))
# plt.legend()
# plt.xlabel("$|S|$")
# plt.ylabel("$\omega$")
# plt.title("Estimated frequency vs number of selected nodes, wc=0.8")
# plt.show()


# %%


# for plot purposes, use odd number of estimations
Wc_SN = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.3]  # , 1.4, 1.5]
K = 10
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


# %%


G = pygsp2.graphs.Logo()
G.compute_fourier_basis()
f = pygsp2.filters.Expwin(G)
G.compute_fourier_basis()
y = f.evaluate(G.e)
plt.plot(G.e, y[0])
