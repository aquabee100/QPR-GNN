from __future__ import annotations

import numpy as np

FIFO, SP, WFQ, DRR = 0, 1, 2, 3


def mm1k_sojourn(lam_pk, mu_pk, K, return_loss=False):
    lam = np.asarray(lam_pk, float); mu = np.asarray(mu_pk, float)
    K = np.maximum(np.asarray(K, float), 1.0)
    rho = lam / mu
    out = 1.0 / mu                                   # empty-queue limit: one service time
    pos = lam > 1e-12
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        lo = pos & (rho < 1 - 1e-6)
        hi = pos & (rho > 1 + 1e-6)
        mid = pos & ~lo & ~hi
        L = np.zeros_like(rho); PK = np.zeros_like(rho)
        x = np.exp((K + 1) * np.log(np.where(lo, rho, 0.5)))           # rho^(K+1), underflows safely
        L = np.where(lo, rho / (1 - rho) - (K + 1) * x / (1 - x), L)
        PK = np.where(lo, (1 - rho) * x / np.where(lo, rho, 1) / (1 - x), PK)
        r = np.where(hi, 1.0 / rho, 0.5)
        y = np.exp((K + 1) * np.log(r))                                 # r^(K+1)
        L = np.where(hi, rho / (1 - rho) + (K + 1) / (1 - y), L)
        PK = np.where(hi, (1 - r) / (1 - y), PK)
        L = np.where(mid, K / 2, L); PK = np.where(mid, 1 / (K + 1), PK)
        T = L / np.maximum(lam * (1 - PK), 1e-12)
    T = np.where(pos & (L > 0), T, out)
    if return_loss:
        return T, np.where(pos, PK, 0.0)
    return T


def queue_delays(s, sp_high_first=True, buffer_scale=1.0, sched=True, finite=True):
    links, cap = s["links"], s["capacity"]
    L = len(links)
    src = links[:, 0]
    MAXQ = s["node_buffer"].shape[1]
    lam = np.zeros((L, MAXQ)); pk = np.zeros((L, MAXQ))
    for f in range(len(s["flow_src"])):
        p = s["flow_path"][f]
        q = s["tos_to_queue"][src[p], s["flow_tos"][f]]
        np.add.at(lam, (p, q), s["flow_bw"][f])
        np.add.at(pk, (p, q), s["flow_pkts"][f])
    size = np.where(pk > 0, lam / np.maximum(pk, 1e-12), 0.0)
    tot_lam, tot_pk = lam.sum(1), pk.sum(1)
    mean_size = np.where(tot_pk > 0, tot_lam / np.maximum(tot_pk, 1e-12), 1000.0)
    size = np.where(size > 0, size, mean_size[:, None])

    pol = s["node_policy"][src]
    nq = s["node_nq"][src]
    w = s["node_weights"][src]
    B = s["node_buffer"][src] * buffer_scale
    C = cap[:, None]
    c = np.repeat(C, MAXQ, 1).astype(float)
    lam_q, size_q = lam.copy(), size.copy()

    fifo = (pol == FIFO) | (nq == 1) | (not sched)
    # FIFO: collapse to one queue
    lam_q[fifo, 0] = tot_lam[fifo]; size_q[fifo, 0] = mean_size[fifo]
    rho = lam_q / C
    sp = (pol == SP) & ~fifo
    for l in np.where(sp)[0]:
        order = range(nq[l]) if sp_high_first else range(nq[l] - 1, -1, -1)
        higher = 0.0
        for q in order:
            c[l, q] = cap[l] * max(1e-3, 1 - higher)
            higher += rho[l, q]
    fq = ((pol == WFQ) | (pol == DRR)) & ~fifo
    for l in np.where(fq)[0]:
        for q in range(nq[l]):
            others = lam_q[l, : nq[l]].sum() - lam_q[l, q]
            c[l, q] = max(w[l, q] * cap[l], cap[l] - others, 1e-3 * cap[l])
    K = np.maximum(np.floor(np.where(B > 0, B, B.max(1, keepdims=True)) / size_q), 1)
    if finite:
        T = mm1k_sojourn(lam_q / size_q, c / size_q, K)
        W = np.maximum(T - size_q / c, 0.0)  # waiting = sojourn - service time
    else:
        r = np.minimum(lam_q / c, 0.99)
        W = (size_q / c) * r / (1 - r)           # M/M/1 waiting time
    return W, fifo


def flow_hop_prior(s, **kw):
    W, fifo = queue_delays(s, **kw)
    cap = s["capacity"]
    src = s["links"][:, 0]
    F = len(s["flow_src"])
    P = max(len(p) for p in s["flow_path"])
    out = np.zeros((F, P))
    for f, p in enumerate(s["flow_path"]):
        q = np.where(fifo[p], 0, s["tos_to_queue"][src[p], s["flow_tos"][f]])
        pkt = s["flow_bw"][f] / max(s["flow_pkts"][f], 1e-12)
        out[f, : len(p)] = W[p, q] + pkt / cap[p]
    return out


def mg1_flow_delay(s):
    cap = s["capacity"]; L = len(cap); lp = np.zeros(L); lb = np.zeros(L); es2 = np.zeros(L)
    for f, p in enumerate(s["flow_path"]):
        pk = s["flow_pkts"][f]; sz = s["flow_bw"][f] / max(pk, 1e-12)
        lp[p] += pk; lb[p] += s["flow_bw"][f]; es2[p] += pk * (sz / cap[p]) ** 2
    rho = np.minimum(lb / cap, 0.99); W = es2 / (2 * (1 - rho))          # lambda_p E[S^2] = sum pk (s/C)^2
    out = []
    for f, p in enumerate(s["flow_path"]):
        sz = s["flow_bw"][f] / max(s["flow_pkts"][f], 1e-12)
        out.append((W[p] + sz / cap[p]).sum())
    return np.array(out)
