from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch

from .queueing import flow_hop_prior

POL = 4
MAXQ = 3
LINK_F = 9  # total load, per-ToS load, policy, #queues


def sample_tensors(s):
    L = len(s["links"])
    cap = s["capacity"]
    src_node = s["links"][:, 0]
    F = len(s["flow_src"])
    pkt_bits = s["flow_bw"] / np.maximum(s["flow_pkts"], 1e-9)
    mean_pkt = float(np.average(pkt_bits, weights=s["flow_pkts"]))

    # offered load per link and per ToS
    load = np.zeros((L, MAXQ))
    for f in range(F):
        load[s["flow_path"][f], s["flow_tos"][f]] += s["flow_bw"][f]
    rho = load / cap[:, None]
    pol = np.eye(POL)[s["node_policy"][src_node]]
    nq = s["node_nq"][src_node][:, None] / MAXQ
    link_x = np.concatenate([rho.sum(1, keepdims=True), rho, pol, nq], 1)

    # per-hop features of the used queue
    plen = np.array([len(p) for p in s["flow_path"]])
    P = int(plen.max())
    path = np.full((F, P), -1, np.int64)
    hop_x = np.zeros((F, P, 5), np.float32)
    for f, p in enumerate(s["flow_path"]):
        path[f, : len(p)] = p
        n = src_node[p]
        q = s["tos_to_queue"][n, s["flow_tos"][f]]
        w = s["node_weights"][n, q]
        buf_pk = s["node_buffer"][n, q] / mean_pkt
        rho_q = np.array([rho[l, s["tos_to_queue"][src_node[l]] == qq].sum() for l, qq in zip(p, q)])
        hop_x[f, : len(p)] = np.stack([w, np.log1p(buf_pk) / 5, rho_q, q / 2.0, s["flow_bw"][f] / cap[p]], 1)

    bottleneck = np.array([cap[p].min() for p in s["flow_path"]])
    flow_x = np.concatenate([np.eye(MAXQ)[s["flow_tos"]],
                             (s["flow_bw"] / bottleneck)[:, None] * 10,
                             plen[:, None] / 10.0], 1)
    tx = np.zeros((F, P))  # own transmission time per hop
    prior = np.zeros((F, P))  # M/M/1 per-hop delay
    rho_tot = rho.sum(1)
    for f, p in enumerate(s["flow_path"]):
        tx[f, : len(p)] = pkt_bits[f] / cap[p]
        prior[f, : len(p)] = tx[f, : len(p)] / (1.0 - np.minimum(rho_tot[p], 0.99))

    # own packet size over port mean
    bits = np.zeros(L); pk = np.zeros(L)
    for f, p in enumerate(s["flow_path"]):
        bits[p] += s["flow_bw"][f]; pk[p] += s["flow_pkts"][f]
    sbar = bits / np.maximum(pk, 1e-12)
    own = s["flow_bw"] / np.maximum(s["flow_pkts"], 1e-12)
    hop_ratio = np.ones((F, P), np.float32)
    for f, p in enumerate(s["flow_path"]):
        hop_ratio[f, : len(p)] = own[f] / sbar[p]

    def wait(**kw):
        return torch.tensor(np.maximum(flow_hop_prior(s, **kw)[:, :P] - tx, 0.0), dtype=torch.float32)

    return dict(
        L=L, F=F, P=P,
        link_x=torch.tensor(link_x, dtype=torch.float32),
        flow_x=torch.tensor(flow_x, dtype=torch.float32),
        hop_x=torch.tensor(hop_x),
        path=torch.tensor(path),
        tx=torch.tensor(tx, dtype=torch.float32),
        prior=torch.tensor(prior, dtype=torch.float32),
        wait_nosched=wait(sched=False),               # OUR PRIOR: FIFO M/M/1/K waiting time
        wait=wait(),                                  # ablation: QoS-aware (SP/WFQ/DRR) M/M/1/K
        wait_mm1=wait(sched=False, finite=False),     # ablation: FIFO M/M/1 (infinite buffer)
        hop_ratio=torch.tensor(hop_ratio),
        y=torch.tensor(s["y_delay"], dtype=torch.float32),
        tos=torch.tensor(s["flow_tos"]),
    )


@dataclass
class Batch:
    link_x: torch.Tensor        # (L, LINK_F)
    flow_x: torch.Tensor        # (F, 5)
    hop_x: torch.Tensor         # (F, P, 5)
    path: torch.Tensor  # link ids, -1 = padding
    tx: torch.Tensor            # (F, P) own transmission time per hop
    prior: torch.Tensor         # (F, P) M/M/1 per-hop delay
    wait_nosched: torch.Tensor  # FIFO M/M/1/K waiting time (prior)
    wait: torch.Tensor          # (F, P) QoS-aware M/M/1/K waiting time
    wait_mm1: torch.Tensor      # (F, P) FIFO M/M/1 waiting time
    hop_ratio: torch.Tensor     # (F, P) s_f / s_bar_port
    y: torch.Tensor             # (F,) measured mean delay (s)
    tos: torch.Tensor           # (F,)


def collate(items):
    P = max(it["P"] for it in items)
    pad = lambda t, it, v=0.0: torch.nn.functional.pad(t, (0, P - it["P"]), value=v)
    paths, loff = [], 0
    for it in items:
        p = it["path"].clone(); p[p >= 0] += loff
        paths.append(pad(p, it, -1)); loff += it["L"]
    cat = lambda k: torch.cat([pad(it[k], it) for it in items])
    return Batch(
        link_x=torch.cat([it["link_x"] for it in items]),
        flow_x=torch.cat([it["flow_x"] for it in items]),
        hop_x=torch.cat([torch.nn.functional.pad(it["hop_x"], (0, 0, 0, P - it["P"])) for it in items]),
        path=torch.cat(paths), tx=cat("tx"), prior=cat("prior"), wait_nosched=cat("wait_nosched"),
        wait=cat("wait"), wait_mm1=cat("wait_mm1"), hop_ratio=cat("hop_ratio"),
        y=torch.cat([it["y"] for it in items]), tos=torch.cat([it["tos"] for it in items]))


def link_adjacency(links):
    L = len(links)
    A = np.zeros((L, L), np.float32)
    for i in range(L):
        a, b = links[i]
        for j in range(L):
            if i != j and len({a, b} & {links[j][0], links[j][1]}):
                A[i, j] = 1.0
    return A


def valid_samples(name, cache_dir="cache"):
    S = pickle.load(open(f"{cache_dir}/{name}.pkl", "rb"))
    S = [s for s in S if len(s["y_delay"]) and np.all(s["y_delay"] > 0)]
    for s in S:
        assert np.array_equal(s["links"], S[0]["links"]), "topology must be fixed within a dataset"
    return S


def load_topology(name, cache_dir="cache"):
    p = f"{cache_dir}/tensors_{name}.pkl"
    if os.path.exists(p):
        items, S = pickle.load(open(p, "rb"))
    else:
        S = valid_samples(name, cache_dir)
        items = [sample_tensors(s) for s in S]
        pickle.dump((items, S), open(p, "wb"), protocol=4)
    A = torch.tensor(link_adjacency(S[0]["links"]))      # used by the link-graph baselines
    for it in items:
        it["A"] = A
    return items, S


def file_split(S, seed=0, frac=(0.8, 0.1, 0.1)):
    files = sorted({s["source_file"] for s in S})
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    n = len(files)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    part = {f: 0 for f in files[:a]} | {f: 1 for f in files[a:b]} | {f: 2 for f in files[b:]}
    return [[i for i, s in enumerate(S) if part[s["source_file"]] == k] for k in range(3)]
