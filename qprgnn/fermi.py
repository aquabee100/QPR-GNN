from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MAXQ = 3
FIFO_CODE = 0          # qprgnn.parse.POLICIES["FIFO"]
N_MODELS = 7
N_POL = 4


def sample_to_fermi(s):
    links = s["links"]; cap = s["capacity"]; src = links[:, 0]
    L = len(links)
    nq = s["node_nq"][src]
    qbase = np.concatenate([[0], np.cumsum(nq)[:-1]])
    Q = int(nq.sum())
    q_size = np.zeros(Q); q_prio = np.zeros(Q, np.int64); q_w = np.zeros(Q); q_link = np.zeros(Q, np.int64)
    for l in range(L):
        for q in range(nq[l]):
            i = qbase[l] + q
            q_size[i] = s["node_buffer"][src[l], q] if s["node_buffer"][src[l], q] > 0 else s["node_buffer"][src[l]].max()
            q_prio[i] = q; q_w[i] = s["node_weights"][src[l], q]; q_link[i] = l
    F = len(s["flow_src"])
    P = max(len(p) for p in s["flow_path"])
    path_links = np.full((F, P), -1, np.int64); path_queues = np.full((F, P), -1, np.int64)
    for f, p in enumerate(s["flow_path"]):
        path_links[f, : len(p)] = p
        q = s["tos_to_queue"][src[p], s["flow_tos"][f]]
        q = np.minimum(q, nq[p] - 1)
        path_queues[f, : len(p)] = qbase[p] + q
    q_used = np.zeros(Q, bool); q_used[path_queues[path_queues >= 0]] = True
    q_w_off = np.where(s["node_policy"][src][q_link] == FIFO_CODE, 0.0, q_w)   # official: "-" weights -> 0
    traffic = s["flow_bw"]; packets = s["flow_pkts"]
    eq = s.get("flow_eqlambda", None)
    load = np.zeros(L)
    for f, p in enumerate(s["flow_path"]):
        load[p] += traffic[f]
    load = load / cap
    pol = s["node_policy"][src]
    return dict(L=L, Q=Q, F=F, P=P, cap=cap.astype(np.float32), load=load.astype(np.float32), policy=pol,
                q_size=q_size.astype(np.float32), q_prio=q_prio, q_w=q_w.astype(np.float32), q_link=q_link,
                q_w_off=q_w_off.astype(np.float32), q_used=q_used,
                eq=(eq if eq is not None else np.full(F, np.nan)).astype(np.float32),
                q_slot=np.concatenate([np.arange(n) for n in nq]),
                traffic=traffic.astype(np.float32), packets=packets.astype(np.float32),
                path_links=path_links, path_queues=path_queues,
                y=s["y_delay"].astype(np.float32), tos=s["flow_tos"])


def zstats(items):
    def ms(x):
        x = np.concatenate(x); return [float(x.mean()), float(x.std() + 1e-9)]
    return {"traffic": ms([it["traffic"] for it in items]), "packets": ms([it["packets"] for it in items]),
            "eq_lambda": ms([it["eq"] for it in items]),
            "capacity": ms([it["cap"] for it in items]), "queue_size": ms([it["q_size"] for it in items])}


def collate_fermi(items):
    P = max(it["P"] for it in items)
    lo = qo = 0
    out = {k: [] for k in ["cap", "load", "policy", "q_size", "q_prio", "q_w", "q_link", "q_slot", "traffic",
                           "packets", "path_links", "path_queues", "y", "tos", "F", "q_w_off", "q_used", "eq"]}
    for it in items:
        pl = np.full((it["F"], P), -1, np.int64); pq = np.full((it["F"], P), -1, np.int64)
        m = it["path_links"] >= 0
        pl[:, : it["P"]] = np.where(m, it["path_links"] + lo, -1)
        pq[:, : it["P"]] = np.where(m, it["path_queues"] + qo, -1)
        out["path_links"].append(pl); out["path_queues"].append(pq)
        out["q_link"].append(it["q_link"] + lo)
        for k in ["cap", "load", "policy", "q_size", "q_prio", "q_w", "q_slot", "traffic", "packets", "y", "tos",
                  "q_w_off", "q_used", "eq"]:
            out[k].append(it[k])
        out["F"].append(it["F"])
        lo += it["L"]; qo += it["Q"]
    b = {k: torch.tensor(np.concatenate(v)) for k, v in out.items() if k != "F"}
    b["F"] = out["F"]
    return b


class RouteNetFermi(nn.Module):
    def __init__(self, z, H=32, T=8):
        super().__init__()
        self.z, self.T, self.H = z, T, H
        self.path_embedding = nn.Sequential(nn.Linear(10 + N_MODELS, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.queue_embedding = nn.Sequential(nn.Linear(MAXQ + 2, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.link_embedding = nn.Sequential(nn.Linear(N_POL + 1, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.path_update = nn.GRUCell(2 * H, H)
        self.queue_update = nn.GRUCell(H, H)
        self.link_update = nn.GRUCell(H, H)
        self.readout = nn.Sequential(nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))

    def _z(self, x, k):
        m, s = self.z[k]; return (x - m) / s

    def forward(self, b):
        F_ = b["traffic"].shape[0]
        zt = self._z(b["traffic"], "traffic")[:, None]
        path_in = torch.cat([zt, self._z(b["packets"], "packets")[:, None],
                             torch.nn.functional.one_hot(torch.zeros(F_, dtype=torch.long), N_MODELS).float(),
                             self._z(b["eq"], "eq_lambda")[:, None],
                             torch.zeros(F_, 7)], 1)
        h_p = self.path_embedding(path_in)
        h_l = self.link_embedding(torch.cat([b["load"][:, None],
                                             torch.nn.functional.one_hot(b["policy"], N_POL).float()], 1))
        h_q = self.queue_embedding(torch.cat([self._z(b["q_size"], "queue_size")[:, None],
                                              torch.nn.functional.one_hot(b["q_prio"], MAXQ).float(),
                                              b["q_w_off"][:, None]], 1))
        pl, pq = b["path_links"], b["path_queues"]
        mask = pl >= 0
        pli, pqi = pl.clamp(min=0), pq.clamp(min=0)
        P = pl.shape[1]
        Lk = h_l.shape[0]
        q_link, q_slot = b["q_link"], b["q_slot"]
        for _ in range(self.T):
            prev = h_p
            seq = [prev]
            hp = h_p
            for k in range(P):
                inp = torch.cat([h_q[pqi[:, k]], h_l[pli[:, k]]], 1)
                new = self.path_update(inp, hp)
                m = mask[:, k:k + 1].float()
                hp = m * new + (1 - m) * hp
                seq.append(hp)
            S = torch.stack(seq, 1)                                # (F, P+1, H); S[:,0] = previous state
            # path -> queue aggregation
            agg = torch.zeros_like(h_q).index_add_(0, pqi[mask], S[:, :P][mask])
            h_q = self.queue_update(agg, h_q)
            # queue -> link GRU
            hl = h_l
            for slot in range(MAXQ):
                sel = (q_slot == slot) & b["q_used"]
                if not sel.any():
                    continue
                x = torch.zeros(Lk, self.H); has = torch.zeros(Lk, 1)
                x[q_link[sel]] = h_q[sel]; has[q_link[sel]] = 1.0
                hl = has * self.link_update(x, hl) + (1 - has) * hl
            h_l = hl
            h_p = hp
        occ = self.readout(S[:, 1:]).squeeze(-1)                  # (F, P) occupancy in bits
        cap = b["cap"][pli]
        qd = (occ / cap * mask).sum(1)
        pkt = b["traffic"] / b["packets"].clamp(min=1e-9)
        td = pkt * ((1.0 / cap) * mask).sum(1)
        return qd + td
