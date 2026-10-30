from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GATConv

P_DIM, L_DIM, H_P, H_L = 5, 1, 8, 8


def sample_to_qt(s):
    links = s["links"]; cap = s["capacity"].astype(np.float64); L = len(links); src = links[:, 0]
    F_ = len(s["flow_src"]); P = max(len(p) for p in s["flow_path"])
    path = np.full((F_, P), -1, np.int64)
    for f, p in enumerate(s["flow_path"]):
        path[f, : len(p)] = p
    bw = s["flow_bw"].astype(np.float64); pk = np.maximum(s["flow_pkts"].astype(np.float64), 1e-12)
    bits = np.zeros(L); pkts = np.zeros(L)
    for f, p in enumerate(s["flow_path"]):
        bits[p] += bw[f]; pkts[p] += pk[f]
    sbar = np.where(pkts > 0, bits / np.maximum(pkts, 1e-12), 1000.0)
    qbits = s["node_buffer"][src, 0].astype(np.float64)
    qbits = np.where(qbits > 0, qbits, s["node_buffer"][src].max(1))
    B = np.maximum(np.round(qbits / sbar), 1.0)
    pfeat = np.stack([np.log(bw), np.log(pk), np.log(bw / pk), np.array([len(p) for p in s["flow_path"]], float),
                      s["flow_tos"].astype(float)], 1)
    return dict(F=F_, L=L, P=P, path=path, cap=cap.astype(np.float32), qbits=qbits.astype(np.float32),
                B=B.astype(np.float32), sbar=sbar.astype(np.float32), pkts=pk.astype(np.float32),
                sf=(bw / pk).astype(np.float32), pfeat=pfeat.astype(np.float32), lfeat=np.log(cap)[:, None].astype(np.float32),
                y=s["y_delay"].astype(np.float32), tos=s["flow_tos"])


def qt_stats(items):
    pf = np.concatenate([it["pfeat"] for it in items]); lf = np.concatenate([it["lfeat"] for it in items])
    return {"p": [pf.mean(0).tolist(), (pf.std(0) + 1e-8).tolist()], "l": [lf.mean(0).tolist(), (lf.std(0) + 1e-8).tolist()]}


def collate_qt(items):
    nP = sum(it["F"] for it in items); nL = sum(it["L"] for it in items); Pm = max(it["P"] for it in items)
    path = torch.full((nP, Pm), -1, dtype=torch.long)
    po = lo = 0
    cat = {k: [] for k in ["cap", "qbits", "B", "pkts", "sf", "pfeat", "lfeat", "y", "tos"]}
    for it in items:
        p = torch.tensor(it["path"]); m = p >= 0
        path[po:po + it["F"], : it["P"]] = torch.where(m, p + lo, p)
        for k in cat:
            cat[k].append(torch.as_tensor(it[k]))
        po += it["F"]; lo += it["L"]
    b = {k: torch.cat(v) for k, v in cat.items()}
    b.update(nP=nP, nL=nL, path=path)
    return b


class GConvGRU(nn.Module):
    def __init__(self, cin, cout, K=2):
        super().__init__()
        mk = lambda a: ChebConv(a, cout, K=K, normalization="sym")
        self.xz, self.hz, self.xr, self.hr, self.xh, self.hh = mk(cin), mk(cout), mk(cin), mk(cout), mk(cin), mk(cout)
        self.cout = cout

    def forward(self, X, ei, H=None):
        if H is None:
            H = torch.zeros(X.shape[0], self.cout)
        Z = torch.sigmoid(self.xz(X, ei) + self.hz(H, ei))
        R = torch.sigmoid(self.xr(X, ei) + self.hr(H, ei))
        Ht = torch.tanh(self.xh(X, ei) + self.hh(H * R, ei))
        return Z * H + (1 - Z) * Ht


def _mm1b(rho, Bk):
    jmax = int(Bk.max().item())
    j = torch.arange(0, jmax + 1, dtype=torch.float32)[None, :]
    lw = j * torch.log(rho.clamp(min=1e-12))[:, None]
    lw = torch.where(j <= Bk[:, None], lw, torch.full_like(lw, -float("inf")))
    p = torch.softmax(lw, 1)
    occ = (p * j).sum(1) / Bk
    pB = p.gather(1, Bk.long()[:, None]).squeeze(1)
    return occ, p[:, 0], pB


def mm1b_baseline(b, iters=3):
    path, nL = b["path"], b["nL"]
    m = path >= 0; pc = path.clamp(min=0)
    mu = b["cap"] / torch.clamp(b["qbits"] / b["B"], min=1e-9)          # service rate, packets/s
    block = torch.full((nL,), 0.3)
    for _ in range(iters):
        keep = torch.where(m, 1 - block[pc], torch.ones_like(block[pc]))
        surv = torch.cumprod(torch.cat([torch.ones(path.shape[0], 1), keep], 1)[:, :-1], 1)
        T = torch.zeros(nL).index_add_(0, pc[m], (b["pkts"][:, None] * surv)[m])
        rho = T / mu
        occ, pi0, block = _mm1b(rho, b["B"])
    link_delay = occ * b["qbits"] / b["cap"]
    base = (link_delay[pc] * m).sum(1)
    return base, torch.stack([occ, rho, block], 1)


class QTRouteNet(nn.Module):
    def __init__(self, stats, iters=3, tx_term=False):
        super().__init__()
        self.st = stats; self.iters = iters; self.tx_term = tx_term
        self.lin_all = L_DIM + H_L + P_DIM + H_P
        self.gru = nn.ModuleList([GConvGRU(H_L + L_DIM, H_P, K=2) for _ in range(iters)])
        self.gat = nn.ModuleList([GATConv(H_P + P_DIM, H_L) for _ in range(iters)])
        self.xlin1 = nn.Linear(self.lin_all, 128); self.xlin2 = nn.Linear(128, self.lin_all)
        self.lin1 = nn.Linear(L_DIM + H_L, 128); self.lin2 = nn.Linear(128, 32); self.lin3 = nn.Linear(32, 1)

    def forward(self, b):
        nP, nL = b["nP"], b["nL"]; N = nP + nL
        base, bocc = mm1b_baseline(b)
        pm, ps = (torch.tensor(v) for v in self.st["p"]); lm, ls = (torch.tensor(v) for v in self.st["l"])
        # column layout of X
        lh, lo, ph, po = slice(0, H_L), slice(H_L, H_L + L_DIM), slice(H_L + L_DIM, H_L + L_DIM + H_P), slice(H_L + L_DIM + H_P, self.lin_all)
        link_all, path_all = slice(0, H_L + L_DIM), slice(H_L + L_DIM, self.lin_all)
        X = torch.zeros(N, self.lin_all)
        X[nP:, lo] = (b["lfeat"] - lm) / ls
        X[:nP, po] = (b["pfeat"] - pm) / ps
        X[nP:, 0:3] = bocc
        X[:nP, ph.start] = base
        X = self.xlin2(self.xlin1(X))
        path = b["path"]; m = path >= 0
        fid = torch.arange(nP)[:, None].expand_as(path)
        pos_edges = [torch.stack([path[:, k][m[:, k]] + nP, fid[:, k][m[:, k]]]) for k in range(path.shape[1])]
        pl = torch.stack([fid[m], path[m] + nP])                     # path -> link
        for i in range(self.iters):
            x = X[:, link_all].clone(); H = None
            for e in pos_edges:
                H = self.gru[i](x, e, H)
            X = X.clone()
            X[:nP, ph] = F.leaky_relu(H[:nP] / len(pos_edges))
            X[:nP, ph.start] = base
            X[nP:, lh] = F.leaky_relu(self.gat[i](X[:, path_all].clone(), pl)[nP:])
            X[nP:, 0:3] = bocc
        Lh = X[nP:, link_all]
        occ = torch.sigmoid(self.lin3(F.leaky_relu(self.lin2(F.leaky_relu(self.lin1(Lh)))))).squeeze(-1)
        link_delay = occ * b["qbits"] / b["cap"]
        d = (link_delay[path.clamp(min=0)] * m).sum(1)
        if self.tx_term:
            d = d + (b["sf"][:, None] / b["cap"][path.clamp(min=0)] * m).sum(1)
        return d
