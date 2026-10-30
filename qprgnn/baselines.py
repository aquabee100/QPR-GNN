from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import LINK_F


# ----------------------------------------------------------------- features
def flow_table(it):
    p = it["path"]; m = (p >= 0).float(); pc = p.clamp(min=0)
    n = m.sum(1, keepdim=True).clamp(min=1)
    lx = it["link_x"]
    rho = lx[:, 0][pc] * m
    hop = it["hop_x"]
    pol = lx[:, 4:8][pc] * m[..., None]
    feats = [it["flow_x"],
             rho.max(1, keepdim=True).values, rho.sum(1, keepdim=True) / n, rho.sum(1, keepdim=True),
             (hop[..., 2] * m).max(1, keepdim=True).values, (hop[..., 2] * m).sum(1, keepdim=True) / n,
             (hop[..., 0] * m + (1 - m) * 9).min(1, keepdim=True).values, (hop[..., 0] * m).sum(1, keepdim=True) / n,
             (hop[..., 1] * m + (1 - m) * 9).min(1, keepdim=True).values, (hop[..., 1] * m).sum(1, keepdim=True) / n,
             (hop[..., 4] * m).max(1, keepdim=True).values,
             pol.sum(1) / n,
             (lx[:, 8][pc] * m).sum(1, keepdim=True) / n]
    return torch.cat(feats, 1)


def tx_sum(it):
    return (it["tx"] * (it["path"] >= 0)).sum(1)


def prior_sum(it):
    return ((it["tx"] + it["wait_nosched"]) * (it["path"] >= 0)).sum(1)


def prior_table(it):
    m = (it["path"] >= 0).float(); n = m.sum(1, keepdim=True).clamp(min=1)
    tx, W = it["tx"], it["wait_nosched"]
    lr = torch.log1p(W / tx.clamp(min=1e-12)) * m
    lq = torch.log(it["hop_ratio"].clamp(min=1e-6)) * m
    return torch.cat([torch.log(prior_sum(it) / tx_sum(it))[:, None], lr.max(1, keepdim=True).values,
                      lr.sum(1, keepdim=True) / n,
                      (lq + (1 - m) * 99).min(1, keepdim=True).values, lq.sum(1, keepdim=True) / n,
                      (lq - (1 - m) * 99).max(1, keepdim=True).values], 1)


def link_wait(it):
    L = it["link_x"].shape[0]; m = it["path"] >= 0
    w_over_sbar = (it["wait_nosched"] / (it["tx"] / it["hop_ratio"]).clamp(min=1e-12))[m]
    out = torch.zeros(L).index_reduce_(0, it["path"][m], w_over_sbar, "amax", include_self=False)
    return torch.log1p(out)


# ----------------------------------------------------------------- batching
def collate_graph(items, prior=False):
    Ls = [it["link_x"].shape[0] for it in items]
    tot = sum(Ls); P = max(it["path"].shape[1] for it in items)
    A = torch.zeros(tot, tot); same = torch.zeros(tot, tot, dtype=torch.bool)
    paths, off = [], 0
    for it, L in zip(items, Ls):
        A[off:off + L, off:off + L] = it["A"]
        same[off:off + L, off:off + L] = True
        p = it["path"].clone(); p[p >= 0] += off
        paths.append(F.pad(p, (0, P - p.shape[1]), value=-1)); off += L
    x = torch.cat([it["link_x"] for it in items])
    flow = torch.cat([flow_table(it) for it in items])
    base = torch.cat([tx_sum(it) for it in items])
    if prior:
        x = torch.cat([x, torch.cat([link_wait(it) for it in items])[:, None]], 1)
        flow = torch.cat([flow, torch.cat([prior_table(it) for it in items])], 1)
        base = torch.cat([prior_sum(it) for it in items])
    return dict(x=x, A=A, same=same,
                path=torch.cat(paths), flow=flow,
                txs=base, y=torch.cat([it["y"] for it in items]),
                tos=torch.cat([it["tos"] for it in items]))


# ----------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, d_in, H=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))

    def forward(self, b):
        return b["txs"] * torch.exp(self.net(b["flow"]).squeeze(-1).clamp(-6, 8))


def _gcn_norm(A):
    A = A + torch.eye(A.shape[0])
    d = A.sum(1).rsqrt()
    return d[:, None] * A * d[None, :]


class GCNLayer(nn.Module):
    def __init__(self, H):
        super().__init__(); self.lin = nn.Linear(H, H)

    def forward(self, h, An, b):
        return h + F.relu(self.lin(An @ h))


class GATLayer(nn.Module):
    def __init__(self, H, heads=4):
        super().__init__()
        self.h, self.d = heads, H // heads
        self.W = nn.Linear(H, H, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.d) * 0.1); self.a_dst = nn.Parameter(torch.randn(heads, self.d) * 0.1)

    def forward(self, h, An, b):
        N = h.shape[0]
        z = self.W(h).view(N, self.h, self.d)
        e = (z * self.a_src).sum(-1)[:, None, :] + (z * self.a_dst).sum(-1)[None, :, :]     # (N,N,heads)
        e = F.leaky_relu(e, 0.2)
        mask = (b["A"] + torch.eye(N)) > 0
        e = e.masked_fill(~mask[..., None], float("-inf"))
        att = torch.softmax(e, dim=1)
        out = torch.einsum("ijh,jhd->ihd", att, z).reshape(N, -1)
        return h + F.elu(out)


class GPSLayer(nn.Module):
    def __init__(self, H, heads=4):
        super().__init__()
        self.local = nn.Linear(H, H)
        self.attn = nn.MultiheadAttention(H, heads, batch_first=True)
        self.n1, self.n2, self.n3 = nn.LayerNorm(H), nn.LayerNorm(H), nn.LayerNorm(H)
        self.ffn = nn.Sequential(nn.Linear(H, 2 * H), nn.ReLU(), nn.Linear(2 * H, H))

    def forward(self, h, An, b):
        loc = self.n1(h + F.relu(self.local(An @ h)))
        g, _ = self.attn(h[None], h[None], h[None], attn_mask=~b["same"])
        glob = self.n2(h + g[0])
        return self.n3(loc + glob + self.ffn(loc + glob))


class LinkGNN(nn.Module):
    def __init__(self, kind, d_flow, H=64, layers=4, d_link=LINK_F):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_link, H), nn.ReLU(), nn.Linear(H, H))
        Layer = {"gcn": GCNLayer, "gat": GATLayer, "gtrans": GPSLayer}[kind]
        self.layers = nn.ModuleList([Layer(H) for _ in range(layers)])
        self.read = nn.Sequential(nn.Linear(2 * H + d_flow, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, b):
        An = _gcn_norm(b["A"])
        h = self.enc(b["x"])
        for layer in self.layers:
            h = layer(h, An, b)
        p = b["path"]; m = (p >= 0).float()[..., None]
        hp = h[p.clamp(min=0)] * m
        s = hp.sum(1); mx = (hp + (m - 1) * 1e4).max(1).values
        r = self.read(torch.cat([s, mx, b["flow"]], 1)).squeeze(-1).clamp(-6, 8)
        return b["txs"] * torch.exp(r)
