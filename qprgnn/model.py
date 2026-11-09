from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import LINK_F

WAIT = {"qosres_nosched": "wait_nosched", "qosres": "wait", "qosres_mm1": "wait_mm1"}
QUEUE = ("queueres", "queueres_pure")
READOUTS = list(WAIT) + ["mm1res", "tx_res", "fermi"] + list(QUEUE) + ["decoupled"]
MAXQ = 3


class QPRGNN(nn.Module):
    def __init__(self, H=32, T=4, readout="qosres_nosched", prior_feats=True, flow_in=5, hop_in=5):
        super().__init__()
        assert readout in READOUTS, readout
        self.readout, self.prior_feats, self.T, self.H = readout, prior_feats, T, H
        self.link_enc = nn.Sequential(nn.Linear(LINK_F, H), nn.ReLU(), nn.Linear(H, H))
        self.flow_enc = nn.Sequential(nn.Linear(flow_in, H), nn.ReLU(), nn.Linear(H, H))
        self.hop_enc = nn.Linear(hop_in + 2 * prior_feats, H)
        self.path_gru = nn.GRUCell(2 * H, H)
        self.link_gru = nn.GRUCell(2 * H, H)
        self.read = nn.Sequential(nn.Linear(3 * H, H), nn.ReLU(), nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))
        if readout == "decoupled":
            self.read[-1] = nn.Linear(H // 2, 2)
        nn.init.zeros_(self.read[-1].weight); nn.init.zeros_(self.read[-1].bias)
        if readout in QUEUE:
            self.qread = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))
            nn.init.zeros_(self.qread[-1].weight); nn.init.zeros_(self.qread[-1].bias)

    def forward(self, bt):
        h_l = self.link_enc(bt.link_x)
        h_f = self.flow_enc(bt.flow_x)
        mask = bt.path >= 0
        hx = bt.hop_x
        if self.prior_feats:
            m = mask.float()
            ex = torch.stack([torch.log1p(bt.wait_nosched / bt.tx.clamp(min=1e-12)) * m,
                              torch.log(bt.hop_ratio.clamp(min=1e-6)) * m], -1)
            hx = torch.cat([hx, ex], -1)
        hop = self.hop_enc(hx)
        pidx = bt.path.clamp(min=0)
        P = bt.path.shape[1]
        for _ in range(self.T):
            hf, msgs = h_f, []
            for k in range(P):
                new = self.path_gru(torch.cat([h_l[pidx[:, k]], hop[:, k]], 1), hf)
                m = mask[:, k:k + 1].float()
                hf = m * new + (1 - m) * hf
                msgs.append(hf)
            M = torch.stack(msgs, 1)
            agg = torch.zeros_like(h_l).index_add_(0, pidx[mask], M[mask])
            cnt = torch.zeros(h_l.shape[0], 1).index_add_(0, pidx[mask], torch.ones(int(mask.sum()), 1)).clamp(min=1)
            h_l = self.link_gru(torch.cat([agg, agg / cnt], 1), h_l)
            h_f = hf
        if self.readout in QUEUE:
            q = torch.round(bt.hop_x[..., 3] * 2).long().clamp(0, MAXQ - 1)
            gid = pidx * MAXQ + q
            G = h_l.shape[0] * MAXQ
            qs = torch.zeros(G, self.H).index_add_(0, gid[mask], M[mask])
            qn = torch.zeros(G, 1).index_add_(0, gid[mask], torch.ones(int(mask.sum()), 1)).clamp(min=1)
            qlink = h_l.repeat_interleave(MAXQ, 0)
            b = self.qread(torch.cat([qs / qn, qlink], 1)).squeeze(-1).clamp(-6, 6)[gid]
            wq = bt.wait_nosched * torch.exp(b)
            if self.readout == "queueres":
                a = self.read(torch.cat([M, h_l[pidx], hop], 2)).squeeze(-1).clamp(-6, 6)
                per_hop = bt.tx * torch.exp(a) + wq
            else:
                per_hop = bt.tx + wq
            return (per_hop * mask).sum(1)
        if self.readout == "decoupled":
            ab = self.read(torch.cat([M, h_l[pidx], hop], 2)).clamp(-6, 6)
            per_hop = bt.tx * torch.exp(ab[..., 0]) + bt.wait_nosched * torch.exp(ab[..., 1])
            return (per_hop * mask).sum(1)
        r = self.read(torch.cat([M, h_l[pidx], hop], 2)).squeeze(-1).clamp(-6, 6)
        if self.readout in WAIT:
            per_hop = (bt.tx + getattr(bt, WAIT[self.readout])) * torch.exp(r)
        elif self.readout == "mm1res":
            per_hop = bt.prior * torch.exp(r)
        elif self.readout == "tx_res":
            per_hop = bt.tx * torch.exp(r)
        else:
            per_hop = bt.tx * (1.0 + F.softplus(r))
        return (per_hop * mask).sum(1)
