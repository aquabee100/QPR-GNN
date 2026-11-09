from __future__ import annotations

import torch
import torch.nn as nn

N_TOS = 3


def _selu_dense(i, o):
    lin = nn.Linear(i, o)
    nn.init.kaiming_uniform_(lin.weight, a=0, mode="fan_in", nonlinearity="linear")
    nn.init.zeros_(lin.bias)
    return nn.Sequential(lin, nn.SELU())


class MSMPAttention(nn.Module):
    def __init__(self, z, H=16, T=12):
        super().__init__()
        self.z, self.H, self.T = z, H, T
        self.path_embedding = nn.Sequential(_selu_dense(4 + N_TOS, H), _selu_dense(H, H))
        self.link_embedding = nn.Sequential(_selu_dense(3, H), _selu_dense(H, H))
        self.attention = nn.Sequential(nn.Linear(H, H), nn.LeakyReLU(0.01))
        self.path_update = nn.GRUCell(H, H)
        self.link_update = nn.GRUCell(H, H)
        self.readout = nn.Sequential(_selu_dense(H, H // 2), _selu_dense(H // 2, H // 4),
                                     nn.Linear(H // 4, 1), nn.Softplus())

    def _z(self, x, k):
        m, s = self.z[k]; return (x - m) / s

    def forward(self, b):
        pl = b["path_links"]; mask = pl >= 0; pli = pl.clamp(min=0); P = pl.shape[1]
        pkt = b["traffic"] / b["packets"].clamp(min=1e-9)
        tos = b["tos"].long().clamp(0, N_TOS - 1)
        h_p = self.path_embedding(torch.cat([
            self._z(b["traffic"], "traffic")[:, None], self._z(b["packets"], "packets")[:, None],
            self._z(pkt, "pkt_size")[:, None], mask.sum(1, keepdim=True).float(),
            torch.nn.functional.one_hot(tos, N_TOS).float()], 1))
        L = b["cap"].shape[0]; sid = torch.zeros(L, dtype=torch.long)
        if "link_sample" in b:
            sid = b["link_sample"]
        mx = torch.zeros(int(sid.max()) + 1).scatter_reduce(0, sid, b["load"], "amax", include_self=False)
        h_l = self.link_embedding(torch.stack([self._z(b["cap"], "capacity"), b["load"],
                                               b["load"] / mx[sid].clamp(min=1e-9)], 1))
        for _ in range(self.T):
            prev = h_p; seq = [prev]; hp = h_p
            for k in range(P):
                new = self.path_update(h_l[pli[:, k]], hp)
                m = mask[:, k:k + 1].float()
                hp = m * new + (1 - m) * hp
                seq.append(hp)
            S = torch.stack(seq, 1)
            g = S[:, :P][mask]
            w = torch.softmax(self.attention(g), -1) * g
            agg = torch.zeros_like(h_l).index_add_(0, pli[mask], w)
            h_l = self.link_update(agg, h_l)
            h_p = hp
        occ = self.readout(S[:, 1:]).squeeze(-1)
        cap_pkts = b["cap"][pli] / self.z["pkt_size"][0]
        return (occ / cap_pkts * mask).sum(1)


def zstats_msmp(items):
    import numpy as np
    from qprgnn.fermi import zstats
    z = zstats(items)
    x = np.concatenate([it["traffic"] / np.maximum(it["packets"], 1e-9) for it in items])
    z["pkt_size"] = [float(x.mean()), float(x.std() + 1e-9)]
    return z


def collate_msmp(items):
    from qprgnn.fermi import collate_fermi
    b = collate_fermi(items)
    b["link_sample"] = torch.cat([torch.full((it["L"],), i, dtype=torch.long) for i, it in enumerate(items)])
    return b
