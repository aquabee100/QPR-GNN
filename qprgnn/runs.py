from __future__ import annotations

import glob
import json
import os
import pickle

import numpy as np
import torch

from .baselines import MLP, LinkGNN, collate_graph, flow_table, prior_sum, prior_table, tx_sum
from .data import LINK_F, collate, load_topology, valid_samples
from .fermi import RouteNetFermi, collate_fermi, sample_to_fermi
from .model import QPRGNN
from .qtroutenet import QTRouteNet, collate_qt, sample_to_qt
from .queueing import mg1_flow_delay


def _cached(path, build):
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    out = build()
    pickle.dump(out, open(path, "wb"), protocol=4)
    return out


def fermi_data(name, cache_dir="cache"):
    def build():
        S = valid_samples(name, cache_dir); return S, [sample_to_fermi(s) for s in S]
    return _cached(f"{cache_dir}/fermi_{name}.pkl", build)


def qt_data(name, cache_dir="cache"):
    def build():
        S = valid_samples(name, cache_dir); return S, [sample_to_qt(s) for s in S]
    return _cached(f"{cache_dir}/qt_{name}.pkl", build)


class Topology:

    def __init__(self, name, cache_dir="cache"):
        self.name, self.cache_dir = name, cache_dir
        self.items, self.S = load_topology(name, cache_dir)
        self.y = torch.cat([it["y"] for it in self.items]).numpy()
        self.flows = np.array([it["F"] for it in self.items])
        self._fermi = self._qt = None

    @property
    def fermi(self):
        if self._fermi is None:
            self._fermi = fermi_data(self.name, self.cache_dir)[1]; assert len(self._fermi) == len(self.items)
        return self._fermi

    @property
    def qt(self):
        if self._qt is None:
            self._qt = qt_data(self.name, self.cache_dir)[1]; assert len(self._qt) == len(self.items)
        return self._qt


def file_key(kind, key):
    return f"bl_{key}" if kind == "baseline" else key


def run_files(kind, key, run_dir, holdout):
    return sorted(glob.glob(f"{run_dir}/{file_key(kind, key)}_short_{holdout}_s*.json"))


def _batched(model, items, collate_fn, bs):
    return torch.cat([model(collate_fn(items[i:i + bs])) for i in range(0, len(items), bs)])


def load_model(kind, key, json_path, item=None):
    d = json.load(open(json_path)); stem = json_path[:-5]
    if kind == "qprgnn":
        assert d.get("model_variant", "routenet") == "routenet", json_path
        m = QPRGNN(H=d.get("H", 32), T=d.get("T", 4), readout=d.get("readout") or "mm1res",
                   prior_feats=bool(d.get("prior_feats", False)))
        sd = torch.load(stem + ".pt"); sd.pop("ystats", None)
        m.load_state_dict(sd)
    elif kind == "baseline":
        if key.startswith("gbdt"):
            return d, pickle.load(open(stem + ".pkl", "rb"))
        prior = key.endswith("_p")
        d_flow = flow_table(item).shape[1] + (prior_table(item).shape[1] if prior else 0)
        m = MLP(d_flow) if key.startswith("mlp") else LinkGNN(key.replace("_p", ""), d_flow, d_link=LINK_F + prior)
        m.load_state_dict(torch.load(stem + ".pt"))
    elif kind == "fermi":
        assert d.get("fermi_version") == 2, f"{json_path}: superseded RouteNet-Fermi port"
        m = RouteNetFermi(d["z"]); m.load_state_dict(torch.load(stem + ".pt"))
    elif kind == "qt":
        m = QTRouteNet(d["z"], tx_term=d.get("tx_term", False)); m.load_state_dict(torch.load(stem + ".pt"))
    else:
        raise ValueError(kind)
    m.eval()
    return d, m


@torch.no_grad()
def predict(kind, key, topo: Topology, run_dir="results/loto", holdout="*"):
    items = topo.items
    if kind == "analytic":
        if key == "an_mm1":
            p = torch.cat([(it["prior"] * (it["path"] >= 0)).sum(1) for it in items])
        elif key == "an_mm1k":
            p = torch.cat([prior_sum(it) for it in items])
        else:
            p = torch.tensor(np.concatenate([mg1_flow_delay(s) for s in topo.S]), dtype=torch.float32)
        yield key, p.numpy().astype(np.float32)
        return
    for j in run_files(kind, key, run_dir, holdout):
        d, m = load_model(kind, key, j, items[0])
        stem = os.path.basename(j)[:-5]
        if kind == "qprgnn":
            p = _batched(m, items, collate, 8)
        elif kind == "baseline":
            prior = key.endswith("_p")
            if key.startswith("gbdt"):
                feats = torch.cat([torch.cat([flow_table(it), prior_table(it)], 1) if prior else flow_table(it) for it in items])
                base = torch.cat([prior_sum(it) if prior else tx_sum(it) for it in items])
                p = torch.tensor(np.exp(m.predict(feats.numpy())), dtype=torch.float32) * base
            else:
                p = _batched(m, items, lambda b: collate_graph(b, prior=prior), 8)
        elif kind == "fermi":
            p = _batched(m, topo.fermi, collate_fermi, 8)
        else:
            p = _batched(m, topo.qt, collate_qt, 16)
        yield stem, p.numpy().astype(np.float32)


def seed_of(stem):
    return int(stem.rsplit("_s", 1)[1]) if "_s" in stem and stem.rsplit("_s", 1)[1].isdigit() else 0
