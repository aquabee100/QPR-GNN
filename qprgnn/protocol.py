from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .data import file_split

ARCHIVE = ["train_geant", "test_abilene", "test_nobel", "test_germany50"]
TARGETS = ARCHIVE[1:]  # GEANT train/test archives are identical


def run_tag(model, holdout, seed):
    return f"{model}_short_{holdout}_s{seed}"


def add_args(ap: argparse.ArgumentParser, epochs=25, bs=8, lr=2e-3, cap=250):
    ap.add_argument("--holdout", required=True, help="held-out topology, or the fold label with --test_topos")
    ap.add_argument("--test_topos", default="", help="comma list held out together (simulator folds)")
    ap.add_argument("--universe", default="", help="comma list of cached topologies (default: the 4 archive ones)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=epochs)
    ap.add_argument("--bs", type=int, default=bs)
    ap.add_argument("--lr", type=float, default=lr)
    ap.add_argument("--cap_train", type=int, default=cap, help="max training samples per topology (0 = all)")
    ap.add_argument("--steps_per_epoch", type=int, default=0,
                    help="RouteNet-Fermi-style schedule: this many optimiser steps per epoch (0 = one pass)")
    ap.add_argument("--out", default="results/loto")
    return ap


def setup(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(int(os.environ.get("NTHREADS", "1")))
    os.makedirs(a.out, exist_ok=True)
    universe = [t for t in a.universe.split(",") if t] or ARCHIVE
    held_out = {t for t in a.test_topos.split(",") if t} or {a.holdout}
    return universe, held_out


def splits(universe, held_out, cap, load):
    train, val, tests = [], [], {}
    for n in universe:
        S, items = load(n)
        if n in held_out:
            tests[n] = items
            continue
        tr, va, te = file_split(S, seed=0)
        train += [items[i] for i in tr][: cap or None]
        val += [items[i] for i in va][: max(cap // 8, 20) if cap else None]
        tests[f"{n}(id)"] = [items[i] for i in te]
    return train, val, tests


def _y(b):
    return b.y if hasattr(b, "y") else b["y"]


def _tos(b):
    return b.tos if hasattr(b, "tos") else b["tos"]


def metrics(pred, y, tos):
    ape = (pred - y).abs() / y
    out = dict(mape=float(ape.mean()), medape=float(ape.median()), p90ape=float(ape.quantile(0.9)),
               logmae=float((torch.log(pred.clamp(min=1e-15)) - torch.log(y)).abs().mean()), n=int(len(y)))
    for t in range(3):
        m = tos == t
        if m.any():
            out[f"mape_tos{t}"] = float(ape[m].mean())
    return out


@torch.no_grad()
def predict(model, items, collate, bs=8):
    model.eval(); P, Y, T = [], [], []
    for i in range(0, len(items), bs):
        b = collate(items[i:i + bs]); P.append(model(b)); Y.append(_y(b)); T.append(_tos(b))
    return torch.cat(P), torch.cat(Y), torch.cat(T)


def fit(model, train, val, collate, loss_fn, a, tag, cosine=False, val_bs=8):
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs) if cosine else None
    rng = np.random.default_rng(a.seed)
    best, best_state, hist, stream, t0 = 1e9, None, [], [], time.time()

    def batches():
        if not a.steps_per_epoch:
            order = rng.permutation(len(train))
            for i in range(0, len(order), a.bs):
                yield [train[j] for j in order[i:i + a.bs]]
            return
        for _ in range(a.steps_per_epoch):          # endless shuffled stream (tf: shuffle + repeat)
            while len(stream) < a.bs:
                stream.extend(rng.permutation(len(train)).tolist())
            yield [train[stream.pop(0)] for _ in range(a.bs)]

    for ep in range(a.epochs):
        model.train(); tot = n = 0
        for bl in batches():
            b = collate(bl)
            loss = loss_fn(model(b), _y(b))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss.detach()) * len(_y(b)); n += len(_y(b))
        if sched:
            sched.step()
        v = metrics(*predict(model, val, collate, val_bs))["mape"]
        hist.append(dict(ep=ep, train=tot / n, val=v, t=time.time() - t0))
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
        print(f"[{tag}] ep {ep} train {tot / n:.4f} val {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return dict(best_val=best, history=hist, train_time_s=time.time() - t0)


def log_ratio_loss(pred, y):
    return (torch.log(pred) - torch.log(y)).abs().mean()


def mape_loss(pred, y):
    return ((pred - y).abs() / y).mean()


def save_run(a, tag, info, tests, model, collate, pred_bs=8):
    info["results"] = {k: metrics(*predict(model, it, collate, pred_bs)) for k, it in tests.items()}
    info.update(holdout=a.holdout, seed=a.seed, params=sum(p.numel() for p in model.parameters()), args=vars(a))
    json.dump(info, open(f"{a.out}/{tag}.json", "w"), indent=1)
    torch.save(model.state_dict(), f"{a.out}/{tag}.pt")
    print(tag, {k: round(v["mape"] * 100, 2) for k, v in info["results"].items()})
