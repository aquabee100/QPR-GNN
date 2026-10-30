import argparse, json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from qprgnn.baselines import MLP, LinkGNN, collate_graph, flow_table, prior_sum, prior_table, tx_sum
from qprgnn.data import LINK_F, load_topology
from qprgnn import protocol as P


def main():
    ap = P.add_args(argparse.ArgumentParser(), epochs=25, bs=8, lr=2e-3)
    ap.add_argument("--model", required=True, choices=["mlp", "gbdt", "gcn", "gat", "gtrans", "mlp_p", "gbdt_p", "gat_p"])
    a = ap.parse_args()
    universe, held_out = P.setup(a)
    train, val, tests = P.splits(universe, held_out, a.cap_train, lambda n: load_topology(n)[::-1])
    tag = P.run_tag(f"bl_{a.model}", a.holdout, a.seed)
    kind, prior = a.model.replace("_p", ""), a.model.endswith("_p")
    feats = lambda it: torch.cat([flow_table(it), prior_table(it)], 1) if prior else flow_table(it)
    base = lambda it: prior_sum(it) if prior else tx_sum(it)
    info = dict(variant=f"bl_{a.model}", model_variant=a.model)

    if kind == "gbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor
        t0, hist = time.time(), []
        X = torch.cat([feats(it) for it in train]).numpy()
        T = torch.cat([torch.log(it["y"] / base(it)) for it in train]).numpy()
        Xv = torch.cat([feats(it) for it in val]).numpy()
        Tv = torch.cat([torch.log(it["y"] / base(it)) for it in val]).numpy()
        best, model = 1e9, None
        for lr in (0.05, 0.1):                                   # tiny grid, selected on validation
            m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=lr, max_iter=600,
                                              max_leaf_nodes=63, early_stopping=False, random_state=a.seed)
            m.fit(X, T)
            v = float(np.mean(np.abs(np.exp(m.predict(Xv) - Tv) - 1)))
            hist.append(dict(lr=lr, val=v))
            if v < best:
                best, model = v, m
        pickle.dump(model, open(f"{a.out}/{tag}.pkl", "wb"))
        pred = lambda items: torch.tensor(np.exp(model.predict(torch.cat([feats(it) for it in items]).numpy())),
                                          dtype=torch.float32) * torch.cat([base(it) for it in items])
        info.update(best_val=best, history=hist, train_time_s=time.time() - t0, holdout=a.holdout, seed=a.seed,
                    params=int(sum(len(p.nodes) for ps in model._predictors for p in ps)), args=vars(a),
                    results={k: P.metrics(pred(it), torch.cat([x["y"] for x in it]), torch.cat([x["tos"] for x in it]))
                             for k, it in tests.items()})
        json.dump(info, open(f"{a.out}/{tag}.json", "w"), indent=1)
        print(tag, {k: round(v["mape"] * 100, 2) for k, v in info["results"].items()})
        return

    collate = lambda items: collate_graph(items, prior=prior)
    d_flow = feats(train[0]).shape[1]
    model = MLP(d_flow) if kind == "mlp" else LinkGNN(kind, d_flow, d_link=LINK_F + int(prior))
    info.update(P.fit(model, train, val, collate, P.log_ratio_loss, a, tag, cosine=True, val_bs=a.bs))
    P.save_run(a, tag, info, tests, model, collate, pred_bs=a.bs)


if __name__ == "__main__":
    main()
