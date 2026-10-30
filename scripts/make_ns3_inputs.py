import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networkx as nx
import numpy as np
import pandas as pd

import make_sim_inputs as M

SCALE = 1000.0
BUFFER_BYTES = [8_000, 16_000, 32_000, 64_000]
PKT_MIX = {"0,300,0.5,1700,0.5": [(300, .5), (1700, .5)], "0,500,0.6,1000,0.2,1400,0.2": [(500, .6), (1000, .2), (1400, .2)],
           "0,1000,1": [(1000, 1.0)]}
TARGET_PKTS, T_MIN, T_MAX = 400, 5.0, 60.0


def routings(G, n, rng):
    out = []
    for r in range(n):
        for u, v in G.edges:
            G[u][v]["w"] = 1.0 + (0.0 if r == 0 else rng.random() * 0.9)
        nxt = {}
        for d in sorted(G):
            pred, _ = nx.dijkstra_predecessor_and_distance(G, d, weight="w")
            for v in sorted(G):
                if v != d:
                    nxt[(v, d)] = min(pred[v])           # deterministic tie-break
        paths = {}
        for s in G:
            for d in G:
                if s != d:
                    p = [s]
                    while p[-1] != d:
                        p.append(nxt[(p[-1], d)])
                    paths[(s, d)] = p
        out.append((nxt, paths))
    return out


def node_config(rng):
    sp = rng.random() < 0.5
    buf = int(rng.choice(BUFFER_BYTES))
    if not sp:
        return 0, 1, buf, (0, 0, 0)
    return (1, 3, buf, (0, 1, 2)) if rng.random() < 0.5 else (1, 2, buf, (0, 1, 1))


def write_sample(path, G, caps, nodes, nxt, flows, rng, seed):
    N = G.number_of_nodes()
    lam, lines = [], []
    for i, (s, t, bw, td, sd, tos) in enumerate(flows):
        mix = PKT_MIX[sd]; mean_size = sum(a * p for a, p in mix)
        rate = bw * SCALE / (8 * mean_size)                      # packets/s (time-average)
        lam.append(rate)
        if td == "0":
            on, off = 0.0, 0.0
        else:
            on, off = float(rng.uniform(0.2, 1.0)), float(rng.uniform(0.2, 1.0))
        lines.append(f"{i} {s} {t} {tos} {rate:.6f} {int(td != '0')} {on:.4f} {off:.4f} {len(mix)} "
                     + " ".join(f"{a} {p}" for a, p in mix))
    T = float(np.clip(TARGET_PKTS / np.median(lam), T_MIN, T_MAX))
    with open(path, "w") as fd:
        fd.write("# qpr-ns3 sample v1\n")
        fd.write(f"nodes {N}\nlinks {G.number_of_edges()}\n")
        for (u, v) in sorted(caps):
            fd.write(f"{u} {v} {int(caps[(u, v)] * SCALE)}\n")
        for n in range(N):
            pol, nq, buf, q = nodes[n]
            fd.write(f"node {n} {pol} {nq} {buf} {q[0]} {q[1]} {q[2]}\n")
        fd.write(f"routes {len(nxt)}\n")
        for (v, d), h in sorted(nxt.items()):
            fd.write(f"{v} {d} {h}\n")
        fd.write(f"flows {len(flows)}\n" + "\n".join(lines) + "\n")
        fd.write(f"sim {T:.3f} {0.1 * T:.3f} {seed}\n")
    return T


def make_topology(name, G, labels, out, n_samples, n_routings, rng, seed0):
    root = os.path.join(out, name); os.makedirs(root, exist_ok=True)
    R = routings(G, n_routings, rng)
    plan, fixed = [], None
    for i in range(n_samples):
        r = i % n_routings
        if i < n_samples // 4:                                    # fixed-capacity family, traffic scaled
            if fixed is None:
                mass0 = np.random.default_rng(12345).lognormal(0, 1, G.number_of_nodes())
                fixed = M.fixed_capacities(G, M.make_tm(G, np.random.default_rng(1), 1000.0, mass0), R[0][1])
            r = 0
            mean_bw = 1000.0 * float(rng.uniform(0.5, 2.5)); mode, stratum = "fixed_capacity", "traffic_scaled"
            flows = M.make_tm(G, rng, mean_bw, mass0); caps = fixed
        else:
            mean_bw = float(rng.choice([200, 500, 1000, 2000]))
            stratum = str(rng.choice(list(M.STRATA), p=M.STRATA_P)); mode = "sized"
            flows = M.make_tm(G, rng, mean_bw)
            caps = M.size_capacities(G, flows, R[r][1], rng, stratum)
        ld = M.link_loads(flows, R[r][1])
        util = [ld.get((u, v), 0) / caps[(u, v)] for u, v in G.edges] + [ld.get((v, u), 0) / caps[(u, v)] for u, v in G.edges]
        nodes = [node_config(rng) for _ in range(G.number_of_nodes())]
        T = write_sample(os.path.join(root, f"sample_{i}.txt"), G, caps, nodes, R[r][0], flows, rng, seed0 + i)
        plan.append(dict(sample=i, routing=r, capacity_mode=mode, stratum=stratum, mean_bw=mean_bw,
                         max_util=float(max(util)), median_util=float(np.median(util)), sim_time=T))
    pd.DataFrame(plan).to_csv(os.path.join(root, "samples.csv"), index=False)
    json.dump(dict(name=name, N=G.number_of_nodes(), E=G.number_of_edges(), samples=n_samples,
                   routings=n_routings, labels=labels, scale=SCALE), open(os.path.join(root, "manifest.json"), "w"), indent=1)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", default="data_zoo")
    ap.add_argument("--topologies", default="results/sim_topologies.csv")
    ap.add_argument("--out", default="ns3_inputs")
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--routings", type=int, default=4)
    ap.add_argument("--only", default="", help="comma list of topologies (default: all 16)")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    names = pd.read_csv(a.topologies).query("kind == 'cross'").name.tolist()
    only = [t for t in a.only.split(",") if t]
    for k, z in enumerate(names):
        G, labels = M.load_zoo(os.path.join(a.zoo, z + ".graphml"))
        sub = np.random.default_rng([a.seed, k])                 # per-topology stream: --only gives the same files
        if only and z not in only:
            continue
        plan = make_topology(z, G, labels, a.out, a.samples, a.routings, sub, 1000 * (k + 1))
        T = [p["sim_time"] for p in plan]
        print(f"{z:16s} N={G.number_of_nodes():3d} E={G.number_of_edges():3d} samples={len(plan)} "
              f"sim time median {np.median(T):.1f}s max {max(T):.1f}s")


if __name__ == "__main__":
    main()
