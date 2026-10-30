import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx
import numpy as np
import pandas as pd


CAPS = [10_000, 25_000, 40_000, 100_000, 250_000, 400_000, 1_000_000, 2_500_000, 4_000_000, 10_000_000]
BUFFERS = [8000, 16000, 32000, 64000]
POLICIES = ["FIFO", "SP", "WFQ", "DRR"]
TOS_P = [0.05, 0.05, 0.90]          # matches the RouteNet-Fermi real-trace ToS mix
PKT_DISTS = ["0,300,0.5,1700,0.5", "0,500,0.6,1000,0.2,1400,0.2", "0,1000,1"]


def load_zoo(path):
    G = nx.Graph(nx.read_graphml(path))
    G.remove_edges_from(nx.selfloop_edges(G))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    order = sorted(G.nodes, key=str)
    labels = {i: str(G.nodes[n].get("label", n)) for i, n in enumerate(order)}
    H = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return nx.Graph(H.edges()), labels


def select_topologies(zoo_csv, per_bin=2, min_n=10, max_n=45):
    g = pd.read_csv(zoo_csv)
    c = g[(g.N >= min_n) & (g.N <= max_n) & (g.components == 1)]
    c = c[~c.name.str.match(r".*\d{4,6}$")]          # snapshots handled separately
    picks = []
    for b, grp in c.groupby("b1_short"):
        grp = grp.sort_values("N")
        idx = np.linspace(0, len(grp) - 1, min(per_bin, len(grp))).round().astype(int)
        picks += grp.iloc[sorted(set(idx))].name.tolist()
    return picks


def node_config(rng):
    pol = rng.choice(POLICIES)
    buf = int(rng.choice(BUFFERS))
    d = {"schedulingPolicy": pol}
    if pol == "FIFO":
        d["tosToQoSqueue"] = "0,1,2"; k = 1
    else:
        if rng.random() < 0.5:
            d["tosToQoSqueue"] = "0;1;2"; k = 3
        else:
            d["tosToQoSqueue"] = "0;1,2"; k = 2
    # one integer buffer per node
    d["bufferSizes"] = buf
    if pol != "FIFO":
        if pol in ("WFQ", "DRR"):
            w = rng.dirichlet(np.ones(k) * 2) * 100
            w = np.maximum(np.round(w).astype(int), 5)
            w[-1] = 100 - w[:-1].sum()
            if w[-1] < 5:
                w = np.array([60, 40] if k == 2 else [50, 30, 20])
            d["schedulingWeights"] = ",".join(map(str, w))
    return d


def routings(G, n, rng):
    out = []
    for r in range(n):
        for u, v in G.edges:
            G[u][v]["w"] = 1.0 + (0.0 if r == 0 else rng.random() * 0.9)   # r=0: plain hop-count SP
        P = dict(nx.all_pairs_dijkstra_path(G, weight="w"))
        out.append({(s, t): P[s][t] for s in G for t in G if s != t})
    return out


def make_tm(G, rng, mean_bw, mass=None):
    N = G.number_of_nodes()
    if mass is None:
        mass = rng.lognormal(0, 1, N)
    D = np.outer(mass, mass); np.fill_diagonal(D, 0)
    D = D / D[D > 0].mean() * mean_bw
    D = np.maximum(D * rng.lognormal(0, 0.3, D.shape), 10)          # simulator minimum: 10 bps
    flows = []
    for s in range(N):
        for t in range(N):
            if s == t:
                continue
            tos = int(rng.choice(3, p=TOS_P))
            td = "0" if rng.random() < 0.7 else f"2,{rng.uniform(2, 10):.1f},{rng.uniform(2, 10):.1f}"
            flows.append((s, t, int(D[s, t]), td, str(rng.choice(PKT_DISTS)), tos))
    return flows


# declared load strata
STRATA = {"moderate": (0.10, 0.60, None), "near_saturation": (0.10, 0.60, (0.70, 0.95)),
          "overload": (0.10, 0.60, (1.00, 1.20))}
STRATA_P = [0.5, 0.3, 0.2]


def link_loads(flows, paths):
    load = {}
    for s, t, bw, *_ in flows:
        p = paths[(s, t)]
        for a, b in zip(p[:-1], p[1:]):
            load[(a, b)] = load.get((a, b), 0) + bw
    return load


def size_capacities(G, flows, paths, rng, stratum):
    load = link_loads(flows, paths)
    lo, hi, stress = STRATA[stratum]
    need = {(u, v): max(load.get((u, v), 0), load.get((v, u), 0), 1) for u, v in G.edges}
    hot = set()
    if stress:
        k = max(1, int(round(0.1 * len(need))))
        hot = set(sorted(need, key=need.get, reverse=True)[:k])
    caps = {}
    for e, n in need.items():
        target = rng.uniform(*stress) if e in hot else rng.uniform(lo, hi)
        caps[e] = sim_cap(n / target)
    return caps


def fixed_capacities(G, flows, paths):
    load = link_loads(flows, paths)
    return {(u, v): sim_cap(max(load.get((u, v), 0), load.get((v, u), 0), 1) / 0.4) for u, v in G.edges}


def sim_cap(c):
    return int(max(10_000, 1000 * round(c / 1000)))


def write_dataset(name, G, labels, out, n_samples, n_routings, rng, meta):
    root = os.path.join(out, name)
    for d in ("graphs", "routings", "tm"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    R = routings(G, n_routings, rng)
    for r, paths in enumerate(R):
        with open(os.path.join(root, "routings", f"routing_{r}.txt"), "w") as fd:
            for s in sorted(G):
                for t in sorted(G):
                    if s != t:
                        fd.write(",".join(map(str, paths[(s, t)])) + "\n")
    lines, plan = [], []
    fixed = None
    for i in range(n_samples):
        r = i % n_routings
        # first quarter: fixed capacities, scaled traffic
        if i < n_samples // 4:
            if fixed is None:
                mass0 = np.random.default_rng(12345).lognormal(0, 1, G.number_of_nodes())
                fixed = fixed_capacities(G, make_tm(G, np.random.default_rng(1), 1000.0, mass0), R[0])
            r = 0
            mean_bw = 1000.0 * float(rng.uniform(0.5, 2.5)); mode, stratum = "fixed_capacity", "traffic_scaled"
            flows = make_tm(G, rng, mean_bw, mass0); caps = fixed
        else:
            mean_bw = float(rng.choice([200, 500, 1000, 2000]))
            stratum = str(rng.choice(list(STRATA), p=STRATA_P)); mode = "sized"
            flows = make_tm(G, rng, mean_bw)
            caps = size_capacities(G, flows, R[r], rng, stratum)
        ld = link_loads(flows, R[r])
        util = [ld.get((u, v), 0) / caps[(u, v)] for u, v in G.edges] + [ld.get((v, u), 0) / caps[(u, v)] for u, v in G.edges]
        plan.append(dict(sample=i, routing=r, capacity_mode=mode, stratum=stratum, mean_bw=mean_bw,
                         max_util=float(max(util)), median_util=float(np.median(util))))
        H = nx.Graph()
        H.graph["levelsToS"] = 3
        for n in sorted(G):
            H.add_node(n, **node_config(rng))
        for (u, v), c in caps.items():
            H.add_edge(u, v, bandwidth=int(c))
        nx.write_gml(H, os.path.join(root, "graphs", f"graph_{i}.txt"))
        with open(os.path.join(root, "tm", f"tm_{i}.txt"), "w") as fd:
            for s, t, bw, td, sd, tos in flows:
                fd.write(f"{s},{t},{bw},{td},{sd},{tos}\n")
        lines.append(f"graphs/graph_{i}.txt,routings/routing_{r}.txt,tm/tm_{i}.txt\n")
    with open(os.path.join(root, "simulation.txt"), "w") as fd:
        fd.writelines(lines)
    pd.DataFrame(plan).to_csv(os.path.join(root, "samples.csv"), index=False)   # declared stratum per sample
    with open(os.path.join(root, "conf.yml"), "w") as fd:
        fd.write(f"threads: 6\ndataset_name: {name}\nsamples_per_file: 10\nrm_prev_results: n\nwrite_pkt_info: n\n")
    meta.update(dict(name=name, N=G.number_of_nodes(), E=G.number_of_edges(),
                     b1_graph=G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G),
                     samples=n_samples, routings=n_routings, labels=labels))
    json.dump(meta, open(os.path.join(root, "manifest.json"), "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", default="data_zoo")
    ap.add_argument("--zoo_csv", default="results/zoo_graphs.csv")
    ap.add_argument("--out", default="sim_inputs")
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--routings", type=int, default=4)
    ap.add_argument("--per_bin", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    zoo = pd.read_csv(a.zoo_csv).set_index("name")
    for kind, names in [("cross", select_topologies(a.zoo_csv, a.per_bin))]:
        for z in names:
            G, labels = load_zoo(os.path.join(a.zoo, z + ".graphml"))
            m = write_dataset(z, G, labels, a.out, a.samples, a.routings, rng,
                              dict(kind=kind, zoo=z, b1_short=int(zoo.loc[z, "b1_short"])))
            rows.append({k: v for k, v in m.items() if k != "labels"})
            print(f"{kind:8s} {z:28s} N={m['N']:3d} E={m['E']:3d} b1={m['b1_graph']:3d} b1_short={m['b1_short']:2d}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "topologies.csv"), index=False)
    with open(os.path.join(a.out, "run_all.sh"), "w") as fd:
        fd.write("#!/bin/bash\n# Run BNNetSimulator on every generated topology (needs Docker).\n"
                 "# Usage: ./run_all.sh [topology ...]   (default: all)\n"
                 "cd \"$(dirname \"$0\")\"\n"
                 f"ALL=\"{' '.join(r['name'] for r in rows)}\"   \n"
                 "topos=\"$@\"; [ -z \"$topos\" ] && topos=$ALL\n"
                 "count() { for f in \"$1\"/results/\"$1\"/*.tar.gz; do [ -f \"$f\" ] && tar -xzOf \"$f\" --wildcards '*/simulationResults.txt' | wc -l; done | awk '{s+=$1} END {print s+0}'; }\n"
                 "for t in $topos; do\n"
                 "  want=$(wc -l < \"$t/simulation.txt\")\n"
                 "  # skip only a topology whose DONE marker exists AND whose results hold every declared sample\n"
                 "  if [ -f \"$t/DONE\" ] && [ \"$(count $t)\" -ge \"$want\" ]; then echo \"skip $t (complete)\"; continue; fi\n"
                 "  rm -f \"$t/DONE\"; echo \"== $t  $(date)\" | tee -a \"$t/out.log\"\n"
                 "  docker run --rm --mount type=bind,src=\"$(pwd)/$t\",dst=/data bnnupc/bnnetsimulator 2>&1 | tee -a \"$t/out.log\"\n"
                 "  rc=${PIPESTATUS[0]}; got=$(count $t)\n"
                 "  echo \"$t: exit $rc, $got / $want samples\" | tee -a \"$t/out.log\" status.log\n"
                 "  [ \"$rc\" -eq 0 ] && [ \"$got\" -ge \"$want\" ] && touch \"$t/DONE\" || echo \"INCOMPLETE: $t (kept log)\" | tee -a status.log\n"
                 "done\n")
    os.chmod(os.path.join(a.out, "run_all.sh"), 0o755)
    # no-Docker variant with proot
    with open(os.path.join(a.out, "run_proot.sh"), "w") as fd:
        fd.write("#!/bin/bash\n# Run BNNetSimulator without Docker: unpacked image (ROOTFS) + proot.\n"
                 "# Usage: THREADS=8 ROOTFS=../bnnetsim_root PROOT=~/bin/proot ./run_proot.sh [topology ...]\n"
                 "cd \"$(dirname \"$0\")\"\n"
                 ": ${THREADS:=8} ${ROOTFS:=../bnnetsim_root} ${PROOT:=$HOME/bin/proot}\n"
                 f"ALL=\"{' '.join(r['name'] for r in rows)}\"\n"
                 "topos=\"$@\"; [ -z \"$topos\" ] && topos=$ALL\n"
                 "count() { for f in \"$1\"/results/\"$1\"/*.tar.gz; do [ -f \"$f\" ] && tar -xzOf \"$f\" --wildcards '*/simulationResults.txt' | wc -l; done | awk '{s+=$1} END {print s+0}'; }\n"
                 "for t in $topos; do\n"
                 "  want=$(wc -l < \"$t/simulation.txt\")\n"
                 "  if [ -f \"$t/DONE\" ] && [ \"$(count $t)\" -ge \"$want\" ]; then echo \"skip $t (complete)\"; continue; fi\n"
                 "  rm -f \"$t/DONE\"; sed -i \"s/^threads:.*/threads: $THREADS/\" \"$t/conf.yml\"\n"
                 "  echo \"== $t  $(date)\" | tee -a status.log\n"
                 "  rm -rf \"$ROOTFS\"/usr/src/NetSimulator/data.* \"$t/results\" \"$t/tmp\"   # leftovers of an earlier run\n"
                 "  start=$(date +%s)\n"
                 "  \"$PROOT\" -R \"$ROOTFS\" -b \"$(pwd)/$t:/data\" -w /usr/src/NetSimulator /bin/sh -c 'python3 /usr/src/NetSimulator/Simulate.py' > \"$t/run_stdout.log\" 2>&1\n"
                 "  rc=$?; got=$(count $t)\n"
                 "  echo \"$t: exit $rc, $got / $want samples, $(( $(date +%s) - start )) s\" | tee -a status.log\n"
                 "  [ \"$rc\" -eq 0 ] && [ \"$got\" -ge \"$want\" ] && touch \"$t/DONE\" || echo \"INCOMPLETE: $t (see $t/out.log, $t/run_stdout.log)\" | tee -a status.log\n"
                 "done\n")
    os.chmod(os.path.join(a.out, "run_proot.sh"), 0o755)
    print(f"{len(rows)} topologies, {sum(r['samples'] for r in rows)} samples -> {a.out}/")


if __name__ == "__main__":
    main()
