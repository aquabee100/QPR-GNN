from __future__ import annotations

import os
import re
import tarfile
from functools import lru_cache

import networkx as nx
import numpy as np

POLICIES = {"FIFO": 0, "SP": 1, "WFQ": 2, "DRR": 3}
MAXQ = 3


def _read_gml(path: str) -> nx.MultiDiGraph:
    return nx.read_gml(path, destringizer=int, label="id")


@lru_cache(maxsize=4096)
def load_graph(path: str):
    G = _read_gml(path)
    N = G.number_of_nodes()
    links, cap, port = [], [], {}
    if G.is_directed():
        E = G.edges(keys=True, data=True) if G.is_multigraph() else ((u, v, 0, d) for u, v, d in G.edges(data=True))
        for u, v, k, d in sorted(E, key=lambda e: e[:3]):
            if k != 0:
                continue
            port[(u, d["port"])] = v
            links.append((u, v))
            cap.append(float(d["bandwidth"]))
    else:
        # BNNetSimulator input: undirected graph
        for u, v, d in G.edges(data=True):
            for a, b in ((u, v), (v, u)):
                links.append((a, b)); cap.append(float(d["bandwidth"]))
        order = sorted(range(len(links)), key=lambda i: links[i])
        links = [links[i] for i in order]; cap = [cap[i] for i in order]
    lidx = {e: i for i, e in enumerate(links)}

    policy = np.zeros(N, np.int64)
    weights = np.zeros((N, MAXQ), np.float32)
    buffers = np.zeros((N, MAXQ), np.float32)
    nq = np.ones(N, np.int64)
    tos2q = np.zeros((N, MAXQ), np.int64)
    for n in range(N):
        d = G.nodes[n]
        policy[n] = POLICIES[d.get("schedulingPolicy", "FIFO")]
        t = str(d.get("tosToQoSqueue", "0,1,2"))
        q = int(d.get("levelsQoS", len([g for g in t.split(";") if g.strip()])))
        nq[n] = q
        b = [float(x) for x in str(d.get("bufferSizes", "32000")).split(",")]
        if len(b) == 1:
            b = b * q  # scalar buffer applies to every queue
        buffers[n, : len(b)] = b[:MAXQ]
        w = str(d.get("schedulingWeights", "-"))
        if w not in ("-", ""):
            wv = np.array([float(x) for x in w.split(",")][:MAXQ])
            weights[n, : len(wv)] = wv / max(wv.sum(), 1e-9)
        elif q == 1:
            weights[n, 0] = 1.0
        # ';' separates queues, ',' separates ToS
        for qi, grp in enumerate(t.split(";")):
            for tos in grp.split(","):
                if tos.strip() != "":
                    tos2q[n, int(tos)] = min(qi, q - 1)
    node = dict(policy=policy, weights=weights, buffers=buffers, nq=nq, tos2q=tos2q)
    return G, links, np.array(cap, np.float64), lidx, port, node


@lru_cache(maxsize=256)
def load_routing(path: str, graph_path: str):
    G, _, _, _, port, _ = load_graph(graph_path)
    N = G.number_of_nodes()
    with open(path) as fd:
        rows = [l.strip().rstrip(",") for l in fd if l.strip()]
    is_matrix = len(rows) == N and all(len(r.split(",")) == N and "-1" in r.split(",") for r in rows)
    if not is_matrix:
        paths = {}
        for r in rows:
            p = [int(x) for x in r.replace(";", ",").split(",")]
            paths[(p[0], p[-1])] = p
        return paths
    R = np.array([[int(float(x)) for x in r.split(",")] for r in rows])
    paths = {}
    for s in range(N):
        for t in range(N):
            if s == t:
                continue
            node, p = s, [s]
            while R[node, t] != -1:
                node = port[(node, int(R[node, t]))]
                p.append(node)
                if len(p) > N + 1:
                    raise ValueError(f"routing loop {path} {s}->{t}")
            paths[(s, t)] = p
    return paths


def _resolve(root, rel, sub):
    cands = [os.path.join(root, sub, os.path.basename(rel)), os.path.join(root, rel)]
    up = root
    for _ in range(3):
        up = os.path.dirname(up)
        cands.append(os.path.join(up, rel))
        # simulator routings may lack an extension
        cands.append(os.path.join(up, sub, os.path.basename(rel) + ".txt"))
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"{rel} not found near {root}")


def _parse_sample(lines, root):
    res, tra, sta, inp, lnk = lines
    if ";OK;" not in sta:
        return None
    used = inp.strip().split(";")
    gpath = _resolve(root, used[1], "graphs")
    rpath = _resolve(root, used[2], "routings")
    if len(used) == 4 and "links_bw" in used[3]:
        raise NotImplementedError("links_bw override files not present in this dataset")
    G, links, cap, lidx, _, node = load_graph(gpath)
    paths = load_routing(rpath, gpath)
    N = G.number_of_nodes()

    r = res.strip().rstrip(";")
    r = r[r.find("|") + 1 :].split(";")
    t = tra.strip().rstrip(";")
    t = t[t.find("|") + 1 :].split(";")
    assert len(r) == N * N == len(t), (len(r), len(t), N)

    fs, fd, ftos, fbw, fpk, fpath, yd, yj, yl, feq = ([] for _ in range(10))
    for j in range(N * N):
        s, d = divmod(j, N)
        if s == d:
            continue
        a = [float(x) for x in r[j].split(",")]
        tt = t[j].split(",")
        if tt[0] == "-1" or a[1] <= 0:
            continue
        fs.append(s); fd.append(d); ftos.append(int(tt[-1]))
        fbw.append(a[0] * 1000.0); fpk.append(a[1])
        feq.append(float(tt[1]))  # configured EqLambda (traffic.txt field 1)
        yl.append(a[2]); yd.append(a[3]); yj.append(a[10])
        p = paths[(s, d)]
        fpath.append(np.array([lidx[(p[i], p[i + 1])] for i in range(len(p) - 1)], np.int64))

    L = len(links)
    lu = np.zeros(L, np.float32); qu = np.zeros((L, MAXQ), np.float32); qo = np.zeros((L, MAXQ), np.float32)
    if lnk:
        l = lnk.strip().split(";")
        for i in range(N):
            for k in range(N):
                ent = l[i * N + k].split(":")
                if ent[0].split(",")[0] == "-1":
                    continue
                e = lidx[(i, k)]
                lu[e] = float(ent[0].split(",")[0])
                for qi, qs in enumerate(ent[1 : 1 + MAXQ]):
                    v = qs.split(",")
                    qu[e, qi] = float(v[0]); qo[e, qi] = float(v[2])

    return dict(
        topo=os.path.basename(root), graph_file=used[1], routing_file=used[2],
        n_nodes=N, links=np.array(links, np.int64), capacity=cap,
        node_policy=node["policy"], node_weights=node["weights"], node_buffer=node["buffers"],
        node_nq=node["nq"], tos_to_queue=node["tos2q"],
        flow_src=np.array(fs), flow_dst=np.array(fd), flow_tos=np.array(ftos),
        flow_bw=np.array(fbw), flow_pkts=np.array(fpk), flow_eqlambda=np.array(feq), flow_path=fpath,
        y_delay=np.array(yd), y_jitter=np.array(yj), y_drop=np.array(yl),
        link_util=lu, queue_util=qu, queue_occ=qo,
    )


def iter_samples(topo_dir: str):
    files = sorted(f for f in os.listdir(topo_dir) if re.match(r"results_.*\.tar\.gz$", f))
    for f in files:
        with tarfile.open(os.path.join(topo_dir, f), "r:gz") as tar:
            names = tar.getnames()
            base = names[0].split("/")[0]
            def rd(n):
                p = f"{base}/{n}"
                return tar.extractfile(p).read().decode().splitlines() if p in names else None
            res, tra, sta, inp = rd("simulationResults.txt"), rd("traffic.txt"), rd("stability.txt"), rd("input_files.txt")
            lnk = rd("linkUsage.txt")
            if None in (res, tra, sta, inp):
                print(f"[parse] skipping incomplete archive {f}")
                continue
            counts = {"simulationResults": len(res), "traffic": len(tra), "stability": len(sta), "input_files": len(inp)}
            if lnk is not None:
                counts["linkUsage"] = len(lnk)
            if len(set(counts.values())) != 1:
                # report mismatches, keep common prefix
                print(f"[parse] WARNING line-count mismatch in {f}: {counts}")
            for i in range(min(counts.values())):
                s = _parse_sample((res[i], tra[i], sta[i], inp[i], lnk[i] if lnk is not None else None), topo_dir)
                if s is not None:
                    s["source_file"] = f
                    s["source_line"] = i
                    yield s
