from __future__ import annotations

import glob
import os
import re

import numpy as np

MAXQ = 3


def read_input(path):
    tok = open(path).read().split("\n", 1)[1].split()
    it = iter(tok)
    nxt = lambda: next(it)
    assert nxt() == "nodes"; N = int(nxt())
    assert nxt() == "links"; L = int(nxt())
    edges = [(int(nxt()), int(nxt()), float(nxt())) for _ in range(L)]
    nodes = {}
    for _ in range(N):
        assert nxt() == "node"
        n, pol, nq, buf, q0, q1, q2 = (int(nxt()) for _ in range(7))
        nodes[n] = (pol, nq, buf, (q0, q1, q2))
    assert nxt() == "routes"; R = int(nxt())
    nexthop = {}
    for _ in range(R):
        v, d, h = int(nxt()), int(nxt()), int(nxt())
        nexthop[(v, d)] = h
    assert nxt() == "flows"; F = int(nxt())
    flows = []
    for _ in range(F):
        fid, s, d, tos = int(nxt()), int(nxt()), int(nxt()), int(nxt())
        rate = float(nxt()); onoff = int(nxt()); on, off = float(nxt()), float(nxt()); k = int(nxt())
        mix = [(int(nxt()), float(nxt())) for _ in range(k)]
        flows.append(dict(id=fid, src=s, dst=d, tos=tos, rate=rate, onoff=onoff, on=on, off=off, mix=mix))
    assert nxt() == "sim"; T, warm, seed = float(nxt()), float(nxt()), int(nxt())
    return dict(N=N, edges=edges, nodes=nodes, nexthop=nexthop, flows=flows, T=T, warm=warm, seed=seed)


def read_output(path):
    lines = open(path).read().splitlines()
    window = float(lines[0].split()[2])
    a = np.array([[float(x) for x in l.split()] for l in lines[1:]])
    return window, a


def to_sample(inp, outp, topo, i):
    window, a = read_output(outp)
    N = inp["N"]
    links, cap, lidx = [], [], {}
    for u, v, c in inp["edges"]:
        for x, y in ((u, v), (v, u)):
            lidx[(x, y)] = len(links); links.append((x, y)); cap.append(c)
    pol = np.zeros(N, np.int64); nq = np.ones(N, np.int64); buf = np.zeros((N, MAXQ)); t2q = np.zeros((N, MAXQ), np.int64)
    for n, (p, k, b, q) in inp["nodes"].items():
        pol[n] = p; nq[n] = k; buf[n, :k] = 8.0 * b; t2q[n] = q
    F = len(inp["flows"])
    assert a.shape == (F, 8), (outp, a.shape)
    fs, fd, ftos, fpath, feq = [], [], [], [], []
    for fl in inp["flows"]:
        s, d = fl["src"], fl["dst"]
        p = [s]
        while p[-1] != d:
            p.append(inp["nexthop"][(p[-1], d)])
        fs.append(s); fd.append(d); ftos.append(fl["tos"])
        fpath.append(np.array([lidx[(p[j], p[j + 1])] for j in range(len(p) - 1)], np.int64))
        feq.append(fl["rate"] * 8 * sum(sz * pr for sz, pr in fl["mix"]))
    txp, txb, rxp, rxb, dsum, jsum, lost = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6], a[:, 7]
    with np.errstate(divide="ignore", invalid="ignore"):
        yd = np.where(rxp > 0, dsum / np.maximum(rxp, 1), 0.0)
        yj = np.where(rxp > 1, jsum / np.maximum(rxp - 1, 1), 0.0)
    L = len(links)
    return dict(
        topo=topo, graph_file=f"graph_{i}.txt", routing_file="", n_nodes=N, links=np.array(links, np.int64),
        capacity=np.array(cap), node_policy=pol, node_weights=np.zeros((N, MAXQ)), node_buffer=buf,
        node_nq=nq, tos_to_queue=t2q, flow_src=np.array(fs), flow_dst=np.array(fd), flow_tos=np.array(ftos),
        flow_bw=txb * 8.0 / window, flow_pkts=txp / window, flow_eqlambda=np.array(feq), flow_path=fpath,
        y_delay=yd, y_jitter=yj, y_drop=lost, link_util=np.zeros(L, np.float32),
        queue_util=np.zeros((L, MAXQ), np.float32), queue_occ=np.zeros((L, MAXQ), np.float32),
        source_file=f"sample_{i}", source_line=0)


def iter_ns3_samples(topo_dir, results_dir=None):
    topo = os.path.basename(topo_dir.rstrip("/"))
    res = results_dir or os.path.join(topo_dir, "results")
    ids = sorted(int(re.findall(r"sample_(\d+)\.txt", f)[0]) for f in glob.glob(os.path.join(topo_dir, "sample_*.txt")))
    for i in ids:
        o = os.path.join(res, f"sample_{i}.out")
        if os.path.exists(o) and os.path.getsize(o) > 0:
            yield to_sample(read_input(os.path.join(topo_dir, f"sample_{i}.txt")), o, topo, i)
