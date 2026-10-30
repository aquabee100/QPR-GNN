import glob, math, sys
import networkx as nx
bad = n = 0
for g in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "sim_inputs/*/graphs/graph_*.txt")):
    G = nx.read_gml(g, destringizer=int)
    L = int(G.graph.get("levelsToS", 3))
    for _, d in G.nodes(data=True):
        n += 1; err = []
        pol = d.get("schedulingPolicy")
        if pol not in ("FIFO", "SP", "WFQ", "DRR"): err.append("policy")
        groups = [q for q in str(d.get("tosToQoSqueue", "")).split(";")] if "tosToQoSqueue" in d else []
        tos = [int(t) for q in groups for t in q.split(",")]
        if groups and sorted(tos) != list(range(L)): err.append("tos map")
        if pol in ("WFQ", "DRR"):
            w = [float(x) for x in str(d.get("schedulingWeights", "")).split(",")]
            if len(w) != len(groups) or not math.isclose(sum(w), 100): err.append("weights")
        b = d.get("bufferSizes")
        if not isinstance(b, int) or b < 8000: err.append(f"bufferSizes={b!r}")
        if err:
            bad += 1
            if bad <= 5: print(g, err)
    for u, v, d in G.edges(data=True):
        bw = d.get("bandwidth")
        if not isinstance(bw, int) or bw % 1000 or bw < 10000:
            bad += 1
            if bad <= 5: print(g, "edge", u, v, bw)
    if G.is_directed() or G.is_multigraph() or not nx.is_connected(G):
        bad += 1; print(g, "graph must be undirected, simple, connected")
print(f"nodes checked: {n}, graph/node/edge problems: {bad}")

# routing and traffic-matrix files referenced by simulation.txt
rbad = tbad = lines = 0
for sim in sorted(glob.glob("sim_inputs/*/simulation.txt")):
    root = sim[: -len("simulation.txt")]
    for line in open(sim):
        gp, rp, tp = line.strip().split(",")
        G = nx.read_gml(root + gp, destringizer=int); N = len(G)
        for l in open(root + rp):
            p = list(map(int, l.split(",")))
            if len(p) > 1 and (len(set(p)) != len(p) or max(p) >= N or any(not G.has_edge(a, b) for a, b in zip(p[:-1], p[1:]))):
                rbad += 1
        for l in open(root + tp):
            f = l.strip().split(","); lines += 1
            try:
                td = int(f[3]); sd = 6 if td == 2 else 4
                ok = len(f) >= 7 and 0 <= int(f[0]) < N and 0 <= int(f[1]) < N and float(f[2]) >= 10 and int(f[sd]) == 0
                prm = list(map(float, f[sd + 1:-1])); pairs = list(zip(prm[::2], prm[1::2]))
                ok = ok and len(prm) % 2 == 0 and len(pairs) <= 8 and all(50 <= a <= 72000 and 0 <= b <= 1 for a, b in pairs)
                ok = ok and math.isclose(sum(b for _, b in pairs), 1) and td in (0, 1, 2)
            except Exception:
                ok = False
            tbad += not ok
print(f"routing paths failing: {rbad}; traffic lines checked: {lines}, failing: {tbad}")
