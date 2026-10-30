import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx
from qprgnn.parse import load_graph, load_routing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_archive_format():
    g = sorted(glob.glob(f"{ROOT}/data_traces/test/abilene/graphs/*.txt"))[0]
    r = sorted(glob.glob(f"{ROOT}/data_traces/test/abilene/routings/*.txt"))[0]
    G, links, cap, lidx, port, node = load_graph(g)
    P = load_routing(r, g)
    assert len(links) == 30 and len(P) == 12 * 11
    assert all((a, b) in lidx for p in P.values() for a, b in zip(p[:-1], p[1:]))


def test_simulator_input_format():
    d = f"{ROOT}/sim_inputs/Abilene"
    if not os.path.isdir(d):
        print("skip: run scripts/make_sim_inputs.py first"); return
    g, r = f"{d}/graphs/graph_0.txt", f"{d}/routings/routing_0.txt"
    G, links, cap, lidx, port, node = load_graph(g)
    H = nx.read_gml(g, destringizer=int, label="id")
    assert len(links) == 2 * H.number_of_edges()
    P = load_routing(r, g)
    N = H.number_of_nodes()
    assert len(P) == N * (N - 1)
    assert all((a, b) in lidx for p in P.values() for a, b in zip(p[:-1], p[1:]))
    # queue count from tosToQoSqueue
    for n in range(N):
        t = H.nodes[n]["tosToQoSqueue"]
        assert node["nq"][n] == len(t.split(";"))
        assert node["tos2q"][n].max() == node["nq"][n] - 1


if __name__ == "__main__":
    test_archive_format(); print("ok archive format")
    test_simulator_input_format(); print("ok simulator input format")
