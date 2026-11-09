import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from qprgnn.data import collate, load_topology
from qprgnn.model import QPRGNN, READOUTS


def _items(n=2):
    return load_topology("test_abilene")[0][:n]


def test_starts_at_the_prior():
    bt = collate(_items())
    torch.manual_seed(0)
    m = QPRGNN(readout="qosres_nosched", prior_feats=True).eval()
    with torch.no_grad():
        d = m(bt)
    prior = ((bt.tx + bt.wait_nosched) * (bt.path >= 0)).sum(1)
    assert torch.allclose(d, prior, rtol=1e-6)


def test_all_readouts_positive():
    bt = collate(_items())
    for r in READOUTS:
        torch.manual_seed(0)
        m = QPRGNN(readout=r, prior_feats=r == "qosres_nosched")
        torch.nn.init.normal_(m.read[-1].weight, std=0.1)
        d = m(bt)
        assert d.shape == bt.y.shape and torch.all(d > 0), r


def test_link_relabelling_invariance():
    it = _items(1)[0]
    L = it["L"]; perm = torch.randperm(L, generator=torch.Generator().manual_seed(1)); inv = torch.argsort(perm)
    it2 = dict(it); it2["link_x"] = it["link_x"][perm]
    p = it["path"].clone(); p[p >= 0] = inv[p[p >= 0]]; it2["path"] = p
    torch.manual_seed(0)
    m = QPRGNN().eval(); torch.nn.init.normal_(m.read[-1].weight, std=0.1)
    with torch.no_grad():
        assert torch.allclose(m(collate([it])), m(collate([it2])), rtol=1e-5)


if __name__ == "__main__":
    test_starts_at_the_prior(); test_all_readouts_positive(); test_link_relabelling_invariance()
    print("model tests passed")
