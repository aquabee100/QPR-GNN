import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qprgnn.data import collate, load_topology
from qprgnn.model import QPRGNN, READOUTS
from qprgnn import protocol as P


def main():
    ap = P.add_args(argparse.ArgumentParser(), epochs=25, bs=8, lr=2e-3)
    ap.add_argument("--name", required=True, help="run name (file prefix)")
    ap.add_argument("--readout", default="qosres_nosched", choices=READOUTS)
    ap.add_argument("--prior_feats", action="store_true", help="prior also as per-hop input features")
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--T", type=int, default=4)
    a = ap.parse_args()
    universe, held_out = P.setup(a)
    train, val, tests = P.splits(universe, held_out, a.cap_train, lambda n: load_topology(n)[::-1])
    tag = P.run_tag(a.name, a.holdout, a.seed)
    model = QPRGNN(H=a.H, T=a.T, readout=a.readout, prior_feats=a.prior_feats)
    info = P.fit(model, train, val, collate, P.log_ratio_loss, a, tag, cosine=True)
    info.update(variant=a.name, model_variant="routenet", H=a.H, T=a.T, readout=a.readout, prior_feats=a.prior_feats)
    P.save_run(a, tag, info, tests, model, collate)


if __name__ == "__main__":
    main()
