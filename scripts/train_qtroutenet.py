import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qprgnn.qtroutenet import QTRouteNet, collate_qt, qt_stats
from qprgnn.runs import qt_data
from qprgnn import protocol as P


def main():
    ap = P.add_args(argparse.ArgumentParser(), epochs=75, bs=16, lr=1e-3)
    ap.add_argument("--tx_term", action="store_true", help="+ flow's own transmission time per hop")
    a = ap.parse_args()
    universe, held_out = P.setup(a)
    train, val, tests = P.splits(universe, held_out, a.cap_train, qt_data)
    name = "qtroutenet" + ("_tx" if a.tx_term else "")
    tag = P.run_tag(name, a.holdout, a.seed)
    z = qt_stats(train)
    model = QTRouteNet(z, tx_term=a.tx_term)
    info = P.fit(model, train, val, collate_qt, lambda p, y: P.mape_loss(p, y).sqrt(), a, tag)
    info.update(variant=name, model_variant="qtroutenet", tx_term=a.tx_term, z=z)
    P.save_run(a, tag, info, tests, model, collate_qt)


if __name__ == "__main__":
    main()
