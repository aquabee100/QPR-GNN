import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qprgnn.fermi import RouteNetFermi, collate_fermi, zstats
from qprgnn.runs import fermi_data
from qprgnn import protocol as P


def main():
    a = P.add_args(argparse.ArgumentParser(), epochs=75, bs=8, lr=1e-3).parse_args()
    universe, held_out = P.setup(a)
    train, val, tests = P.splits(universe, held_out, a.cap_train, fermi_data)
    tag = P.run_tag("fermi_official", a.holdout, a.seed)
    z = zstats(train)
    model = RouteNetFermi(z)
    info = P.fit(model, train, val, collate_fermi, P.mape_loss, a, tag)
    info.update(variant="fermi_official", model_variant="fermi_official", fermi_version=2, H=32, T=8, z=z)
    P.save_run(a, tag, info, tests, model, collate_fermi)


if __name__ == "__main__":
    main()
