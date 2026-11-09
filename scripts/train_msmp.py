import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qprgnn.msmp import MSMPAttention, collate_msmp, zstats_msmp
from qprgnn.runs import fermi_data
from qprgnn import protocol as P


def main():
    a = P.add_args(argparse.ArgumentParser(), epochs=75, bs=8, lr=1e-3).parse_args()
    universe, held_out = P.setup(a)
    train, val, tests = P.splits(universe, held_out, a.cap_train, fermi_data)
    tag = P.run_tag("msmp_attn", a.holdout, a.seed)
    z = zstats_msmp(train)
    model = MSMPAttention(z)
    info = P.fit(model, train, val, collate_msmp, P.mape_loss, a, tag)
    info.update(variant="msmp_attn", model_variant="msmp_attn", H=16, T=12, z=z)
    P.save_run(a, tag, info, tests, model, collate_msmp)


if __name__ == "__main__":
    main()
