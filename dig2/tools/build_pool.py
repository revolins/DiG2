"""Build a fresh sbm / planar training pool for anchored fine-tuning and run its fidelity gate.

    python -m dig2.tools.build_pool --domain sbm
    python -m dig2.tools.build_pool --domain planar

writes ``data/pools/<domain>/raw/{train,val,test}.pt`` and ``fidelity_gate.json``; the defaults
are the arguments the released pools were built with. ``+experiment=anchor_ft`` reads the pools
through ``dataset.unified.domain_root_overrides``, whose paths are resolved against the repo
root, so run this from the repo root (or pass ``--out`` and point the override at it). The exit
status is non-zero when the fidelity gate fails; the pool files are written either way so the
failure can be inspected. See dig2.procedures.pools for what the gate checks.
"""
import argparse
import os
import sys

from dig2 import paths
from dig2.procedures import pools


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=pools.POOL_DOMAINS)
    ap.add_argument("--out", default=None, help="pool root (default: data/pools/<domain>, relative to the cwd)")
    ap.add_argument("--source-root", default=None,
                    help="original dataset root holding raw/{train,val,test}.pt; downloaded if missing "
                         "(default: <repo>/data/<domain>)")
    ap.add_argument("--n", type=int, default=pools.DEFAULT_N, help="pool size")
    ap.add_argument("--seed", type=int, default=pools.DEFAULT_SEED)
    ap.add_argument("--gate-n", type=int, default=pools.DEFAULT_GATE_N, help="pool graphs the gate scores")
    ap.add_argument("--mmd-null-b", type=int, default=pools.DEFAULT_MMD_NULL_B,
                    help="pool subsets drawn for each MMD null")
    ap.add_argument("--no-gate", action="store_true", help="write the pool without running the fidelity gate")
    a = ap.parse_args()

    out = os.path.abspath(a.out or os.path.join("data", "pools", a.domain))
    source_root = os.path.abspath(a.source_root or os.path.join(paths.repo_root(), "data", a.domain))
    report = pools.build_pool(a.domain, out, source_root, n=a.n, seed=a.seed, gate_n=a.gate_n,
                              mmd_null_b=a.mmd_null_b, run_gate=not a.no_gate,
                              log=lambda m: print(m, flush=True))
    sys.exit(0 if report is None or report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
