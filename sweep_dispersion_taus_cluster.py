"""Cluster copy of sweep_dispersion_taus.py: indices baked in, not read off disk.

Same sweep, same outputs, same filenames.  The one thing that changes is where
the replicate index comes from.

The original picks its first index with `next_index()`, which globs the output
directory and continues past the highest file it finds.  That is right on this
machine and wrong on a cluster: a fresh scratch directory has no files, so every
pair restarts at 0 and the job writes reconnect_0_age_dur.json, poly_0_sbm_age_dur.json
and so on.  Nothing fails -- the run looks perfectly healthy -- but rsyncing the
results back overwrites the real replicate 0 of every pair, and `path.exists()`
cannot catch it because it only ever sees the cluster's own directory.

So this copy never asks the filesystem.  BASE_INDEX below is a constant, audited
against duration+ages/seir_sims on this machine and set 10 clear of the highest
index any (dataset, model) pair uses, so the cluster's output cannot land on
anything that already exists here even if local runs continue while it is queued.
Re-run `--audit` before shipping to refresh it.

Everything from BASE_INDEX up is from this sweep, which also makes the batch easy
to identify and, if it needs to be, delete.

Examples
--------
    # re-audit the local sims and print the BASE_INDEX this file should carry
    python sweep_dispersion_taus_cluster.py --audit

    # what would be run, at which taus, under which filenames
    python sweep_dispersion_taus_cluster.py --dry-run

    # a real (small) end-to-end check here before shipping
    python sweep_dispersion_taus_cluster.py --datasets poly --models sbm \
        --networks 1 --n 20000 --iterations 8 --r-points 3 --out /tmp/sc_test

    # on the cluster, as a 20-way array job: each task takes its own index block
    python sweep_dispersion_taus_cluster.py --networks 2 --task-id $SLURM_ARRAY_TASK_ID

Ship alongside this file: input_data/, duration+ages/data/gmm_opt_comp/, and the
two tau -> R0 caches named in R0_CACHE_NAMES.  Needs the `phd2` environment: it
imports nd_python_avon, which needs nd_rust.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import sklearn.mixture

import nd_python_avon as nd_p

# Audited 2026-08-30 against duration+ages/seir_sims: the highest replicate index
# in use across all (dataset, model) pairs was 293 (reconnect, GMM + duration), so
# the next free index is 294 and this leaves 10 spare for local runs still to come.
# `--audit` recomputes it; bump this constant, do not lower it.
BASE_INDEX = 304

SHARES = [0.058, 0.145, 0.212, 0.364, 0.497, 0.623, 0.759, 0.866, 1.0]
FIG_DIR = Path("output_data/figs")      # default; see --fig-dir
SIM_DIR = Path("duration+ages/seir_sims")

# The tau -> R0 curves already computed elsewhere in the pipeline.  "fs" is the
# cache Figure 3 writes, "extra" the one the Table 2 band section writes for the
# models Figure 3 does not sweep.  Both must be copied to the cluster: without
# them every pair is skipped and the job exits successfully having done nothing,
# which is why --strict (the default here) turns a missing curve into an error.
R0_CACHE_NAMES = {"fs": "fig3_fs_r0.npz",
                  "extra": "table2_band_extra_r0.npz"}

# key -> how to build the network, what to call, and where the output goes.
# `suffix` reproduces the names Table 2 reads: {data}_{k}{suffix}_age_dur.json.
MODELS = {
    "sbm": {
        "label": "SBM", "suffix": "_sbm", "kind": "sbm",
        "r0_key": "{data}|sbm", "r0_cache": "fs",
    },
    "sbm_dur": {
        "label": "SBM + duration", "suffix": "_sbm_dur", "kind": "sbm_dur",
        "r0_key": "{data}|sbm_dur", "r0_cache": "extra",
    },
    "nodur": {
        "label": "GMM", "suffix": "_nodur", "kind": "gmm",
        "r0_key": "{data}|nodur", "r0_cache": "fs",
        "egos": "input_data/egos/{data}.json",
        "components": "input_data/gmm/optimal_components_{data}_log.json",
    },
    "dur": {
        "label": "GMM + duration", "suffix": "", "kind": "gmm_dur",
        "r0_key": "{data}|dur", "r0_cache": "fs",
        "egos": "input_data/egos/{data}_dur_small.json",
        "components": ("duration+ages/data/gmm_opt_comp/"
                       "optimal_components_{data}_log_smalldur.json"),
    },
}

AUDIT_DATASETS = ["poly", "comixa", "comixb", "comixc", "comix3", "reconnect"]
INDEX_HEADROOM = 10


# -------------------------------------------------------------------- audit

def max_index(out_dir, data, suffix):
    """Highest replicate index on disk for this pair, or -1 if there are none.

    The same parse `next_index()` uses in the original, kept here only so --audit
    can reproduce the number BASE_INDEX is set from.  Nothing in the run path calls
    it: the whole point of this copy is that indices do not come from the disk.
    """
    tail = f"{suffix}_age_dur.json"
    used = [int(p.name[len(data) + 1:-len(tail)])
            for p in Path(out_dir).glob(f"{data}_*{tail}")
            if p.name[len(data) + 1:-len(tail)].isdigit()]
    return max(used) if used else -1


def audit(out_dir):
    """Print the per-pair maxima and the BASE_INDEX they imply."""
    print(f"replicate indices in {out_dir}\n")
    highest = -1
    for data in AUDIT_DATASETS:
        for model, spec in MODELS.items():
            top = max_index(out_dir, data, spec["suffix"])
            highest = max(highest, top)
            print(f"  {data:>10s} {spec['label']:<16s} "
                  f"{'max ' + str(top) if top >= 0 else 'none'}")
    suggested = highest + 1 + INDEX_HEADROOM
    print(f"\nhighest index in use: {highest}"
          f"\nnext free index:      {highest + 1}"
          f"\nBASE_INDEX = {suggested}   ({INDEX_HEADROOM} clear of the next free index)")
    if suggested > BASE_INDEX:
        print(f"\nthis file carries BASE_INDEX = {BASE_INDEX}, which is now too low."
              f"\nEdit it to {suggested} before shipping.")
    else:
        print(f"\nthis file carries BASE_INDEX = {BASE_INDEX}, still clear.")
    return suggested


# --------------------------------------------------------------------- taus

def r0_curve(data, model, fig_dir=FIG_DIR):
    """(tau, R0) for one pair from the cached sweeps, sorted by tau."""
    spec = MODELS[model]
    path = Path(fig_dir) / R0_CACHE_NAMES[spec["r0_cache"]]
    if not path.exists():
        return None
    store = np.load(path, allow_pickle=False)
    key = spec["r0_key"].format(data=data)
    if f"{key}|tau" not in store.files:
        return None
    taus = np.asarray(store[f"{key}|tau"], dtype=float)
    _, sum_i2, sum_i3 = np.asarray(store[f"{key}|stats"], dtype=float).T[:3]
    r0 = np.divide(sum_i3, sum_i2, out=np.zeros_like(sum_i3), where=sum_i2 > 0)
    keep = np.isfinite(r0) & (taus > 0)
    order = np.argsort(taus[keep])
    return taus[keep][order], r0[keep][order]


# Some networks cannot reach a high R at any tau: once every contact transmits,
# R0 stops rising and the sweep flattens (the SBM on the lockdown waves tops out
# near 2.2).  A target above that ceiling is dropped rather than turned into an
# absurd tau by extrapolating a flat curve.
CLIMB_PER_EFOLD = 0.15    # dR0 per e-fold of tau, below which the sweep is saturated
MAX_TAU_FACTOR = 10.0     # never ask for a tau more than this beyond the sweep


def taus_for(data, model, targets, fig_dir=FIG_DIR):
    """A tau per target R, by inverting the cached curve.

    R0 rises with tau but the sweeps are noisy, so the curve is made monotone
    with a running maximum before inverting.  Targets a little above what the
    sweep reached are extrapolated log-linearly from its top third, capped at
    MAX_TAU_FACTOR times the largest tau swept; the simulation reports the R it
    actually produced, so a rough tau still lands a usable point on the figure.
    Targets beyond a saturated curve are reported as unreachable.

    Returns (taus, extrapolated, reached, ceiling): the first three aligned with
    the targets that are usable, `reached` being the targets they aim at.
    """
    curve = r0_curve(data, model, fig_dir)
    if curve is None:
        return None, None, None, None
    taus, r0 = curve
    rising = np.maximum.accumulate(r0)
    log_tau = np.log(taus)

    top = slice(max(len(taus) - max(4, len(taus) // 3), 0), len(taus))
    climb = np.ptp(rising[top]) / max(np.ptp(log_tau[top]), 1e-9)
    if climb > CLIMB_PER_EFOLD:
        slope, intercept = np.polyfit(rising[top], log_tau[top], 1)
        ceiling = float(rising[-1] + climb * math.log(MAX_TAU_FACTOR))
    else:                       # saturated: nothing above the sweep is reachable
        slope, intercept, ceiling = 0.0, log_tau[-1], float(rising[-1])

    chosen, extrapolated, reached = [], [], []
    for target in targets:
        if target <= rising[-1]:
            chosen.append(float(np.exp(np.interp(target, rising, log_tau))))
            extrapolated.append(False)
        elif target <= ceiling:
            chosen.append(float(min(np.exp(slope * target + intercept),
                                    taus[-1] * MAX_TAU_FACTOR)))
            extrapolated.append(True)
        else:
            continue            # unreachable on this network
        reached.append(float(target))
    if not chosen:
        return None, None, None, float(rising[-1])
    return np.array(chosen), np.array(extrapolated), np.array(reached), ceiling


# ---------------------------------------------------------------- preflight

def inputs_for(data, model):
    """Every file `simulate` will open for this pair, so a missing one is caught
    before the queue spends an hour discovering it."""
    spec = MODELS[model]
    needed = [Path(FIG_DIR) / R0_CACHE_NAMES[spec["r0_cache"]]]
    if spec["kind"] == "sbm":
        needed.append(Path(f"input_data/contact_matrices/contact_matrix_{data}.csv"))
        return needed
    needed.append(Path(f"input_data/durations/{data}.csv"))
    if spec["kind"] == "sbm_dur":
        needed.append(Path(f"input_data/egos/{data}_dur_small.json"))
        return needed
    needed.append(Path(spec["egos"].format(data=data)))
    needed.append(Path(spec["components"].format(data=data)))
    return needed


def missing_inputs(data, model, fig_dir):
    out = []
    for path in inputs_for(data, model):
        if path.name in R0_CACHE_NAMES.values():
            path = Path(fig_dir) / path.name
        if not path.exists():
            out.append(path)
    return out


# ---------------------------------------------------------------- networks

def partitions_for(n):
    return [int(round(s * n)) for s in SHARES]


def gmm_samples(data, spec, n, seed):
    """Degree profiles: a Gaussian mixture per age bucket, fitted in log(k+1).

    The same construction the *_sims_*.py scripts use, so the networks this
    writes are drawn from the same ensemble as the runs already on disk.
    """
    with open(spec["egos"].format(data=data)) as f:
        egos = json.load(f)
    with open(spec["components"].format(data=data)) as f:
        components = json.load(f)[data]
    partitions = partitions_for(n)
    samples = []
    for bucket, size in enumerate(np.diff([0] + partitions)):
        X = [[math.log(c + 1) for c in e["contacts"]] for e in egos
             if e["age"] == bucket]
        gm = sklearn.mixture.GaussianMixture(n_components=components[bucket],
                                             covariance_type="full",
                                             random_state=seed).fit(X)
        drawn, _ = gm.sample(size)
        samples += [[max(int(round(math.exp(v) - 1)), 0) for v in row]
                    for row in drawn]
    return samples


def duration_contact_matrices(data, num_durs=3):
    """Per-duration contact matrices, as in r_sbmdur_sc.py."""
    with open(f"input_data/egos/{data}_dur_small.json") as f:
        egos = json.load(f)
    n_buckets = max(e["age"] for e in egos) + 1
    per_bucket = np.zeros(n_buckets)
    matrices = [np.zeros((n_buckets, n_buckets)) for _ in range(num_durs)]
    for ego in egos:
        per_bucket[ego["age"]] += 1
        for j, val in enumerate(ego["contacts"]):
            matrices[j % num_durs][ego["age"], j // num_durs] += val
    out = []
    for m in matrices:
        m = np.divide(m.T, per_bucket).T
        out.append(((m + m.T) / 2).tolist())
    return out


def simulate(data, model, taus, n, iterations, seed):
    """Run one replicate network over the whole tau grid."""
    spec = MODELS[model]
    partitions = partitions_for(n)
    taus = list(map(float, taus))

    if spec["kind"] == "sbm":
        cm = np.genfromtxt(f"input_data/contact_matrices/contact_matrix_{data}.csv",
                           delimiter=",").tolist()
        return nd_p.sbm_gillesp_sc(contact_matrix=cm, partitions=partitions,
                                   taus=taus, iterations=iterations, num_infec=1)

    props = np.genfromtxt(f"input_data/durations/{data}.csv", delimiter=",").tolist()

    if spec["kind"] == "sbm_dur":
        return nd_p.sbm_gillesp_dur_sc(contact_matrix=duration_contact_matrices(data),
                                       partitions=partitions, taus=taus,
                                       iterations=iterations, num_infec=1,
                                       props=props, num_dur=3)

    samples = gmm_samples(data, spec, n, seed)
    if spec["kind"] == "gmm_dur":
        return nd_p.gmm_dur_gillesp_sc(samples, partitions=partitions, num_dur=3,
                                       taus=taus, iterations=iterations,
                                       props=props, num_infec=1)
    return nd_p.gmm_gillesp_sc(samples, partitions=partitions, taus=taus,
                               iterations=iterations, num_infec=1)


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+",
                    default=["poly", "comixb", "reconnect"],
                    help="comixa (Lockdown 1) is out of the analysis, so it is not "
                         "swept by default; name it explicitly to include it")
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS))
    ap.add_argument("--r-min", type=float, default=0.25)
    ap.add_argument("--r-max", type=float, default=5.0)
    ap.add_argument("--r-points", type=int, default=60,
                    help="taus per pair, spread evenly over [r-min, r-max]")
    ap.add_argument("--networks", type=int, default=1,
                    help="replicate networks per pair; one network over a dense tau "
                         "grid buys more of the k-against-R curve than three over a "
                         "coarse one, for the same number of outbreaks")
    ap.add_argument("--iterations", type=int, default=24,
                    help="outbreaks per tau per network, as in the *_sc.py scripts")
    ap.add_argument("--n", type=int, default=100_000, help="nodes per network")
    ap.add_argument("--out", type=Path, default=SIM_DIR)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR,
                    help=f"where the cached tau -> R0 curves live "
                         f"({', '.join(R0_CACHE_NAMES.values())}); point it at a "
                         f"notebook's own figure directory to invert the curves it built")
    ap.add_argument("--base-index", type=int, default=BASE_INDEX,
                    help=f"first replicate index, for every pair (default {BASE_INDEX}, "
                         f"audited against this machine).  Unlike the original this is "
                         f"never read off the output directory, so a cluster with an "
                         f"empty scratch dir still writes non-colliding indices")
    ap.add_argument("--task-id", type=int, default=0,
                    help="array task index; task t takes the block starting at "
                         "base-index + t * networks, so concurrent tasks writing to a "
                         "shared filesystem cannot pick the same replicate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-missing", action="store_true",
                    help="carry on past pairs whose tau -> R0 curve or input files are "
                         "absent.  Off by default: on a cluster that silently turns the "
                         "whole job into a no-op that still exits 0")
    ap.add_argument("--audit", action="store_true",
                    help="report the highest replicate index per pair in --out and the "
                         "BASE_INDEX it implies, then stop.  Run this here before shipping")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan -- taus, R targets, filenames -- and stop")
    args = ap.parse_args()

    if args.audit:
        audit(args.out)
        return

    targets = np.linspace(args.r_min, args.r_max, args.r_points)
    first = args.base_index + args.task_id * args.networks
    last = first + args.networks - 1
    args.out.mkdir(parents=True, exist_ok=True)
    plan, problems = [], []

    print(f"target R: {args.r_min} to {args.r_max} in {args.r_points} steps")
    print(f"tau -> R0 curves from {args.fig_dir}")
    print(f"replicate indices {first}..{last} "
          f"(base {args.base_index} + task {args.task_id} x {args.networks})\n")
    for data in args.datasets:
        for model in args.models:
            spec = MODELS[model]
            absent = missing_inputs(data, model, args.fig_dir)
            if absent:
                print(f"  {data:>10s} {spec['label']:<16s} "
                      f"missing {', '.join(str(p) for p in absent)}")
                problems.append(f"{data}/{model}: missing input files")
                continue
            taus, extrapolated, reached, ceiling = taus_for(data, model, targets,
                                                            args.fig_dir)
            if taus is None:
                why = ("no cached tau -> R0 curve"
                       if ceiling is None else
                       f"cannot exceed R = {ceiling:.2f} at any tau")
                print(f"  {data:>10s} {spec['label']:<16s} {why}, skipped")
                if ceiling is None:         # a saturated curve is a fact about the
                    problems.append(f"{data}/{model}: {why}")   # network, not a fault
                continue
            plan.append((data, model, taus))
            dropped = len(targets) - len(reached)
            print(f"  {data:>10s} {spec['label']:<16s} {len(taus):2d} taus for R "
                  f"{reached.min():.2f}-{reached.max():.2f}, tau "
                  f"{taus.min():.4g}-{taus.max():.4g}"
                  f"{f', {extrapolated.sum()} extrapolated' if extrapolated.any() else ''}"
                  f"{f', {dropped} above the R ceiling' if dropped else ''}"
                  f"\n{'':>28s}-> {data}_{{{first}..{last}}}"
                  f"{spec['suffix']}_age_dur.json")

    if problems and not args.skip_missing:
        print("\n" + "\n".join(f"  ! {p}" for p in problems))
        print("\nrefusing to run: copy the missing files across, or pass --skip-missing.")
        sys.exit(1)
    if not plan:
        print("\nnothing to run.")
        sys.exit(1)
    if args.dry_run:
        print("\ndry run: nothing written")
        return

    print(f"\n{len(plan)} pairs x {args.networks} networks x {len(targets)} taus x "
          f"{args.iterations} iterations, n = {args.n:,}\n")
    started = time.time()
    for data, model, taus in plan:
        spec = MODELS[model]
        for k in range(first, last + 1):
            path = args.out / f"{data}_{k}{spec['suffix']}_age_dur.json"
            if path.exists():
                print(f"  {path.name} exists, skipped", flush=True)
                continue
            # Claim the index before simulating, as the r_*.py scripts do: an hour
            # of simulation is long enough for a second task on the same filesystem
            # to glob the directory, see nothing, and pick the same replicate.
            # Aggregators skip files they cannot parse, so a placeholder left by a
            # job that died is ignored rather than counted.
            path.touch()
            t0 = time.time()
            res = simulate(data, model, taus, args.n, args.iterations,
                           args.seed + k)
            with open(path, "w") as f:
                json.dump(res, f)
            print(f"  wrote {path.name}  ({time.time() - t0:.0f}s, "
                  f"{time.time() - started:.0f}s total)", flush=True)

    print(f"\ndone in {(time.time() - started) / 60:.1f} min")
    print(f"wrote replicate indices {first}..{last}.  Copy the files back into\n"
          f"duration+ages/seir_sims -- the indices are above everything that was\n"
          f"there when this file was audited, so nothing is overwritten.  Table 2\n"
          f"takes its own bound from the highest index on disk and its cache is\n"
          f"keyed on the files it read, so re-running the Table 2 and Figure S4\n"
          f"cells picks these up.")


if __name__ == "__main__":
    main()
