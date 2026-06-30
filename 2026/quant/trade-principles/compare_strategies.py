#!/usr/bin/env python3
"""
Compare betting strategies: bust time distribution and profit trajectories.

Generates:
  fig6_bust_survival.png     - P(T > n) survival curves, 4 strategies, log-log
  fig7_bust_median_funds.png - Median bust time vs initial funds
  fig8_traj_compare.png      - 2x2 profit density heatmaps
  fig9_population.png         - 2x2 busted / alive-losing / alive-profitable

Usage:
  python compare_strategies.py --output-dir ./
"""

import argparse
import os
import time
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from martingale_sim import (
    simulate, simulate_hist, _seed_rng, _smooth2d, DPI,
    MARTINGALE, FLAT, ANTI_MARTINGALE, PROPORTIONAL,
)

# ---- Config ----

class Strategy(NamedTuple):
    id: int
    min_bet: int
    label: str
    color: str


STRATEGIES = [
    Strategy(MARTINGALE,      100, 'Martingale',         'C0'),
    Strategy(ANTI_MARTINGALE, 100, 'Anti-Martingale',    'C1'),
    Strategy(PROPORTIONAL,    100, 'Proportional',       'C2'),
    Strategy(FLAT,            100, 'Random Walk (±100)', 'C3'),
]

FIGSIZE_LINE = (12, 8)
FIGSIZE_PANEL = (18, 12)


# ---- Small helpers ----

@contextmanager
def timed(label):
    t0 = time.time()
    yield
    print(f'  {label}: {time.time() - t0:.1f}s')


def save_fig(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name), dpi=DPI)
    plt.close(fig)
    print(f'  saved {name}')


def alive_clip(trajs, a0, n_pts, thresh, cap):
    """Last sample index where >`thresh` of paths are still alive, plus a margin."""
    alive = (trajs > -a0 + 1).mean(axis=0)
    idx = np.where(alive > thresh)[0]
    if len(idx) == 0:
        return cap
    return min(idx[-1] + max(1, n_pts // 20), cap)


# ---- Workers (explicit strategy + seed: deterministic, no shared globals) ----

def _bust_worker(args):
    """Return bust time (rounds) for one trial."""
    a0, b0, strategy, min_bet, max_rounds, seed = args
    _seed_rng(seed)
    _, rnd, _ = simulate(a0, b0, strategy, min_bet, max_rounds)
    return rnd


def _traj_worker(args):
    """Return one profit trajectory, subsampled to bound IPC payload."""
    a0, b0, strategy, min_bet, max_rounds, seed, sample_idx = args
    _seed_rng(seed)
    hist, rnd = simulate_hist(a0, b0, strategy, min_bet, max_rounds)
    max_len = min(rnd, max_rounds)
    # Ship only sampled points (5000 * 8B = 40KB, not 5M * 8B = 40MB)
    valid = sample_idx[sample_idx < max_len]
    result = np.empty(len(sample_idx), dtype=np.int64)
    result[:len(valid)] = hist[valid]
    result[len(valid):] = hist[max_len - 1] if max_len > 0 else -a0
    return result, rnd


# ---- Plotting ----

def fig6_survival(outdir, a0=10000, n_trials=3000, max_rounds=10_000_000,
                  n_workers=8):
    """Fig 6: bust-time survival curves for all strategies."""
    print(f'Fig 6: survival curves, A={a0}, B=INF, {n_trials} trials')
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    for s in STRATEGIES:
        tasks = [(a0, -1, s.id, s.min_bet, max_rounds, i * 31 + s.id * 7)
                 for i in range(n_trials)]
        with timed(s.label):
            with Pool(n_workers) as pool:
                bust_times = np.array(pool.map(_bust_worker, tasks))
        print(f'    median={int(np.median(bust_times)):,}')

        surv_y = 1.0 - np.arange(1, n_trials + 1) / n_trials
        ax.loglog(np.sort(bust_times), surv_y, color=s.color,
                  linewidth=1.2, alpha=0.85, label=s.label)

    n_ref = np.logspace(2, 7, 300)
    ax.loglog(n_ref, a0 / (a0 + n_ref), 'k:', alpha=0.3, lw=1,
              label=r'$a/(a+n)$ bound')
    ax.set_xlabel('Rounds')
    ax.set_ylabel('P(T > n)')
    ax.set_title(f'Bust Time Survival (A={a0:,}, B=INF)')
    ax.legend(fontsize=10)
    ax.set_xlim(100, max_rounds)
    ax.set_ylim(1e-4, 1.1)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save_fig(fig, outdir, 'fig6_bust_survival.png')


def fig7_median_funds(outdir, fund_range=(100, 100000), n_points=30,
                      n_trials=500, max_rounds=10_000_000, n_workers=8):
    """Fig 7: median bust time vs initial funds for all strategies."""
    funds = np.unique(np.geomspace(*fund_range, n_points).astype(int))
    print(f'Fig 7: median vs funds, {len(funds)} points, {n_trials} trials')
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    for s in STRATEGIES:
        with timed(s.label):
            # One pool per strategy, reused across all fund levels
            with Pool(n_workers) as pool:
                medians = []
                for a0 in funds:
                    tasks = [(int(a0), -1, s.id, s.min_bet, max_rounds,
                              i * 19 + s.id) for i in range(n_trials)]
                    bt = np.array(pool.map(_bust_worker, tasks))
                    medians.append(np.median(bt))
        ax.plot(funds, medians, 'o-', color=s.color, label=s.label,
                markersize=3, linewidth=1.5)

    ax.set_xlabel('Initial Funds')
    ax.set_ylabel('Median Bust Time (rounds)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Median Bust Time vs Initial Funds (B=INF)')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, outdir, 'fig7_bust_median_funds.png')


class TrajData(NamedTuple):
    label: str
    trajs: np.ndarray      # (n_trials, n_sample_pts)
    round_idx: np.ndarray  # actual round numbers of each sample
    color: str
    a0: int


def collect_trajs(a0, strategies, n_trials, max_rounds, n_workers,
                  max_sample_pts=5000):
    """Collect subsampled trajectories for each strategy.

    Workers subsample before returning via IPC, avoiding 40MB-per-result
    pipe congestion at multi-million-round horizons.
    """
    n_pts = min(max_sample_pts, max_rounds)
    sample_idx = np.linspace(0, max_rounds - 1, n_pts, dtype=np.int64)

    out = []
    for s in strategies:
        tasks = [(a0, -1, s.id, s.min_bet, max_rounds, i * 23 + s.id, sample_idx)
                 for i in range(n_trials)]
        with timed(s.label):
            with Pool(n_workers) as pool:
                results = list(pool.map(_traj_worker, tasks))
        max_len = min(max(r[1] for r in results), max_rounds)
        print(f'    max_len={max_len:,}, sampled {n_pts} pts')
        out.append(TrajData(s.label, np.array([r[0] for r in results]),
                            sample_idx, s.color, a0))
    return out


def fig8_trajectories(outdir, traj_data):
    """Fig 8: 2x2 profit density heatmaps, per-strategy time scale."""
    print('Fig 8: trajectory density')
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_PANEL)

    for ax, d in zip(axes.flat, traj_data):
        ridx = d.round_idx
        clip = max(alive_clip(d.trajs, d.a0, len(ridx), 0.01, len(ridx)), 2)
        too_sparse = clip < 10

        trajs_c = d.trajs[:, :clip]
        ridx_c = ridx[:clip]
        x_max = int(ridx_c[-1]) if len(ridx_c) > 0 else 1

        n_pbins = 50
        y_lo = float(np.percentile(trajs_c, 1))
        y_hi = float(np.percentile(trajs_c, 99))
        margin = max((y_hi - y_lo) * 0.05, 100)
        edges = np.linspace(y_lo - margin, y_hi + margin, n_pbins + 1)

        density = np.zeros((n_pbins, len(ridx_c)))
        for j in range(len(ridx_c)):
            counts, _ = np.histogram(trajs_c[:, j], bins=edges)
            total = counts.sum()
            if total > 0:
                density[:, j] = counts / total
        if min(density.shape) > 6:
            density = _smooth2d(density)

        ax.imshow(density, origin='lower', aspect='auto',
                  extent=[0, x_max, edges[0], edges[-1]], cmap='inferno')
        ax.axhline(0, color='white', lw=0.5, ls='--', alpha=0.5)
        if too_sparse:
            ax.text(0.5, 0.5, f'Busted within {x_max:,} rounds\n'
                    '(too fast for density at this sampling rate)',
                    transform=ax.transAxes, ha='center', va='center',
                    color='white', fontsize=11, alpha=0.8)
        ax.set_xlabel('Round')
        ax.set_ylabel("A's Profit")
        ax.set_title(d.label)

    plt.tight_layout()
    save_fig(fig, outdir, 'fig8_traj_compare.png')


def fig9_population(outdir, traj_data):
    """Fig 9: 2x2 stacked area, busted / alive-losing / alive-profitable."""
    print('Fig 9: population state')
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_PANEL)

    for idx, (ax, d) in enumerate(zip(axes.flat, traj_data)):
        trajs, a0, ridx = d.trajs, d.a0, d.round_idx
        n_trials, n_pts = trajs.shape[0], len(ridx)

        busted = (trajs <= -a0 + 1).mean(axis=0)
        profit = (trajs > 0).mean(axis=0)
        losing = 1.0 - busted - profit

        clip = alive_clip(trajs, a0, n_pts, 0.005, n_pts - 1)
        x_max = int(ridx[clip])

        ax.fill_between(ridx, 0, busted, color='#e74c3c', alpha=0.8,
                        label='Busted')
        ax.fill_between(ridx, busted, busted + losing, color='#f39c12',
                        alpha=0.8, label='Alive, losing')
        ax.fill_between(ridx, busted + losing, 1.0, color='#2ecc71',
                        alpha=0.8, label='Alive, profitable')
        if 'Proportional' in d.label or 'Random' in d.label:
            ax.axhline(0.5, color='black', lw=0.8, ls='--', alpha=0.4)

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Round')
        ax.set_ylabel('Fraction')
        ax.set_title(d.label)
        if idx == 0:
            ax.legend(loc='center right', fontsize=9)

    plt.tight_layout()
    save_fig(fig, outdir, 'fig9_population.png')


# ---- Main ----

def warmup():
    """Trigger Numba JIT for every strategy before the Pool forks."""
    print('JIT warmup ...')
    for sid in (MARTINGALE, FLAT, ANTI_MARTINGALE, PROPORTIONAL):
        simulate(100, -1, sid, 1, 100)
        simulate_hist(100, -1, sid, 1, 100)
    _seed_rng(0)


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', default='./')
    ap.add_argument('--n-workers', type=int, default=8)
    ap.add_argument('--n-trials-surv', type=int, default=3000,
                    help='Trials for survival curve')
    ap.add_argument('--n-trials-median', type=int, default=500,
                    help='Trials per fund level for median chart')
    ap.add_argument('--n-trials-traj', type=int, default=100,
                    help='Trials for trajectory density')
    ap.add_argument('--a0', type=int, default=100000,
                    help='Initial funds for survival/trajectory')
    ap.add_argument('--max-rounds', type=int, default=10_000_000)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    warmup()

    # Short-lived strategies get dense sampling; long-lived get full range
    short = [s for s in STRATEGIES if s.id in (MARTINGALE, ANTI_MARTINGALE)]
    long = [s for s in STRATEGIES if s.id in (PROPORTIONAL, FLAT)]

    print(f'Collecting trajectories, A={args.a0}, B=INF')
    print('  -- short-lived strategies (100K rounds) --')
    traj_short = collect_trajs(args.a0, short, args.n_trials_traj,
                               100_000, args.n_workers)
    print('  -- long-lived strategies (full range) --')
    traj_long = collect_trajs(args.a0, long, args.n_trials_traj,
                              args.max_rounds, args.n_workers)
    traj_data = traj_short + traj_long

    fig8_trajectories(args.output_dir, traj_data)
    fig9_population(args.output_dir, traj_data)
    fig6_survival(args.output_dir, a0=args.a0, n_trials=args.n_trials_surv,
                  max_rounds=args.max_rounds, n_workers=args.n_workers)
    fig7_median_funds(args.output_dir, n_trials=args.n_trials_median,
                      max_rounds=args.max_rounds, n_workers=args.n_workers)


if __name__ == '__main__':
    main()
