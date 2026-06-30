#!/usr/bin/env python3
"""
Betting Strategy Monte Carlo Simulation

Pluggable strategies (Martingale / Flat / Anti-Martingale / Proportional).
Supports B=INF (market mode: the house has infinite funds).

Figures generated:
  1. Heatmap: average game length (rounds) for different (A, B) fund pairs
  2. Heatmap: P(A takes all of B's money)
  3. Heatmap: per-bet win rate (fair coin => uniformly ~0.5)
  4. 2D density heatmap: profit distribution over rounds

Requirements: pip install numpy matplotlib numba

Usage:
  python martingale_sim.py --output-dir ./
  python martingale_sim.py --strategy flat --log-grid --traj-b -1
"""

import argparse
import os
import time
from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from multiprocessing import Pool
from numba import njit

# ---- Strategy IDs (Numba-compatible integer dispatch) ----
MARTINGALE = 0
FLAT = 1
ANTI_MARTINGALE = 2
PROPORTIONAL = 3

STRATEGY_MAP = {
    'martingale': MARTINGALE,
    'flat': FLAT,
    'anti-martingale': ANTI_MARTINGALE,
    'proportional': PROPORTIONAL,
}

# ---- Plot style constants ----
DPI = 150
FIGSIZE_HEATMAP = (10, 8)
FIGSIZE_DENSITY = (14, 8)
N_PROFIT_BINS = 60
N_ROUND_BINS_MAX = 500
SMOOTH_SIGMA = 1.5


# ---- Numba core ----

@njit
def _seed_rng(s):
    np.random.seed(s % (2 ** 31))


@njit
def _next_bet(strategy, actual_bet, won, broken, min_bet, current_funds, initial_funds):
    """Determine next round's bet based on strategy and last outcome."""
    if strategy == MARTINGALE:
        return min_bet if (won or broken) else actual_bet * 2
    if strategy == FLAT:
        return min_bet
    if strategy == ANTI_MARTINGALE:
        return actual_bet * 2 if (won and not broken) else min_bet
    if strategy == PROPORTIONAL:
        # Bet proportional to current/initial ratio, scaled by min_bet
        if initial_funds > 0 and current_funds > 0:
            return max(1, int(np.ceil(min_bet * current_funds / initial_funds)))
        return min_bet
    return min_bet


# simulate() and simulate_hist() share the same inner loop on purpose. The split
# is a deliberate performance trade-off, NOT leftover duplication: simulate() is
# allocation-free and runs millions of times for the fund grid; simulate_hist()
# allocates a max_rounds array to record the full path and is used only for the
# (far fewer) trajectory plots. Keep them separate.

@njit
def simulate(a0, b0, strategy, min_bet=1, max_rounds=200_000):
    """Run one complete game (allocation-free).

    b0 < 0 means B has infinite funds (market mode).

    Returns: (a_won, rounds_played, bets_won_by_A)
    """
    a, b = a0, b0
    b_inf = b < 0
    bet = min_bet
    rnd = 0
    bet_wins = 0

    while a > 0 and (b_inf or b > 0) and rnd < max_rounds:
        actual = min(bet, a) if b_inf else min(bet, a, b)
        broken = actual < bet

        won = np.random.random() < 0.5
        delta = actual if won else -actual   # zero-sum: A and B move oppositely
        a += delta
        if not b_inf:
            b -= delta
        if won:
            bet_wins += 1

        bet = _next_bet(strategy, actual, won, broken, min_bet, a, a0)
        rnd += 1

    # A wins only when B is actually bust (impossible if B infinite)
    a_won = 1 if (not b_inf and b <= 0) else 0
    return a_won, rnd, bet_wins


@njit
def simulate_hist(a0, b0, strategy, min_bet, max_rounds):
    """Same game as simulate() but records A's profit at every round."""
    a, b = a0, b0
    b_inf = b < 0
    bet = min_bet
    hist = np.empty(max_rounds, dtype=np.int64)
    rnd = 0

    while a > 0 and (b_inf or b > 0) and rnd < max_rounds:
        actual = min(bet, a) if b_inf else min(bet, a, b)
        broken = actual < bet

        won = np.random.random() < 0.5
        delta = actual if won else -actual
        a += delta
        if not b_inf:
            b -= delta

        hist[rnd] = a - a0
        bet = _next_bet(strategy, actual, won, broken, min_bet, a, a0)
        rnd += 1

    # Pad remaining slots with final profit
    final = a - a0
    for i in range(rnd, max_rounds):
        hist[i] = final
    return hist, rnd


# ---- Parallel helpers ----

_g_strategy = MARTINGALE  # per-worker global, set by the Pool initializer below
                          # (avoids threading the strategy id through every task)


def _init_worker(strategy_id):
    global _g_strategy
    _g_strategy = strategy_id
    _seed_rng(int.from_bytes(os.urandom(4), 'big'))


def parallel_map(worker, tasks, strategy_id, n_workers):
    """Run `worker` over `tasks` across a process pool seeded for `strategy_id`."""
    with Pool(n_workers, _init_worker, (strategy_id,)) as pool:
        return list(pool.imap_unordered(worker, tasks))


class GridCell(NamedTuple):
    a_funds: int
    b_funds: int
    win_prob: float
    avg_rounds: float
    bet_win_rate: float


def _grid_worker(args):
    af, bf, n_trials, min_bet = args
    wins = sum_rounds = sum_bet_wins = 0
    for _ in range(n_trials):
        w, r, bw = simulate(af, bf, _g_strategy, min_bet)
        wins += w
        sum_rounds += r
        sum_bet_wins += bw
    return GridCell(af, bf,
                    wins / n_trials,
                    sum_rounds / n_trials,
                    sum_bet_wins / max(sum_rounds, 1))


def _traj_worker(args):
    a0, b0, min_bet, max_rounds = args
    return simulate_hist(a0, b0, _g_strategy, min_bet, max_rounds)


# ---- Plotting ----

def _heatmap(grid, funds, title, cbar_label, path,
             vmin=None, vmax=None, cmap='viridis', log=False, diag=False):
    ext = [funds[0], funds[-1], funds[0], funds[-1]]
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)

    norm = None
    if log:
        lo = max(grid[grid > 0].min(), 1) if (grid > 0).any() else 1
        norm = mcolors.LogNorm(vmin=lo, vmax=grid.max())

    im = ax.imshow(grid, origin='lower', aspect='auto', extent=ext,
                   vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
    if diag:
        ax.plot([funds[0], funds[-1]], [funds[0], funds[-1]],
                'k--', alpha=0.4, lw=1)
    ax.set_xlabel("B's Funds")
    ax.set_ylabel("A's Funds")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def _smooth2d(arr, sigma=SMOOTH_SIGMA):
    """Separable 2D Gaussian smooth, pure numpy, no scipy."""
    sz = int(3 * sigma + 0.5)
    if sz == 0:
        return arr
    x = np.arange(-sz, sz + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()

    out = np.zeros_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(arr[:, j], k, mode='same')
    tmp = out.copy()
    for i in range(arr.shape[0]):
        out[i, :] = np.convolve(tmp[i, :], k, mode='same')
    return out


def _density_plot(trajectories, max_len, path):
    """2D density heatmap: x = round, y = profit, color = density."""
    n_round_bins = min(N_ROUND_BINS_MAX, max_len)
    round_idx = np.linspace(0, max_len - 1, n_round_bins, dtype=int)

    # Auto-scale y range from data
    y_lo, y_hi = float(trajectories.min()), float(trajectories.max())
    margin = max((y_hi - y_lo) * 0.05, 100)
    edges = np.linspace(y_lo - margin, y_hi + margin, N_PROFIT_BINS + 1)

    density = np.zeros((N_PROFIT_BINS, n_round_bins))
    for j, r in enumerate(round_idx):
        counts, _ = np.histogram(trajectories[:, r], bins=edges)
        total = counts.sum()
        if total > 0:
            density[:, j] = counts / total

    density = _smooth2d(density)

    fig, ax = plt.subplots(figsize=FIGSIZE_DENSITY)
    ax.imshow(density, origin='lower', aspect='auto',
              extent=[0, max_len, edges[0], edges[-1]],
              cmap='inferno')
    ax.axhline(0, color='white', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel('Round')
    ax.set_ylabel("A's Profit")
    ax.set_title('Profit Density Over Rounds')
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


# ---- Figure builders ----

def make_funds(args):
    """Fund grid, linear or log-spaced."""
    if args.log_grid:
        return np.unique(np.geomspace(
            args.fund_min, args.fund_max, args.n_steps).astype(int))
    return np.linspace(args.fund_min, args.fund_max, args.n_steps, dtype=int)


def warmup(strategy_id):
    """Trigger Numba JIT before the Pool forks (compiled code is inherited)."""
    print('JIT warmup ...')
    simulate(100, 100, strategy_id, 1, 100)
    simulate_hist(100, 100, strategy_id, 1, 100)
    _seed_rng(0)


def run_grid(args, strategy_id, funds):
    """Figures 1-3: (A, B) fund-grid heatmaps."""
    n = len(funds)
    tasks = [(int(a), int(b), args.n_trials, args.min_bet)
             for a in funds for b in funds]
    print(f'Grid: {n}x{n} = {n * n} cells, '
          f'{args.n_trials} trials each, {args.n_workers} workers')
    t0 = time.time()
    cells = parallel_map(_grid_worker, tasks, strategy_id, args.n_workers)
    print(f'  done in {time.time() - t0:.1f}s')

    rounds_grid = np.zeros((n, n))
    win_grid = np.zeros((n, n))
    betwr_grid = np.zeros((n, n))
    for c in cells:
        i = np.searchsorted(funds, c.a_funds)
        j = np.searchsorted(funds, c.b_funds)
        rounds_grid[i, j] = c.avg_rounds
        win_grid[i, j] = c.win_prob
        betwr_grid[i, j] = c.bet_win_rate

    out = lambda name: os.path.join(args.output_dir, name)
    _heatmap(rounds_grid, funds, 'Average Game Length', 'Rounds (log scale)',
             out('fig1_avg_rounds.png'), log=True)
    print('Fig 1 saved')
    _heatmap(win_grid, funds, "P(A wins all of B's money)", 'P(A wins)',
             out('fig2_a_win_prob.png'), vmin=0, vmax=1, cmap='RdYlGn', diag=True)
    print('Fig 2 saved')
    _heatmap(betwr_grid, funds, 'Per-Bet Win Rate (fair coin = 0.5)', 'Win Rate',
             out('fig3_bet_win_rate.png'), vmin=0.48, vmax=0.52, cmap='coolwarm')
    print('Fig 3 saved')


def run_trajectory(args, strategy_id):
    """Figure 4: profit density over rounds for a single (A, B)."""
    tasks = [(args.traj_a, args.traj_b, args.min_bet, args.traj_rounds)
             for _ in range(args.traj_trials)]
    b_label = 'INF' if args.traj_b < 0 else str(args.traj_b)
    print(f'Trajectories: {args.traj_trials} trials, '
          f'A={args.traj_a} B={b_label}, max {args.traj_rounds} rounds')
    t0 = time.time()
    results = parallel_map(_traj_worker, tasks, strategy_id, args.n_workers)
    print(f'  done in {time.time() - t0:.1f}s')

    max_len = min(max(r[1] for r in results), args.traj_rounds)
    trajectories = np.array([r[0][:max_len] for r in results])
    _density_plot(trajectories, max_len,
                  os.path.join(args.output_dir, 'fig4_profit_density.png'))
    print('Fig 4 saved')


# ---- Main ----

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', default='./')
    ap.add_argument('--strategy', default='martingale', choices=STRATEGY_MAP,
                    help='Betting strategy')
    ap.add_argument('--n-trials', type=int, default=100,
                    help='Monte Carlo trials per grid cell')
    ap.add_argument('--n-steps', type=int, default=50,
                    help='Grid resolution per axis')
    ap.add_argument('--n-workers', type=int, default=8)
    ap.add_argument('--min-bet', type=int, default=1)
    ap.add_argument('--fund-min', type=int, default=100)
    ap.add_argument('--fund-max', type=int, default=10000)
    ap.add_argument('--log-grid', action='store_true',
                    help='Logarithmic spacing for fund grid')
    ap.add_argument('--traj-rounds', type=int, default=100_000,
                    help='Max rounds per trajectory game')
    ap.add_argument('--traj-trials', type=int, default=100)
    ap.add_argument('--traj-a', type=int, default=10000,
                    help="A's initial funds for trajectory plot")
    ap.add_argument('--traj-b', type=int, default=10000,
                    help="B's initial funds (-1 = infinite, market mode)")
    return ap.parse_args()


def main():
    args = parse_args()
    strategy_id = STRATEGY_MAP[args.strategy]
    os.makedirs(args.output_dir, exist_ok=True)

    funds = make_funds(args)
    warmup(strategy_id)
    run_grid(args, strategy_id, funds)
    run_trajectory(args, strategy_id)

    print(f'\nAll figures saved to {os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
    main()
